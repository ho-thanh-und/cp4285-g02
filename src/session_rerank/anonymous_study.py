"""Run validation tuning, selection, and frozen confirmation for the anonymous study."""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd

from .diversity import MEASURE_DEFINITION, ProductDiversity
from .ranker_features import build_ranker_features
from .reranking import (
    RELEVANCE_LOSS_LIMIT, choose_policy, evaluate_policies, history_context,
    policy_grid, summarize_metrics,
)
from .samples import (
    build_candidates_and_clicks, build_events, build_histories, file_hash,
    load_query_rows, select_sessions,
)
from .study_files import finish_stage, save_table, stage_complete, verify_files, write_json

MODEL_NAME = "query_product_session"
CODE_FILES = (
    "anonymous_study.py", "reranking.py", "diversity.py", "ranker_features.py",
    "samples.py", "core.py", "study_files.py", "study_report.py",
)


def utc_now():
    """Use an explicit UTC timestamp for freeze and outcome-opening records."""
    return datetime.now(timezone.utc).isoformat()


def open_study(dataset_folder, ranker_run, study_folder=None):
    """Create a new study or verify the inputs of an explicitly resumed study."""
    dataset_folder, ranker_run = Path(dataset_folder).resolve(), Path(ranker_run).resolve()
    manifest = json.loads((dataset_folder / "manifest.json").read_text())
    model_run = json.loads((ranker_run / "run.json").read_text())
    if model_run["sample_output_sha256"] != manifest["output_sha256"]:
        raise ValueError("The ranker was trained on different sample tables.")
    code = {name: file_hash(Path(__file__).parent / name) for name in CODE_FILES}
    model_files = {
        name: file_hash(ranker_run / name)
        for name in ("run.json", f"{MODEL_NAME}.txt", "validation_scores.parquet")
    }
    identity = dict(
        dataset_folder=str(dataset_folder), ranker_run=str(ranker_run),
        sample_files=manifest["output_sha256"], model_files=model_files, code_files=code,
    )
    if study_folder is not None:
        folder = Path(study_folder)
        saved = json.loads((folder / "study.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Study inputs or scientific code changed. Resume with the recorded versions.")
    else:
        root = Path("results/anonymous_study")
        root.mkdir(parents=True, exist_ok=True)
        folder = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        folder.mkdir()
        write_json(folder / "study.json", dict(
            created_at_utc=utc_now(), identity=identity, model=MODEL_NAME,
            loss_allowance=RELEVANCE_LOSS_LIMIT, measure=MEASURE_DEFINITION,
            choice_rule="Maximize combined ILD with at most 5% relative NDCG loss and no ILD loss versus relevance. Ties prefer NDCG, simplicity, then ID.",
            bootstrap=dict(seed=4285, replicates=2000, unit="daily session", estimator="query-weighted mean"),
        ))
    print(f"Study folder: {folder.resolve()}", flush=True)
    return folder


def study_inputs(folder):
    """Read the saved paths and recheck model/source identity before each stage."""
    study = json.loads((folder / "study.json").read_text())
    identity = study["identity"]
    dataset = Path(identity["dataset_folder"])
    ranker = Path(identity["ranker_run"])
    verify_files(dataset, identity["sample_files"])
    verify_files(ranker, identity["model_files"])
    verify_files(Path(__file__).parent, identity["code_files"])
    return dataset, ranker, study


def score_split(folder, split, tables=None):
    """Score a later split with the frozen trained tree and training-only statistics."""
    dataset, ranker, _ = study_inputs(folder)
    if split == "confirmation" and not (folder / "frozen_choices.json").exists():
        raise ValueError("Confirmation scoring requires saved frozen choices.")
    started = perf_counter()
    rows = build_ranker_features(dataset, splits=(split,), tables=tables)
    if rows.empty or not rows.split.eq(split).all():
        raise ValueError(f"Expected candidates for {split}.")
    spec = json.loads((ranker / "run.json").read_text())["models"][MODEL_NAME]
    model = lgb.Booster(model_file=str(ranker / f"{MODEL_NAME}.txt"))
    scores = rows[["query_id", "session_key", "item_id", "original_rank", "clicked", "split"]].copy()
    scores["relevance_score"] = model.predict(rows[spec["features"]], num_threads=4)
    write_json(folder / f"{split}_scoring_time.json", dict(
        seconds=perf_counter() - started, queries=int(rows.query_id.nunique()),
        scope="Feature loading, feature construction, model loading, and tree prediction",
    ))
    return scores


def tune_diversification(folder):
    """Choose one setting per family on validation before examining selection outcomes."""
    if stage_complete(folder, "validation"):
        return
    dataset, ranker, _ = study_inputs(folder)
    print("[3/6] Tune quota, fixed-weight, and adaptive diversification on validation", flush=True)
    catalog = ProductDiversity(pd.read_parquet(dataset / "products.parquet"))
    # Quantile thresholds come from training prefixes, never from future outcomes.
    queries = pd.read_parquet(dataset / "queries.parquet", filters=[("split", "==", "train")])
    histories = pd.read_parquet(dataset / "histories.parquet", filters=[("split", "==", "train")])
    train_context = history_context(queries, histories, catalog)
    valid_history = train_context.history_ild.dropna()
    thresholds = sorted(set(valid_history.quantile([0.25, 0.5, 0.75]).tolist())) if len(valid_history) else [0.0]
    segment_threshold = float(valid_history.median()) if len(valid_history) else 0.0
    policies = policy_grid(thresholds)
    initial = [policy for policy in policies if policy["family"] != "adaptive"]
    # This source snapshot is the exact score file reviewed in earlier steps.
    scores = pd.read_parquet(ranker / "validation_scores.parquet")
    scores["relevance_score"] = scores[f"{MODEL_NAME}_score"]
    queries = pd.read_parquet(dataset / "queries.parquet", filters=[("split", "==", "validation")])
    histories = pd.read_parquet(dataset / "histories.parquet", filters=[("split", "==", "validation")])
    contexts = history_context(queries, histories, catalog)
    first = evaluate_policies(scores, contexts, catalog, initial, segment_threshold)
    first_summary = summarize_metrics(first)
    fixed_ids = [p["id"] for p in initial if p["family"] == "fixed"]
    default_id = choose_policy(first_summary, fixed_ids)
    default_weight = next(p["weight"] for p in initial if p["id"] == default_id)
    adaptive = [p for p in policies if p["family"] == "adaptive"]
    for policy in adaptive:
        policy["default_weight"] = default_weight
    # An exact relevance equivalent ensures the adaptive family can always fall back.
    adaptive.append(dict(
        id="adaptive_relevance", family="adaptive", threshold=segment_threshold,
        focused_weight=1.0, broad_weight=1.0, default_weight=1.0,
    ))
    second = evaluate_policies(scores, contexts, catalog, adaptive, segment_threshold)
    metrics = pd.concat([first, second], ignore_index=True)
    summary = summarize_metrics(metrics)
    all_policies = initial + adaptive
    selected_ids = ["relevance"] + [
        choose_policy(summary, [p["id"] for p in all_policies if p["family"] == family])
        for family in ("quota", "fixed", "adaptive")
    ]
    shortlist = [p for p in all_policies if p["id"] in selected_ids]
    save_table(metrics, folder / "validation_by_query.parquet")
    save_table(summary, folder / "validation_grid.csv")
    save_table(train_context, folder / "training_history_diversity.csv")
    # The shortlist and deterministic choice rule are committed to disk before
    # selection scores are built. Selection is allowed to choose only among these.
    write_json(folder / "shortlist.json", dict(
        policies=shortlist, search_grid=all_policies, segment_threshold=segment_threshold,
        threshold_candidates=thresholds, default_weight=default_weight,
        relevance_loss_limit=RELEVANCE_LOSS_LIMIT,
    ))
    finish_stage(folder, "validation", [
        "validation_by_query.parquet", "validation_grid.csv",
        "training_history_diversity.csv", "shortlist.json",
    ])


def compare_on_selection(folder):
    """Compare the four validation-nominated settings and select the final policy."""
    if not stage_complete(folder, "validation"):
        raise ValueError("Tune validation and save the shortlist first.")
    if stage_complete(folder, "selection"):
        return
    dataset, _, _ = study_inputs(folder)
    print("[4/6] Compare the fixed shortlist on reserved selection queries", flush=True)
    shortlist = json.loads((folder / "shortlist.json").read_text())
    scores = score_split(folder, "selection")
    catalog = ProductDiversity(pd.read_parquet(dataset / "products.parquet"))
    queries = pd.read_parquet(dataset / "queries.parquet", filters=[("split", "==", "selection")])
    histories = pd.read_parquet(dataset / "histories.parquet", filters=[("split", "==", "selection")])
    contexts = history_context(queries, histories, catalog)
    metrics = evaluate_policies(scores, contexts, catalog, shortlist["policies"], shortlist["segment_threshold"])
    summary = summarize_metrics(metrics, segments=True)
    winner = choose_policy(summary, [p["id"] for p in shortlist["policies"]])
    save_table(scores, folder / "selection_scores.parquet")
    save_table(metrics, folder / "selection_by_query.parquet")
    save_table(summary, folder / "selection_metrics.csv")
    write_json(folder / "selection_choice.json", dict(
        winner=winner, shortlist_sha256=file_hash(folder / "shortlist.json"),
        chosen_at_utc=utc_now(),
    ))
    finish_stage(folder, "selection", [
        "selection_scores.parquet", "selection_by_query.parquet", "selection_metrics.csv",
        "selection_choice.json", "selection_scoring_time.json",
    ])


def load_confirmation(folder):
    """Materialize only the exact reserved sessions after the freeze has been saved."""
    dataset, _, study = study_inputs(folder)
    frozen = json.loads((folder / "frozen_choices.json").read_text())
    if frozen["study_sha256"] != file_hash(folder / "study.json"):
        raise ValueError("The study definition changed after the freeze.")
    manifest = json.loads((dataset / "manifest.json").read_text())
    protocol = manifest["protocol"]
    raw = Path(protocol["raw_directory"])
    verify_files(raw, manifest["source_sha256"])
    verify_files(dataset, manifest["output_sha256"])
    selected, metadata = select_sessions(raw, protocol)
    metadata = metadata[metadata.split.eq("confirmation")]
    selected = selected[selected.split.eq("confirmation")]
    reserved = pd.read_parquet(dataset / "confirmation_ids.parquet")
    expected = metadata.rename(columns={"queryId": "query_id", "timeframe": "query_timeframe_ms"})
    keys = ["query_id", "session_key", "eventdate", "query_timeframe_ms"]
    pd.testing.assert_frame_equal(
        reserved[keys].sort_values("query_id").reset_index(drop=True),
        expected[keys].sort_values("query_id").reset_index(drop=True),
    )
    known_sessions = pd.read_parquet(dataset / "sessions.parquet")
    visible = known_sessions[~known_sessions.split.eq("confirmation")]
    if set(selected.session_key) & set(visible.session_key):
        raise ValueError("Confirmation sessions overlap development sessions.")
    if (folder / "confirmation_opened.json").exists():
        opened = json.loads((folder / "confirmation_opened.json").read_text())
        if opened["frozen_sha256"] != file_hash(folder / "frozen_choices.json"):
            raise ValueError("Frozen choices changed after confirmation was opened.")
    else:
        # The opening timestamp is written before reading candidate lists or clicks.
        write_json(folder / "confirmation_opened.json", dict(
            opened_at_utc=utc_now(), frozen_sha256=file_hash(folder / "frozen_choices.json"),
        ))
    source = load_query_rows(raw, metadata)
    candidates, clicks = build_candidates_and_clicks(raw, source)
    events = build_events(raw, source, clicks, selected, include_confirmation=True)
    queries = source.rename(columns={
        "queryId": "query_id", "timeframe": "query_timeframe_ms",
        "searchstring.tokens": "query_tokens", "categoryId": "query_category_id",
    })[["query_id", "session_key", "split", "eventdate", "query_timeframe_ms", "query_tokens", "query_category_id"]]
    queries["query_tokens"] = queries.query_tokens.fillna("")
    queries["candidate_count"] = queries.query_id.map(candidates.groupby("query_id").size()).astype("int64")
    queries, histories = build_histories(queries, events, protocol["history_max_events"])
    checked = histories.merge(queries[["query_id", "session_key"]], on="query_id", suffixes=("", "_query"))
    if not checked.session_key.eq(checked.session_key_query).all() or not checked.age_ms.gt(0).all():
        raise ValueError("Confirmation history violates session or chronological boundaries.")
    if checked.source_query_id.eq(checked.query_id).fillna(False).any():
        raise ValueError("A confirmation query's own outcome entered its history.")
    tables = dict(candidates=candidates, clicks=clicks, events=events, queries=queries, histories=histories)
    output = folder / "confirmation_tables"
    output.mkdir(exist_ok=True)
    for name, frame in tables.items():
        save_table(frame, output / f"{name}.parquet")
    write_json(output / "audit.json", dict(
        queries=len(queries), sessions=int(queries.session_key.nunique()),
        candidates=len(candidates), clicks=len(clicks),
        click_rank_status=clicks.rank_match_status.value_counts().to_dict(),
        zero_click_queries=int((~candidates.groupby("query_id").clicked.any()).sum()),
        source_verified=True, reserved_ids_verified=True, causal_histories_verified=True,
        frozen_sha256=file_hash(folder / "frozen_choices.json"),
    ))
    return tables


def freeze_and_evaluate(folder):
    """Write immutable choices first, then evaluate every frozen approach on confirmation."""
    if not stage_complete(folder, "selection"):
        raise ValueError("Selection comparison must be completed before freezing.")
    if stage_complete(folder, "confirmation"):
        return
    dataset, _, _ = study_inputs(folder)
    print("[5/6] Freeze choices and evaluate reserved confirmation queries", flush=True)
    freeze_path = folder / "frozen_choices.json"
    if not freeze_path.exists():
        write_json(freeze_path, dict(
            frozen_at_utc=utc_now(), study_sha256=file_hash(folder / "study.json"),
            shortlist_sha256=file_hash(folder / "shortlist.json"),
            selection_choice_sha256=file_hash(folder / "selection_choice.json"),
            shortlist=json.loads((folder / "shortlist.json").read_text()),
            selection_choice=json.loads((folder / "selection_choice.json").read_text()),
        ))
    frozen = json.loads(freeze_path.read_text())
    for name, key in (
        ("shortlist.json", "shortlist_sha256"), ("selection_choice.json", "selection_choice_sha256"),
    ):
        if file_hash(folder / name) != frozen[key]:
            raise ValueError("Choices changed after they were frozen.")
    tables = load_confirmation(folder)
    scores = score_split(folder, "confirmation", tables)
    catalog = ProductDiversity(pd.read_parquet(dataset / "products.parquet"))
    contexts = history_context(tables["queries"], tables["histories"], catalog)
    policies = frozen["shortlist"]["policies"] + [dict(id="original_order", family="original")]
    metrics = evaluate_policies(scores, contexts, catalog, policies, frozen["shortlist"]["segment_threshold"])
    save_table(scores, folder / "confirmation_scores.parquet")
    save_table(contexts, folder / "confirmation_context.parquet")
    save_table(metrics, folder / "confirmation_by_query.parquet")
    save_table(summarize_metrics(metrics, segments=True), folder / "confirmation_metrics.csv")
    table_files = [str(path.relative_to(folder)) for path in (folder / "confirmation_tables").iterdir()]
    finish_stage(folder, "confirmation", [
        "frozen_choices.json", "confirmation_opened.json", "confirmation_scores.parquet",
        "confirmation_context.parquet", "confirmation_by_query.parquet", "confirmation_metrics.csv",
        "confirmation_scoring_time.json", *table_files,
    ])
