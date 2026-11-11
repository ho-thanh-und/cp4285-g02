"""Phase 5 (category pages): freeze the choices, then score untouched held-out queries.

Order of operations, each step recorded before the next:

1. Freeze. Write frozen_choices.json with every choice made on validation: data
   choices, the lead ranker (popularity features + rank_xendcg, 50k queries),
   the leaf/min-rows setting and tree count of each model's selected trial,
   item-kNN settings, the metrics and the selection rule. Nothing about the
   held-out split has been read at this point.
2. Retrain the frozen LightGBM models with exactly that setting and tree count.
   Check: on validation they must reproduce the Phase 2 per-query scores exactly,
   which proves the frozen model is the one chosen.
3. Build the held-out split with the pipeline's own sampling (select_sessions)
   and database code (create_database). Check: its query IDs must equal the
   pilot's reserved IDs for that split.
4. Features (clicks only, end-of-training popularity, window popularity rates)
   and click labels (clicks strictly after the query on a logged product),
   exactly as for validation.
5. Evaluate site order, popularity sort, item-kNN and both LightGBM models;
   apply the declared selection rule (higher category-page click NDCG@10).

Splits: selection (the declared rule chooses between the two frozen LightGBM
models) and confirmation (opened once, after every page is frozen; it needs
frozen_selection_run, and the model chosen there is the reported model; the
rule is not applied again).
"""

import gc
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .event_ablation import (
    MODELS, filter_events, lightgbm_dataset, per_query_metrics, popularity_snapshots, ranker_params,
    session_bootstrap, training_queries,
)
from .expanded_data import FEATURES, PURCHASE_FEATURES, create_database, query_features
from .expanded_training import training_settings
from .item_knn import KnnData, read_knn_settings
from .ranker_phase2 import ALL_FEATURES, Phase2Data, read_phase2_settings
from .ranking_settings import top_k
from .samples import select_sessions
from .study_files import save_table, write_json
from .windows import check_not_opened

METRICS = ["click_ndcg", "recall_at_k", "mrr"]
POPULARITY_COLUMNS = [ALL_FEATURES.index("window_click_rate"), ALL_FEATURES.index("window_purchase_rate")]
OPENABLE_SPLITS = ("selection", "confirmation")


def read_heldout_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if settings["split"] not in OPENABLE_SPLITS:
        raise ValueError(f"split must be one of {OPENABLE_SPLITS}")
    if settings["split"] == "confirmation":
        choice = Path(settings.get("frozen_selection_run") or "missing") / "selection_choice.json"
        if not choice.exists():
            raise ValueError("The confirmation split needs frozen_selection_run: the selection run whose "
                             "selection_choice.json fixes the model")
    if settings["bootstrap_replicates"] < 1:
        raise ValueError("bootstrap_replicates must be positive")
    return settings


def git_state(folder):
    """Commit and whether the working tree had uncommitted changes, for the freeze record."""
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return dict(commit=run("rev-parse", "HEAD"), uncommitted_changes=None if status is None else bool(status))


def frozen_models(phase2_run, phase2_settings):
    """Selected grid trial of each model in the frozen Phase 2 run."""
    audit = json.loads((Path(phase2_run) / "audit.json").read_text())
    variant = phase2_settings["variants"][0]
    if len(phase2_settings["variants"]) != 1:
        raise ValueError("The frozen Phase 2 config must list exactly one variant")
    models = {}
    for entry in audit:
        if entry["variant"] == variant["id"]:
            models[entry["model"]] = dict(leaves=entry["selected_leaves"], min_rows=entry["selected_min_rows"],
                                          iterations=entry["selected_iterations"])
    return variant, models


def split_database(config, split, path):
    """Rebuild one reserved split with the pipeline's sampling and database code."""
    pilot = Path(config["pilot_directory"])
    protocol = json.loads((pilot / "manifest.json").read_text())["protocol"]
    raw = Path(protocol["raw_directory"])
    _, meta = select_sessions(raw, protocol)
    meta = meta[meta.split.eq(split)]
    if split == "confirmation":
        # The pilot keeps its confirmation reservation in its own file, apart from the opened splits.
        reserved = set(pd.read_parquet(pilot / "confirmation_ids.parquet").query_id)
    else:
        reserved = pd.read_parquet(pilot / "queries.parquet")
        reserved = set(reserved.loc[reserved.split.eq(split), "query_id"])
    if set(meta.queryId) != reserved:
        raise ValueError(f"Rebuilt {split} query IDs differ from the pilot's reserved IDs")
    create_database(raw, meta, path)
    return len(meta)


def split_features(data, path):
    """Features, candidates and click labels of a held-out split, as built for validation."""
    inputs = data.inputs
    train_events = filter_events(inputs.events, "clicks_only", 60_000)
    _, counters = popularity_snapshots(train_events, inputs.train_sessions, inputs.training_dates)
    with sqlite3.connect(path) as db:
        queries = db.execute("SELECT * FROM queries ORDER BY eventdate,session_key,timeframe,query_id").fetchall()
        events = pd.read_sql("""SELECT event_id,session_key,eventdate,kind,timeframe,query_id,item_id
                                FROM events ORDER BY timeframe,event_id""", db)
    kept = filter_events(events, "clicks_only", 60_000)
    by_session = {}
    for row in kept[["session_key", "timeframe", "kind", "item_id", "query_id"]].itertuples(index=False):
        by_session.setdefault(row.session_key, []).append(
            (row.timeframe, row.kind, None if pd.isna(row.item_id) else int(row.item_id),
             None if pd.isna(row.query_id) else int(row.query_id)))
    clicks = events[events.kind.eq("click")]
    later = {}
    times = {q[0]: q[4] for q in queries}
    for row in clicks.itertuples(index=False):
        qid = int(row.query_id)
        if row.timeframe > times[qid]:
            later.setdefault(qid, set()).add(int(row.item_id))
    width = len(FEATURES) + len(PURCHASE_FEATURES)
    blocks, frames = [], []
    for query in queries:
        qid, session = query[0], query[1]
        found, block, _, _ = query_features(query, by_session.get(session, []), inputs.catalog, counters,
                                            inputs.history_limit)
        extra = np.zeros((len(found), len(POPULARITY_COLUMNS)), dtype=np.float32)
        index = data.item_index.reindex(found).to_numpy()
        known = ~np.isnan(index)
        extra[known] = data.rates[data.days, index[known].astype(int)]
        blocks.append(np.hstack([block[:, :width], extra]))
        frames.append(pd.DataFrame(dict(query_id=qid, session_key=session, item_id=found,
                                        original_rank=np.arange(1, len(found) + 1),
                                        clicked=[item in later.get(qid, ()) for item in found], purchased=False)))
    matrix = np.vstack(blocks)
    candidates = pd.concat(frames, ignore_index=True)
    query_table = pd.DataFrame([dict(query_id=q[0], session_key=q[1], timeframe=q[4], category=q[6])
                                for q in queries]).set_index("query_id")
    click_table = clicks[["session_key", "timeframe", "item_id", "query_id"]]
    return matrix, candidates, query_table, click_table


def train_frozen(data, variant, models, training, config):
    """Retrain each frozen model with its selected setting and tree count (no early stopping)."""
    boosters = {}
    folder = data.inputs.folder
    train_category = data.query_type["train"] == "category"
    for model, spec in models.items():
        objective, indices = MODELS[model]
        indices = list(indices) + (POPULARITY_COLUMNS if variant["popularity"] else [])
        label_file = "click_labels.npy" if objective == "click" else "grades.npy"
        labels = np.load(folder / "train_features" / label_file)
        mask = training_queries(data.meta["train"], objective, "all_sessions", 1) & train_category
        dataset = lightgbm_dataset(data.matrices["train"], labels, data.meta["train"], mask, indices,
                                   config["seed"], config["threads"], training["max_bin"],
                                   feature_names=ALL_FEATURES)
        fit_gains = [0, 1] if objective == "click" else config["label_gains"]
        params = ranker_params(variant["objective"], fit_gains, training | dict(learning_rate=variant["learning_rate"]),
                               spec["leaves"], spec["min_rows"], config["seed"], config["threads"])
        print(f"[freeze] Retrain {model}: {spec['leaves']} leaves, min rows {spec['min_rows']}, "
              f"{spec['iterations']} trees", flush=True)
        boosters[model] = (lgb.train(params, dataset, num_boost_round=spec["iterations"]), indices)
        del dataset
        gc.collect()
    return boosters


def compare(per_query, segments, settings):
    rng = np.random.default_rng(settings["seed"])
    rows = []
    for reference in ("popularity_sort", "item_knn"):
        base = per_query[per_query.method.eq(reference)].set_index("query_id")
        for method, frame in per_query.groupby("method", sort=False):
            if method == reference:
                continue
            frame = frame.set_index("query_id").reindex(base.index)
            for name, select in segments.items():
                mask = select(base).to_numpy(bool)
                for metric in METRICS:
                    diff = (frame[metric] - base[metric])[mask]
                    low, high = session_bootstrap(diff, base.session_key[mask], settings["bootstrap_replicates"], rng)
                    rows.append(dict(method=method, reference=reference, segment=name, metric=metric,
                                     value=float(frame[metric][mask].mean()),
                                     reference_value=float(base[metric][mask].mean()),
                                     difference=float(diff.mean()), ci_low=low, ci_high=high,
                                     queries=int(diff.notna().sum())))
    return pd.DataFrame(rows)


def run_heldout(settings_path=Path("config/heldout.json"), output_root=None):
    settings = read_heldout_settings(settings_path)
    phase2_settings = read_phase2_settings(settings["phase2_config"])
    knn_settings = read_knn_settings(settings["knn_config"])
    variant, models = frozen_models(settings["phase2_run"], phase2_settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    check_not_opened(Path(output_root or settings["output_directory"]), settings["split"])
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{settings['split']}"
    folder.mkdir(parents=True)
    confirmation = settings["split"] == "confirmation"
    chosen = (json.loads((Path(settings["frozen_selection_run"]) / "selection_choice.json").read_text())["chosen"]
              if confirmation else None)

    # 1. Freeze before anything about the held-out split is read.
    write_json(folder / "frozen_choices.json", dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), code=git_state(Path.cwd()),
        page="category pages", trained_run=settings["trained_run"], phase2_run=settings["phase2_run"],
        data_choices=dict(event_variant=phase2_settings["event_variant"], training_sessions="all",
                          exclude_text_search_from_training=phase2_settings["exclude_text_search_from_training"]),
        ranker=dict(variant=variant, models=models), item_knn=knn_settings | dict(compare_with=[]),
        metrics=["click NDCG@10", "Recall@10", "MRR@10"], primary_segment="category pages",
        selection_rule=("Choose between click_session and purchase_session by higher category-page click "
                        "NDCG@10 on the selection split; ties go to click_session. Report every method."),
        heldout_split=settings["split"],
        chosen_model=chosen, selection_run=settings.get("frozen_selection_run"),
    ))
    print(f"[freeze] Choices written to {folder / 'frozen_choices.json'}", flush=True)

    # 2. Retrain the frozen models and check they reproduce validation exactly.
    data = Phase2Data(settings["trained_run"], phase2_settings)
    config = data.inputs.config
    k = top_k(config)
    boosters = train_frozen(data, variant, models, training_settings(config), config)
    phase2 = pd.read_parquet(Path(settings["phase2_run"]) / "validation_per_query.parquet")
    from .expanded_data import read_candidates
    validation = read_candidates(data.inputs.folder, "validation")
    for model, (booster, indices) in boosters.items():
        booster.save_model(str(folder / f"{model}.txt"))
        measured = per_query_metrics(validation, booster.predict(data.matrices["validation"][:, indices]),
                                     config["label_gains"], k).set_index("query_id")
        expected = phase2[phase2.variant.eq(variant["id"]) & phase2.model.eq(model)
                          & phase2.statistic.eq("selected_trial")].set_index("query_id")
        if not np.allclose(measured.click_ndcg.reindex(expected.index), expected.click_ndcg, equal_nan=True):
            raise ValueError(f"Frozen {model} does not reproduce its validation scores")
    print("[freeze] Frozen models reproduce their validation scores exactly", flush=True)
    del data.matrices["train"]
    gc.collect()

    # 3–4. Open the held-out split.
    database = folder / f"{settings['split']}.sqlite"
    count = split_database(config, settings["split"], database)
    print(f"[{settings['split']}] {count:,} queries rebuilt; IDs match the pilot's reservation", flush=True)
    matrix, candidates, query_table, click_table = split_features(data, database)

    # 5. Score every method on identical queries.
    knn = KnnData(settings["trained_run"], knn_settings)
    knn.use_queries(query_table, click_table)
    knn_scores, _, history = knn.scores(candidates)
    methods = {"site_order": -candidates.original_rank.to_numpy(float),
               "popularity_sort": matrix[:, POPULARITY_COLUMNS[0]].astype(float),
               "item_knn": knn_scores}
    for model, (booster, indices) in boosters.items():
        methods[f"lightgbm_{model}"] = booster.predict(matrix[:, indices])
    query_type = query_table.category.fillna(0).map(lambda c: "text" if c == 0 else "category")
    rows = [per_query_metrics(candidates, score, config["label_gains"], k).assign(method=name)
            for name, score in methods.items()]
    per_query = pd.concat(rows, ignore_index=True)
    per_query["query_type"] = per_query.query_id.map(query_type).to_numpy()
    per_query["has_history"] = per_query.query_id.map(lambda q: history.get(q, 0) > 0).to_numpy()
    segments = {
        "category_pages": lambda f: f.query_type.eq("category"),
        "category_with_history": lambda f: f.query_type.eq("category") & f.has_history,
        "category_no_history": lambda f: f.query_type.eq("category") & ~f.has_history.astype(bool),
        "all_queries": lambda f: pd.Series(True, index=f.index),
    }
    summary = pd.concat([per_query[select(per_query)].groupby("method", sort=False)[METRICS].mean()
                         .join(per_query[select(per_query)].groupby("method", sort=False)
                               .click_ndcg.count().rename("queries"))
                         .reset_index().assign(segment=name) for name, select in segments.items()],
                        ignore_index=True)
    category = summary[summary.segment.eq("category_pages")].set_index("method").click_ndcg
    if confirmation:
        # The model was chosen on the selection split; only report it here.
        write_json(folder / "confirmation_model.json", dict(
            chosen=chosen, chosen_on=settings["frozen_selection_run"],
            click_session=float(category["lightgbm_click_session"]),
            purchase_session=float(category["lightgbm_purchase_session"])))
    else:
        chosen = ("purchase_session" if category["lightgbm_purchase_session"] > category["lightgbm_click_session"]
                  else "click_session")
        write_json(folder / "selection_choice.json", dict(
            rule="higher category-page click NDCG@10 on the selection split; ties to click_session",
            click_session=float(category["lightgbm_click_session"]),
            purchase_session=float(category["lightgbm_purchase_session"]), chosen=chosen))
    save_table(candidates, folder / f"{settings['split']}_candidates.parquet")
    save_table(per_query, folder / f"{settings['split']}_per_query.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(compare(per_query, segments, settings), folder / "differences.csv")
    write_json(folder / "audit.json", dict(
        split=settings["split"], queries=int(candidates.query_id.nunique()),
        click_queries=int(candidates.groupby("query_id").clicked.any().sum()),
        candidate_rows=len(candidates), sessions=int(candidates.session_key.nunique()),
        queries_with_history=int(sum(v > 0 for v in history.values())), knn=knn.stats))
    print(f"Held-out {settings['split']} evaluation saved to {folder}; model: {chosen}", flush=True)
    return folder
