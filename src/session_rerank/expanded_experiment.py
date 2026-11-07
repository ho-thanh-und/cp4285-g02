"""Orchestrate the expanded study with explicit stage checkpoints and frozen choices."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from .adaptive_policies import expanded_order, relevance_weight, relevance_metrics, weighted_history_context
from .diversity import ProductDiversity, MEASURE_DEFINITION
from .expanded_data import read_histories
from .expanded_training import score_expanded_split
from .reranking import policy_grid, choose_policy as choose_pilot_policy
from .ranking_settings import top_k
from .study_files import write_json, save_table, verify_files, stage_complete, finish_stage
from .samples import file_hash

SOURCES = (
    "expanded_data.py", "expanded_training.py", "adaptive_policies.py", "expanded_experiment.py",
    "expanded_report.py", "core.py", "diversity.py", "samples.py", "study_files.py",
    "ranker_features.py", "reranking.py", "train_rankers.py", "study_report.py", "learning_curve.py", "ranking_settings.py", "configured_reranking.py",
)


def choose_policy(summary, policy_ids):
    """Adapt the pilot helper's column name to the actual configured cutoff.

    The numeric endpoint is NDCG at config.top_k. Only this local compatibility
    adapter uses the pilot's historical column name; saved metrics say at_k.
    """
    return choose_pilot_policy(summary.rename(columns={"ndcg_at_k": "ndcg_at_10"}), policy_ids)


def open_expanded_run(config_path=None, resume=None, *, config=None, run_name=None):
    """Record the exact configuration and scientific source versions for a run."""
    if resume is not None:
        folder = Path(resume)
        saved = json.loads((folder / "expanded_run.json").read_text())
        verify_files(Path(__file__).parent, saved["source_sha256"])
        return saved["config"], folder
    config = json.loads(Path(config_path).read_text()) if config is None else dict(config)
    from .expanded_training import training_settings
    training_settings(config)
    if config["query_batch_size"] < 1 or (not config.get("validation_only", False) and config["confirmation_target_queries"] < 1):
        raise ValueError("Batch and confirmation sizes must be positive.")
    if not config.get("validation_only", False) and top_k(config) < 2:
        raise ValueError("The expanded diversification study needs top_k >= 2 to define pairwise ILD")
    if config["label_gains"] != [0, 1, 3] or (not config.get("validation_only", False) and config["relevance_loss_limit"] != .05):
        raise ValueError("This declared study uses gains [0,1,3] and a 5% relevance loss allowance.")
    root = Path(config["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    folder = root / (run_name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
    folder.mkdir()
    write_json(folder / "expanded_run.json", dict(
        config=config, created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: file_hash(Path(__file__).parent / name) for name in SOURCES},
        diversity=MEASURE_DEFINITION | {"top_k": top_k(config), "ild": f"Mean known pair distance in top {top_k(config)}"},
        attribution="Unique latest preceding same-session query containing the purchased candidate",
        exponential="Normalized exponential with common alpha_min/alpha_max endpoints",
    ))
    print(f"Expanded run: {folder.resolve()}", flush=True)
    return config, folder


def context_variants(config, folder, split):
    """Cache no-decay and time-decayed histories independently from relevance scoring."""
    query_info = pd.read_parquet(folder / f"{split}_features" / "queries.parquet")
    histories = read_histories(folder, split)
    catalog = ProductDiversity(pd.read_parquet(folder / "products.parquet"))
    contexts = {"none": weighted_history_context(query_info, histories, catalog)}
    for kind in ("linear", "exponential"):
        for minutes in config["history_windows_minutes"]:
            contexts[f"{kind}_{minutes}"] = weighted_history_context(
                query_info, histories, catalog, kind, minutes,
            )
    return contexts, catalog


def expanded_evaluate(config, folder, split, policies):
    """Evaluate graded relevance, purchase retrieval and diversity on the same queries."""
    k = top_k(config)
    scores = score_expanded_split(config, folder, split)
    contexts, catalog = context_variants(config, folder, split)
    primary = json.loads((folder / "objective.json").read_text())["primary"]
    lookup = {key: frame.set_index("query_id") for key, frame in contexts.items()}
    records = []
    for number, (qid, group) in enumerate(scores.groupby("query_id", sort=False), start=1):
        items = group.item_id.to_numpy(dtype=int)
        prediction = group.relevance_score.to_numpy()
        clicked, purchased = group.clicked.to_numpy(), group.purchased.to_numpy()
        started = perf_counter()
        # Catalog omissions remain unknown metadata with explicit metric coverage.
        for item in items:
            if item not in catalog.name_tokens:
                catalog.name_tokens[item] = frozenset()
                catalog.categories[item] = None
        matrices = catalog.distances(items)
        pairwise_ms = (perf_counter()-started)*1000
        reward = np.nan_to_num(matrices["combined"], nan=0)
        for policy in policies:
            context = lookup[policy.get("history_decay", "none")].loc[qid]
            history_ild = float(context.history_ild)
            started = perf_counter()
            order = (np.arange(min(k, len(items))) if policy["family"] == "original"
                     else expanded_order(prediction, reward, policy, history_ild, k))
            rerank_ms = (perf_counter()-started)*1000
            relevance = relevance_metrics(order, clicked, purchased, config["label_gains"], k)
            pairs = np.triu_indices(len(order), 1)
            diversity = {}
            for name, matrix in matrices.items():
                values = matrix[np.ix_(order,order)][pairs]
                known = values[np.isfinite(values)]
                diversity[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
                diversity[f"{name}_coverage"] = len(known)/len(values) if len(values) else np.nan
            records.append(dict(
                query_id=int(qid), session_key=group.session_key.iloc[0], policy_id=policy["id"],
                top_k=k, family=policy["family"], history_decay=policy.get("history_decay","none"),
                history_ild=history_ild, alpha=relevance_weight(policy,history_ild),
                history_unique_items=int(context.history_unique_items),
                history_items=context.history_items, ndcg_at_k=relevance[primary],
                clicked_candidates=int(clicked.sum()), purchased_candidates=int(purchased.sum()),
                topk_item_ids=json.dumps(items[order].tolist()),
                clicked_item_ids=json.dumps(items[clicked].tolist()),
                purchased_item_ids=json.dumps(items[purchased].tolist()),
                pairwise_ms=pairwise_ms, rerank_ms=rerank_ms, **relevance, **diversity,
            ))
        if number % 250 == 0:
            print(f"{split}: evaluated {number:,} queries across {len(policies)} policies", flush=True)
    return pd.DataFrame(records), contexts


def expanded_summary(frame):
    """Report each endpoint with its actual eligible-query denominator."""
    metrics = ["ndcg_at_k","click_ndcg","gain_ndcg","recall_at_k","purchase_recall",
               "token_ild","category_ild","combined_ild","combined_coverage","rerank_ms","pairwise_ms"]
    means = frame.groupby(["policy_id","family"],sort=False)[metrics].mean().reset_index()
    counts = frame.groupby("policy_id",sort=False).agg(
        queries=("query_id","size"), relevance_queries=("ndcg_at_k","count"),
        purchase_queries=("purchase_recall","count"), diversity_queries=("combined_ild","count"),
    ).reset_index()
    return means.merge(counts,on="policy_id").assign(segment="all",top_k=int(frame.top_k.iloc[0]))


def tune_expanded(config, folder):
    """Tune continuous adaptation first, then isolate history-decay effects."""
    if stage_complete(folder, "tuning"):
        return
    print("[4/7] Tune threshold, linear, exponential and time-decay policies",flush=True)
    train_queries = pd.read_parquet(folder / "train_features" / "queries.parquet")
    train_history = read_histories(folder, "train")
    catalog = ProductDiversity(pd.read_parquet(folder / "products.parquet"))
    train_context = weighted_history_context(train_queries, train_history, catalog)
    save_table(train_context, folder / "training_history_context.parquet")
    values = train_context.history_ild.dropna()
    thresholds = sorted(set(values.quantile([.25,.5,.75]).tolist())) if len(values) else [0.]
    initial = [p for p in policy_grid(thresholds) if p["family"] != "adaptive"]
    # Express the pilot quota fractions at this K; deduplicate rounded quotas.
    converted = {}
    for policy in initial:
        if policy["family"] == "quota":
            policy["keep"] = round(policy["keep"] / 10 * top_k(config))
            policy["id"] = f"quota_{policy['keep']}"
        converted[policy["id"]] = policy
    initial = list(converted.values())
    first, _ = expanded_evaluate(config,folder,"validation",initial)
    summary = expanded_summary(first)
    default_id = choose_policy(summary,[p["id"] for p in initial if p["family"]=="fixed"])
    default = next(p["weight"] for p in initial if p["id"]==default_id)
    adaptive = [p for p in policy_grid(thresholds) if p["family"]=="adaptive"]
    for p in adaptive:
        p["default_weight"]=default
    for low,high in config["alpha_bounds"] + [[1.,1.]]:
        for family in ("linear","exponential"):
            betas = config["exponential_beta"] if family=="exponential" else [0]
            for beta in betas:
                adaptive.append(dict(
                    id=f"{family}_{low}_{high}_{beta}",family=family,
                    alpha_min=low,alpha_max=high,beta=beta,
                    default_weight=1. if low==high==1 else default,
                ))
    adaptive.append(dict(id="adaptive_relevance",family="adaptive",threshold=0.,
                         focused_weight=1.,broad_weight=1.,default_weight=1.))
    second, _ = expanded_evaluate(config,folder,"validation",adaptive)
    combined = pd.concat([first,second],ignore_index=True)
    summary = expanded_summary(combined)
    all_policies=initial+adaptive
    nominees=["relevance"]
    for family in ("quota","fixed","adaptive","linear","exponential"):
        nominees.append(choose_policy(summary,[p["id"] for p in all_policies if p["family"]==family]))
    # Hold each family's no-decay winner fixed for the recency ablation.
    decay_policies=[]
    for policy in all_policies:
        if policy["id"] in nominees and policy["family"] in ("adaptive","linear","exponential"):
            for kind in ("linear","exponential"):
                for minutes in config["history_windows_minutes"]:
                    variant=dict(policy)
                    variant["history_decay"]=f"{kind}_{minutes}"
                    variant["id"]=policy["id"]+"__"+variant["history_decay"]
                    decay_policies.append(variant)
    third,_=expanded_evaluate(config,folder,"validation",decay_policies)
    combined=pd.concat([combined,third],ignore_index=True)
    summary=expanded_summary(combined)
    all_policies += decay_policies
    # Keep the no-decay nominee and one best time-decay variant per adaptive family.
    shortlist=[p for p in all_policies if p["id"] in nominees]
    for family in ("adaptive","linear","exponential"):
        choices=[p["id"] for p in all_policies if p["family"]==family and
                 (p["id"] in nominees or "history_decay" in p)]
        chosen=choose_policy(summary,choices)
        if chosen not in {p["id"] for p in shortlist}:
            shortlist.append(next(p for p in all_policies if p["id"]==chosen))
    save_table(combined,folder/"validation_policies.parquet")
    save_table(summary,folder/"validation_grid.csv")
    write_json(folder/"shortlist.json",dict(policies=shortlist,grid=all_policies,
        default_weight=default,thresholds=thresholds,
        segment_threshold=float(values.median()) if len(values) else 0.))
    finish_stage(folder,"tuning",["validation_policies.parquet","validation_grid.csv","shortlist.json"])


def select_expanded(config,folder):
    """Choose among validation nominees before any fresh confirmation outcomes."""
    if not stage_complete(folder,"tuning"):
        raise ValueError("Tune validation before selection.")
    if stage_complete(folder,"selection"):
        return
    print("[5/7] Compare the fixed shortlist on selection",flush=True)
    policies=json.loads((folder/"shortlist.json").read_text())["policies"]
    metrics,_=expanded_evaluate(config,folder,"selection",policies)
    summary=expanded_summary(metrics)
    winner=choose_policy(summary,[p["id"] for p in policies])
    save_table(metrics,folder/"selection_policies.parquet")
    save_table(summary,folder/"selection_metrics.csv")
    write_json(folder/"selection_choice.json",dict(winner=winner,
        shortlist_sha256=file_hash(folder/"shortlist.json")))
    finish_stage(folder,"selection",["selection_policies.parquet","selection_metrics.csv","selection_choice.json"])


def freeze_expanded(folder):
    """Record choices, model hashes and completed-stage hashes before final outcomes."""
    if not stage_complete(folder,"selection"):
        raise ValueError("Complete selection before freezing.")
    identity={
        name:file_hash(folder/name) for name in (
            "expanded_run.json","objective.json","shortlist.json","selected_model.json",
            "selection_choice.json","data_complete.json","features_complete.json","training_complete.json",
            "selection_complete.json")
    }
    selected=json.loads((folder/"selected_model.json").read_text())["model"]
    identity[selected+".txt"]=file_hash(folder/(selected+".txt"))
    path=folder/"frozen_choices.json"
    if path.exists():
        frozen=json.loads(path.read_text())
        if frozen["files"]!=identity:
            raise ValueError("Frozen model, data, or choices changed.")
    else:
        write_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),files=identity,
                            policies=json.loads((folder/"shortlist.json").read_text())["policies"],
                            winner=json.loads((folder/"selection_choice.json").read_text())["winner"]))
    return json.loads(path.read_text())


def evaluate_expanded_confirmation(config,folder):
    """Evaluate exactly the frozen policies, preserving every measured gain and loss."""
    frozen=freeze_expanded(folder)
    if stage_complete(folder,"confirmation_evaluation"):
        return
    print("[6/7] Evaluate frozen approaches on fresh confirmation queries",flush=True)
    verify_files(folder,frozen["files"])
    opened=json.loads((folder/"confirmation_opened.json").read_text())
    if opened["frozen_sha256"]!=file_hash(folder/"frozen_choices.json"):
        raise ValueError("Choices changed after confirmation was opened.")
    policies=frozen["policies"]+[dict(id="original_order",family="original")]
    metrics,contexts=expanded_evaluate(config,folder,"confirmation",policies)
    save_table(metrics,folder/"confirmation_policies.parquet")
    save_table(expanded_summary(metrics),folder/"confirmation_metrics.csv")
    finish_stage(folder,"confirmation_evaluation",["confirmation_policies.parquet","confirmation_metrics.csv"])
