"""Evaluate a manually chosen decay policy after training, using validation only."""

import json
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from .adaptive_policies import expanded_order, relevance_weight, relevance_metrics, weighted_history_context
from .diversity import ProductDiversity
from .expanded_data import read_candidates, read_histories
from .ranking_settings import top_k, reranking_settings, configured_policy, history_decay_minutes
from .study_files import write_json, save_table, stage_complete, finish_stage


def evaluate_manual_reranking(config, folder):
    """Compare relevance-only and the configured policy for every fitted model.

    Decay is a reranking control: it does not change LambdaRank's loss. We keep
    the relevance-only score so changing the decay cannot masquerade as a better
    relevance model. All model families use the same histories and distances.
    There is no tuning on selection and no confirmation access in this stage.
    """
    if stage_complete(folder, "manual_reranking"):
        return
    if not stage_complete(folder, "training"):
        raise ValueError("Train the relevance models before reviewing reranking.")
    settings, k = reranking_settings(config), top_k(config)
    write_json(folder / "reranking_settings.json", dict(top_k=k, **settings))
    outputs = ["reranking_settings.json"]
    if not settings["enabled"]:
        finish_stage(folder, "manual_reranking", outputs)
        return
    print(f"[4/4] Validate top-{k} lists with {settings['policy_type']} policy "
          f"and {settings['history_decay_type']} history decay", flush=True)
    scores = read_candidates(folder, "validation")
    queries = pd.read_parquet(folder / "validation_features" / "queries.parquet")
    catalog = ProductDiversity(pd.read_parquet(folder / "products.parquet"))
    context = weighted_history_context(
        queries, read_histories(folder, "validation"), catalog,
        settings["history_decay_type"], history_decay_minutes(settings),
        metric=settings["diversity_metric"], minimum_items=settings["minimum_history_items"],
    ).set_index("query_id")
    matrix = np.load(folder / "validation_features" / "X.npy", mmap_mode="r")
    predictions = {}
    for model in pd.read_csv(folder / "model_comparison.csv").model:
        details = json.loads((folder / f"{model}_trials.json").read_text())["result"]
        booster = lgb.Booster(model_file=str(folder / f"{model}.txt"))
        predictions[model] = booster.predict(matrix[:, details["feature_indices"]], num_threads=config["threads"])
    policies = [dict(id="relevance", family="relevance"), configured_policy(config)]
    rows = []
    for number, (qid, group) in enumerate(scores.groupby("query_id", sort=False), start=1):
        items = group.item_id.to_numpy(dtype=int)
        for item in items:
            if item not in catalog.name_tokens:
                catalog.name_tokens[item], catalog.categories[item] = frozenset(), None
        started = perf_counter()
        distances = catalog.distances(items)
        pairwise_ms = (perf_counter()-started)*1000
        reward = np.nan_to_num(distances[settings["diversity_metric"]], nan=0.)
        history_ild = float(context.loc[qid, "history_ild"])
        for model, prediction in predictions.items():
            relevance = prediction[group.index.to_numpy()]
            for policy in policies:
                started = perf_counter()
                order = expanded_order(relevance, reward, policy, history_ild, k)
                elapsed = (perf_counter()-started)*1000
                measured = relevance_metrics(order, group.clicked, group.purchased, config["label_gains"], k)
                pairs = np.triu_indices(len(order), 1)
                # Review the relevance head and diverse tail separately. The
                # head cutoff is fixed by the configured quota, for both policies.
                head_count = min(settings["relevance_slots"], len(order))
                head_metrics = (relevance_metrics(order[:head_count], group.clicked, group.purchased,
                                                  config["label_gains"], head_count)
                                if head_count else None)
                tail_order = order[head_count:]
                tail_pairs = np.triu_indices(len(tail_order), 1)
                tail_values = distances[settings["diversity_metric"]][np.ix_(tail_order, tail_order)][tail_pairs]
                tail_known = tail_values[np.isfinite(tail_values)]
                tail_ild = float(tail_known.mean()) if len(tail_known) else np.nan
                diversity = {}
                for name, distance in distances.items():
                    values = distance[np.ix_(order, order)][pairs]
                    known = values[np.isfinite(values)]
                    diversity[name+"_ild"] = float(known.mean()) if len(known) else np.nan
                    diversity[name+"_coverage"] = len(known)/len(values) if len(values) else np.nan
                rows.append(dict(
                    model=model, policy_id=policy["id"], query_id=int(qid), session_key=group.session_key.iloc[0],
                    top_k=k, topk_item_ids=json.dumps(items[order].tolist()), history_ild=history_ild,
                    candidate_pool_size=(min(len(items), settings["k1"]) if policy["id"] == "configured"
                                         and settings["policy_type"] == "diverse_tail" else len(items)),
                    alpha=relevance_weight(policy, history_ild), rerank_ms=elapsed, pairwise_ms=pairwise_ms,
                    diversity_ild=diversity[settings["diversity_metric"]+"_ild"],
                    head_click_ndcg=head_metrics["click_ndcg"] if head_metrics else np.nan,
                    head_gain_ndcg=head_metrics["gain_ndcg"] if head_metrics else np.nan,
                    tail_diversity_ild=tail_ild, **measured, **diversity,
                ))
        if number % 100 == 0 or number == len(queries):
            print(f"Reranking validation: {number:,}/{len(queries):,} queries, {len(predictions)} models", flush=True)
    measured = pd.DataFrame(rows)
    metrics = ["click_ndcg", "gain_ndcg", "recall_at_k", "purchase_recall", "diversity_ild",
               "token_ild", "category_ild", "combined_ild", "head_click_ndcg", "head_gain_ndcg",
               "tail_diversity_ild", "rerank_ms", "pairwise_ms"]
    summary = measured.groupby(["model", "policy_id", "top_k"], sort=False)[metrics].mean().reset_index()
    counts = measured.groupby(["model", "policy_id"], sort=False).agg(
        queries=("query_id", "size"), mean_candidate_pool_size=("candidate_pool_size", "mean"),
        click_queries=("click_ndcg", "count"),
        gain_queries=("gain_ndcg", "count"), purchase_queries=("purchase_recall", "count"),
        diversity_queries=("diversity_ild", "count"), tail_diversity_queries=("tail_diversity_ild", "count"),
    ).reset_index()
    save_table(measured, folder / "validation_reranking.parquet")
    save_table(summary.merge(counts, on=["model", "policy_id"]), folder / "reranking_metrics.csv")
    finish_stage(folder, "manual_reranking", outputs + ["validation_reranking.parquet", "reranking_metrics.csv"])
