"""The four list builders, evaluated on identical query candidates."""

import json
from time import perf_counter

import numpy as np
import pandas as pd

from .core import greedy_rerank, ranking_metrics

K = 10
# The allowed drop is relative to the relevance-only model on the same split.
# Both the loss allowance and the search grid are saved before selection is opened.
RELEVANCE_LOSS_LIMIT = 0.05
WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)
QUOTAS = (0, 2, 4, 6, 8, 10)


def quota_order(scores, distances, keep, k=K):
    """Keep the first m relevance items, then greedily increase whole-list diversity.

    Each extra item maximizes its total distance to every already selected item.
    This is a greedy approximation to the set objective, rather than an exact
    search over all possible subsets. Ties follow original candidate order.
    """
    if keep < 0 or keep > k:
        raise ValueError("The relevance quota must lie between zero and k.")
    k = min(k, len(scores))
    # With m=0 start from the highest relevance item, since an empty list has no
    # pair distances. With m=k the result is exactly the relevance-only list.
    selected = np.argsort(-np.asarray(scores), kind="stable")[:min(max(1, keep), k)].tolist()
    available = np.ones(len(scores), dtype=bool)
    available[selected] = False
    while len(selected) < k:
        gain = distances[:, selected].sum(axis=1)
        gain[~available] = -np.inf
        item = int(np.argmax(gain))
        selected.append(item)
        available[item] = False
    return np.array(selected, dtype=int)


def recommendation_order(scores, distances, policy, history_ild):
    """Dispatch to relevance, quota, fixed-weight, or session-adaptive ranking."""
    family = policy["family"]
    if family == "relevance":
        return np.argsort(-np.asarray(scores), kind="stable")[:K]
    if family == "quota":
        return quota_order(scores, distances, policy["keep"])
    if family == "fixed":
        return greedy_rerank(scores, distances, policy["weight"], K)
    if family == "adaptive":
        # Sparse or unmeasurable history uses the fixed default tuned on validation.
        # Larger weights favor relevance. Broad history can receive more diversity.
        if not np.isfinite(history_ild):
            weight = policy["default_weight"]
        elif history_ild > policy["threshold"]:
            weight = policy["broad_weight"]
        else:
            weight = policy["focused_weight"]
        return greedy_rerank(scores, distances, weight, K)
    raise ValueError(f"Unknown policy family: {family}")


def history_context(queries, histories, catalog):
    """Describe unique earlier clicked/viewed products for each query.

    Histories were already cut strictly before the query. De-duplication prevents
    repeated views of one product from creating artificial evidence of variety.
    A session needs two distinct products with known metadata to have defined ILD.
    """
    interactions = histories[histories.event_type.isin(["click", "view"])].dropna(subset=["item_id"])
    groups = {query: group.item_id.astype(int).drop_duplicates().tolist()
              for query, group in interactions.groupby("query_id", sort=False)}
    result = []
    for query in queries.itertuples(index=False):
        items = groups.get(query.query_id, [])
        measured = catalog.measure_list(items, k=max(1, len(items)))
        result.append(dict(
            query_id=int(query.query_id), history_items=json.dumps(items),
            history_unique_items=len(items), history_ild=measured["combined_ild"],
            history_pair_coverage=measured["combined_pair_coverage"],
        ))
    return pd.DataFrame(result)


def evaluate_policies(scores, contexts, catalog, policies, segment_threshold):
    """Measure all policies on each candidate list, sharing its distance computation.

    Ranking metrics are undefined for queries with no clicked candidate. Their
    lists still contribute to ILD. Unknown distances earn zero diversity reward
    during selection, while reported ILD uses only known pairs and shows coverage.
    """
    contexts = contexts.set_index("query_id")
    records = []
    for count, (query_id, group) in enumerate(scores.sort_values(
        ["query_id", "original_rank"]
    ).groupby("query_id", sort=False), start=1):
        items = group.item_id.to_numpy(dtype=int)
        relevance = group.relevance_score.to_numpy(dtype=float)
        clicked = group.clicked.to_numpy(dtype=bool)
        if len(set(items)) != len(items) or not np.isfinite(relevance).all():
            raise ValueError("Each query needs unique candidates and finite relevance scores.")
        started = perf_counter()
        matrices = catalog.distances(items)
        pairwise_ms = (perf_counter() - started) * 1000
        # NaN means unknown metadata, and receives a conservative zero reward.
        reward_distances = np.nan_to_num(matrices["combined"], nan=0.0)
        context = contexts.loc[query_id]
        history_ild = float(context.history_ild)
        if not np.isfinite(history_ild):
            segment = "sparse_history"
        elif history_ild > segment_threshold:
            segment = "varied_history"
        else:
            segment = "focused_history"
        for policy in policies:
            started = perf_counter()
            if policy["family"] == "original":
                order = np.arange(min(K, len(items)))
            else:
                order = recommendation_order(relevance, reward_distances, policy, history_ild)
            rerank_ms = (perf_counter() - started) * 1000
            # Reuse the existing relevance metric implementation. Unknown metadata
            # affects only the diversity metrics calculated separately below.
            ndcg, recall, _ = ranking_metrics(order, clicked, reward_distances, K)
            pairs = np.triu_indices(len(order), k=1)
            measured = {}
            for name, matrix in matrices.items():
                values = matrix[np.ix_(order, order)][pairs]
                known = values[np.isfinite(values)]
                measured[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
                measured[f"{name}_coverage"] = len(known) / len(values) if len(values) else np.nan
            records.append(dict(
                query_id=int(query_id), session_key=group.session_key.iloc[0],
                policy_id=policy["id"], family=policy["family"], segment=segment,
                ndcg_at_10=ndcg if clicked.any() else np.nan,
                recall_at_10=recall if clicked.any() else np.nan,
                clicked_candidates=int(clicked.sum()), list_size=len(order),
                history_unique_items=int(context.history_unique_items),
                history_ild=history_ild, rerank_ms=rerank_ms, pairwise_ms=pairwise_ms,
                top10_item_ids=json.dumps(items[order].tolist()),
                clicked_item_ids=json.dumps(items[clicked].tolist()), **measured,
            ))
        if count % 250 == 0:
            print(f"Reviewed {count} query lists", flush=True)
    return pd.DataFrame(records)


def summarize_metrics(metrics, segments=False):
    """Give every query equal weight and expose the denominator for each metric."""
    frames = [metrics.assign(segment="all")]
    if segments:
        frames.append(metrics)
    return pd.concat(frames).groupby(["segment", "policy_id", "family"], sort=False, as_index=False).agg(
        queries=("query_id", "size"), relevance_queries=("ndcg_at_10", "count"),
        diversity_queries=("combined_ild", "count"),
        ndcg_at_10=("ndcg_at_10", "mean"), recall_at_10=("recall_at_10", "mean"),
        token_ild=("token_ild", "mean"), category_ild=("category_ild", "mean"),
        combined_ild=("combined_ild", "mean"), combined_coverage=("combined_coverage", "mean"),
        rerank_ms=("rerank_ms", "mean"), pairwise_ms=("pairwise_ms", "mean"),
    )


def choose_policy(summary, policy_ids):
    """Maximize combined ILD under the prespecified relevance loss allowance.

    A policy also needs at least the baseline ILD. Equivalent relevance-only
    settings are present in every family so a fallback always exists. Ties use
    NDCG, then simplicity, then the ID for a deterministic result.
    """
    all_rows = summary[summary.segment.eq("all")].set_index("policy_id")
    baseline = all_rows.loc["relevance"]
    allowed = all_rows.loc[list(policy_ids)]
    allowed = allowed[
        allowed.ndcg_at_10.ge(baseline.ndcg_at_10 * (1 - RELEVANCE_LOSS_LIMIT) - 1e-12)
        & allowed.combined_ild.ge(baseline.combined_ild - 1e-12)
    ]
    if allowed.empty:
        raise ValueError("No policy meets the recorded relevance/diversity constraints.")
    ranked = allowed.reset_index()
    # If measured quality is equal, prefer the simplest family. Rounding here
    # prevents floating-point noise from choosing a more complex equivalent.
    ranked["diversity_tie"] = ranked.combined_ild.round(12)
    ranked["relevance_tie"] = ranked.ndcg_at_10.round(12)
    ranked["complexity"] = ranked.policy_id.str.split("_").str[0].map(
        {"relevance": 0, "fixed": 1, "quota": 2, "adaptive": 3}
    ).fillna(4)
    ranked = ranked.sort_values(
        ["diversity_tie", "relevance_tie", "complexity", "policy_id"],
        ascending=[False, False, True, True],
    )
    return ranked.policy_id.iloc[0]


def policy_grid(thresholds):
    """Build a small declared search space, including baseline-equivalent settings."""
    policies = [dict(id="relevance", family="relevance")]
    policies += [dict(id=f"quota_{keep}", family="quota", keep=keep) for keep in QUOTAS]
    policies += [dict(id=f"fixed_{weight:.1f}", family="fixed", weight=weight) for weight in WEIGHTS]
    # The adaptive default is filled in after fixed-weight tuning on validation.
    for index, threshold in enumerate(thresholds):
        for focused in (0.6, 0.8, 1.0):
            for broad in (0.2, 0.4, 0.6, 0.8, 1.0):
                if broad <= focused:
                    policies.append(dict(
                        id=f"adaptive_{index}_{focused:.1f}_{broad:.1f}", family="adaptive",
                        threshold=float(threshold), focused_weight=focused, broad_weight=broad,
                    ))
    return policies
