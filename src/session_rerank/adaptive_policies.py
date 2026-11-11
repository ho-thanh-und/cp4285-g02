"""Continuous adaptation, history time decay, and graded relevance calculations."""

import json

import numpy as np
import pandas as pd

from .core import greedy_rerank
from .reranking import quota_order


def relevance_weight(policy, history_ild):
    """Map observed session variety to a relevance weight between zero and one.

    The normalized exponential shares both endpoints with the linear policy.
    expm1 avoids cancellation when beta is small, where the curve becomes linear.
    Sparse history uses the saved validation default, independent of query count.
    """
    family = policy["family"]
    if family in ("relevance", "original"):
        return 1.0
    if family in ("quota", "diverse_tail"):
        return np.nan
    if family == "fixed":
        return float(policy["weight"])
    if not np.isfinite(history_ild):
        return float(policy["default_weight"])
    diversity = float(np.clip(history_ild, 0, 1))
    if family == "adaptive":
        return float(policy["broad_weight"] if diversity > policy["threshold"] else policy["focused_weight"])
    low, high = policy["alpha_min"], policy["alpha_max"]
    if not 0 <= low <= high <= 1:
        raise ValueError("Adaptive bounds must satisfy 0 <= min <= max <= 1.")
    if family == "linear":
        # rate=1 is the original endpoint-preserving linear curve. A larger
        # slope reaches alpha_min sooner; a smaller slope decays more slowly.
        fraction = max(0., 1 - policy.get("rate", 1.) * diversity)
    elif family == "exponential":
        beta = policy["beta"]
        if beta <= 0:
            raise ValueError("Exponential beta must be positive.")
        # Algebraically equivalent to (exp(-beta*H)-exp(-beta))/(1-exp(-beta)).
        fraction = 1 - np.expm1(-beta * diversity) / np.expm1(-beta)
    else:
        raise ValueError(f"Unknown adaptive family: {family}")
    return float(np.clip(low + (high - low) * fraction, low, high))


def diverse_tail_order(scores, distances, keep, k, k1=None):
    """Keep top relevance items; greedily maximize diversity within the tail.

    Restrict the pool to the top K1 by relevance before selecting either part.
    Seed with the most distant pair among remaining products, then repeatedly
    add the item with the largest total distance to the selected tail. This is
    an approximation to the maximum-sum subset. Sort only the chosen tail by
    relevance so positions 1..keep retain the highest relevance products.
    Unknown distances are supplied as zero by the caller; ties are stable.
    """
    scores = np.asarray(scores)
    k = min(k, len(scores))
    # K1 is the candidate pool after ranking by relevance. If the logged list
    # has fewer than K1 products, use all of them; do not invent candidates.
    relevance_order = np.argsort(-scores, kind="stable")[:k1]
    head = relevance_order[:min(keep, k)].tolist()
    remaining = relevance_order[min(keep, k):].tolist()
    needed = min(k - len(head), len(remaining))
    tail = []
    if needed >= 2:
        sub = distances[np.ix_(remaining, remaining)]
        i, j = np.triu_indices(len(remaining), 1)
        best = int(np.argmax(sub[i, j]))
        tail = [remaining[i[best]], remaining[j[best]]]
        remaining = [item for item in remaining if item not in tail]
    elif needed == 1:
        tail = [remaining.pop(0)]
    while len(tail) < needed:
        values = distances[np.ix_(remaining, tail)].sum(axis=1)
        tail.append(remaining.pop(int(np.argmax(values))))
    tail.sort(key=lambda item: (-scores[item], item))
    return np.asarray(head + tail, dtype=int)


def expanded_order(scores, distances, policy, history_ild, k=10):
    """Build exactly min(k, candidates) items without changing pilot constants."""
    family = policy["family"]
    if family == "relevance":
        return np.argsort(-np.asarray(scores), kind="stable")[:k]
    if family == "quota":
        return quota_order(scores, distances, min(policy["keep"], k), k=k)
    if family == "diverse_tail":
        return diverse_tail_order(scores, distances, policy["keep"], k, policy.get("k1"))
    return greedy_rerank(scores, distances, relevance_weight(policy, history_ild), k)


def weighted_history_context(queries, histories, catalog, decay="none", minutes=None, *,
                             metric="combined", minimum_items=2):
    """Measure query-time session variety with an optional time-decay function.

    Each distinct product keeps its latest earlier click/view, so duplicate view
    and click logs do not inflate pair counts. Pair weights are w_i*w_j.
    Missing metadata is excluded from both numerator and denominator.
    """
    if decay not in ("none", "linear", "exponential"):
        raise ValueError("Unknown history decay.")
    if decay != "none" and (minutes is None or minutes <= 0):
        raise ValueError("Decay window or half-life must be positive.")
    interacted = histories[histories.event_type.isin(["click", "view"])].dropna(subset=["item_id"])
    if (interacted.age_ms <= 0).any():
        raise ValueError("History interactions must strictly precede their query.")
    groups = dict(tuple(interacted.groupby("query_id", sort=False)))
    result = []
    for query in queries.itertuples(index=False):
        history = groups.get(query.query_id)
        latest = history.groupby("item_id").age_ms.min() if history is not None else pd.Series(dtype=float)
        items = latest.index.astype(int).tolist()
        ages = latest.to_numpy(dtype=float) / 60000
        if decay == "linear":
            weights = np.maximum(0, 1 - ages / minutes)
        elif decay == "exponential":
            # Subtracting the smallest age rescales all pair weights equally,
            # preserving the ratio while preventing needless underflow.
            weights = np.exp(-np.log(2) * (ages - ages.min()) / minutes) if len(ages) else ages
        else:
            weights = np.ones(len(ages))
        known_items = [item for item in items if item in catalog.name_tokens]
        known_mask = np.array([item in catalog.name_tokens for item in items])
        matrix = catalog.distances(known_items)[metric]
        pairs = np.triu_indices(len(known_items), 1)
        values = matrix[pairs]
        selected_weights = weights[known_mask] if len(items) else weights
        pair_weights = selected_weights[pairs[0]] * selected_weights[pairs[1]]
        valid = np.isfinite(values) & (pair_weights > 0)
        denominator = pair_weights[valid].sum()
        # Count products that actually participate in a known, positive-weight
        # pair. An item with missing metadata must not satisfy the evidence gate.
        usable_items = np.unique(np.r_[pairs[0][valid], pairs[1][valid]])
        enough_history = len(usable_items) >= minimum_items
        diversity = (float(np.dot(values[valid], pair_weights[valid]) / denominator)
                     if denominator > 0 and enough_history else np.nan)
        result.append(dict(
            query_id=int(query.query_id), history_ild=diversity,
            history_unique_items=len(items), history_usable_items=len(usable_items), history_items=json.dumps(items),
            history_pair_coverage=float(valid.sum() / (len(items)*(len(items)-1)/2)) if len(items)>1 else np.nan,
        ))
    return pd.DataFrame(result)


def ndcg(order, gains, k=10):
    """NDCG@k with gains already mapped to business utility, including zero positives."""
    gains = np.asarray(gains, dtype=float)
    order = np.asarray(order[:k], dtype=int)
    discounts = 1 / np.log2(np.arange(min(k, len(gains))) + 2)
    ideal = np.sort(gains)[::-1][:k] @ discounts
    return float(gains[order] @ discounts / ideal) if ideal > 0 else np.nan


def relevance_metrics(order, clicked, purchased, label_gains=(0, 1, 3), k=10):
    """Keep click and purchase measurements visible beside the combined objective."""
    order = np.asarray(order[:k], dtype=int)
    clicked, purchased = np.asarray(clicked, bool), np.asarray(purchased, bool)
    grades = np.where(purchased, 2, clicked.astype(int))
    return dict(
        click_ndcg=ndcg(order, clicked.astype(float), k),
        gain_ndcg=ndcg(order, np.asarray(label_gains)[grades], k),
        recall_at_k=float(clicked[order].sum()/clicked.sum()) if clicked.any() else np.nan,
        purchase_recall=float(purchased[order].sum()/purchased.sum()) if purchased.any() else np.nan,
    )
