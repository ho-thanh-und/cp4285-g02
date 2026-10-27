"""Pure ranking, distance and metric functions, independent of dataset loading."""

import numpy as np


def tokens(value):
    if not isinstance(value, str) or not value:
        return frozenset()
    return frozenset(int(x) for x in value.split(",") if x)


def jaccard_distance(a, b):
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0


def distance_matrix(item_ids, item_tokens):
    result = np.zeros((len(item_ids), len(item_ids)), dtype=np.float32)
    for i, item in enumerate(item_ids):
        for j in range(i):
            result[i, j] = result[j, i] = jaccard_distance(
                item_tokens[item], item_tokens[item_ids[j]]
            )
    return result


def normalize(scores):
    scores = np.asarray(scores, dtype=float)
    span = np.ptp(scores)
    return (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)


def greedy_rerank(scores, distances, weight, k=10):
    """Average-distance MMR variant; keep best relevance item as the first item.

    Relevance is min-max scaled once over the candidate list. Jaccard distances
    already lie in [0,1] and are not re-scaled at each greedy step. Stable ties
    preserve the supplied candidate order. weight=1 exactly recovers relevance.
    """
    if not 0 <= weight <= 1:
        raise ValueError("weight must lie in [0,1]")
    scores = normalize(scores)
    k = min(k, len(scores))
    if not k:
        return np.array([], dtype=int)
    selected = [int(np.argmax(scores))]
    available = np.ones(len(scores), dtype=bool)
    available[selected] = False
    distance_sum = distances[:, selected[0]].copy()
    while len(selected) < k:
        objective = weight * scores + (1 - weight) * distance_sum / len(selected)
        objective[~available] = -np.inf
        nxt = int(np.argmax(objective))
        selected.append(nxt)
        available[nxt] = False
        distance_sum += distances[:, nxt]
    return np.asarray(selected)


def ranking_metrics(order, relevant, distances, k=10):
    order = np.asarray(order[:k], dtype=int)
    hits = np.asarray(relevant, dtype=bool)[order]
    n_relevant = int(np.sum(relevant))
    discounts = 1 / np.log2(np.arange(len(order)) + 2)
    ideal = discounts[:min(n_relevant, len(order))].sum()
    ndcg = float(np.dot(hits, discounts) / ideal) if ideal else 0.0
    recall = float(hits.sum() / n_relevant) if n_relevant else 0.0
    ild = float(distances[np.ix_(order, order)].sum() / (len(order) * (len(order) - 1))) if len(order) > 1 else 0.0
    return ndcg, recall, ild


def prior_click_items(events, query_id, query_time, limit=10):
    """events are (click_time, source_query_id, item_id), all on the same date.

    The current query's labels are forbidden even if inconsistent times occur.
    Equal timestamps are excluded. Keep the most recent occurrence of each item.
    """
    result = []
    seen = set()
    for timestamp, source, item in reversed(events):
        if timestamp < query_time and source != query_id and item not in seen:
            result.append(item)
            seen.add(item)
            if len(result) == limit:
                break
    return result


def history_diversity(history, item_tokens):
    if len(history) < 2:
        return float("nan")
    distances = distance_matrix(history, item_tokens)
    return float(distances.sum() / (len(history) * (len(history) - 1)))
