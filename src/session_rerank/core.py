"""Pure ranking, distance and metric functions, independent of dataset loading."""

import numpy as np


def tokens(value):
    """Convert comma-separated token IDs into an immutable set of integers.

    Missing or non-string input becomes an empty set. Tokens are anonymized
    word IDs, so set overlap can measure product-name similarity.
    """
    if not isinstance(value, str) or not value:
        return frozenset()
    return frozenset(int(x) for x in value.split(",") if x)


def jaccard_distance(a, b):
    """Return token dissimilarity: 1 - shared_tokens / all_distinct_tokens.

    0 means identical nonempty sets; 1 means no shared tokens. Two empty
    sets return 0. Used to reward variety between recommended products.
    """
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0


def distance_matrix(item_ids, item_tokens):
    """Return a square matrix of token distances between candidate products.

    item_tokens maps each product ID to its token set. Entry [i, j] is the
    distance between item_ids[i] and item_ids[j]; the diagonal stays zero.
    """
    result = np.zeros((len(item_ids), len(item_ids)), dtype=np.float32)
    for i, item in enumerate(item_ids):
        for j in range(i):
            result[i, j] = result[j, i] = jaccard_distance(
                item_tokens[item], item_tokens[item_ids[j]]
            )
    return result


def normalize(scores):
    """Scale one query's relevance scores into the range 0 to 1.

    This makes tree and neural scores comparable before mixing them.
    A constant score vector becomes all zeros.
    """
    scores = np.asarray(scores, dtype=float)
    span = np.ptp(scores)
    return (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)


def greedy_rerank(scores, distances, weight, k=10):
    """Choose up to k candidate positions balancing relevance and token variety.

    scores gives relevance; distances gives pairwise token distances.
    Start with the best relevance item. Repeatedly choose the candidate
    maximizing weight * relevance + (1 - weight) * mean distance to the
    items already selected. This is an average-distance MMR variant.
    Relevance is min-max scaled once; distances already lie in [0, 1].
    Stable ties preserve candidate order. weight=1 recovers relevance order.
    Returns candidate positions, not product IDs.
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
        # Relevance rewards likely clicks; mean distance rewards adding a
        # product unlike the ones already selected. Larger weight favors relevance.
        objective = weight * scores + (1 - weight) * distance_sum / len(selected)
        objective[~available] = -np.inf
        nxt = int(np.argmax(objective))
        selected.append(nxt)
        available[nxt] = False
        distance_sum += distances[:, nxt]
    return np.asarray(selected)


def ranking_metrics(order, relevant, distances, k=10):
    """Return (NDCG, recall, token ILD) for the first k ranked positions.

    order contains candidate positions; relevant contains binary click labels;
    distances is the candidate distance matrix. NDCG rewards relevant items
    near the top, recall measures how many clicked candidates were retrieved,
    and ILD is the average distance between recommended items.
    """
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
    """Return up to limit distinct earlier clicks, ordered newest first.

    events must be sorted (click_time, source_query_id, item_id) tuples from
    the same date/session. Only click_time < query_time is allowed, and the
    current query's own labels are always excluded. Repeated items keep their
    latest eligible occurrence. These rules prevent future/target leakage.
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
    """Return mean token distance between distinct history positions.

    history contains product IDs; item_tokens maps IDs to token sets.
    Fewer than two history items returns NaN because variety is undefined.
    """
    if len(history) < 2:
        return float("nan")
    distances = distance_matrix(history, item_tokens)
    return float(distances.sum() / (len(history) * (len(history) - 1)))
