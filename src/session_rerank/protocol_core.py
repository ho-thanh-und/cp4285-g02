"""Pure functions for the session benchmark protocol; earlier frozen experiments are untouched."""
import numpy as np

WINDOW = 1_200_000
DELTAS = (.5, .7, .85, .95, 1.)
TAUS = (1, 2, 3, 5)
LAMBDAS = (0., .25, .5, .75, .9, 1.)


def causal_indices(times, query_time, start=0):
    """Sorted events in this sub-session: include exactly the closed 20m lookback,
    exclude tied/future events. start is the first event in the current episode.
    """
    left = max(start, int(np.searchsorted(times, query_time-WINDOW, side='left')))
    right = int(np.searchsorted(times, query_time, side='left'))
    return np.arange(left, right)


def calibrated_metrics(order, labels, distances=None, k=10):
    order = np.asarray(order[:k], dtype=int)
    labels = np.asarray(labels) > 0
    hits = labels[order]
    discounts = 1/np.log2(np.arange(len(order))+2)
    ideal = discounts[:min(int(labels.sum()), k)].sum()
    ndcg = float(hits @ discounts / ideal) if ideal else 0.
    denominator = min(k, int(labels.sum()))
    recall = float(hits.sum()/denominator) if denominator else 0.
    ild = (float(distances[np.ix_(order, order)].sum()/(len(order)*(len(order)-1)))
           if distances is not None and len(order)>1 else 0.)
    return np.array([ndcg, recall, ild])


def exposure_weights(labels, viewed=None, unseen=.2):
    """SkipAbove is an examination heuristic, not an estimated propensity."""
    positive = np.asarray(labels)>0
    weight = np.full(len(labels), unseen, dtype=np.float32)
    if positive.any():
        weight[:np.flatnonzero(positive)[-1]] = 1.
    if viewed is not None:
        weight[(~positive) & np.asarray(viewed, dtype=bool)] *= .5
    weight[positive] = 1.
    return weight


def floor_order(scores, distances, weight=1., floor='none', k=10):
    """Average-distance greedy MMR; eligibility is fixed before selection.
    A threshold with <K eligible items is backfilled in relevance order.
    """
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        return np.array([], dtype=int)
    rank = np.argsort(-scores, kind='stable')
    span = np.ptp(scores)
    relevance = (scores-scores.min())/span if span else np.zeros(len(scores))
    eligible = np.ones(len(scores), dtype=bool)
    if floor == 'top30':
        eligible[:] = False
        eligible[rank[:max(k, 30)]] = True
    elif floor == 'score03':
        eligible = relevance >= .3
        if eligible.sum() < min(k, len(scores)):
            eligible[rank[:min(k,len(scores))]] = True
    elif floor != 'none':
        raise ValueError(floor)
    if weight == 1.:
        return rank[eligible[rank]][:k]
    chosen = [int(rank[0])]
    eligible[chosen] = False
    total = distances[:, chosen[0]].copy()
    while len(chosen) < min(k, len(scores)):
        objective = weight*relevance + (1-weight)*total/len(chosen)
        objective[~eligible] = -np.inf
        nxt = int(np.argmax(objective))
        chosen.append(nxt)
        eligible[nxt] = False
        total += distances[:, nxt]
    return np.asarray(chosen)


def nested_fractions(frame, seed=4285):
    """Whole episode ranks within week, length, identity, and click strata.
    Midpoint ranks ensure nested, near-exact fractions even in small strata.
    """
    frame = frame.copy()
    rng = np.random.default_rng(seed)
    frame['_random'] = rng.random(len(frame))
    keys = ['split', 'week', 'length_bucket', 'known', 'positive']
    group = frame.groupby(keys, dropna=False)
    frame['fraction_rank'] = (group['_random'].rank(method='first')-.5)/group['_random'].transform('size')
    return frame.drop(columns='_random')
