"""Step 4: compare diversification policies on the original validation sample."""

from .state import FROZEN_CHOICES, ModelComparison, read_json


def choose_diversification(comparison: ModelComparison):
    """Choose how strongly to trade relevance for variety in the top 10.

    comparison identifies the selected relevance model. On original validation
    queries, try fixed weights, minimum-history gates, diversity thresholds,
    and history-length progression. Policies maximize token variety subject
    to at most a 2% relative mean NDCG loss on that validation sample.
    Returns a dictionary of chosen policies, one per policy family.
    """
    from .. import iteration3_report as scoring
    from ..optimization_report import evaluate, policy_search

    print("[4/6] Choose diversification settings", flush=True)
    if FROZEN_CHOICES.exists():
        return read_json(FROZEN_CHOICES)["policies"]

    validation = scoring.load_data("validation")
    relevance_scores = scoring.score_descriptor(
        comparison.selected, "validation", validation, {}
    )
    # sweep[query, weight, metric] records the outcome at every fixed weight.
    # Policy search can select different weights per query from this cache.
    _, sweep, _ = evaluate(
        validation, {"selected_relevance": relevance_scores}, relevance_scores
    )
    return policy_search(validation, sweep)
