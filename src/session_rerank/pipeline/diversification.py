"""Step 4: compare diversification policies on the original validation sample."""

from .state import FROZEN_CHOICES, ModelComparison, read_json


def choose_diversification(comparison: ModelComparison):
    from .. import iteration3_report as scoring
    from ..optimization_report import evaluate, policy_search

    print("[4/6] Choose diversification settings", flush=True)
    if FROZEN_CHOICES.exists():
        return read_json(FROZEN_CHOICES)["policies"]

    validation = scoring.load_data("validation")
    relevance_scores = scoring.score_descriptor(
        comparison.selected, "validation", validation, {}
    )
    _, sweep, _ = evaluate(
        validation, {"selected_relevance": relevance_scores}, relevance_scores
    )
    return policy_search(validation, sweep)
