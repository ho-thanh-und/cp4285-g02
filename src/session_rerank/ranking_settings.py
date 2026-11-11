"""Validate the ranking controls shared by manual and expanded experiments."""

from math import isfinite, log


DEFAULT_RERANKING = dict(
    enabled=True, policy_type="adaptive", relevance_slots=6, k1=500,
    decay_function_type="linear", decay_function_rate=1.0,
    alpha_min=.2, alpha_max=1., fallback_alpha=.8, fixed_alpha=.8,
    history_decay_type="none", history_decay_rate_per_minute=log(2)/5,
    history_max_events=100, minimum_history_items=2, diversity_metric="combined",
)


def top_k(config):
    """Number of recommended products, also used for NDCG and recall cutoffs."""
    value = config.get("top_k", 10)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("top_k must be a positive integer")
    return value


def reranking_settings(config):
    """Reject misspelled controls and invalid decay rates before processing data.

    History decay uses age in minutes. For exponential history decay, the rate
    lambda means w=exp(-lambda*age); half-life is ln(2)/lambda minutes. For linear
    decay, w=max(0,1-rate*age), reaching zero after 1/rate minutes.
    The adaptive relevance curve instead uses dimensionless session variety H.
    These two rates are independent and deliberately have different names.
    """
    top_k(config)
    supplied = config.get("reranking", {})
    unknown = set(supplied) - set(DEFAULT_RERANKING)
    if unknown:
        raise ValueError(f"Unknown reranking settings: {sorted(unknown)}")
    settings = DEFAULT_RERANKING | supplied
    if not isinstance(settings["enabled"], bool):
        raise ValueError("reranking.enabled must be true or false")
    if settings["policy_type"] not in ("adaptive", "diverse_tail"):
        raise ValueError("policy_type must be adaptive or diverse_tail")
    slots = settings["relevance_slots"]
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 0:
        raise ValueError("relevance_slots must be a nonnegative integer")
    if settings["policy_type"] == "diverse_tail" and slots > top_k(config):
        raise ValueError("relevance_slots must not exceed top_k")
    k1 = settings["k1"]
    if isinstance(k1, bool) or not isinstance(k1, int) or k1 < 1:
        raise ValueError("k1 must be a positive integer")
    if settings["policy_type"] == "diverse_tail" and k1 < top_k(config):
        raise ValueError("k1 must be at least top_k, so the pool can fill the list")
    if settings["decay_function_type"] not in ("fixed", "linear", "exponential"):
        raise ValueError("decay_function_type must be fixed, linear, or exponential")
    if settings["history_decay_type"] not in ("none", "linear", "exponential"):
        raise ValueError("history_decay_type must be none, linear, or exponential")
    for name in ("decay_function_rate", "history_decay_rate_per_minute"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ValueError(f"reranking.{name} must be finite and positive")
    for name in ("alpha_min", "alpha_max", "fallback_alpha", "fixed_alpha"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"reranking.{name} must be in [0,1]")
    if settings["alpha_min"] > settings["alpha_max"]:
        raise ValueError("alpha_min must not exceed alpha_max")
    for name, minimum in (("history_max_events", 1), ("minimum_history_items", 2)):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"reranking.{name} must be an integer >= {minimum}")
    if settings["diversity_metric"] not in ("combined", "token", "category"):
        raise ValueError("diversity_metric must be combined, token, or category")
    return settings


def configured_policy(config):
    """Translate readable JSON settings into the list builder's policy format."""
    settings = reranking_settings(config)
    return dict(
        id="configured", family=("diverse_tail" if settings["policy_type"] == "diverse_tail" else settings["decay_function_type"]),
        keep=settings["relevance_slots"], k1=settings["k1"],
        alpha_min=settings["alpha_min"], alpha_max=settings["alpha_max"],
        rate=settings["decay_function_rate"], beta=settings["decay_function_rate"],
        default_weight=settings["fallback_alpha"], weight=settings["fixed_alpha"],
    )


def history_decay_minutes(settings):
    """Convert an explicit rate to the existing window/half-life implementation."""
    rate = settings["history_decay_rate_per_minute"]
    return log(2)/rate if settings["history_decay_type"] == "exponential" else 1/rate
