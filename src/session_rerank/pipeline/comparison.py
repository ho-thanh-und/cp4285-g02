"""Step 3: compare relevance models and hybrids, without tuning diversity."""

from .state import (
    RESULTS, FROZEN_CHOICES, CANDIDATES, SELECTION_SCORES, PROTOCOL,
    ModelComparison, read_json, write_json, verify_models,
)


def _nominate_models():
    """Reduce the validation search to five representative candidates.

    A descriptor is a small dictionary saying how to compute scores, for
    example {"kind": "tree", "name": "collab31"}. Compare individual trees,
    neural models, the previous hybrid, and several tree/neural mixtures.
    Returns nominee descriptors, validation scores, and model-file hashes.
    Confirmation data is never used to nominate candidates.
    """
    from .. import iteration3_report as scoring

    for name in ("collab31", "collab63", "anonymous31", "gru_ce", "sas_ce"):
        if not (RESULTS / f"{name}.json").exists():
            raise RuntimeError(f"Finish training {name} before comparing models")

    validation = scoring.load_data("validation")
    cache = {}
    records = []

    def assess(descriptor):
        """Score one model description on validation and remember its mean NDCG@10.

        NDCG rewards clicked items appearing near the top of the list.
        Returns the descriptor and score, also appending them to records.
        """
        scores = scoring.score_descriptor(descriptor, "validation", validation, cache)
        value = float(scoring.ndcg_by_query(
            scores, validation["y"], validation["offsets"]
        ).mean())
        result = dict(descriptor=descriptor, validation_ndcg10=value)
        records.append(result)
        return result

    trees = ["previous_views_60k", "previous_views_60k_63",
             "collab31", "collab63", "anonymous31"]
    tree_results = [assess(dict(kind="tree", name=name)) for name in trees]
    neural_results = [assess(dict(kind="neural", name=name))
                      for name in ("gru_ce", "sas_ce")]
    previous = assess(dict(kind="previous"))
    best_tree = max(tree_results, key=lambda row: row["validation_ndcg10"])
    # Try both neural models with several tree contributions. A weight of
    # .9 means 90% tree score and 10% neural score for nonempty click histories.
    fusions = [
        assess(dict(kind="fusion", tree=best_tree["descriptor"]["name"],
                    neural=neural, weight=weight))
        for neural in ("gru_ce", "sas_ce")
        for weight in (.25, .5, .75, .9, .95)
    ]
    # Only these five representatives enter the fresh selection comparison.
    # This keeps the large validation search separate from the final choice.
    nominees = dict(
        previous_hybrid=previous["descriptor"],
        best_tree=best_tree["descriptor"],
        best_neural=max(neural_results, key=lambda row: row["validation_ndcg10"])["descriptor"],
        best_fusion=max(fusions, key=lambda row: row["validation_ndcg10"])["descriptor"],
        anonymous_control=dict(kind="tree", name="anonymous31"),
    )
    return nominees, records, scoring.fingerprints()


def compare_models():
    """Choose the relevance model using the separate model-selection sample.

    Validation nominates five candidates; fresh selection picks the highest
    mean NDCG@10 among them. Saved scores are reused when restarting.
    Returns a ModelComparison containing the winner, other candidates,
    measured scores, and hashes identifying the exact fitted models.
    """
    from .. import iteration3_report as scoring
    import pandas as pd

    print("[3/6] Compare models", flush=True)
    if FROZEN_CHOICES.exists():
        frozen = read_json(FROZEN_CHOICES)
        verify_models(frozen["hashes"])
        return ModelComparison.from_frozen(frozen)

    if CANDIDATES.exists():
        candidates = read_json(CANDIDATES)
        nominees, hashes = candidates["nominees"], candidates["hashes"]
        verify_models(hashes)
        records = read_json(RESULTS / "validation_search.json")
    else:
        if SELECTION_SCORES.exists():
            raise RuntimeError("Selection scores exist without their recorded candidates")
        nominees, records, hashes = _nominate_models()
        write_json(RESULTS / "validation_search.json", records)
        # Record fitted candidates before opening fresh selection outcomes.
        write_json(CANDIDATES, dict(nominees=nominees, hashes=hashes))

    if not SELECTION_SCORES.exists():
        scoring.build_split("selection")
        selection = scoring.load_data("selection")
        cache = {}
        rows = []
        for name, descriptor in nominees.items():
            scores = scoring.score_descriptor(descriptor, "selection", selection, cache)
            mean = float(scoring.ndcg_by_query(
                scores, selection["y"], selection["offsets"]
            ).mean())
            rows.append(dict(candidate=name, ndcg10=mean, queries=len(selection["meta"])))
        temporary = SELECTION_SCORES.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(SELECTION_SCORES)

    # Choose using fresh selection scores, not confirmation outcomes.
    rows = pd.read_csv(SELECTION_SCORES, float_precision="round_trip")
    means = dict(zip(rows.candidate, rows.ndcg10))
    return ModelComparison(
        selected_name=max(means, key=means.get),
        nominees=nominees,
        validation_scores=records,
        selection_scores=means,
        model_hashes=hashes,
        selection_queries=int(rows.queries.iloc[0]),
        protocol_sha256=hashes[PROTOCOL.as_posix()],
    )
