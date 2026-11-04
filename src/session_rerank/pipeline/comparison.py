"""Step 3: compare relevance models and hybrids, without tuning diversity."""

from .state import (
    RESULTS, FROZEN_CHOICES, CANDIDATES, SELECTION_SCORES, PROTOCOL,
    ModelComparison, read_json, write_json, verify_models,
)


def _nominate_models():
    """Use original validation data to nominate the same five candidate types."""
    from .. import iteration3_report as scoring

    for name in ("collab31", "collab63", "anonymous31", "gru_ce", "sas_ce"):
        if not (RESULTS / f"{name}.json").exists():
            raise RuntimeError(f"Finish training {name} before comparing models")

    validation = scoring.load_data("validation")
    cache = {}
    records = []

    def assess(descriptor):
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
    fusions = [
        assess(dict(kind="fusion", tree=best_tree["descriptor"]["name"],
                    neural=neural, weight=weight))
        for neural in ("gru_ce", "sas_ce")
        for weight in (.25, .5, .75, .9, .95)
    ]
    nominees = dict(
        previous_hybrid=previous["descriptor"],
        best_tree=best_tree["descriptor"],
        best_neural=max(neural_results, key=lambda row: row["validation_ndcg10"])["descriptor"],
        best_fusion=max(fusions, key=lambda row: row["validation_ndcg10"])["descriptor"],
        anonymous_control=dict(kind="tree", name="anonymous31"),
    )
    return nominees, records, scoring.fingerprints()


def compare_models():
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
