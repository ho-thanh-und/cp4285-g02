"""Step 5: freeze choices, then evaluate the reserved confirmation queries once."""

from .state import (
    RESULTS, FROZEN_CHOICES, EVALUATION_CACHE, PROTOCOL,
    ModelComparison, EvaluationResults, read_json, write_json, verify_models,
)


def freeze_choices(comparison: ModelComparison, policies):
    """Preserve the existing frozen format and refuse to replace saved choices."""
    expected = comparison.frozen_payload(policies)
    verify_models(expected["hashes"])
    if FROZEN_CHOICES.exists():
        frozen = read_json(FROZEN_CHOICES)
        if any(frozen.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Choices differ from the existing frozen selection")
        return frozen
    if EVALUATION_CACHE.exists():
        raise RuntimeError("Confirmation results exist without a frozen selection")
    write_json(FROZEN_CHOICES, expected)
    return expected


def _read_cached_results():
    """Reuse numerical outcomes; never rerun confirmation to repair a report."""
    import json
    import numpy as np
    from ..iteration3 import load_data

    with np.load(EVALUATION_CACHE) as cache:
        metrics = {str(name): cache["values"][:, i]
                   for i, name in enumerate(cache["names"])}
        sweep = cache["sweep"]
        # Old completed experiments store these JSON files beside the cache.
        examples = (json.loads(str(cache["examples_json"].item()))
                    if "examples_json" in cache else read_json(RESULTS / "confirmation_examples.json"))
        receipt = (json.loads(str(cache["receipt_json"].item()))
                   if "receipt_json" in cache else read_json(RESULTS / "confirmation_receipt.json"))
    return EvaluationResults(metrics, sweep, load_data("confirmation"), examples, receipt)


def freeze_and_evaluate(comparison: ModelComparison, policies):
    import json
    import numpy as np
    import pandas as pd
    from .. import iteration3_report as scoring
    from ..optimization_report import WEIGHTS, evaluate
    from ..policies import apply_policy, file_hash

    print("[5/6] Freeze choices and evaluate", flush=True)
    frozen = freeze_choices(comparison, policies)
    if EVALUATION_CACHE.exists():
        return _read_cached_results()

    reserved = pd.read_parquet(scoring.ROOT / "confirmation_ids.parquet")
    protocol = read_json(PROTOCOL)
    assert scoring.hash_ids(reserved.queryId) == protocol["confirmation_query_ids_sha256"]
    used = (set(pd.read_parquet("data/processed/test.parquet").sessionId)
            | set(pd.read_parquet("data/processed_v2/confirmation_ids.parquet").sessionId))
    assert not set(reserved.sessionId) & used

    scoring.build_split("confirmation")
    data = scoring.load_data("confirmation")
    cache = {}
    scores = {name: scoring.score_descriptor(descriptor, "confirmation", data, cache)
              for name, descriptor in frozen["nominees"].items()}
    original = scoring.lgb.Booster(model_file="results/model.txt")
    scores["original_ranker"] = original.predict(data["X"][:, :12], num_threads=4)
    scores["selected_relevance"] = scores[frozen["selected_name"]]
    metrics, sweep, examples = evaluate(data, scores, scores["selected_relevance"])
    for name, policy in frozen["policies"].items():
        metrics[name] = apply_policy(policy, data["meta"], WEIGHTS, sweep)

    receipt = dict(
        frozen_selection_sha256=file_hash(FROZEN_CHOICES),
        confirmation_ids_sha256=protocol["confirmation_query_ids_sha256"],
        queries=len(data["meta"]),
        candidate_rows=int(data["offsets"][-1]),
        no_post_confirmation_tuning=True,
    )
    # Persist outcomes atomically before reporting, so a chart failure cannot
    # cause a second confirmation evaluation. Extra keys preserve report inputs.
    temporary = EVALUATION_CACHE.with_name("confirmation_per_query.pending.npz")
    np.savez_compressed(
        temporary, values=np.stack(list(metrics.values()), axis=1),
        names=list(metrics), sweep=sweep, weights=WEIGHTS,
        examples_json=json.dumps(examples), receipt_json=json.dumps(receipt),
    )
    temporary.replace(EVALUATION_CACHE)
    return EvaluationResults(metrics, sweep, data, examples, receipt)
