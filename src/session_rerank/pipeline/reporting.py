"""Step 6: save tables, recommendation examples, statistics and figures."""

from .state import RESULTS, EvaluationResults, write_json


def save_results(results: EvaluationResults):
    """Turn EvaluationResults into the files used to write the final report.

    Writes summary tables, paired statistical comparisons, subgroup results,
    recommendation examples, charts, and a receipt under results/iteration3/.
    Uses already measured outcomes; it does not fit models or tune policies.
    Returns nothing.
    """
    from ..iteration3_report import report

    print("[6/6] Save results", flush=True)
    write_json(RESULTS / "confirmation_examples.json", results.examples)
    report(results.metrics, results.diversity_sweep, results.dataset)
    write_json(RESULTS / "confirmation_receipt.json", results.receipt)
    print(f"Results saved to {RESULTS.resolve()}", flush=True)
