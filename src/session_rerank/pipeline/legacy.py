"""The earlier full benchmark, kept separate from the new anonymous pipeline."""

from .dataset import prepare_dataset
from .training import train_models
from .comparison import compare_models
from .diversification import choose_diversification
from .evaluation import freeze_and_evaluate
from .reporting import save_results


def run_legacy_experiment():
    """Run the earlier six-stage experiment using its original feature definitions.

    This workflow includes personal and older-session context. It is retained
    for reproducing the earlier benchmark through `uv run cp4285 run`.
    The new root main.py does not call this workflow.
    """
    prepare_dataset()
    train_models()
    comparison = compare_models()
    settings = choose_diversification(comparison)
    results = freeze_and_evaluate(comparison, settings)
    save_results(results)
