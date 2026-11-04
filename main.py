"""Read this file first: the complete experiment in six steps."""

import argparse
import os
from pathlib import Path

# Import each function from its implementation file so Go to Definition
# can take you directly to the stage you want to understand.
from session_rerank.pipeline.dataset import prepare_dataset
from session_rerank.pipeline.training import train_models
from session_rerank.pipeline.comparison import compare_models
from session_rerank.pipeline.diversification import choose_diversification
from session_rerank.pipeline.evaluation import freeze_and_evaluate
from session_rerank.pipeline.reporting import save_results


def main():
    """Run the complete experiment in the order described in the report.

    Each call below is one stage. Data and fitted models are saved on disk;
    comparison, settings, and results carry the choices between stages.
    Returns nothing: the final tables, examples, and charts are saved by step 6.
    """
    # 1. Turn raw CSVs into samples, past histories, and model features.
    prepare_dataset()

    # 2. Learn relevance scores using trees, GRU4Rec, and SASRec.
    train_models()

    # 3. Compare relevance models and mixtures; carry the winner forward.
    comparison = compare_models()

    # 4. Decide how much variety to add, using validation queries only.
    settings = choose_diversification(comparison)

    # 5. Lock those decisions, then evaluate the reserved confirmation queries.
    results = freeze_and_evaluate(comparison, settings)

    # 6. Save tables, example recommendations, statistics, and charts.
    save_results(results)


def cli(argv=None):
    """Read command-line options, then start the experiment from its own folder.

    argv is an optional list of arguments; None uses the terminal arguments.
    Changing folders makes paths such as data/raw work consistently.
    --help exits before any preparation or training begins.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["run"], default="run")
    parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parent)
    main()


if __name__ == "__main__":
    cli()
