"""Read this file first: the complete experiment in six steps."""

import argparse
import os
from pathlib import Path

from session_rerank.pipeline import (
    prepare_dataset,
    train_models,
    compare_models,
    choose_diversification,
    freeze_and_evaluate,
    save_results,
)


def main():
    """Prepare data, learn recommendations, evaluate them, and save the evidence."""
    prepare_dataset()
    train_models()
    comparison = compare_models()
    settings = choose_diversification(comparison)
    results = freeze_and_evaluate(comparison, settings)
    save_results(results)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["run"], default="run")
    parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parent)
    main()


if __name__ == "__main__":
    cli()
