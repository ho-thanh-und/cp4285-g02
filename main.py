"""Run the anonymous-session study, from sample preparation to the final report."""

import argparse
import os
from pathlib import Path

# Ctrl+click this function to read the anonymous sample preparation code.
from session_rerank.samples import prepare_samples
from session_rerank.train_rankers import train_rankers
# Ctrl+click define_product_diversity to follow the five-step diversity review.
from session_rerank.diversity import define_product_diversity
from session_rerank.anonymous_study import (
    open_study, tune_diversification, compare_on_selection, freeze_and_evaluate,
)
from session_rerank.study_report import save_study_report
from session_rerank.study_files import study_lock

# This JSON file sets the CSV folder, output folder, split dates, and sample sizes.
DEFAULT_CONFIG = Path("config/anonymous_samples.json")


def main(config_path=DEFAULT_CONFIG, *, train=False, ranker_run=None, run=False, study_run=None):
    """Read the high-level stages here. Ctrl+click each call to follow its implementation."""
    # 1. Prepare the anonymous samples and their query-time histories.
    dataset_folder = prepare_samples(config_path)

    # 2. Learn relevance scores with and without current-session features.
    if train:
        return train_rankers(dataset_folder)

    # The run action reuses a trained ranker. A resume action reads its recorded
    # paths and validates completed stages before continuing after an interruption.
    if run:
        folder = open_study(dataset_folder, ranker_run, study_run)
        with study_lock(folder):
            # 3. Tune the quota, fixed weight, and session gate using validation.
            tune_diversification(folder)

            # 4. Choose among the four shortlisted settings on reserved selection queries.
            compare_on_selection(folder)

            # 5. Save frozen choices before opening reserved confirmation outcomes.
            freeze_and_evaluate(folder)

            # 6. Export figures, session confidence intervals, examples, and the report.
            return save_study_report(folder)

    # 3a. Define product diversity and measure saved validation recommendation lists.
    # Read the chosen training run, take its top 10 products per query,
    # calculate category/token distances, average the pairs, and save the review.
    # Run this path with main.py diversity --ranker-run <saved training folder>.
    if ranker_run is not None:
        return define_product_diversity(dataset_folder, ranker_run)
    return dataset_folder


def cli(argv=None):
    """Read optional preparation settings and start the new pipeline.

    `uv run main.py train` prepares samples and fits the rankers.
    `uv run main.py diversity --ranker-run results/anonymous_rankers/RUN` reviews ILD.
    `uv run main.py run --ranker-run results/anonymous_rankers/RUN` completes the study.
    `uv run main.py resume --study-run results/anonymous_study/RUN` continues a saved study.
    --config selects the sample protocol. --help performs no preparation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["prepare", "train", "diversity", "run", "resume"], default="prepare",
                        help="Prepare, train, review diversity, run remaining stages, or resume a study")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Sample sizes, date periods, and history limit")
    parser.add_argument("--ranker-run", type=Path,
                        help="Saved ranker folder to review with the diversity action")
    parser.add_argument("--study-run", type=Path, help="Existing study folder for the resume action")
    args = parser.parse_args(argv)
    if args.action in ("diversity", "run") and args.ranker_run is None:
        parser.error("--ranker-run is required for diversity and run")
    if args.action not in ("diversity", "run") and args.ranker_run is not None:
        parser.error("--ranker-run is used with diversity or run")
    if (args.action == "resume") != (args.study_run is not None):
        parser.error("--study-run is required with resume")
    # Resolve paths such as data/raw from the project folder, wherever the script was launched.
    os.chdir(Path(__file__).resolve().parent)
    if args.action == "resume":
        # Read only the saved protocol and paths. Resume uses the original study inputs.
        import json
        identity = json.loads((args.study_run / "study.json").read_text())["identity"]
        main(args.config, ranker_run=Path(identity["ranker_run"]), run=True, study_run=args.study_run)
    else:
        main(args.config, train=args.action == "train", ranker_run=args.ranker_run, run=args.action == "run")


# Start the CLI when this file is run as a script. Importing main leaves it idle.
if __name__ == "__main__":
    cli()
