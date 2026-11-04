"""Start the new anonymous-session pipeline. Prepare samples and train LightGBM rankers."""

import argparse
import os
from pathlib import Path

# Ctrl+click this function to read the anonymous sample preparation code.
from session_rerank.samples import prepare_samples
from session_rerank.train_rankers import train_rankers

# This JSON file sets the CSV folder, output folder, split dates, and sample sizes.
DEFAULT_CONFIG = Path("config/anonymous_samples.json")


def main(config_path=DEFAULT_CONFIG, *, train=False):
    """Prepare the sample tables and optionally fit the two relevance models."""
    # 1. Prepare the anonymous samples and their query-time histories.
    dataset_folder = prepare_samples(config_path)

    # 2. Learn relevance scores with and without current-session features.
    if train:
        return train_rankers(dataset_folder)
    return dataset_folder


def cli(argv=None):
    """Read optional preparation settings and start the new pipeline.

    `uv run main.py train` prepares samples and fits the rankers.
    --config selects the sample protocol. --help performs no preparation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["prepare", "train"], default="prepare",
                        help="Prepare samples, or prepare and train both rankers")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Sample sizes, date periods, and history limit")
    args = parser.parse_args(argv)
    # Resolve paths such as data/raw from the project folder, wherever the script was launched.
    os.chdir(Path(__file__).resolve().parent)
    main(args.config, train=args.action == "train")


# Start the CLI when this file is run as a script. Importing main leaves it idle.
if __name__ == "__main__":
    cli()
