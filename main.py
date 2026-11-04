"""Start the new anonymous-session pipeline. Currently, step 1 is implemented."""

import argparse
import os
from pathlib import Path

# Ctrl+click this function to read the anonymous sample preparation code.
from session_rerank.samples import prepare_samples

DEFAULT_CONFIG = Path("config/anonymous_samples.json")


def main(config_path=DEFAULT_CONFIG):
    """Run the implemented stages of the new anonymous-session study.

    Step 1 creates complete session samples, candidate/click tables, and
    strictly earlier session histories. Confirmation has reserved IDs only.
    Returns the prepared-data folder. Model training and later stages will
    be added here when they support this new anonymous table format.
    """
    # 1. Prepare the anonymous samples and their query-time histories.
    dataset_folder = prepare_samples(config_path)
    return dataset_folder


def cli(argv=None):
    """Read optional preparation settings and start the new pipeline.

    Both `uv run main.py` and `uv run main.py prepare` call main() above.
    --config selects the sample protocol. --help performs no preparation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["prepare"], default="prepare",
                        help="Prepare the new anonymous-session sample tables")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Sample sizes, date periods, and history limit")
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parent)
    main(args.config)


if __name__ == "__main__":
    cli()
