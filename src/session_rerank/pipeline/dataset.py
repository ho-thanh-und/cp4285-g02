"""Step 1: prepare samples, histories and features without fitting models."""

from pathlib import Path
import subprocess
import sys

from .state import RESULTS


def prepare_dataset():
    """Create the saved data files needed for training and later evaluation.

    Reads the six raw CSVs in data/raw and reuses completed preparation work.
    Builds chronological samples, past-click histories, popularity statistics,
    view features, and collaborative features for training and validation.
    Reserves selection/confirmation queries, but does not measure their results.
    Returns nothing; the next stages read these files from data/ on disk.
    """
    from .. import prepare, optimization_data, view_features, iteration3

    print("[1/6] Prepare the dataset", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Download only when local raw data is absent. The audit also creates
    # the original chronological train/validation/test samples.
    if not Path("data/raw/train-queries.csv").exists():
        subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
    if not Path("data/processed/queries_metadata.parquet").exists():
        prepare.audit_and_split()
    # Prepare daily product counts and the ordered click sequences used by
    # RecBole. Later feature builders enforce which past days are allowed.
    if not Path("data/processed_v2/daily_counts.npz").exists():
        optimization_data.prepare()

    # Each candidate product becomes one feature row. A manifest records
    # query boundaries and completion; existing caches are reused.
    for split in ("validation", "train"):
        folder = Path("data/processed_v2") / split
        if not (folder / "manifest.json").exists():
            optimization_data.build_features(split)
        if not (folder / "features_views.npy").exists():
            view_features.build(split)

    # Reserving query IDs now prevents us choosing convenient test examples
    # after seeing which models perform well. This does not score them.
    required = ("confirmation_ids.parquet", "selection_queries.parquet",
                "confirmation_queries.parquet")
    if any(not (iteration3.ROOT / name).exists() for name in required):
        iteration3.reserve_and_prepare()

    # Add historical co-click graph and identity-context features for trees.
    for split in ("validation", "train"):
        iteration3.build_split(split)
    # Selection features are opened during comparison. Confirmation features
    # remain locked until choices have been frozen in step 5.
