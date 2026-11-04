"""Step 1: prepare samples, histories and features without fitting models."""

from pathlib import Path
import subprocess
import sys

from .state import RESULTS


def prepare_dataset():
    from .. import prepare, optimization_data, view_features, iteration3

    print("[1/6] Prepare the dataset", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not Path("data/raw/train-queries.csv").exists():
        subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
    if not Path("data/processed/queries_metadata.parquet").exists():
        prepare.audit_and_split()
    if not Path("data/processed_v2/daily_counts.npz").exists():
        optimization_data.prepare()

    for split in ("validation", "train"):
        folder = Path("data/processed_v2") / split
        if not (folder / "manifest.json").exists():
            optimization_data.build_features(split)
        if not (folder / "features_views.npy").exists():
            view_features.build(split)

    required = ("confirmation_ids.parquet", "selection_queries.parquet",
                "confirmation_queries.parquet")
    if any(not (iteration3.ROOT / name).exists() for name in required):
        iteration3.reserve_and_prepare()

    for split in ("validation", "train"):
        iteration3.build_split(split)
    # Selection features are opened during comparison. Confirmation features
    # remain locked until choices have been frozen in step 5.
