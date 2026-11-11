"""Check, on the cluster, that every input of the sequence-model grid is present.

Run from the repository root before submitting the job. It verifies the files
the grid reads (raw CSVs, the product table, the saved home-page cases and the
configs) and that the Python packages import. It prints what is missing and
exits with status 1, so nothing is submitted against an incomplete copy.

    python scripts/cluster/check_inputs.py   (with the conda environment active)
"""

import importlib
import json
import sys
from pathlib import Path


def required_files(index="config/sequence_grid/grid_index.txt"):
    files = [Path(index)]
    configs = Path(index).read_text(encoding="utf-8").split() if Path(index).exists() else []
    files += [Path(c) for c in configs]
    if configs:
        first = json.loads(Path(configs[0]).read_text(encoding="utf-8"))
        home = json.loads(Path(first["home_config"]).read_text(encoding="utf-8"))
        files += [Path(first["home_config"]), Path(first["home_run"]) / "per_case.parquet",
                  Path(home["trained_run"]) / "products.parquet"]
        raw = Path(home["raw_directory"])
        files += [raw / "train-queries.csv", raw / "train-clicks.csv"]
    return files


def main():
    problems = [f"missing file: {path}" for path in required_files() if not path.exists()]
    for package in ("numpy", "pandas", "pyarrow", "scipy", "torch", "lightgbm"):
        try:
            importlib.import_module(package)
        except ImportError as error:
            problems.append(f"cannot import {package}: {error}")
    if problems:
        print("Not ready:\n" + "\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    import torch
    print(f"Ready. Python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"{len(required_files()) - 1} input files present.")


if __name__ == "__main__":
    main()
