"""Write one sequence-models config per grid setting, plus the index the job array reads.

Every config starts from config/sequence_models_feb_apr.json (February–April
training data, validation-week evaluation on the home-page cases) and trains a
single model, so each array task runs one setting. The best setting is later
chosen on validation (collect_sequence_grid.py) and must be frozen before any
selection-week run.

    python scripts/cluster/make_sequence_grid.py --threads 8
"""

import argparse
import itertools
import json
from pathlib import Path

GRID = {
    "sasrec": dict(embedding_dim=[64, 128], sasrec_blocks=[1, 2], dropout=[0.2, 0.5],
                   epochs=[30], learning_rate=[0.001]),
    "gru4rec": dict(embedding_dim=[64, 128], learning_rate=[0.001, 0.003], dropout=[0.2], epochs=[60]),
}


def settings_name(model, values):
    parts = [model] + [f"{key.replace('embedding_dim', 'dim').replace('sasrec_blocks', 'blocks').replace('learning_rate', 'lr')}"
                       f"{value:g}" for key, value in values.items()]
    return "_".join(parts)


def make_grid(base_path, output_dir, threads, grid=GRID):
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, options in grid.items():
        keys = list(options)
        for combination in itertools.product(*(options[key] for key in keys)):
            values = dict(zip(keys, combination))
            name = settings_name(model, values)
            config = base | values | dict(models=[model], threads=threads,
                                          output_directory=f"results/sequence_grid/{name}")
            path = output_dir / f"{name}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            paths.append(path.as_posix())
    (output_dir / "grid_index.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="config/sequence_models_feb_apr.json")
    parser.add_argument("--output", default="config/sequence_grid")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads per task (match the job request)")
    args = parser.parse_args()
    paths = make_grid(args.base, args.output, args.threads)
    print(f"Wrote {len(paths)} configs and {Path(args.output) / 'grid_index.txt'}; "
          f"submit an array of 1-{len(paths)}.")


if __name__ == "__main__":
    main()
