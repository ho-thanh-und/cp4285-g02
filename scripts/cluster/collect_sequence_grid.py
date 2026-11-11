"""Rank the finished grid settings by validation NDCG@10 on the home-page cases.

For each config in the grid index it reads the newest run folder under the
config's output directory and reports the model's validation metrics, best
epoch, and difference from item-kNN and from the frozen home-page policy (with
95% session-bootstrap intervals). Missing runs are listed, not skipped silently.

The ranking is a validation result. Pick one setting, freeze it, and only then
evaluate it on the selection week (as for every other model in the study).

    python scripts/cluster/collect_sequence_grid.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(index_path):
    rows, missing = [], []
    for line in Path(index_path).read_text(encoding="utf-8").split():
        config = json.loads(Path(line).read_text(encoding="utf-8"))
        runs = sorted(Path(config["output_directory"]).glob("*_validation"))
        runs = [run for run in runs if (run / "summary.csv").exists()]
        if not runs:
            missing.append(line)
            continue
        run = runs[-1]
        model = config["models"][0]
        summary = pd.read_csv(run / "summary.csv")
        mine = summary[summary.segment.eq("all cases") & summary.method.eq(model)].iloc[0]
        differences = pd.read_csv(run / "differences.csv")
        training = json.loads((run / "sequence_settings.json").read_text())["training"][model]
        row = dict(config=Path(line).stem, model=model, run=run.as_posix(), best_epoch=training.get("best_epoch"),
                   epochs=config["epochs"], recall=mine.recall, mrr=mine.mrr, ndcg=mine.ndcg,
                   combined_ild=mine.combined_ild)
        for reference in ("item_knn", "fixed_0.8_item_knn"):
            d = differences[differences.reference.eq(reference) & differences.metric.eq("ndcg")].iloc[0]
            row[f"ndcg_vs_{reference}"] = d.difference
            row[f"ndcg_vs_{reference}_ci"] = f"({d.ci_low:+.4f}, {d.ci_high:+.4f})"
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("ndcg", ascending=False)
        # A best epoch equal to the last epoch means the model was still improving.
        table["still_improving"] = table.best_epoch.eq(table.epochs)
    return table, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", default="config/sequence_grid/grid_index.txt")
    parser.add_argument("--output", default="results/sequence_grid/grid_summary.csv")
    args = parser.parse_args()
    table, missing = collect(args.index)
    if not table.empty:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(table.drop(columns=["run"]).round(4).to_string(index=False))
        print(f"\nSaved {args.output}")
    if missing:
        print(f"\n{len(missing)} setting(s) without a finished run:")
        print("\n".join(f"  {m}" for m in missing))


if __name__ == "__main__":
    main()
