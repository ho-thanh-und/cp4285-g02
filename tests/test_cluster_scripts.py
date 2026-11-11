"""Check the cluster grid tools (generator, input check, collector) without a cluster."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from session_rerank.sequence_models import read_sequence_settings

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "cluster" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_writes_twelve_valid_single_model_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    paths = load("make_sequence_grid").make_grid("config/sequence_models_feb_apr.json", tmp_path / "grid", threads=8)
    assert len(paths) == 12 and len(set(paths)) == 12
    index = (tmp_path / "grid" / "grid_index.txt").read_text().split()
    assert index == paths
    for path in paths:
        settings, _ = read_sequence_settings(path)
        assert len(settings["models"]) == 1 and settings["threads"] == 8
        assert settings["training_window"] == dict(start="2016-02-01", end_exclusive="2016-04-30")
        assert settings["output_directory"].startswith("results/sequence_grid/")
    names = [Path(p).stem for p in paths]
    assert sum(n.startswith("sasrec") for n in names) == 8 and sum(n.startswith("gru4rec") for n in names) == 4


def test_input_check_lists_exactly_the_files_the_guide_copies(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    load("make_sequence_grid").make_grid("config/sequence_models_feb_apr.json", tmp_path / "grid", threads=8)
    files = {p.as_posix() for p in load("check_inputs").required_files(tmp_path / "grid" / "grid_index.txt")}
    data = {f for f in files if not f.startswith(tmp_path.as_posix())}
    assert data == {"config/home_page.json", "data/raw/train-queries.csv", "data/raw/train-clicks.csv",
                    "results/learning_curve_v2/queries_50000/products.parquet",
                    "results/home_page/20261108T095146Z_validation/per_case.parquet"}
    guide = (ROOT / "scripts" / "cluster" / "CLUSTER_GUIDE.md").read_text(encoding="utf-8")
    for path in data - {"config/home_page.json"}:
        assert path.split("/")[-1] in guide or "data/raw" in path


def test_collector_ranks_finished_runs_and_lists_missing_ones(tmp_path):
    index = tmp_path / "grid_index.txt"
    configs = []
    for name, ndcg in (("a", 0.12), ("b", 0.13), ("c", None)):
        out = tmp_path / "runs" / name
        config = tmp_path / f"{name}.json"
        config.write_text(json.dumps(dict(models=["sasrec"], epochs=5, output_directory=out.as_posix())))
        configs.append(config.as_posix())
        if ndcg is None:
            continue
        run = out / "20261108T000000Z_validation"
        run.mkdir(parents=True)
        pd.DataFrame([dict(segment="all cases", method="sasrec", recall=0.1, mrr=0.1, ndcg=ndcg,
                           combined_ild=0.5)]).to_csv(run / "summary.csv", index=False)
        pd.DataFrame([dict(reference=r, metric="ndcg", difference=0.01, ci_low=0.0, ci_high=0.02)
                      for r in ("item_knn", "fixed_0.8_item_knn")]).to_csv(run / "differences.csv", index=False)
        (run / "sequence_settings.json").write_text(json.dumps(dict(training=dict(sasrec=dict(best_epoch=5)))))
    index.write_text("\n".join(configs))
    table, missing = load("collect_sequence_grid").collect(index)
    assert table.config.tolist() == ["b", "a"] and missing == [configs[2]]
    assert table.still_improving.all()
