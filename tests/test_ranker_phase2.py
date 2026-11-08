"""Check the Phase 2 category-page ranker experiment on synthetic data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main
from test_samples import raw_experiment  # noqa: F401  (fixture used by learning_config)
from test_learning_curve import learning_config  # noqa: F401
from session_rerank.ranker_phase2 import position_buckets, read_phase2_settings, run_phase2

VARIANTS = [
    {"id": "base", "popularity": False, "position_debias": False, "objective": "lambdarank", "learning_rate": 0.1},
    {"id": "popularity_position", "popularity": True, "position_debias": True, "objective": "lambdarank",
     "learning_rate": 0.1},
    {"id": "popularity_xendcg", "popularity": True, "position_debias": False, "objective": "rank_xendcg",
     "learning_rate": 0.1},
]
SETTINGS = dict(output_directory="unused", event_variant="clicks_only", exclude_text_search_from_training=True,
                raw_directory="unused", position_bucket_edges=[1, 2, 3, 6], models=["click_session"],
                variants=VARIANTS, bootstrap_replicates=20, seed=1)


def test_cli_prepare_data_requires_a_size(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "prepare_data_main", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main.cli(["prepare-data"])
    main.cli(["prepare-data", "--queries", "50000", "--curve-root", "results/learning_curve_v2"])
    assert calls == [(Path("config/learning_curve.json"), "50000", Path("results/learning_curve_v2"))]


def test_position_buckets_group_deep_ranks():
    assert position_buckets([1, 2, 3, 5, 6, 200], [1, 2, 3, 6]).tolist() == [0, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("update", [{"event_variant": "views"}, {"models": ["neumf"]},
                                    {"variants": VARIANTS + [VARIANTS[0]]},
                                    {"variants": [VARIANTS[0] | {"objective": "ranknet"}]},
                                    {"position_bucket_edges": [2, 3]}])
def test_invalid_settings_are_rejected(tmp_path, update):
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_phase2_settings(path)


def test_phase2_trains_variants_and_compares_with_popularity_sort(learning_config, tmp_path):
    path, config = learning_config
    # Data-only preparation is enough: Phase 2 fits its own models.
    run = main.prepare_data_main(path, queries="all")
    assert not (run / "training_complete.json").exists()
    raw = json.loads((Path(config["pilot_directory"]) / "manifest.json").read_text())["protocol"]["raw_directory"]
    settings = tmp_path / "phase2.json"
    settings.write_text(json.dumps(SETTINGS | {"raw_directory": raw}))
    output = run_phase2(run, settings, output_root=tmp_path / "phase2")
    summary = pd.read_csv(output / "summary.csv")
    assert {"site_order", "popularity_sort", "base", "popularity_position", "popularity_xendcg"} <= set(summary.variant)
    assert summary.click_ndcg.dropna().between(0, 1).all()
    differences = pd.read_csv(output / "differences.csv")
    same = differences[differences.variant.eq("base") & differences.reference.eq("base")]
    assert np.allclose(same.difference.dropna(), 0)
    gains = pd.read_csv(output / "feature_gain.csv")
    assert "window_click_rate" in set(gains[gains.variant.eq("popularity_position")].feature)
    assert "window_click_rate" not in set(gains[gains.variant.eq("base")].feature)
