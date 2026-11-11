"""Check the freeze-then-evaluate held-out pipeline on synthetic data."""

import json
from pathlib import Path

import pandas as pd
import pytest

import main
from test_samples import raw_experiment  # noqa: F401  (fixture used by learning_config)
from test_learning_curve import learning_config  # noqa: F401
from session_rerank.heldout import read_heldout_settings, run_heldout
from session_rerank.ranker_phase2 import run_phase2

LEAD = {"id": "popularity_xendcg", "popularity": True, "position_debias": False, "objective": "rank_xendcg",
        "learning_rate": 0.1}


def test_confirmation_needs_the_model_chosen_on_selection(tmp_path):
    base = dict(output_directory="x", split="confirmation", trained_run="x", phase2_run="x",
                phase2_config="x", knn_config="x", bootstrap_replicates=10, seed=1)
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="frozen_selection_run"):
        read_heldout_settings(path)
    selection = tmp_path / "selection"
    selection.mkdir()
    (selection / "selection_choice.json").write_text(json.dumps(dict(chosen="purchase_session")))
    path.write_text(json.dumps(base | dict(frozen_selection_run=str(selection))))
    assert read_heldout_settings(path)["split"] == "confirmation"
    path.write_text(json.dumps(base | dict(split="test")))
    with pytest.raises(ValueError, match="split"):
        read_heldout_settings(path)


def test_freeze_retrain_and_selection_evaluation(learning_config, tmp_path):
    path, config = learning_config
    run = main.prepare_data_main(path, queries="all")
    raw = json.loads((Path(config["pilot_directory"]) / "manifest.json").read_text())["protocol"]["raw_directory"]
    phase2_config = tmp_path / "lead.json"
    phase2_config.write_text(json.dumps(dict(
        output_directory="unused", event_variant="clicks_only", exclude_text_search_from_training=True,
        raw_directory=raw, position_bucket_edges=[1, 2, 3, 6], models=["click_session", "purchase_session"],
        variants=[LEAD], bootstrap_replicates=10, seed=1)))
    phase2 = run_phase2(run, phase2_config, output_root=tmp_path / "phase2")
    knn_config = tmp_path / "knn.json"
    knn_config.write_text(json.dumps(dict(output_directory="unused", raw_directory=raw, max_items_per_session=50,
                                          shrinkage=0, compare_with=[], bootstrap_replicates=10, seed=1)))
    settings = tmp_path / "heldout.json"
    settings.write_text(json.dumps(dict(output_directory="unused", split="selection", trained_run=str(run),
                                        phase2_run=str(phase2), phase2_config=str(phase2_config),
                                        knn_config=str(knn_config), bootstrap_replicates=10, seed=1)))
    # The run stops if the frozen models do not reproduce validation or the
    # rebuilt selection IDs differ from the pilot's reservation.
    output = run_heldout(settings, output_root=tmp_path / "heldout")
    frozen = json.loads((output / "frozen_choices.json").read_text())
    assert set(frozen["ranker"]["models"]) == {"click_session", "purchase_session"}
    assert frozen["heldout_split"] == "selection"
    per_query = pd.read_parquet(output / "selection_per_query.parquet")
    assert set(per_query.method) == {"site_order", "popularity_sort", "item_knn",
                                     "lightgbm_click_session", "lightgbm_purchase_session"}
    pilot = pd.read_parquet(Path(config["pilot_directory"]) / "queries.parquet")
    assert set(per_query.query_id) == set(pilot.loc[pilot.split.eq("selection"), "query_id"])
    choice = json.loads((output / "selection_choice.json").read_text())
    assert choice["chosen"] in ("click_session", "purchase_session")
    assert (output / "click_session.txt").exists() and (output / "summary.csv").exists()
