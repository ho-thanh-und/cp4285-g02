"""Check the item-kNN co-click baseline and MRR on synthetic data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main
from test_samples import raw_experiment  # noqa: F401  (fixture used by learning_config)
from test_learning_curve import learning_config  # noqa: F401
from session_rerank.event_ablation import reciprocal_rank
from session_rerank.item_knn import cosine_similarity, read_knn_settings, run_item_knn

SETTINGS = dict(output_directory="unused", raw_directory="unused", max_items_per_session=50, shrinkage=0,
                compare_with=[], bootstrap_replicates=20, seed=1)


def test_reciprocal_rank_uses_the_first_click_in_the_top_k():
    clicked = np.array([False, True, False, True])
    assert reciprocal_rank([2, 3, 1], clicked) == pytest.approx(1 / 2)
    assert reciprocal_rank([0, 2], clicked) == 0
    assert np.isnan(reciprocal_rank([0, 1], np.zeros(4, bool)))


def test_cosine_similarity_counts_shared_sessions():
    # Sessions: {0,1}, {0,1,2}, {2}. Products 0 and 1 share 2 sessions, each in 2.
    session = np.array([0, 0, 1, 1, 1, 2])
    item = np.array([0, 1, 0, 1, 2, 2])
    similarity, support = cosine_similarity(session, item, products=4, shrinkage=0)
    assert support.tolist() == [2, 2, 2, 0]
    assert similarity[0, 1] == pytest.approx(2 / np.sqrt(2 * 2))
    assert similarity[0, 2] == pytest.approx(1 / np.sqrt(2 * 2))
    assert similarity[0, 0] == 0 and similarity[3].nnz == 0
    shrunk, _ = cosine_similarity(session, item, products=4, shrinkage=2)
    assert shrunk[0, 1] == pytest.approx(2 / (2 + 2))


@pytest.mark.parametrize("update", [{"max_items_per_session": 1}, {"shrinkage": -1},
                                    {"compare_with": [{"label": "x"}]}])
def test_invalid_settings_are_rejected(tmp_path, update):
    path = tmp_path / "knn.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_knn_settings(path)


def test_item_knn_ranks_validation_and_matches_popularity_without_history(learning_config, tmp_path):
    path, config = learning_config
    run = main.prepare_data_main(path, queries="all")
    raw = json.loads((Path(config["pilot_directory"]) / "manifest.json").read_text())["protocol"]["raw_directory"]
    settings = tmp_path / "knn.json"
    settings.write_text(json.dumps(SETTINGS | {"raw_directory": raw}))
    output = run_item_knn(run, settings, output_root=tmp_path / "knn")
    per_query = pd.read_parquet(output / "validation_per_query.parquet")
    assert set(per_query.method) == {"site_order", "popularity_sort", "item_knn"}
    assert per_query.mrr.dropna().between(0, 1).all()
    # Without earlier clicks item-kNN falls back to the popularity sort exactly.
    cold = per_query[~per_query.has_history].pivot(index="query_id", columns="method", values="click_ndcg")
    assert np.allclose(cold.item_knn.dropna(), cold.popularity_sort.dropna())
    stats = json.loads((output / "knn_settings.json").read_text())["stats"]
    assert stats["sessions"] > 0 and stats["validation_queries"] == per_query.query_id.nunique()


def test_cli_dispatches_item_knn(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("session_rerank.item_knn.run_item_knn", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main.cli(["item-knn"])
    main.cli(["item-knn", "--trained-run", str(tmp_path)])
    assert len(calls) == 1 and calls[0][0] == tmp_path
