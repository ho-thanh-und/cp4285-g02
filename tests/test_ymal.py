"""Check the item page "You may also like" evaluation on a small synthetic catalog."""

import json

import numpy as np
import pandas as pd
import pytest

import main
from session_rerank.diversity import ProductDiversity
from session_rerank.item_knn import cosine_similarity
from session_rerank.ymal import candidate_pool, list_metrics, read_ymal_settings, run_ymal

SETTINGS = dict(output_directory="unused", raw_directory="unused", trained_run="unused", knn_config="unused",
                window=dict(name="validation", start="2016-05-01", end_exclusive="2016-05-08"),
                top_k=4, relevance_slots=2, pool_sizes=[4, 6], fixed_alphas=[0.8], max_seeds=1000,
                bootstrap_replicates=20, seed=1)


@pytest.mark.parametrize("update", [
    {"window": dict(name="confirmation", start="2016-05-16", end_exclusive="2016-06-02")},
    {"window": dict(name="validation", start="2016-05-01", end_exclusive="2016-05-20")},
    {"relevance_slots": 4}, {"pool_sizes": [3]}, {"fixed_alphas": [1.5]}])
def test_invalid_settings_and_sealed_weeks_are_rejected(tmp_path, update):
    path = tmp_path / "ymal.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_ymal_settings(path)


def test_selection_week_needs_a_frozen_policy(tmp_path):
    path = tmp_path / "ymal.json"
    selection = dict(name="selection", start="2016-05-09", end_exclusive="2016-05-15")
    for frozen in (None, "six_four_99"):
        path.write_text(json.dumps(SETTINGS | {"window": selection, "frozen_policy": frozen}))
        with pytest.raises(ValueError, match="frozen_policy"):
            read_ymal_settings(path)
    path.write_text(json.dumps(SETTINGS | {"window": selection, "frozen_policy": "fixed_0.8_4"}))
    assert read_ymal_settings(path)["frozen_policy"] == "fixed_0.8_4"


def test_pool_uses_neighbours_first_then_fills_without_the_item():
    session = np.array([0, 0, 1, 1])
    item = np.array([0, 1, 0, 2])
    similarity, _ = cosine_similarity(session, item, products=6, shrinkage=0)
    members, scores, neighbours = candidate_pool(similarity.tocsr(), 0, 4, fill_order=np.array([0, 5, 1, 4]))
    assert neighbours == 2 and set(members[:2]) == {1, 2}
    assert members.tolist()[2:] == [5, 4]  # Item 0 itself and the duplicate 1 are skipped.
    assert (np.diff(scores) <= 0).all() and scores[2] < scores[1]


def test_tail_to_head_distance_compares_tail_with_head():
    products = pd.DataFrame(dict(item_id=[1, 2, 3, 4], price_log2=1, category_id=[7, 7, 8, 8],
                                 name_tokens=["1,2", "1,2", "9", "9"]))
    diversity = ProductDiversity(products)
    categories = products.category_id.to_numpy()
    metrics = list_metrics(np.array([0, 1, 2, 3]), {2}, 0, products.item_id.to_numpy(), categories, diversity, 2)
    # Head (products 1, 2) and tail (3, 4) share no tokens and no category: distance 1.
    assert metrics["tail_to_head_distance"] == pytest.approx(1.0)
    assert metrics["recall"] == 1 and metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["outside_category_share"] == pytest.approx(0.5) and metrics["distinct_categories"] == 2


def test_run_compares_policies_on_identical_item_pages(tmp_path):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    rng = np.random.default_rng(0)
    items = np.arange(1, 31)
    pd.DataFrame(dict(item_id=items, price_log2=items % 5, category_id=items % 3,
                      name_tokens=[f"{i % 7},{i % 4 + 10}" for i in items])).to_parquet(run / "products.parquet")
    queries, clicks, qid = [], [], 0
    for day, count in (("2016-04-03", 60), ("2016-05-02", 30), ("2016-05-10", 20)):
        for session in range(count):
            qid += 1
            queries.append((qid, f"{day}-{session}", day, False))
            for position, item in enumerate(rng.choice(items, 3, replace=False)):
                clicks.append((qid, 100 * position, int(item)))
    pd.DataFrame(queries, columns=["queryId", "sessionId", "eventdate", "is.test"]).to_csv(
        raw / "train-queries.csv", sep=";", index=False)
    pd.DataFrame(clicks, columns=["queryId", "timeframe", "itemId"]).to_csv(raw / "train-clicks.csv", sep=";", index=False)
    knn = tmp_path / "knn.json"
    knn.write_text(json.dumps(dict(output_directory="x", raw_directory=str(raw), max_items_per_session=50,
                                   shrinkage=0, compare_with=[], bootstrap_replicates=10, seed=1)))
    settings = tmp_path / "ymal.json"
    settings.write_text(json.dumps(SETTINGS | dict(raw_directory=str(raw), trained_run=str(run), knn_config=str(knn))))
    output = run_ymal(settings, output_root=tmp_path / "ymal")
    per_page = pd.read_parquet(output / "per_page.parquet")
    methods = {"category_popularity", "relevance_4", "six_four_4", "fixed_0.8_4", "relevance_6", "six_four_6",
               "fixed_0.8_6"}
    assert set(per_page.method) == methods
    # Every method scores the same item pages (2 evaluable pages per 3-click session).
    assert per_page.groupby("method").size().nunique() == 1 and per_page.groupby("method").size().iloc[0] == 60
    assert per_page.recall.between(0, 1).all()
    assert (output / "summary.csv").exists() and (output / "differences.csv").exists()
    assert not (output / "frozen_choices.json").exists()
    # The selection week records the frozen policy and scores only its own pages.
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(SETTINGS | dict(
        raw_directory=str(raw), trained_run=str(run), knn_config=str(knn), frozen_policy="fixed_0.8_4",
        window=dict(name="selection", start="2016-05-09", end_exclusive="2016-05-15"))))
    held_out = run_ymal(selection, output_root=tmp_path / "ymal")
    assert json.loads((held_out / "frozen_choices.json").read_text())["policy"] == "fixed_0.8_4"
    assert pd.read_parquet(held_out / "per_page.parquet").groupby("method").size().iloc[0] == 40
