"""Check the home page session two-tower evaluation on a small synthetic catalog."""

import json

import numpy as np
import pandas as pd
import pytest

from session_rerank.home_page import evaluation_cases, read_home_settings, run_home_page, training_examples

SETTINGS = dict(output_directory="unused", raw_directory="unused", trained_run="unused", knn_config="unused",
                training_window=dict(start="2016-04-01", end_exclusive="2016-04-30"),
                window=dict(name="validation", start="2016-05-01", end_exclusive="2016-05-08"),
                history_items=3, max_name_tokens=4, embedding_dim=8, hidden_dim=8, epochs=2, batch_size=16,
                learning_rate=0.01, uniform_negatives=2, popular_negatives=2, popular_negative_power=0.75,
                trending_days=3, top_k=4, relevance_slots=2, pool_size=6, fixed_alphas=[0.8],
                diversity_bases=["two_tower", "item_knn"], max_cases=1000, epoch_check_cases=10,
                bootstrap_replicates=10, seed=1, threads=1)


@pytest.mark.parametrize("update", [
    {"window": dict(name="confirmation", start="2016-05-16", end_exclusive="2016-06-02")},
    {"window": dict(name="selection", start="2016-05-09", end_exclusive="2016-05-15")},
    {"relevance_slots": 4}, {"diversity_bases": ["gru4rec"]}, {"epochs": 0}])
def test_invalid_settings_and_sealed_weeks_are_rejected(tmp_path, update):
    path = tmp_path / "home.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_home_settings(path)


def test_training_examples_predict_each_next_click_from_its_prefix():
    sequences = [("a", "2016-04-03", [5, 6, 7]), ("b", "2016-04-01", [9])]
    history, targets, days = training_examples(sequences, "2016-04-01", history=2)
    assert targets.tolist() == [6, 7] and days.tolist() == [2, 2]
    assert history.tolist() == [[5, -1], [5, 6]]
    cases, total = evaluation_cases(sequences, max_cases=10, seed=0)
    assert total == 2 and cases[1] == ("a", [5, 6], {7})


def test_validation_run_then_frozen_selection_run(tmp_path):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    rng = np.random.default_rng(0)
    items = np.arange(1, 41)
    pd.DataFrame(dict(item_id=items, price_log2=items % 5, category_id=items % 4,
                      name_tokens=[f"{i % 7},{i % 5 + 10}" for i in items])).to_parquet(run / "products.parquet")
    queries, clicks, purchases, qid = [], [], [], 0
    for day, count in (("2016-04-03", 80), ("2016-04-20", 80), ("2016-05-02", 25), ("2016-05-10", 20)):
        for session in range(count):
            qid += 1
            queries.append((qid, f"{day}-{session}", day, False))
            for position, item in enumerate(rng.choice(items, 3, replace=False)):
                clicks.append((qid, 100 * position, int(item)))
    pd.DataFrame(queries, columns=["queryId", "sessionId", "eventdate", "is.test"]).to_csv(
        raw / "train-queries.csv", sep=";", index=False)
    pd.DataFrame(clicks, columns=["queryId", "timeframe", "itemId"]).to_csv(raw / "train-clicks.csv", sep=";", index=False)
    pd.DataFrame(dict(itemId=[1, 2], eventdate=["2016-04-05", "2016-04-06"])).to_csv(
        raw / "train-purchases.csv", sep=";", index=False)
    knn = tmp_path / "knn.json"
    knn.write_text(json.dumps(dict(output_directory="x", raw_directory=str(raw), max_items_per_session=50,
                                   shrinkage=0, compare_with=[], bootstrap_replicates=10, seed=1)))
    common = dict(raw_directory=str(raw), trained_run=str(run), knn_config=str(knn))
    validation = tmp_path / "home.json"
    validation.write_text(json.dumps(SETTINGS | common))
    output = run_home_page(validation, output_root=tmp_path / "home")
    per_case = pd.read_parquet(output / "per_case.parquet")
    methods = {"popularity", "trending", "item_knn", "two_tower", "six_four_two_tower", "fixed_0.8_two_tower",
               "six_four_item_knn", "fixed_0.8_item_knn"}
    assert set(per_case.method) == methods
    assert per_case.groupby("method").size().nunique() == 1 and per_case.groupby("method").size().iloc[0] == 50
    assert per_case.recall.between(0, 1).all()
    assert (output / "session_two_tower.pt").exists() and len(pd.read_csv(output / "training_log.csv")) == 2
    # Selection reuses the saved model and records the frozen policy first.
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(SETTINGS | common | dict(
        window=dict(name="selection", start="2016-05-09", end_exclusive="2016-05-15"),
        frozen_run=str(output), frozen_policy="fixed_0.8_two_tower")))
    held_out = run_home_page(selection, output_root=tmp_path / "home")
    assert json.loads((held_out / "frozen_choices.json").read_text())["policy"] == "fixed_0.8_two_tower"
    assert pd.read_parquet(held_out / "per_case.parquet").groupby("method").size().iloc[0] == 40
