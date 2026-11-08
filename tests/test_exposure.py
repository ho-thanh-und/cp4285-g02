"""Check the exposure measures and that the rebuilt lists are the evaluated ones."""

import json

import numpy as np
import pandas as pd
import pytest

import main
from session_rerank.exposure import (HEAD, NO_CLICKS, TAIL, exposure_row, gini, popularity_groups,
                                     read_exposure_settings, run_exposure)
from session_rerank.home_page import run_home_page
from session_rerank.ymal import run_ymal
from test_home_page import SETTINGS as HOME_SETTINGS
from test_ymal import SETTINGS as YMAL_SETTINGS
from session_rerank.sequence_models import run_sequence_models
from test_sequence_models import SEQUENCE


def test_gini_is_zero_for_equal_exposure_and_grows_with_concentration():
    assert gini([3, 3, 3, 3]) == pytest.approx(0)
    assert gini([0, 0, 0, 12]) == pytest.approx(0.75)  # (n - 1) / n for one product of n
    assert gini([0, 0, 0, 0]) == 0
    assert gini([1, 2, 3, 4]) < gini([0, 1, 2, 7])


def test_head_holds_the_requested_share_of_clicks():
    popularity = np.array([50.0, 30.0, 15.0, 5.0, 0.0])
    groups = popularity_groups(popularity, 0.5)
    # 50 of 100 clicks reach the share with the first product alone.
    assert groups.tolist() == [HEAD, TAIL, TAIL, TAIL, NO_CLICKS]
    assert popularity_groups(popularity, 0.6).tolist() == [HEAD, HEAD, TAIL, TAIL, NO_CLICKS]


def test_exposure_row_counts_slots_coverage_and_groups():
    popularity = np.array([50.0, 30.0, 15.0, 5.0, 0.0])
    groups = popularity_groups(popularity, 0.5)
    row = exposure_row([0, 0, 1, 4], eligible=5, popularity=popularity, groups=groups)
    assert row["slots"] == 4 and row["distinct_products"] == 3 and row["coverage"] == pytest.approx(0.6)
    assert row["head_share"] == pytest.approx(0.5) and row["tail_share"] == pytest.approx(0.25)
    assert row["no_click_share"] == pytest.approx(0.25) and row["median_april_clicks"] == pytest.approx(40)


@pytest.mark.parametrize("update", [{"pages": []}, {"pages": ["search"]}, {"head_click_share": 1.0},
                                    {"popularity_window": dict(start="2016-04-01", end_exclusive="2016-05-09")}])
def test_invalid_settings_are_rejected(tmp_path, update):
    settings = dict(pages=["home"], head_click_share=0.5,
                    popularity_window=dict(start="2016-04-01", end_exclusive="2016-04-30")) | update
    path = tmp_path / "exposure.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        read_exposure_settings(path)


def test_cli_dispatches_exposure(monkeypatch):
    calls = []
    monkeypatch.setattr("session_rerank.exposure.run_exposure", lambda *args: calls.append(args))
    main.cli(["exposure"])
    assert len(calls) == 1


def test_home_and_item_lists_reproduce_the_selection_runs(tmp_path):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    rng = np.random.default_rng(0)
    items = np.arange(1, 41)
    pd.DataFrame(dict(item_id=items, price_log2=items % 5, category_id=items % 4,
                      name_tokens=[f"{i % 7},{i % 5 + 10}" for i in items])).to_parquet(run / "products.parquet")
    queries, clicks, qid = [], [], 0
    for day, count in (("2016-04-03", 80), ("2016-04-20", 80), ("2016-05-02", 25), ("2016-05-10", 20)):
        for session in range(count):
            qid += 1
            queries.append((qid, f"{day}-{session}", day, False))
            # Low item numbers are clicked more often, so popularity is uneven.
            for position, item in enumerate(rng.choice(items, 3, replace=False, p=1 / items / (1 / items).sum())):
                clicks.append((qid, 100 * position, int(item)))
    pd.DataFrame(queries, columns=["queryId", "sessionId", "eventdate", "is.test"]).to_csv(
        raw / "train-queries.csv", sep=";", index=False)
    pd.DataFrame(clicks, columns=["queryId", "timeframe", "itemId"]).to_csv(raw / "train-clicks.csv", sep=";", index=False)
    pd.DataFrame(dict(itemId=[1], eventdate=["2016-04-05"])).to_csv(raw / "train-purchases.csv", sep=";", index=False)
    knn = tmp_path / "knn.json"
    knn.write_text(json.dumps(dict(output_directory="x", raw_directory=str(raw), max_items_per_session=50,
                                   shrinkage=0, compare_with=[], bootstrap_replicates=10, seed=1)))
    common = dict(raw_directory=str(raw), trained_run=str(run), knn_config=str(knn))
    selection = dict(name="selection", start="2016-05-09", end_exclusive="2016-05-15")
    home_validation = tmp_path / "home.json"
    home_validation.write_text(json.dumps(HOME_SETTINGS | common))
    home_frozen = run_home_page(home_validation, output_root=tmp_path / "home")
    home_selection = tmp_path / "home_selection.json"
    home_selection.write_text(json.dumps(HOME_SETTINGS | common | dict(
        window=selection, frozen_run=str(home_frozen), frozen_policy="fixed_0.8_item_knn")))
    home_run = run_home_page(home_selection, output_root=tmp_path / "home")
    ymal_selection = tmp_path / "ymal_selection.json"
    ymal_selection.write_text(json.dumps(YMAL_SETTINGS | common | dict(window=selection,
                                                                       frozen_policy="fixed_0.8_4")))
    ymal_run = run_ymal(ymal_selection, output_root=tmp_path / "ymal")
    # SASRec with 6 + 4, frozen on validation and scored on the selection cases.
    sequence = SEQUENCE | dict(models=["sasrec"], diversity=True)
    sequence_validation = tmp_path / "sequence.json"
    sequence_validation.write_text(json.dumps(sequence | dict(home_config=str(home_validation),
                                                              home_run=str(home_frozen))))
    sequence_frozen = run_sequence_models(sequence_validation, output_root=tmp_path / "sequence")
    sequence_selection = tmp_path / "sequence_selection.json"
    sequence_selection.write_text(json.dumps(sequence | dict(
        home_config=str(home_selection), home_run=str(home_run), frozen_run=str(sequence_frozen),
        frozen_policy="six_four_sasrec")))
    sequence_run = run_sequence_models(sequence_selection, output_root=tmp_path / "sequence_selection")
    settings = tmp_path / "exposure.json"
    settings.write_text(json.dumps(dict(
        output_directory="unused", raw_directory=str(raw), trained_run=str(run),
        popularity_window=dict(start="2016-04-01", end_exclusive="2016-04-30"), head_click_share=0.5,
        pages=["home", "item"], home_config=str(home_selection), home_run=str(home_run),
        ymal_config=str(ymal_selection), ymal_run=str(ymal_run), home_sequence_run=str(sequence_run))))
    output = run_exposure(settings, output_root=tmp_path / "exposure")
    summary = pd.read_csv(output / "summary.csv")
    assert set(summary.page) == {"home", "item"}
    assert summary.groupby("page").frozen.sum().tolist() == [1, 1]
    assert summary[summary.method.eq("shopper_clicks")].head_share_ratio.eq(1).all()
    shares = summary.head_share + summary.tail_share + summary.no_click_share
    assert np.allclose(shares, 1) and summary.coverage.between(0, 1).all()
    home = summary[summary.page.eq("home")].set_index("method")
    # The sequence run's lists join the home page, and its frozen policy replaces item-kNN's.
    assert {"sasrec", "six_four_sasrec"} <= set(home.index) and home.loc["six_four_sasrec", "frozen"]
    # Popularity shows the same few products to everyone: more concentrated than item-kNN.
    assert home.loc["popularity", "gini"] > home.loc["item_knn", "gini"]
    assert home.loc["popularity", "distinct_products"] < home.loc["item_knn", "distinct_products"]
    assert home.loc["popularity", "slots"] == 40 * HOME_SETTINGS["top_k"]
