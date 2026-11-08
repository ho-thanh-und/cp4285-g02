"""Check the rank-weighted two-tower filter on a tiny synthetic run folder."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import main
from session_rerank.two_tower import (
    RunData, click_weight, estimate_eta, other_weight, popularity_features, read_two_tower_settings,
    train_two_tower,
)

SETTINGS = dict(
    output_directory="unused", raw_directory="unused", use_popularity=True,
    seed=1, threads=1, embedding_dim=8, hidden_dim=8, max_name_tokens=4,
    max_query_tokens=3, max_history_items=3, max_position=3, epochs=2, batch_size=4, learning_rate=.01,
    weight_decay=0.0, propensity_eta="estimate", propensity_fit_max_rank=10, max_positive_weight=5.0,
    shown_negatives_per_positive=2, unshown_negatives_per_positive=2, unshown_negative_weight=.5,
    k1_values=[2, 5],
)


@pytest.fixture
def settings(run_folder):
    return SETTINGS | {"raw_directory": str(run_folder.parent / "raw")}


@pytest.fixture
def run_folder(tmp_path):
    """Two categories of 12 products, two text searches, sessions with several queries.

    Training queries fall on April 1-3 and validation on May 1. Raw CSVs hold the
    same clicks plus extra views of product 7 on April 2 for the popularity checks.
    """
    folder = tmp_path / "queries_all"
    for split in ("train", "validation"):
        (folder / f"{split}_features").mkdir(parents=True)
    items = np.arange(1, 25)
    pd.DataFrame(dict(item_id=items, price_log2=items % 4, category_id=np.where(items <= 12, 5, 9),
                      name_tokens=[f"{i % 3},{100 + (i % 2)}" for i in items])).to_parquet(folder / "products.parquet")
    queries, events, candidates = [], [], []
    rng = np.random.default_rng(0)
    event_id = 0
    for number in range(16):
        split = "train" if number < 12 else "validation"
        session = f"s{number // 2}"
        date = f"2016-04-0{1 + number // 4}" if split == "train" else "2016-05-01"
        time = 1000 * (number % 2)
        category = 0 if number in (3, 13) else (5 if number % 4 < 2 else 9)
        pool = items if category == 0 else items[(items <= 12) == (category == 5)]
        logged = rng.permutation(pool)[:8]
        clicked = {int(logged[2]), int(logged[5])}
        queries.append((number, session, split, date, time, "1,100" if category == 0 else None, category,
                        ",".join(map(str, logged))))
        event_id += 1
        events.append((event_id, session, date, "query", time, number, None, None))
        for item in clicked:
            event_id += 1
            events.append((event_id, session, date, "click", time + 10, number, item, None))
        for rank, item in enumerate(logged, start=1):
            candidates.append(dict(query_id=number, session_key=session, split=split, item_id=int(item),
                                   original_rank=rank, clicked=int(item) in clicked, purchased=False,
                                   grade=int(int(item) in clicked)))
    with sqlite3.connect(folder / "development.sqlite") as db:
        db.execute("CREATE TABLE queries (query_id INTEGER, session_key TEXT, split TEXT, eventdate TEXT, "
                   "timeframe INTEGER, query_tokens TEXT, category INTEGER, items TEXT)")
        db.execute("CREATE TABLE events (event_id INTEGER, session_key TEXT, eventdate TEXT, kind TEXT, "
                   "timeframe INTEGER, query_id INTEGER, item_id INTEGER, order_number INTEGER)")
        db.executemany("INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)", queries)
        db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", events)
    frame = pd.DataFrame(candidates)
    for split in ("train", "validation"):
        frame[frame.split.eq(split)].reset_index(drop=True).to_parquet(
            folder / f"{split}_features" / "candidates_00000.parquet")
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([(q[0], q[3]) for q in queries], columns=["queryId", "eventdate"]).to_csv(
        raw / "train-queries.csv", sep=";", index=False)
    pd.DataFrame([(e[5], e[4], e[6]) for e in events if e[3] == "click"],
                 columns=["queryId", "timeframe", "itemId"]).to_csv(raw / "train-clicks.csv", sep=";", index=False)
    pd.DataFrame(dict(sessionId=1, userId="NA", itemId=[7, 7, 7, 999], timeframe=1,
                      eventdate=["2016-04-02"] * 3 + ["2016-05-01"])).to_csv(
        raw / "train-item-views.csv", sep=";", index=False)
    return folder


def test_rank_weights_punish_top_skips_and_reward_deep_clicks():
    eta = .5
    # A skipped product at rank 1 is penalised more than one skipped at rank 10.
    assert other_weight(1, eta, .5) == pytest.approx(1)
    assert other_weight(1, eta, .5) > other_weight(10, eta, .5) > other_weight(100, eta, .5)
    # A click at rank 10 counts more than a click at rank 1, up to the cap.
    assert click_weight(1, eta, 5) == pytest.approx(1)
    assert click_weight(10, eta, 5) == pytest.approx(10 ** .5)
    assert click_weight(10_000, eta, 5) == 5
    # Never-shown products use the fixed weight; eta = 0 removes the rank effect.
    assert other_weight(np.nan, eta, .5) == .5
    assert click_weight(50, 0, 5) == other_weight(50, 0, .5) == 1


def test_eta_recovers_a_known_rank_effect():
    ranks = np.repeat(np.arange(1, 21), 4000)
    rng = np.random.default_rng(3)
    clicked = rng.random(len(ranks)) < .2 * ranks ** -.6
    assert estimate_eta(pd.DataFrame(dict(original_rank=ranks, clicked=clicked)), 20) == pytest.approx(.6, abs=.08)


def test_popularity_rows_only_count_earlier_days():
    clicks = np.array([[2, 0], [0, 4], [6, 0]], dtype=np.float32)  # 3 days x 2 products
    views = np.zeros_like(clicks)
    features = popularity_features(clicks, views)
    assert features.shape == (4, 2, 3)
    # Day 0 has no earlier days; the "counts available" flag is off.
    assert (features[0] == 0).all()
    # Day 1 sees only day 0: product 0 had 2 clicks in 1 day.
    assert features[1, 0, 0] == pytest.approx(np.log1p(2)) and features[1, 1, 0] == 0
    # The full window (row 3) averages all three days: (2+6)/3 and 4/3 per day.
    assert features[3, 0, 0] == pytest.approx(np.log1p(8 / 3)) and features[3, 1, 0] == pytest.approx(np.log1p(4 / 3))
    assert features[1:, :, 2].min() == 1


def test_training_queries_never_see_their_own_day(run_folder, settings):
    data = RunData(run_folder, settings)
    # Training runs April 1-3 (3 days); validation on May 1 sees all 3 days.
    assert data.window == dict(start="2016-04-01", days=3)
    assert data.query_day[data.query_row[0]] == 0      # April 1
    assert data.query_day[data.query_row[11]] == 2     # April 3
    assert data.query_day[data.query_row[12]] == 3     # validation
    product7 = data.item_index[7]
    # Views of product 7 on April 2 appear from April 3 onward, not on April 2.
    assert data.popularity_inputs[1, product7, 1] < data.popularity_inputs[2, product7, 1]
    # The May view and the unknown product 999 are outside the window or catalog.
    assert data.popularity.sum() == pytest.approx(len(data.candidates["train"].query("clicked")) + 3)


def test_histories_positions_and_pools_use_only_earlier_session_events(run_folder, settings):
    data = RunData(run_folder, settings)
    # Query 1 is the second query of session s0; it sees query 0's clicks only.
    second, first = data.query_row[1], data.query_row[0]
    assert data.query_position[first] == 1 and data.query_position[second] == 2
    assert (data.history[first] == -1).all()
    seen = set(data.item_ids[data.history[second][data.history[second] >= 0]])
    clicks0 = set(data.candidates["train"].query("query_id == 0 and clicked").item_id)
    assert seen == clicks0
    # Category pages filter their category; text searches filter the whole catalog.
    assert set(data.item_category[data.pool(5)]) == {5} and len(data.pool(5)) == 12
    assert len(data.pool(0)) == 24


@pytest.mark.parametrize("update", [{"epochs": 0}, {"k1_values": []}, {"propensity_eta": -1},
                                    {"shown_negatives_per_positive": 0, "unshown_negatives_per_positive": 0},
                                    {"learning_rate": 0}])
def test_invalid_settings_are_rejected(tmp_path, update):
    path = tmp_path / "tower.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_two_tower_settings(path)


@pytest.mark.parametrize("use_popularity", [True, False])
def test_training_saves_filter_and_recall_tables(run_folder, settings, tmp_path, use_popularity):
    path = tmp_path / "tower.json"
    path.write_text(json.dumps(settings | {"use_popularity": use_popularity}))
    output = train_two_tower(run_folder, path, output_root=tmp_path / "towers")
    for name in ("two_tower_settings.json", "two_tower.pt", "item_embeddings.npy", "item_ids.npy",
                 "training_log.csv", "validation_retrieval.parquet", "retrieval_summary.csv", "recall_at_k1.png",
                 "validation_top10.parquet", "top10_summary.csv"):
        assert (output / name).exists(), name
    top = pd.read_parquet(output / "validation_top10.parquet")
    assert top.click_ndcg.dropna().between(0, 1).all() and top.combined_ild.dropna().between(0, 1).all()
    # Lists drawn only from the logged products never contain unshown products.
    assert top[top.pool.eq("logged")].unshown_share.eq(0).all()
    summary = pd.read_csv(output / "retrieval_summary.csv")
    assert set(summary.method) == {"two_tower", "site_order", "popularity", "random"}
    assert summary.recall.between(0, 1).all()
    assert len(np.load(output / "item_embeddings.npy")) == 24
    assert json.loads((output / "two_tower_settings.json").read_text())["eta"] >= 0


def test_cli_dispatches_two_tower(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("session_rerank.two_tower.train_two_tower", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main.cli(["two-tower"])
    main.cli(["two-tower", "--trained-run", str(tmp_path)])
    assert len(calls) == 1 and calls[0][0] == tmp_path
