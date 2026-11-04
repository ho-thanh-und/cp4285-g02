"""Check session completeness, original ranks, strict prefixes, and held-out protection."""

import json
from pathlib import Path

import pandas as pd
import pytest

from session_rerank.samples import prepare_samples, read_protocol, select_sessions


@pytest.fixture
def raw_experiment(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"itemId": [10, 20, 30], "pricelog2": [2., 3., 4.],
                  "product.name.tokens": ["1,2", "2,3", "4,5"]}).to_csv(raw / "products.csv", sep=";", index=False)
    pd.DataFrame({"itemId": [10, 20, 30], "categoryId": [1, 1, 2]}).to_csv(raw / "product-categories.csv", sep=";", index=False)
    queries, clicks, views, purchases = [], [], [], []
    splits = {}
    for split_index, (name, day, end) in enumerate([
        ("train", "2016-04-01", "2016-04-04"),
        ("validation", "2016-05-01", "2016-05-04"),
        ("selection", "2016-05-09", "2016-05-12"),
        ("confirmation", "2016-05-16", "2016-05-19"),
    ]):
        splits[name] = dict(start=day, end_exclusive=end, target_queries=100)
        for session_offset, count in [(0, 3), (1, 2)]:
            date = day[:-2] + f"{int(day[-2:]) + session_offset:02d}"
            sid = split_index + 1  # Deliberately reused on another date.
            for position in range(count):
                qid = split_index * 100 + session_offset * 10 + position + 1
                queries.append(dict(queryId=qid, sessionId=sid, userId=99999,
                                    timeframe=(position + 1) * 100, duration=999999,
                                    eventdate=date, **{"searchstring.tokens": "1,2", "categoryId": 1,
                                                     "items": "10,20,30", "is.test": False}))
                clicks.append(dict(queryId=qid, itemId=[30, 20, 10][position], timeframe=(position + 1) * 100 + 50))
            views.extend([dict(sessionId=sid, userId=99999, itemId=20, timeframe=t, eventdate=date) for t in [50, 200]])
            purchases.append(dict(sessionId=sid, userId=99999, itemId=30, timeframe=160,
                                  eventdate=date, ordernumber="1234567890123456789"))
    # A visible query shares a session with hidden labels: exclude both.
    for qid, hidden in [(999, False), (1000, True)]:
        queries.append(dict(queryId=qid, sessionId=99, userId=88888, timeframe=qid,
                            duration=1, eventdate="2016-04-03", **{"searchstring.tokens": "1", "categoryId": 1,
                            "items": "10,20,30", "is.test": hidden}))
    pd.DataFrame(queries).to_csv(raw / "train-queries.csv", sep=";", index=False)
    pd.DataFrame(clicks).to_csv(raw / "train-clicks.csv", sep=";", index=False)
    pd.DataFrame(views).to_csv(raw / "train-item-views.csv", sep=";", index=False)
    pd.DataFrame(purchases).to_csv(raw / "train-purchases.csv", sep=";", index=False)
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(dict(raw_directory=str(raw), output_directory=str(tmp_path / "prepared"),
                                     seed=4285, history_max_events=100, splits=splits)))
    return raw, config


def load_tables(folder):
    """Load every prepared table for assertions on its actual contents."""
    return {path.stem: pd.read_parquet(path) for path in folder.glob("*.parquet")}


def test_complete_anonymous_sessions_and_no_personal_fields(raw_experiment):
    _, config = raw_experiment
    tables = load_tables(prepare_samples(config))
    assert set(tables) == {"products", "sessions", "queries", "candidates", "clicks", "events", "histories", "confirmation_ids"}
    for table in tables.values():
        assert all("user" not in name.lower() and name != "duration" for name in table.columns)
    sessions = tables["sessions"]
    assert sessions.groupby("split").query_count.sum().eq(5).all()
    assert sessions.session_key.is_unique
    assert not set(tables["queries"].query_id) & {999, 1000}
    all_queries = pd.concat([tables["queries"][["session_key"]], tables["confirmation_ids"][["session_key"]]])
    actual = all_queries.groupby("session_key").size()
    assert actual.equals(sessions.set_index("session_key").query_count.reindex(actual.index).rename(None))


def test_clicks_keep_original_positions_and_orders(raw_experiment):
    _, config = raw_experiment
    tables = load_tables(prepare_samples(config))
    clicked = tables["clicks"].set_index("query_id")
    assert clicked.loc[1, "original_rank"] == 3
    assert clicked.loc[2, "original_rank"] == 2
    assert clicked.loc[3, "original_rank"] == 1
    assert clicked.rank_match_status.eq("matched").all()
    candidates = tables["candidates"].query("query_id == 1")
    assert candidates.item_id.tolist() == [10, 20, 30]
    assert candidates.clicked.tolist() == [False, False, True]


def test_histories_have_only_strictly_earlier_events_in_this_session(raw_experiment):
    _, config = raw_experiment
    tables = load_tables(prepare_samples(config))
    history = tables["histories"].query("query_id == 2")
    assert history.event_timeframe_ms.tolist() == [50, 100, 150, 160]
    assert set(history.event_type) == {"query", "view", "click", "purchase"}
    assert history.age_ms.gt(0).all()
    assert not history.source_query_id.eq(2).fillna(False).any()
    assert history.session_key.nunique() == 1
    assert tables["events"].query("event_type == 'purchase'").order_number.eq("1234567890123456789").all()


def test_confirmation_has_ids_only(raw_experiment):
    _, config = raw_experiment
    tables = load_tables(prepare_samples(config))
    reserved = tables["confirmation_ids"]
    assert len(reserved) == 5
    assert set(reserved.columns) == {"query_id", "session_key", "eventdate", "query_timeframe_ms"}
    for name in ["queries", "candidates", "clicks", "events", "histories"]:
        assert "confirmation" not in set(tables[name].split)
    assert not set(reserved.query_id) & set(tables["queries"].query_id)


def test_session_sampling_is_reproducible_and_targets_are_soft(raw_experiment):
    raw, config = raw_experiment
    protocol = read_protocol(config)
    for spec in protocol["splits"].values():
        spec["target_queries"] = 1
    first, ids = select_sessions(raw, protocol)
    second, ids_again = select_sessions(raw, protocol)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(ids, ids_again)
    assert first.groupby("split").query_count.sum().ge(2).all()
    assert not set(ids.queryId) & {999, 1000}


@pytest.mark.parametrize("items,status", [("10,20,30,30", "ambiguous_duplicate"), ("10,20", "missing_candidate")])
def test_ambiguous_and_missing_click_ranks_are_not_guessed(raw_experiment, items, status):
    raw, config = raw_experiment
    path = raw / "train-queries.csv"
    queries = pd.read_csv(path, sep=";")
    queries.loc[queries.queryId == 1, "items"] = items
    queries.to_csv(path, sep=";", index=False)
    tables = load_tables(prepare_samples(config))
    click = tables["clicks"].set_index("query_id").loc[1]
    assert click.rank_match_status == status
    assert pd.isna(click.original_rank)


def test_prepared_data_reuses_verified_tables_and_rejects_changes(raw_experiment):
    _, config = raw_experiment
    folder = prepare_samples(config)
    assert prepare_samples(config) == folder
    queries = pd.read_parquet(folder / "queries.parquet")
    queries.loc[0, "candidate_count"] = 9999
    queries.to_parquet(folder / "queries.parquet", index=False)
    with pytest.raises(RuntimeError, match="Prepared table changed"):
        prepare_samples(config)


def test_protocol_prevents_overlapping_splits(raw_experiment):
    _, config = raw_experiment
    protocol = json.loads(config.read_text())
    protocol["splits"]["validation"]["start"] = "2016-04-01"
    config.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Overlapping"):
        read_protocol(config)


def test_invalid_target_click_time_stops_preparation(raw_experiment):
    raw, config = raw_experiment
    path = raw / "train-clicks.csv"
    clicks = pd.read_csv(path, sep=";")
    clicks.loc[clicks.queryId == 1, "timeframe"] = 100
    clicks.to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError, match="strictly after"):
        prepare_samples(config)
    assert not (config.parent / "prepared").exists()


def test_prefix_window_keeps_recent_events_and_reports_the_full_count(raw_experiment):
    _, config = raw_experiment
    protocol = json.loads(config.read_text())
    protocol["history_max_events"] = 2
    config.write_text(json.dumps(protocol))
    tables = load_tables(prepare_samples(config))
    query = tables["queries"].set_index("query_id").loc[2]
    assert query.history_available_count == 4
    assert query.history_used_count == 2
    assert tables["histories"].query("query_id == 2").event_timeframe_ms.tolist() == [150, 160]


def test_equal_click_times_share_order_without_changing_original_rank(raw_experiment):
    raw, config = raw_experiment
    path = raw / "train-clicks.csv"
    clicks = pd.read_csv(path, sep=";")
    clicks = pd.concat([clicks, pd.DataFrame([
        dict(queryId=1, itemId=20, timeframe=175),
        dict(queryId=1, itemId=10, timeframe=175),
    ])], ignore_index=True)
    clicks.to_csv(path, sep=";", index=False)
    tables = load_tables(prepare_samples(config))
    observed = tables["clicks"].query("query_id == 1").set_index("item_id")
    assert observed.loc[30, "click_time_order"] == 1
    assert observed.loc[20, "click_time_order"] == observed.loc[10, "click_time_order"] == 2
    assert observed.loc[30, "original_rank"] == 3
    assert observed.loc[20, "original_rank"] == 2
    assert observed.loc[10, "original_rank"] == 1
