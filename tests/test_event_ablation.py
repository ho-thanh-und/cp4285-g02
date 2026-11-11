"""Check the Phase 1 event and training-set ablation on synthetic data."""

import json

import pandas as pd
import pytest

import main
from test_samples import raw_experiment  # noqa: F401  (fixture used by learning_config)
from test_learning_curve import learning_config  # noqa: F401
from session_rerank.event_ablation import (
    filter_events, read_ablation_settings, run_event_ablation, training_queries,
)

SETTINGS = dict(output_directory="unused", event_variants=["all_events", "clicks_only", "unexplained_views"],
                explained_view_window_seconds=60, training_sets=["all_sessions", "multi_click_sessions"],
                minimum_session_clicks=2, models=["click_session", "purchase_session"],
                bootstrap_replicates=20, seed=1)


def events():
    return pd.DataFrame(dict(
        event_id=range(7), session_key=["a"] * 5 + ["b"] * 2, eventdate="2016-04-01",
        kind=["query", "click", "view", "view", "view", "view", "click"],
        timeframe=[0, 1_000, 2_000, 200_000, 3_000, 5_000, 9_000],
        query_id=[1, 1, None, None, None, None, 2], item_id=[None, 7, 7, 7, 8, 7, 7]))


def test_event_rules_drop_only_the_intended_views():
    frame = events()
    assert filter_events(frame, "all_events", 60_000).equals(frame)
    assert set(filter_events(frame, "clicks_only", 60_000).kind) == {"query", "click"}
    kept = filter_events(frame, "unexplained_views", 60_000)
    # Event 2 views product 7 one second after its click: dropped. Event 3 is
    # outside the window, event 4 is another product, and event 5 is in session b
    # before that session's click: all kept.
    assert set(kept.event_id) == {0, 1, 3, 4, 5, 6}


def test_multi_click_filter_keeps_whole_sessions_with_enough_clicks():
    metadata = pd.DataFrame(dict(session_key=["a", "a", "b", "c"], clicks=[1, 1, 1, 0], purchases=[0, 0, 0, 1]))
    assert training_queries(metadata, "click", "all_sessions", 2).tolist() == [True, True, True, False]
    assert training_queries(metadata, "click", "multi_click_sessions", 2).tolist() == [True, True, False, False]
    assert training_queries(metadata, "purchase", "all_sessions", 2).tolist() == [True, True, True, True]


@pytest.mark.parametrize("update", [{"event_variants": ["views_only"]}, {"training_sets": []},
                                    {"models": ["neumf"]}, {"explained_view_window_seconds": 0}])
def test_invalid_settings_are_rejected(tmp_path, update):
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_ablation_settings(path)


def test_ablation_rebuilds_cached_features_and_compares_every_combination(learning_config, tmp_path):
    path, _ = learning_config
    run = main.manual_training_main(path, queries="all")
    settings = tmp_path / "ablation.json"
    settings.write_text(json.dumps(SETTINGS))
    # The run fails loudly if the rebuilt all_events features differ from the cache.
    output = run_event_ablation(run, settings, output_root=tmp_path / "ablation")
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 3 * 2 * 2 * 2  # events x training sets x models x statistics
    differences = pd.read_csv(output / "difference_vs_current.csv")
    current = differences[differences.event_variant.eq("all_events") & differences.training_set.eq("all_sessions")]
    assert current.difference.dropna().eq(0).all()
    for name in ("validation_per_query.parquet", "feature_gain.csv", "audit.json", "ablation_settings.json"):
        assert (output / name).exists()
