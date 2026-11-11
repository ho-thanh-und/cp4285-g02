import numpy as np
import pytest
from session_rerank.core import (
    greedy_rerank, ranking_metrics, prior_click_items, jaccard_distance,
)


def test_current_future_and_tied_clicks_cannot_enter_history():
    events = [(10, 1, 101), (11, 1, 102), (12, 2, 103), (13, 3, 104), (14, 4, 105)]
    assert prior_click_items(events, query_id=2, query_time=13) == [102, 101]


def test_repeated_history_keeps_latest_unique_items():
    events = [(1, 1, 8), (2, 2, 9), (3, 3, 8), (4, 4, 10)]
    assert prior_click_items(events, 5, 5, limit=2) == [10, 8]


def test_metric_values_against_hand_computation():
    d = np.array([[0, .2, .8], [.2, 0, .6], [.8, .6, 0]])
    ndcg, recall, ild = ranking_metrics([1, 0], [True, False, True], d, k=2)
    assert ndcg == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert recall == .5
    assert ild == pytest.approx(.2)


def test_full_relevance_recovers_stable_order_and_only_candidate_items():
    d = np.ones((4, 4)) - np.eye(4)
    assert greedy_rerank([.8, .9, .8, .1], d, 1, 3).tolist() == [1, 0, 2]


def test_diversification_changes_second_choice():
    d = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    assert greedy_rerank([1, .9, .8], d, .2, 2).tolist() == [0, 2]


def test_distance_empty_and_identical():
    assert jaccard_distance(frozenset(), frozenset()) == 0
    assert jaccard_distance(frozenset([1]), frozenset([1])) == 0
    assert jaccard_distance(frozenset([1]), frozenset([2])) == 1
