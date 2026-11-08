"""Check the error-analysis tables."""

import numpy as np
import pandas as pd
import pytest

import main
from session_rerank.error_analysis import outcome_table, popularity_bucket


def test_outcome_table_counts_wins_ties_and_losses_per_bucket():
    frame = pd.DataFrame(dict(group=["a", "a", "a", "b"], ndcg_model=[0.5, 0.2, 0.3, 0.0],
                              ndcg_reference=[0.1, 0.2, 0.4, 0.0]))
    table = outcome_table(frame, "group").set_index("bucket")
    assert table.loc["a", "cases"] == 3 and table.loc["a", "difference"] == pytest.approx((0.4 + 0 - 0.1) / 3)
    assert table.loc["a", "win_share"] == pytest.approx(1 / 3) and table.loc["a", "loss_share"] == pytest.approx(1 / 3)
    assert table.loc["b", "tie_share"] == pytest.approx(1)
    assert table.case_share.sum() == pytest.approx(1)


def test_popularity_buckets_separate_products_without_clicks():
    buckets = popularity_bucket(pd.Series([0, 1, 9, 10, 250]))
    assert buckets.astype(str).tolist() == ["0 (no April clicks)", "1-9", "1-9", "10-99", "100+"]


def test_cli_dispatches_error_analysis(monkeypatch):
    calls = []
    monkeypatch.setattr("session_rerank.error_analysis.run_error_analysis", lambda *args: calls.append(args))
    main.cli(["error-analysis"])
    assert len(calls) == 1
