"""Check the filter -> LightGBM -> rerank pipeline's settings, metrics and comparisons."""

import json

import numpy as np
import pandas as pd
import pytest

import main
from session_rerank.filter_pipeline import (
    differences_vs_logged, list_metrics, read_pipeline_settings, summarize,
)

SETTINGS = dict(output_directory="unused", top_k=10, k1_values=[50, 100], models=["purchase_session"],
                policies=[{"id": "relevance", "family": "relevance"}], bootstrap_replicates=50, seed=1)


@pytest.mark.parametrize("update", [{"k1_values": []}, {"k1_values": [5]}, {"models": []},
                                    {"policies": [{"id": "a", "family": "relevance"}] * 2},
                                    {"policies": [{"id": "a", "family": "magic"}]}])
def test_invalid_settings_are_rejected(tmp_path, update):
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(SETTINGS | update))
    with pytest.raises(ValueError):
        read_pipeline_settings(path)


def test_list_metrics_judge_against_every_click_and_report_unshown_products():
    top = np.array([10, 11, 12])
    clicked = {11, 99}            # 99 was clicked but the filter dropped it.
    logged = {10, 11, 99}         # 12 was never shown by the site.
    distances = {"combined": np.array([[0, .2, .4], [.2, 0, .6], [.4, .6, 0]], dtype=float)}
    metrics = list_metrics(top, clicked, logged, distances, k=3)
    ideal = 1 + 1 / np.log2(3)
    assert metrics["click_ndcg"] == pytest.approx((1 / np.log2(3)) / ideal)
    assert metrics["click_recall"] == pytest.approx(.5)
    assert metrics["unshown_share"] == pytest.approx(1 / 3)
    assert metrics["combined_ild"] == pytest.approx(.4)


def test_differences_compare_each_pool_with_the_logged_list_per_query():
    rows = []
    for query in range(40):
        for pool, k1, ndcg in (("logged", 0, .2), ("two_tower", 100, .15)):
            rows.append(dict(query_id=query, session_key=f"s{query // 2}", model="m", policy_id="relevance",
                             pool=pool, k1=k1, click_ndcg=ndcg, combined_ild=.5, token_ild=.9))
    frame = pd.DataFrame(rows)
    diff = differences_vs_logged(frame, 50, 0)
    ndcg = diff[diff.metric.eq("click_ndcg")].iloc[0]
    assert ndcg.difference == pytest.approx(-.05) and ndcg.ci_low == pytest.approx(-.05)
    assert diff[diff.metric.eq("combined_ild")].difference.iloc[0] == pytest.approx(0)
    assert set(summarize(frame.assign(click_recall=0, category_ild=0, unshown_share=0,
                                      pool_click_coverage=1, pool_size=10)).pool) == {"logged", "two_tower"}


def test_cli_requires_both_runs_for_filter_pipeline(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("session_rerank.filter_pipeline.run_filter_pipeline", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main.cli(["filter-pipeline", "--trained-run", str(tmp_path)])
    with pytest.raises(SystemExit):
        main.cli(["two-tower", "--trained-run", str(tmp_path), "--two-tower-run", str(tmp_path)])
    main.cli(["filter-pipeline", "--trained-run", str(tmp_path), "--two-tower-run", str(tmp_path)])
    assert len(calls) == 1 and calls[0][:2] == (tmp_path, tmp_path)
