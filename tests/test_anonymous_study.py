"""Regression checks for list construction, session statistics, and the holdout boundary."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from test_samples import raw_experiment
from session_rerank import anonymous_study as study
from session_rerank import train_rankers as training
from session_rerank.ranker_features import build_ranker_features
from session_rerank.reranking import (
    choose_policy, quota_order, recommendation_order, RELEVANCE_LOSS_LIMIT,
)
from session_rerank.samples import prepare_samples
from session_rerank.study_files import stage_complete, study_lock, write_json
from session_rerank.study_report import bootstrap_session_means, save_study_report


def test_quota_accounts_for_all_already_selected_items():
    distances = np.array([[0, .2, .9, .6], [.2, 0, 0, .6],
                          [.9, 0, 0, .8], [.6, .6, .8, 0]])
    # Start with items 0 and 1. Candidate 2 is far from only item 0,
    # while candidate 3 adds more total distance to the whole selected set.
    assert quota_order([4, 3, 2, 1], distances, keep=2, k=3).tolist() == [0, 1, 3]
    assert quota_order([], np.empty((0, 0)), 0).tolist() == []
    assert quota_order([2, 3, 2], np.zeros((3, 3)), 10).tolist() == [1, 0, 2]
    with pytest.raises(ValueError):
        quota_order([1], np.zeros((1, 1)), -1)


def test_adaptive_switch_and_sparse_fallback():
    distances = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    policy = dict(family="adaptive", threshold=.5, focused_weight=1,
                  broad_weight=0, default_weight=1)
    assert recommendation_order([3, 2, 1], distances, policy, .2).tolist() == [0, 1, 2]
    assert recommendation_order([3, 2, 1], distances, policy, .8).tolist() == [0, 2, 1]
    assert recommendation_order([3, 2, 1], distances, policy, np.nan).tolist() == [0, 1, 2]


def test_choice_rejects_extra_diversity_when_relevance_loss_exceeds_budget():
    summary = pd.DataFrame([
        dict(segment="all", policy_id="relevance", ndcg_at_10=.2, combined_ild=.4),
        dict(segment="all", policy_id="too_much_loss", ndcg_at_10=.18, combined_ild=.9),
        dict(segment="all", policy_id="allowed", ndcg_at_10=.195, combined_ild=.5),
    ])
    assert RELEVANCE_LOSS_LIMIT == .05
    assert choose_policy(summary, summary.policy_id) == "allowed"


def test_bootstrap_samples_whole_sessions_and_keeps_query_weights():
    values = bootstrap_session_means([0, 0, 1], ["A", "A", "B"], replicates=500)
    # Drawing AA gives 0, AB or BA gives 1/3, BB gives 1.
    assert set(np.round(values, 6)) == {0, round(1 / 3, 6), 1}
    np.testing.assert_array_equal(values, bootstrap_session_means(
        [0, 0, 1], ["A", "A", "B"], replicates=500,
    ))


def test_lock_blocks_second_writer_and_releases_on_failure(tmp_path):
    with pytest.raises(RuntimeError, match="already running"):
        with study_lock(tmp_path):
            with study_lock(tmp_path):
                pass
    assert not (tmp_path / "running.lock").exists()


def test_study_runs_freeze_before_confirmation_and_resumes_identically(raw_experiment, monkeypatch):
    raw, config = raw_experiment
    monkeypatch.chdir(config.parent)
    data = prepare_samples(config)
    # Small real fitted models keep this a pipeline test, not a full training benchmark.
    monkeypatch.setattr(training, "GRID", ((3, 1),))
    monkeypatch.setattr(training, "MODEL_SETTINGS", dict(
        objective="lambdarank", n_estimators=5, learning_rate=.1,
        random_state=4285, n_jobs=1, verbosity=-1,
    ))
    ranker = training.train_rankers(data, config.parent / "ranker_runs")
    folder = study.open_study(data, ranker)
    with pytest.raises(FileNotFoundError):
        study.load_confirmation(folder)
    with pytest.raises(ValueError, match="requires"):
        study.score_split(folder, "confirmation")

    # Evaluating a later split uses all training counts even without training rows.
    later = build_ranker_features(data, splits=("selection",))
    early = build_ranker_features(data)
    assert later.prior_product_clicks.max() > 0
    spec = json.loads((ranker / "run.json").read_text())["models"][study.MODEL_NAME]
    model = lgb.Booster(model_file=str(ranker / f"{study.MODEL_NAME}.txt"))
    original_scores = pd.read_parquet(ranker / "validation_scores.parquet")
    rows = early[early.split.eq("validation")]
    np.testing.assert_allclose(model.predict(rows[spec["features"]], num_threads=1),
                               original_scores.query_product_session_score)

    study.tune_diversification(folder)
    assert not (folder / "confirmation_opened.json").exists()
    study.compare_on_selection(folder)
    assert not (folder / "confirmation_opened.json").exists()
    study.freeze_and_evaluate(folder)
    frozen_bytes = (folder / "frozen_choices.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    opened = json.loads((folder / "confirmation_opened.json").read_text())
    assert frozen["frozen_at_utc"] <= opened["opened_at_utc"]
    confirmed = pd.read_parquet(folder / "confirmation_scores.parquet")
    assert set(confirmed.query_id) == set(pd.read_parquet(data / "confirmation_ids.parquet").query_id)
    assert confirmed.split.eq("confirmation").all()
    assert stage_complete(folder, "confirmation")
    report = save_study_report(folder)
    assert report.exists() and "data:image/png;base64," in report.read_text()
    assert stage_complete(folder, "report")
    study.freeze_and_evaluate(folder)
    assert frozen_bytes == (folder / "frozen_choices.json").read_bytes()
    assert save_study_report(folder) == report
    assert study.open_study(data, ranker, folder) == folder

    # A completed stage is reusable only while its files match.
    (folder / "confirmation_metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        study.freeze_and_evaluate(folder)
