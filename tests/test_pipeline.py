"""Small end-to-end selection/evaluation regressions, without model training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from session_rerank.pipeline import comparison, diversification, evaluation, reporting, training
from session_rerank.pipeline.state import (
    RESULTS, CANDIDATES, FROZEN_CHOICES, EVALUATION_CACHE, read_json,
)


def make_toy_experiment(directory, monkeypatch):
    """Fixed scores allow real re-ranking and policy search on three queries."""
    from session_rerank import iteration3_report as scoring
    from session_rerank import optimization_report as metrics

    monkeypatch.chdir(directory)
    RESULTS.mkdir(parents=True)
    Path("config").mkdir()
    Path("results/optimization").mkdir()
    Path("data/iteration3").mkdir(parents=True)
    Path("data/processed").mkdir()
    Path("data/processed_v2").mkdir()

    ids = np.arange(11, 23)
    tokens = [frozenset([1, 2])] * 8 + [
        frozenset([3, 4]), frozenset([5, 6]),
        frozenset([7, 8]), frozenset([9, 10]),
    ]

    class Catalog:
        row = dict(zip(ids, range(len(ids))))
        sizes = np.full(12, 2)

        def similarity(self, items, other=None):
            other = items if other is None else other
            return np.array([
                [len(tokens[self.row[a]] & tokens[self.row[b]])
                 / len(tokens[self.row[a]] | tokens[self.row[b]]) for b in other]
                for a in items
            ], dtype=np.float32)

    offsets = np.arange(0, 37, 12)
    labels = np.zeros(36, dtype=np.int8)
    labels[[0, 1, 13, 15, 29]] = 1
    data = {}
    for split, start in [("validation", 10), ("selection", 20), ("confirmation", 30)]:
        data[split] = dict(
            X=np.zeros((36, 63), dtype=np.float32),
            y=labels.copy(), offsets=offsets.copy(), items=np.tile(ids, 3),
            prefixes=np.array([[0] * 10, [11, 12] + [0] * 8, [11, 12, 19, 20] + [0] * 6]),
            meta=pd.DataFrame(dict(
                query_id=[1001, 1002, 1003], session_id=[start + 1, start + 2, start + 3],
                date=["2016-05-20", "2016-05-25", "2016-05-27"],
                history_count=[0, 2, 4], history_diversity=[np.nan, .2, .8],
                category_count=[1, 1, 2],
            )),
        )

    base = np.tile(np.linspace(1, 0, 12), 3)
    tree_scores = {}
    for index, name in enumerate([
        "previous_views_60k", "previous_views_60k_63", "collab31", "collab63", "anonymous31",
    ]):
        scores = base.copy()
        scores[(index * 3) % 12::12] += .25
        tree_scores[name] = scores
    neural_scores = {
        "bpr": np.roll(base.reshape(3, 12), 1, axis=1).ravel(),
        "gru_ce": np.roll(base.reshape(3, 12), 2, axis=1).ravel(),
        "sas_ce": np.roll(base.reshape(3, 12), 3, axis=1).ravel(),
    }
    events = []

    def load_data(split, variant="base"):
        events.append(("load", split))
        return data[split]

    def build_split(split):
        events.append(("build", split))
        if split == "confirmation":
            assert FROZEN_CHOICES.exists(), "Confirmation opened before choices were frozen"

    monkeypatch.setattr(scoring, "load_data", load_data)
    monkeypatch.setattr(scoring, "build_split", build_split)
    monkeypatch.setattr(scoring, "tree_prediction", lambda name, dataset, split: tree_scores[name])
    monkeypatch.setattr(scoring, "neural", lambda variant, split: neural_scores[variant])
    monkeypatch.setattr(metrics, "Catalog", Catalog)
    monkeypatch.setattr(metrics, "OUT", metrics.OUT)  # Restore reporting's legacy global after each test.
    from session_rerank import iteration3
    monkeypatch.setattr(iteration3, "load_data", load_data)

    class ReferenceModel:
        def __init__(self, **kwargs):
            pass

        def predict(self, X, **kwargs):
            return base

    monkeypatch.setattr(scoring.lgb, "Booster", ReferenceModel)

    reserved = pd.DataFrame(dict(queryId=[1001, 1002, 1003], sessionId=[31, 32, 33]))
    reserved.to_parquet("data/iteration3/confirmation_ids.parquet", index=False)
    pd.DataFrame({"sessionId": [1]}).to_parquet("data/processed/test.parquet", index=False)
    pd.DataFrame({"sessionId": [2]}).to_parquet("data/processed_v2/confirmation_ids.parquet", index=False)
    protocol = dict(confirmation_query_ids_sha256=scoring.hash_ids(reserved.queryId))
    Path("config/iteration3_protocol.json").write_text(json.dumps(protocol))
    for name in ("recbole_gru_ce.yaml", "recbole_sas_ce.yaml"):
        Path("config", name).write_text("fixture configuration\n", encoding="utf-8", newline="\n")
    for name in ("collab31", "collab63", "anonymous31", "gru_ce", "sas_ce"):
        (RESULTS / f"{name}.json").write_text("{}\n", encoding="utf-8", newline="\n")
    for name in ("collab31", "collab63", "anonymous31"):
        (RESULTS / f"{name}.txt").write_text(name + "\n", encoding="utf-8", newline="\n")
    for name in ("gru_ce", "sas_ce"):
        (RESULTS / f"{name}.pth").write_text(name + "\n", encoding="utf-8", newline="\n")
    for name in ("views_60k.txt", "views_60k_63.txt", "recbole_gru4rec.pth"):
        Path("results/optimization", name).write_text(name + "\n", encoding="utf-8", newline="\n")
    return data, events


@pytest.fixture
def toy_experiment(tmp_path, monkeypatch):
    return make_toy_experiment(tmp_path, monkeypatch)


def test_legacy_pipeline_calls_the_six_stages_with_their_results(monkeypatch):
    from session_rerank.pipeline import legacy
    events = []

    def record(name, result=None):
        def stage(*arguments):
            events.append((name, arguments))
            return result
        return stage

    for name, result in [
        ("prepare_dataset", None), ("train_models", None), ("compare_models", "comparison"),
        ("choose_diversification", "policies"), ("freeze_and_evaluate", "evaluation"),
        ("save_results", None),
    ]:
        monkeypatch.setattr(legacy, name, record(name, result))
    legacy.run_legacy_experiment()
    assert events == [
        ("prepare_dataset", ()), ("train_models", ()), ("compare_models", ()),
        ("choose_diversification", ("comparison",)),
        ("freeze_and_evaluate", ("comparison", "policies")),
        ("save_results", ("evaluation",)),
    ]


def test_selection_and_evaluation_match_original_pipeline(toy_experiment):
    golden = json.loads((Path(__file__).parent / "fixtures/pipeline_reference.json").read_text())
    _, events = toy_experiment
    selected = comparison.compare_models()
    policies = diversification.choose_diversification(selected)
    actual = selected.frozen_payload(policies)
    actual["hashes"] = {key.replace("\\", "/"): value for key, value in actual["hashes"].items()}
    assert actual == golden["frozen_selection"]
    assert not FROZEN_CHOICES.exists()
    assert all(split in ("validation", "selection") for _, split in events)

    results = evaluation.freeze_and_evaluate(selected, policies)
    assert FROZEN_CHOICES.exists()
    assert EVALUATION_CACHE.exists()
    assert not (RESULTS / "confirmation_summary.csv").exists()
    assert not (RESULTS / "confirmation_examples.json").exists()
    assert list(results.metrics) == golden["methods"]
    np.testing.assert_allclose(np.stack(list(results.metrics.values()), axis=1), golden["values"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(results.diversity_sweep, golden["sweep"], rtol=1e-12, atol=1e-14)
    assert results.examples == golden["examples"]
    assert results.receipt["frozen_selection_sha256"] == hashlib.sha256(FROZEN_CHOICES.read_bytes()).hexdigest()
    assert {key: value for key, value in results.receipt.items() if key != "frozen_selection_sha256"} == {
        key: value for key, value in golden["receipt"].items() if key != "frozen_selection_sha256"
    }

    reporting.save_results(results)
    for relative in [
        "confirmation_summary.csv", "confirmation_paired_differences.csv",
        "confirmation_slices.csv", "confirmation_examples.json", "confirmation_receipt.json",
        "improvement_vs_previous.csv", "figures/confirmation_models.png",
    ]:
        assert (RESULTS / relative).exists(), relative


def test_cached_comparison_does_not_reopen_selection(toy_experiment, monkeypatch):
    comparison.compare_models()
    _, events = toy_experiment
    events.clear()
    monkeypatch.setattr(comparison, "_nominate_models", lambda: pytest.fail("Models nominated twice"))
    comparison.compare_models()
    assert events == []


def test_frozen_pipeline_does_not_fit_or_reevaluate(toy_experiment, monkeypatch):
    selected = comparison.compare_models()
    policies = diversification.choose_diversification(selected)
    first = evaluation.freeze_and_evaluate(selected, policies)
    from session_rerank import optimization_report, benchmark
    monkeypatch.setattr(optimization_report, "evaluate", lambda *a, **k: pytest.fail("Confirmation repeated"))
    monkeypatch.setattr(benchmark, "develop_reference_model", lambda: pytest.fail("Model retrained"))
    training.train_models()
    restored = comparison.compare_models()
    settings = diversification.choose_diversification(restored)
    second = evaluation.freeze_and_evaluate(restored, settings)
    assert first.examples == second.examples
    for name in first.metrics:
        np.testing.assert_array_equal(first.metrics[name], second.metrics[name])


def test_changed_recorded_model_blocks_confirmation(toy_experiment):
    selected = comparison.compare_models()
    policies = diversification.choose_diversification(selected)
    Path("results/iteration3/collab31.txt").write_text("changed model")
    with pytest.raises(RuntimeError, match="changed"):
        evaluation.freeze_and_evaluate(selected, policies)
    assert not EVALUATION_CACHE.exists()


def test_diversification_cannot_replace_frozen_choices(toy_experiment):
    selected = comparison.compare_models()
    policies = diversification.choose_diversification(selected)
    evaluation.freeze_choices(selected, policies)
    changed = dict(policies)
    changed["fixed"] = dict(policies["fixed"], weight=.123)
    with pytest.raises(RuntimeError, match="Choices differ"):
        evaluation.freeze_choices(selected, changed)


def test_partial_selection_prevents_new_fitting(toy_experiment, monkeypatch):
    from session_rerank import benchmark
    CANDIDATES.write_text("{}")
    monkeypatch.setattr(benchmark, "develop_reference_model", lambda: pytest.fail("Late fitting"))
    training.train_models()


def test_legacy_cache_is_still_readable(toy_experiment):
    selected = comparison.compare_models()
    policies = diversification.choose_diversification(selected)
    results = evaluation.freeze_and_evaluate(selected, policies)
    with np.load(EVALUATION_CACHE) as cache:
        values, names, sweep, weights = [cache[key] for key in ("values", "names", "sweep", "weights")]
    np.savez_compressed(EVALUATION_CACHE, values=values, names=names, sweep=sweep, weights=weights)
    (RESULTS / "confirmation_examples.json").write_text(json.dumps(results.examples))
    (RESULTS / "confirmation_receipt.json").write_text(json.dumps(results.receipt))
    restored = evaluation.freeze_and_evaluate(selected, policies)
    assert restored.examples == results.examples
    for name in results.metrics:
        np.testing.assert_array_equal(restored.metrics[name], results.metrics[name])


def test_dataset_stage_prepares_features_without_training(tmp_path, monkeypatch):
    from session_rerank.pipeline.dataset import prepare_dataset
    from session_rerank import prepare, optimization_data, view_features, iteration3, benchmark
    monkeypatch.chdir(tmp_path)
    Path("data/raw").mkdir(parents=True)
    Path("data/raw/train-queries.csv").touch()
    events = []
    monkeypatch.setattr(prepare, "audit_and_split", lambda: events.append("audit"))
    monkeypatch.setattr(optimization_data, "prepare", lambda: events.append("statistics"))
    monkeypatch.setattr(optimization_data, "build_features", lambda split: events.append(("features", split)))
    monkeypatch.setattr(view_features, "build", lambda split: events.append(("views", split)))
    monkeypatch.setattr(iteration3, "reserve_and_prepare", lambda: events.append("reserve"))
    monkeypatch.setattr(iteration3, "build_split", lambda split: events.append(("context", split)))
    monkeypatch.setattr(benchmark, "develop_reference_model", lambda: pytest.fail("Preparation trained a model"))
    prepare_dataset()
    assert events == [
        "audit", "statistics", ("features", "validation"), ("views", "validation"),
        ("features", "train"), ("views", "train"), "reserve",
        ("context", "validation"), ("context", "train"),
    ]


def test_training_stage_keeps_the_original_model_variants(tmp_path, monkeypatch):
    from session_rerank import benchmark, optimize_ranker, recbole_model, iteration3, neural_search
    monkeypatch.chdir(tmp_path)
    events = []
    monkeypatch.setattr(benchmark, "develop_reference_model", lambda: events.append("reference"))
    monkeypatch.setattr(optimize_ranker, "train", lambda name, columns, **kw: events.append((name, columns, kw)))
    monkeypatch.setattr(recbole_model, "train", lambda: events.append("bpr"))
    monkeypatch.setattr(iteration3, "train_tree", lambda *args: events.append(args))
    monkeypatch.setattr(neural_search, "train", lambda variant: events.append(variant))
    training.train_models()
    assert events == [
        "reference", ("views_60k", 48, {"leaves": 31, "variant": "views"}),
        ("views_60k_63", 48, {"leaves": 63, "variant": "views"}), "bpr",
        ("collab31", 31, False), ("collab63", 63, False), ("anonymous31", 31, True),
        "sas_ce", "gru_ce",
    ]


def test_confirmation_rejects_overlapping_sessions(toy_experiment):
    selected = comparison.compare_models()
    policies = diversification.choose_diversification(selected)
    pd.DataFrame({"sessionId": [31]}).to_parquet("data/processed/test.parquet", index=False)
    with pytest.raises(AssertionError):
        evaluation.freeze_and_evaluate(selected, policies)
    assert not EVALUATION_CACHE.exists()


def test_legacy_entry_point_uses_the_legacy_pipeline(monkeypatch):
    from session_rerank.pipeline import legacy
    from session_rerank.iteration3 import run_all
    called = []
    monkeypatch.setattr(legacy, "run_legacy_experiment", lambda: called.append("six stages"))
    run_all()
    assert called == ["six stages"]
