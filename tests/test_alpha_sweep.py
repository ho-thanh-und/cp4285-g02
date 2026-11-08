"""Check the fixed-alpha sweep with synthetic data, never the real course dataset."""

import json

import numpy as np
import pandas as pd
import pytest

import main
from test_samples import raw_experiment  # noqa: F401  (fixture used by learning_config)
from test_learning_curve import learning_config  # noqa: F401
from session_rerank.alpha_sweep import (
    _fixed_ndcg_at, curve_gaps, history_group, policy_alphas, read_sweep_settings, run_alpha_sweep,
)

SETTINGS = dict(alpha_min=.2, alpha_max=1., fallback_alpha=.8)


def sweep(**update):
    return dict(fixed_alphas=[.9, .5], exponential_betas=[1, 5], shuffle_control=True,
                bootstrap_replicates=50, seed=7) | update


def test_history_groups_separate_missing_sparse_and_measured():
    assert history_group(0, np.nan) == "no_history"
    assert history_group(1, np.nan) == "sparse_history"
    assert history_group(3, .4) == "measured_history"


def test_alphas_follow_each_policy_and_shuffle_only_measured_queries():
    context = pd.DataFrame(dict(
        history_ild=[np.nan, np.nan, .1, .5, .9, 1.],
        history_group=["no_history", "sparse_history"] + ["measured_history"] * 4,
    ), index=[10, 11, 12, 13, 14, 15])
    alphas = policy_alphas(context, SETTINGS, sweep())
    assert set(alphas["relevance"].values()) == {1.}
    assert set(alphas["fixed_0.5"].values()) == {.5}
    exponential, shuffled = alphas["exponential_5"], alphas["shuffled_5"]
    # No measurable variety: both the policy and its control use the fallback.
    assert exponential[10] == exponential[11] == shuffled[10] == shuffled[11] == .8
    # More session variety gives a smaller relevance weight, ending at alpha_min.
    measured = [exponential[q] for q in (12, 13, 14, 15)]
    assert measured == sorted(measured, reverse=True) and measured[-1] == pytest.approx(.2)
    # The control reuses the same measured alphas, only reassigned between queries.
    assert sorted(shuffled[q] for q in (12, 13, 14, 15)) == sorted(measured)


def test_fixed_curve_interpolation_and_range():
    ild, ndcg = [.5, .3, .4], [.1, .3, .2]
    assert _fixed_ndcg_at(.35, ild, ndcg) == pytest.approx(.25)
    assert np.isnan(_fixed_ndcg_at(.6, ild, ndcg))


def test_gap_is_positive_when_a_policy_beats_the_fixed_curve():
    rows = []
    # Fixed curve: ILD .2 -> NDCG .5, ILD .4 -> NDCG .3. The tested policy sits at
    # ILD .3 with NDCG .45, so the curve gives .4 there and the gap is +.05.
    for session in range(30):
        for policy, ndcg, ild in (("relevance", .5, .2), ("fixed_0.5", .3, .4), ("exponential_1", .45, .3)):
            rows.append(dict(model="m", policy_id=policy, session_key=f"s{session}", query_id=session,
                             history_group="measured_history", click_ndcg=ndcg, combined_ild=ild))
    gaps = curve_gaps(pd.DataFrame(rows), sweep(fixed_alphas=[.5]))
    row = gaps[gaps.population.eq("measured_history")].iloc[0]
    assert row.ndcg_gap_vs_fixed_curve == pytest.approx(.05)
    assert row.gap_ci_low == pytest.approx(.05) and row.gap_ci_high == pytest.approx(.05)


@pytest.mark.parametrize("update", [{"fixed_alphas": []}, {"fixed_alphas": [1.]}, {"fixed_alphas": [.5, .5]},
                                    {"exponential_betas": [0]}, {"bootstrap_replicates": 0}])
def test_invalid_sweep_settings_are_rejected(tmp_path, update):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep(**update)))
    with pytest.raises(ValueError):
        read_sweep_settings(path)


def test_sweep_reuses_a_trained_run_and_saves_every_table(learning_config, tmp_path):
    path, _ = learning_config
    folder = main.manual_training_main(path, queries="all")
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep()))
    output = run_alpha_sweep(folder, sweep_path, output_root=tmp_path / "sweeps")
    for name in ("sweep_settings.json", "validation_alpha_sweep.parquet", "summary.csv",
                 "history_diagnostics.csv", "adaptive_alpha_spread.csv", "gap_vs_fixed_curve.csv",
                 "alpha_curve_all.png", "alpha_curve_measured_history.png"):
        assert (output / name).exists(), name
    measured = pd.read_parquet(output / "validation_alpha_sweep.parquet")
    assert set(measured.policy_id) == {"relevance", "fixed_0.9", "fixed_0.5", "exponential_1",
                                       "exponential_5", "shuffled_1", "shuffled_5"}
    assert measured[measured.policy_id.eq("relevance")].alpha.eq(1).all()
    unmeasured = measured[measured.family.isin(["exponential", "shuffled"])
                          & ~measured.history_group.eq("measured_history")]
    assert unmeasured.alpha.eq(.8).all()
    # Every policy ranks the same queries for every model.
    assert measured.groupby(["model", "policy_id"]).query_id.nunique().nunique() == 1


def test_cli_requires_a_trained_run_for_alpha_sweep(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("session_rerank.alpha_sweep.run_alpha_sweep", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main.cli(["alpha-sweep"])
    with pytest.raises(SystemExit):
        main.cli(["prepare", "--trained-run", str(tmp_path)])
    main.cli(["alpha-sweep", "--trained-run", str(tmp_path)])
    assert len(calls) == 1 and calls[0][0] == tmp_path
