"""Test whether session-adaptive alpha beats a sweep of fixed alpha values.

alpha is the relevance weight in greedy reranking. Each next product maximizes
    alpha * relevance + (1 - alpha) * mean distance to products already chosen.
alpha = 1 is relevance-only order; a smaller alpha gives more weight to variety.

A fixed policy uses one alpha for every query. The exponential policy chooses
alpha per query from the session variety H of earlier clicked/viewed products;
beta sets how quickly alpha falls from alpha_max toward alpha_min as H rises.
Queries whose H cannot be measured (fewer than two usable products) receive
fallback_alpha, so adaptation only acts on the "measured" history group.

Sweeping fixed alpha draws a relevance/diversity curve. Adaptation adds value
only if an adaptive point lies above that curve: higher NDCG at equal ILD. The
shuffle control keeps each beta's alpha values but permutes them across the
measured queries. If it scores like the matched policy, linking alpha to the
session's own H is not what produces the result.

Validation only. Models and candidates come from one saved learning-curve run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .adaptive_policies import relevance_metrics, relevance_weight, weighted_history_context
from .core import greedy_rerank
from .diversity import ProductDiversity
from .expanded_data import read_candidates, read_histories
from .ranking_settings import reranking_settings, top_k
from .study_files import save_table, write_json

GROUPS = ("no_history", "sparse_history", "measured_history")
METRICS = ["click_ndcg", "gain_ndcg", "recall_at_k", "combined_ild", "token_ild", "category_ild"]


def read_sweep_settings(path):
    """Reject empty grids and alpha values outside [0, 1] before reading data."""
    sweep = json.loads(Path(path).read_text(encoding="utf-8"))
    alphas, betas = sweep["fixed_alphas"], sweep["exponential_betas"]
    if not alphas or any(not 0 <= a < 1 for a in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("fixed_alphas must be distinct values in [0, 1); alpha = 1 is the relevance policy")
    if not betas or any(b <= 0 for b in betas):
        raise ValueError("exponential_betas must be positive")
    if sweep["bootstrap_replicates"] < 1:
        raise ValueError("bootstrap_replicates must be positive")
    return sweep


def history_group(unique_items, history_ild):
    """Name the evidence an adaptive policy has for one query."""
    if unique_items == 0:
        return "no_history"
    return "sparse_history" if not np.isfinite(history_ild) else "measured_history"


def policy_alphas(context, settings, sweep):
    """Return {policy_id: {query_id: alpha}} for every policy in the sweep.

    Fixed and relevance policies ignore the session. Exponential policies use
    relevance_weight, the same function as the configured reranking stage.
    Shuffled controls permute one beta's measured-query alphas with a fixed seed.
    """
    queries = context.index.to_numpy()
    measured = context.history_group.eq("measured_history").to_numpy()
    rng = np.random.default_rng(sweep["seed"])
    alphas = {"relevance": dict.fromkeys(queries, 1.0)}
    for value in sweep["fixed_alphas"]:
        alphas[f"fixed_{value:g}"] = dict.fromkeys(queries, float(value))
    for beta in sweep["exponential_betas"]:
        policy = dict(family="exponential", beta=beta, alpha_min=settings["alpha_min"],
                      alpha_max=settings["alpha_max"], default_weight=settings["fallback_alpha"])
        values = np.array([relevance_weight(policy, h) for h in context.history_ild])
        alphas[f"exponential_{beta:g}"] = dict(zip(queries, values))
        if sweep["shuffle_control"]:
            shuffled = values.copy()
            shuffled[measured] = rng.permutation(values[measured])
            alphas[f"shuffled_{beta:g}"] = dict(zip(queries, shuffled))
    return alphas


def policy_family(policy_id):
    """Group policy IDs for tables and plots."""
    return policy_id.split("_")[0]


def list_diversity(order, distances):
    """Mean known pair distance for each distance definition; NaN if no known pair."""
    pairs = np.triu_indices(len(order), 1)
    result = {}
    for name, matrix in distances.items():
        values = matrix[np.ix_(order, order)][pairs]
        known = values[np.isfinite(values)]
        result[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
    return result


def measure_sweep(run_folder, settings, sweep, k, label_gains, threads):
    """Rerank every validation query with every model and policy; one row each."""
    scores = read_candidates(run_folder, "validation")
    queries = pd.read_parquet(run_folder / "validation_features" / "queries.parquet")
    catalog = ProductDiversity(pd.read_parquet(run_folder / "products.parquet"))
    context = weighted_history_context(
        queries, read_histories(run_folder, "validation"), catalog, "none",
        metric=settings["diversity_metric"], minimum_items=settings["minimum_history_items"],
    ).set_index("query_id")
    context["history_group"] = [history_group(n, h) for n, h in
                                zip(context.history_unique_items, context.history_ild)]
    alphas = policy_alphas(context, settings, sweep)
    matrix = np.load(run_folder / "validation_features" / "X.npy", mmap_mode="r")
    predictions = {}
    for model in pd.read_csv(run_folder / "model_comparison.csv").model:
        details = json.loads((run_folder / f"{model}_trials.json").read_text())["result"]
        booster = lgb.Booster(model_file=str(run_folder / f"{model}.txt"))
        predictions[model] = booster.predict(matrix[:, details["feature_indices"]], num_threads=threads)
    rows = []
    for number, (qid, group) in enumerate(scores.groupby("query_id", sort=False), start=1):
        items = group.item_id.to_numpy(dtype=int)
        for item in items:
            if item not in catalog.name_tokens:
                catalog.name_tokens[item], catalog.categories[item] = frozenset(), None
        distances = catalog.distances(items)
        reward = np.nan_to_num(distances[settings["diversity_metric"]], nan=0.)
        info = context.loc[qid]
        for model, prediction in predictions.items():
            relevance = prediction[group.index.to_numpy()]
            for policy_id, by_query in alphas.items():
                alpha = float(by_query[qid])
                order = (np.argsort(-relevance, kind="stable")[:k] if policy_id == "relevance"
                         else greedy_rerank(relevance, reward, alpha, k))
                rows.append(dict(
                    model=model, policy_id=policy_id, family=policy_family(policy_id),
                    query_id=int(qid), session_key=group.session_key.iloc[0],
                    history_group=info.history_group, history_ild=float(info.history_ild), alpha=alpha,
                    **relevance_metrics(order, group.clicked, group.purchased, label_gains, k),
                    **list_diversity(order, distances),
                ))
        if number % 100 == 0 or number == len(queries):
            print(f"Alpha sweep: {number:,}/{len(queries):,} queries, {len(predictions)} models, "
                  f"{len(alphas)} policies", flush=True)
    return pd.DataFrame(rows), context


def summarize(measured):
    """Equal-query means for all queries and for each history group."""
    frames = [measured.assign(population="all")] + [
        measured[measured.history_group.eq(name)].assign(population=name) for name in GROUPS]
    data = pd.concat(frames, ignore_index=True)
    keys = ["model", "population", "policy_id", "family"]
    means = data.groupby(keys, sort=False)[METRICS + ["alpha"]].mean()
    counts = data.groupby(keys, sort=False).agg(queries=("query_id", "size"),
                                                click_queries=("click_ndcg", "count"),
                                                diversity_queries=("combined_ild", "count"))
    return means.join(counts).reset_index()


def diagnostics(context, measured):
    """How often the adaptive policy actually adapts, and which alphas it uses."""
    total = len(context)
    rows = [dict(statistic=f"share_{name}", value=float(context.history_group.eq(name).sum() / total))
            for name in GROUPS]
    h = context.history_ild.dropna()
    rows += [dict(statistic=f"history_ild_{q}", value=float(h.quantile(p)))
             for q, p in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))] if len(h) else []
    frame = pd.DataFrame(rows)
    alphas = (measured[measured.family.eq("exponential") & measured.history_group.eq("measured_history")]
              .drop_duplicates(["policy_id", "query_id"]).groupby("policy_id").alpha
              .describe(percentiles=[.25, .5, .75]).reset_index())
    return frame, alphas


def _session_sums(frame, policy_ids, metric):
    """Per-session sums and counts, so bootstrap means are weighted by queries."""
    data = frame[frame.policy_id.isin(policy_ids)].dropna(subset=[metric])
    sums = data.pivot_table(index="session_key", columns="policy_id", values=metric, aggfunc="sum")
    counts = data.pivot_table(index="session_key", columns="policy_id", values=metric, aggfunc="count")
    return sums, counts


def _fixed_ndcg_at(ild, fixed_ild, fixed_ndcg):
    """NDCG of the fixed curve at a given ILD, by linear interpolation; NaN outside it."""
    order = np.argsort(fixed_ild)
    x, y = np.asarray(fixed_ild)[order], np.asarray(fixed_ndcg)[order]
    return float(np.interp(ild, x, y)) if x[0] <= ild <= x[-1] else np.nan


def curve_gaps(measured, sweep):
    """Adaptive NDCG minus fixed-curve NDCG at the same ILD, with session bootstrap CI.

    Whole daily sessions are resampled, matching the earlier study's intervals.
    A positive gap means the policy beats every fixed alpha at its diversity level.
    The fixed curve includes relevance (alpha = 1) and all swept fixed alphas.
    """
    rng = np.random.default_rng(sweep["seed"])
    fixed_ids = ["relevance"] + [f"fixed_{a:g}" for a in sweep["fixed_alphas"]]
    rows = []
    for population in ("all", "measured_history"):
        subset = measured if population == "all" else measured[measured.history_group.eq(population)]
        sessions = sorted(subset.session_key.unique())
        weights = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions)),
                                  size=sweep["bootstrap_replicates"])
        for model, frame in subset.groupby("model", sort=False):
            tested = [p for p in frame.policy_id.unique() if p not in fixed_ids]
            means = {}
            for metric in ("click_ndcg", "combined_ild"):
                sums, counts = _session_sums(frame, fixed_ids + tested, metric)
                sums, counts = sums.reindex(sessions).fillna(0), counts.reindex(sessions).fillna(0)
                point = sums.sum() / counts.sum()
                boot = (weights @ sums.to_numpy()) / (weights @ counts.to_numpy())
                means[metric] = (point, pd.DataFrame(boot, columns=sums.columns))
            (ndcg, ndcg_boot), (ild, ild_boot) = means["click_ndcg"], means["combined_ild"]
            fixed_ndcg_boot, fixed_ild_boot = ndcg_boot[fixed_ids].to_numpy(), ild_boot[fixed_ids].to_numpy()
            for policy in tested:
                gap = ndcg[policy] - _fixed_ndcg_at(ild[policy], ild[fixed_ids], ndcg[fixed_ids])
                policy_ndcg, policy_ild = ndcg_boot[policy].to_numpy(), ild_boot[policy].to_numpy()
                boot = np.array([
                    policy_ndcg[b] - _fixed_ndcg_at(policy_ild[b], fixed_ild_boot[b], fixed_ndcg_boot[b])
                    for b in range(len(weights))])
                boot = boot[np.isfinite(boot)]
                rows.append(dict(
                    model=model, population=population, policy_id=policy, family=policy_family(policy),
                    click_ndcg=ndcg[policy], combined_ild=ild[policy], ndcg_gap_vs_fixed_curve=gap,
                    gap_ci_low=float(np.quantile(boot, .025)) if len(boot) else np.nan,
                    gap_ci_high=float(np.quantile(boot, .975)) if len(boot) else np.nan,
                    bootstrap_inside_curve=len(boot) / len(weights), sessions=len(sessions),
                ))
    return pd.DataFrame(rows)


def plot_curves(summary, folder):
    """One panel per model: the fixed-alpha curve with adaptive and shuffled points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outputs = []
    for population in ("all", "measured_history"):
        data = summary[summary.population.eq(population)]
        models = list(data.model.unique())
        columns = 3
        rows = int(np.ceil(len(models) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
        for axis, model in zip(axes.flat, models):
            frame = data[data.model.eq(model)]
            fixed = frame[frame.family.isin(["relevance", "fixed"])].sort_values("alpha")
            axis.plot(fixed.combined_ild, fixed.click_ndcg, "-o", color="0.45", ms=4, label="fixed alpha")
            for _, point in fixed.iterrows():
                axis.annotate(f"{point.alpha:g}", (point.combined_ild, point.click_ndcg),
                              textcoords="offset points", xytext=(3, 3), fontsize=7, color="0.35")
            for family, style in (("exponential", dict(marker="s", color="#1f6fb2")),
                                  ("shuffled", dict(marker="x", color="#c0392b"))):
                points = frame[frame.family.eq(family)]
                if len(points):
                    axis.scatter(points.combined_ild, points.click_ndcg, label=family, zorder=3, **style)
            axis.set_title(model, fontsize=10)
            axis.set_xlabel("combined ILD@10")
            axis.set_ylabel("click NDCG@10")
            axis.grid(alpha=.3)
        for axis in list(axes.flat)[len(models):]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        label = "all validation queries" if population == "all" else "queries with measurable session variety"
        figure.suptitle(f"Fixed alpha curve vs session-adaptive alpha ({label})")
        figure.tight_layout()
        name = f"alpha_curve_{population}.png"
        figure.savefig(folder / name, dpi=150)
        plt.close(figure)
        outputs.append(name)
    return outputs


def run_alpha_sweep(run_folder, sweep_path=Path("config/alpha_sweep.json"), output_root=None):
    """Read one trained run, sweep the policies on validation, and save every table."""
    run_folder = Path(run_folder)
    if not (run_folder / "training_complete.json").exists():
        raise ValueError(f"{run_folder} has no completed training stage")
    sweep = read_sweep_settings(sweep_path)
    config = json.loads((run_folder / "expanded_run.json").read_text())["config"]
    # Use only keys the validator knows, so older saved runs remain readable.
    settings = reranking_settings(config)
    k = top_k(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or sweep["output_directory"]) / f"{stamp}_{run_folder.name}"
    folder.mkdir(parents=True)
    write_json(folder / "sweep_settings.json", dict(
        trained_run=str(run_folder), top_k=k, sweep=sweep,
        reranking={key: settings[key] for key in ("alpha_min", "alpha_max", "fallback_alpha",
                                                  "minimum_history_items", "diversity_metric")},
    ))
    measured, context = measure_sweep(run_folder, settings, sweep, k, config["label_gains"], config["threads"])
    summary = summarize(measured)
    shares, alpha_spread = diagnostics(context, measured)
    gaps = curve_gaps(measured, sweep)
    save_table(measured, folder / "validation_alpha_sweep.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(shares, folder / "history_diagnostics.csv")
    save_table(alpha_spread, folder / "adaptive_alpha_spread.csv")
    save_table(gaps, folder / "gap_vs_fixed_curve.csv")
    plot_curves(summary, folder)
    print(f"Alpha sweep saved to {folder}", flush=True)
    return folder
