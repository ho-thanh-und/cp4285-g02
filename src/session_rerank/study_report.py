"""Export session-bootstrap statistics, figures, examples, and one offline HTML report."""

import base64
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .study_files import finish_stage, save_table, stage_complete, write_json

LABELS = {
    "relevance": "Relevance only", "quota": "Relevance quota",
    "fixed": "Fixed weight", "adaptive": "Session adaptive", "original": "Original order",
}
METRICS = ("ndcg_at_10", "recall_at_10", "token_ild", "category_ild", "combined_ild")


def bootstrap_session_means(values, sessions, seed=4285, replicates=2000):
    """Resample whole sessions while keeping the reported mean query-weighted.

    Queries in one visit are related. Sampling individual queries would treat
    those observations as independent. We instead sample session IDs with
    replacement, bring every query in each sampled session, and take the mean.
    Sums divided by counts retain the original equal weighting of queries.
    """
    frame = pd.DataFrame(dict(value=np.asarray(values, dtype=float), session=np.asarray(sessions)))
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return np.full(replicates, np.nan)
    grouped = frame.groupby("session").value.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    return grouped["sum"].to_numpy()[draws].sum(axis=1) / grouped["count"].to_numpy()[draws].sum(axis=1)


def confidence_intervals(metrics):
    """Calculate 95% percentile intervals for each policy and paired baseline changes."""
    session_map = metrics.drop_duplicates("query_id").set_index("query_id").session_key
    records = []
    for metric in METRICS:
        pivot = metrics.pivot(index="query_id", columns="policy_id", values=metric)
        for policy in pivot.columns:
            values = pivot[policy].dropna()
            sessions = session_map.reindex(values.index)
            distribution = bootstrap_session_means(values, sessions)
            # Pair only queries with both values, then resample their complete sessions.
            # This matters if missing product metadata changes a list's metric coverage.
            pair = pivot[[policy]].rename(columns={policy: "policy"}).join(
                pivot.relevance.rename("baseline")
            ).dropna()
            difference = pair.policy - pair.baseline
            paired = bootstrap_session_means(difference, session_map.reindex(pair.index))
            records.append(dict(
                policy_id=policy, metric=metric, queries=len(values), sessions=sessions.nunique(),
                estimate=float(values.mean()) if len(values) else np.nan,
                lower=float(np.quantile(distribution, 0.025)) if len(values) else np.nan,
                upper=float(np.quantile(distribution, 0.975)) if len(values) else np.nan,
                paired_queries=len(pair), delta=float(difference.mean()) if len(pair) else np.nan,
                delta_lower=float(np.quantile(paired, 0.025)) if len(pair) else np.nan,
                delta_upper=float(np.quantile(paired, 0.975)) if len(pair) else np.nan,
            ))
    return pd.DataFrame(records)


def make_figures(folder, confirmation, intervals, frozen):
    """Create exportable charts directly from saved results, with consistent policy labels."""
    figures = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    validation = pd.read_csv(folder / "validation_grid.csv")
    validation = validation[validation.segment.eq("all")]
    selected = frozen["selection_choice"]["winner"]
    families = dict(confirmation[["policy_id", "family"]].drop_duplicates().values)

    fig, axis = plt.subplots(figsize=(9, 5))
    for family, group in validation.groupby("family", sort=False):
        axis.scatter(group.ndcg_at_10, group.combined_ild, label=LABELS[family], alpha=0.65, s=32)
    baseline = validation[validation.policy_id.eq("relevance")].iloc[0]
    axis.axvline(baseline.ndcg_at_10 * 0.95, linestyle="--", color="#888", label="5% relevance loss boundary")
    axis.set(xlabel="Validation NDCG@10", ylabel="Validation combined ILD@10",
             title="Validation trade-off across the declared search grid")
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "validation_tradeoff.png", dpi=180)
    plt.close(fig)
    figures.append("validation_tradeoff.png")

    all_rows = confirmation[confirmation.segment.eq("all")].set_index("policy_id")
    policy_ids = all_rows.index.tolist()
    names = [LABELS[families[p]] + (" *" if p == selected else "") for p in policy_ids]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, metric, title in zip(axes, ("ndcg_at_10", "recall_at_10", "combined_ild"),
                                   ("NDCG@10", "Recall@10", "Combined ILD@10")):
        points = intervals[intervals.metric.eq(metric)].set_index("policy_id").reindex(policy_ids)
        estimate = points.estimate.to_numpy()
        errors = np.vstack([np.maximum(0, estimate - points.lower), np.maximum(0, points.upper - estimate)])
        axis.bar(np.arange(len(names)), estimate, color="#287f8e", yerr=errors, capsize=3)
        axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        axis.set_title(title)
        axis.set_ylim(bottom=0)
    fig.suptitle("Frozen confirmation results — 95% session-bootstrap intervals\n* policy chosen on selection")
    fig.tight_layout()
    fig.savefig(folder / "confirmation_metrics.png", dpi=180)
    plt.close(fig)
    figures.append("confirmation_metrics.png")

    segments = confirmation[~confirmation.segment.eq("all")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, title in zip(axes, ("ndcg_at_10", "combined_ild"), ("NDCG@10", "Combined ILD@10")):
        pivot = segments.pivot(index="policy_id", columns="segment", values=metric).reindex(policy_ids)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
        axis.set_xticks(range(len(pivot.columns)), [x.replace("_", " ") for x in pivot.columns], rotation=25, ha="right")
        axis.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                axis.text(j, i, f"{value:.3f}" if pd.notna(value) else "undefined",
                          ha="center", va="center", color="#111")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("Confirmation results by earlier session history")
    fig.tight_layout()
    fig.savefig(folder / "history_segments.png", dpi=180)
    plt.close(fig)
    figures.append("history_segments.png")

    fig, axis = plt.subplots(figsize=(12, 3))
    axis.set_axis_off()
    stages = ["Raw CSVs", "Anonymous\nsamples", "LightGBM\nrelevance", "Validation\ntuning",
              "Selection\nchoice", "Freeze", "Confirmation\n+ report"]
    for i, stage in enumerate(stages):
        axis.text(i, 0.5, stage, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.55", facecolor="#e3f1f2", edgecolor="#287f8e"))
        if i < len(stages) - 1:
            axis.annotate("", xy=(i + 0.62, 0.5), xytext=(i + 0.38, 0.5),
                          arrowprops=dict(arrowstyle="->", color="#287f8e"))
    axis.set(xlim=(-0.6, 6.7), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(folder / "architecture.png", dpi=180)
    plt.close(fig)
    figures.append("architecture.png")
    return figures


def recommendation_examples(folder, metrics, frozen, products, contexts):
    """Show deterministic gain/loss cases after final evaluation, labelled as descriptive."""
    selected = frozen["selection_choice"]["winner"]
    policies = frozen["shortlist"]["policies"]
    # If relevance itself won, inspect the adaptive comparison as an additional
    # descriptive example. The selected policy remains unchanged in every result.
    example_policy = selected
    if selected == "relevance":
        example_policy = next(p["id"] for p in policies if p["family"] == "adaptive")
    pivot = metrics.pivot(index="query_id", columns="policy_id", values="ndcg_at_10")
    delta = (pivot[example_policy] - pivot.relevance).dropna().sort_values(kind="stable")
    ids = list(dict.fromkeys(delta.head(3).index.tolist() + delta.tail(3).index.tolist()))
    metadata = products.set_index("item_id")
    contexts = contexts.set_index("query_id")
    records, blocks = [], []
    for query_id in ids:
        group = metrics[metrics.query_id.eq(query_id)].set_index("policy_id")
        blocks.append(f"<h3>Query {query_id}</h3><p>Example policy {escape(example_policy)}. "
                      f"NDCG change {delta.loc[query_id]:+.4f}. Earlier product IDs "
                      f"{escape(contexts.loc[query_id, 'history_items'])}.</p>")
        for policy in ("relevance", example_policy):
            row = group.loc[policy]
            top = json.loads(row.top10_item_ids)
            clicked = set(json.loads(row.clicked_item_ids))
            display = []
            for rank, item in enumerate(top, start=1):
                product = metadata.loc[item]
                display.append(dict(rank=rank, item_id=item, category=product.category_id,
                                    name_token_ids=product.name_tokens, observed_click=item in clicked))
            blocks.append(f"<h4>{escape(policy)}</h4>" + pd.DataFrame(display).to_html(index=False, escape=True))
        records.append(dict(
            query_id=int(query_id), policy_id=example_policy,
            ndcg_change=float(delta.loc[query_id]),
            relevance_top10=json.loads(group.loc["relevance"].top10_item_ids),
            policy_top10=json.loads(group.loc[example_policy].top10_item_ids),
        ))
    write_json(folder / "recommendation_examples.json", records)
    return "\n".join(blocks)


def save_study_report(folder):
    """Save the tables, statistical comparisons, charts and standalone HTML report."""
    if not stage_complete(folder, "confirmation"):
        raise ValueError("Confirmation evaluation must finish before reporting.")
    if stage_complete(folder, "report"):
        return folder / "report.html"
    print("[6/6] Save report, session confidence intervals, figures, and examples", flush=True)
    study = json.loads((folder / "study.json").read_text())
    frozen = json.loads((folder / "frozen_choices.json").read_text())
    data = Path(study["identity"]["dataset_folder"])
    metrics = pd.read_parquet(folder / "confirmation_by_query.parquet")
    summary = pd.read_csv(folder / "confirmation_metrics.csv")
    intervals = confidence_intervals(metrics)
    save_table(intervals, folder / "confirmation_confidence_intervals.csv")
    figures = make_figures(folder, summary, intervals, frozen)
    examples = recommendation_examples(
        folder, metrics, frozen, pd.read_parquet(data / "products.parquet"),
        pd.read_parquet(folder / "confirmation_context.parquet"),
    )
    audit = json.loads((folder / "confirmation_tables" / "audit.json").read_text())
    shortlist = frozen["shortlist"]
    all_rows = summary[summary.segment.eq("all")]
    selected = frozen["selection_choice"]["winner"]
    winner = all_rows[all_rows.policy_id.eq(selected)].iloc[0]
    baseline = all_rows[all_rows.policy_id.eq("relevance")].iloc[0]
    relevance_change = winner.ndcg_at_10 / baseline.ndcg_at_10 - 1 if baseline.ndcg_at_10 else np.nan
    diversity_change = winner.combined_ild - baseline.combined_ild
    # Keep every chart inline so the HTML remains usable when copied by itself.
    def embed(name):
        encoded = base64.b64encode((folder / name).read_bytes()).decode()
        return f'<img alt="{escape(name)}" src="data:image/png;base64,{encoded}">'
    def table(frame):
        return frame.to_html(index=False, float_format=lambda value: f"{value:.4f}", escape=True)
    comparison = intervals[intervals.metric.isin(["ndcg_at_10", "combined_ild"])][[
        "policy_id", "metric", "paired_queries", "delta", "delta_lower", "delta_upper",
    ]]
    display = all_rows[["policy_id", "queries", "relevance_queries", "diversity_queries",
                        "ndcg_at_10", "recall_at_10", "token_ild", "category_ild", "combined_ild", "combined_coverage"]]
    selection = pd.read_csv(folder / "selection_metrics.csv")
    runtime = all_rows[["policy_id", "rerank_ms", "pairwise_ms"]]
    body = f"""
<h1>Anonymous session recommendation study</h1>
<p>CP4285 experiment report. Frozen at {escape(frozen['frozen_at_utc'])}.</p>
<h2>Research question and final result</h2>
<p>Can earlier activity in an anonymous shopping session guide the amount of
diversity added to product search recommendations?</p>
<p>The selection stage chose <strong>{escape(selected)}</strong>. On reserved
confirmation queries, its NDCG@10 was {winner.ndcg_at_10:.4f}, compared with
{baseline.ndcg_at_10:.4f} for relevance-only ranking. The relative change was
{relevance_change:+.1%}. Its combined ILD changed by {diversity_change:+.4f}.
The confidence intervals below quantify uncertainty around the measured changes.</p>
<h2>Architecture and Python execution sequence</h2>
{embed('architecture.png')}
<pre>main.py
  samples.py → prepare_samples()
  train_rankers.py → train_rankers() [existing trained run reused here]
  anonymous_study.py → tune_diversification()
  anonymous_study.py → compare_on_selection()
  anonymous_study.py → freeze_and_evaluate()
  study_report.py → save_study_report()

Shared functions
  ranker_features.py → query-time features and frozen training popularity
  reranking.py → list construction, session context, metrics and policy choice
  diversity.py → category and product-token distances
  study_files.py → fingerprints, atomic files and stage completion records</pre>
<h2>Data preparation and relevance training</h2>
<p>Daily sessions use supplied session ID plus date. Query histories contain
strictly earlier events within that session, with the prepared 100-event cap.
The ranker uses earlier clicks and views, query/product matching, category,
price and aggregate product activity. Aggregate counts use previous training
days for training rows and the frozen training totals for every later split.
The LightGBM LambdaRank model was trained on binary observed-click labels,
with validation NDCG@10 selecting tree settings and early stopping.</p>
{table(pd.DataFrame(json.loads((data / 'audit.json').read_text())['split_counts']).T.reset_index(names='split'))}
<h2>Diversity definition and four approaches</h2>
<p>Category distance is 0 for matching IDs and 1 for different IDs. Token
distance is one minus Jaccard similarity between anonymized product-name
token sets. Combined distance gives both components weight 0.5.
ILD@10 averages all unordered product pairs in the selected list.
Missing comparisons are excluded from reporting with coverage shown.
During list construction, unknown distances receive zero diversity reward.</p>
<ul>
<li>Relevance-only sorts the saved LightGBM scores.</li>
<li>Quota keeps m relevant products and greedily fills the remaining places to
maximize total distance to all products already selected. At m=0, the highest
relevance product seeds the list.</li>
<li>Fixed weight repeatedly maximizes weight × normalized relevance +
(1 − weight) × mean distance to the products already selected.</li>
<li>Adaptive weight uses a threshold on earlier session ILD to choose focused
or broad relevance weights. It uses the validation-chosen fixed default when
fewer than two distinct earlier clicked/viewed products have measurable ILD.</li>
</ul>
<p>Scores are min–max normalized separately within each query for weighted
reranking. Equal scores keep original candidate order. Greedy selection is an
approximation to the list diversity objective. Every list uses the original
logged candidates, with up to 200 products per query in this pilot.</p>
<h2>Validation tuning and reserved selection</h2>
<p>The search covers quota values 0, 2, 4, 6, 8, 10, fixed relevance weights
0.2, 0.4, 0.6, 0.8, 1.0, and adaptive weights recorded below.
Adaptive thresholds come from training-prefix diversity quantiles.
The fixed default weight is tuned on validation.
One setting per family is nominated on validation. Selection then chooses
among those four settings using the recorded rule of maximum combined ILD
with at most 5% relative NDCG loss and no ILD loss against relevance-only.</p>
{embed('validation_tradeoff.png')}
<pre>{escape(json.dumps(shortlist['policies'], indent=2))}</pre>
{table(selection[selection.segment.eq('all')])}
<h2>Frozen confirmation evaluation</h2>
<p>The shortlist, selected policy, input hashes and source-code fingerprints
were saved before confirmation candidate lists and outcomes were opened.
Earlier outcomes in a confirmation session may become history for later
queries in that same session. Every prediction uses the prefix available at
its own query time. Source files and reserved query IDs were verified again.</p>
<p>Confirmation includes {audit['queries']} queries in {audit['sessions']} sessions.
There are {audit['zero_click_queries']} queries with no observed clicked candidate.
NDCG and Recall average queries with an observed candidate click.
Diversity includes every list with at least two products and known comparisons.</p>
{table(display)}
{embed('confirmation_metrics.png')}
<h2>Statistical uncertainty and history segments</h2>
<p>Intervals use 2,000 bootstrap samples of entire daily sessions with seed 4285.
Within each sample the estimator gives equal weight to queries.
Paired changes compare the same queries against relevance-only.
Each interval is a 95% percentile interval, and the comparisons are descriptive
without multiplicity correction. A delta interval spanning zero leaves the
direction of the effect uncertain.</p>
{table(comparison)}
<p>Focused and varied segments use the median of measurable training-prefix
ILD. Sparse history has fewer than two distinct measurable earlier interacted
products. The boundary is {shortlist['segment_threshold']:.4f}.</p>
{embed('history_segments.png')}
{table(summary[~summary.segment.eq('all')][['segment','policy_id','queries','relevance_queries','ndcg_at_10','combined_ild']])}
<h2>Runtime</h2>
<p>Times are local CPU measurements in milliseconds per query.
Reranking time covers list construction. Pairwise time covers distance-matrix
construction and is shared by all approaches for a query.
Model scoring and feature construction are recorded separately in the split
scoring-time JSON files. These timings depend on the local machine and workload.</p>
{table(runtime)}
<h2>Recommendation examples</h2>
<p>Examples show up to three largest gains and losses for the selected policy
against relevance-only. When relevance-only wins selection, examples use the
adaptive shortlist member for descriptive inspection. Product names are
anonymized token IDs. Examples are selected after evaluation and serve as illustrations.</p>
{examples}
<h2>Limitations and interpretation</h2>
<ul>
<li>Logged clicks are a relevance proxy affected by the original display positions.
The experiment cannot establish shopper satisfaction or online exploration gains.</li>
<li>Product-name token overlap measures lexical differences. Category IDs supply
no hierarchy. Increased ILD can include products less aligned with the query.</li>
<li>The pilot uses 2,000 training queries. Sparse session histories and unseen
products limit the evidence available for individual predictions.</li>
<li>Daily session boundaries follow the provided source fields. An inactivity-based
session definition could change the results.</li>
<li>The 5% loss allowance, 50/50 distance blend and limited search grid are declared
experimental choices. Confirmation results retain their measured gains and losses.</li>
<li>The study tests the existing LightGBM ranker and offline rerankers. Additional
neural-model comparisons and online feedback require separate experiments.</li>
</ul>
<h2>Related work</h2>
<p>The session-diversity gate is inspired by Gharahighehi and Vens (2021),
who adapt accuracy/diversity strategies using current-session variety in music
recommendations. This experiment tests a query-conditioned shopping setting
with sparse histories and logged candidate lists.
<a href="https://doi.org/10.1007/s42979-020-00399-2">Personalizing Diversity Versus Accuracy in Session-Based Recommender Systems</a>.</p>
<h2>Reproduction and saved outputs</h2>
<pre>uv run main.py run --ranker-run {escape(study['identity']['ranker_run'])}
uv run main.py resume --study-run {escape(str(folder.resolve()))}</pre>
<p>CSV and Parquet tables, standalone PNG figures, recommendation examples,
frozen choices and stage completion fingerprints accompany this report.
Source code contains comments explaining each stage. The report embeds all
figures and can be copied as a single HTML file.</p>
"""
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CP4285 anonymous session study</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#19363d;line-height:1.65}
h1,h2,h3{line-height:1.25;color:#145466}h2{margin-top:42px;border-bottom:1px solid #c9dfe4;padding-bottom:10px}
img{max-width:100%;height:auto}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px;margin:18px 0}
td,th{padding:9px;border:1px solid #c9dfe4;text-align:left}th{background:#e9f3f5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff5f6;padding:18px;border-radius:8px}a{color:#176f96}
li{margin:8px 0}@media print{body{max-width:none;margin:0}h2{break-after:avoid}img{break-inside:avoid}}
</style><body>""" + body + "</body></html>"
    temporary = folder / "report.html.tmp"
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(folder / "report.html")
    finish_stage(folder, "report", [
        "report.html", "confirmation_confidence_intervals.csv", "recommendation_examples.json", *figures,
    ])
    print(f"Report: {(folder / 'report.html').resolve()}", flush=True)
    return folder / "report.html"
