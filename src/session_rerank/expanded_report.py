"""Build a readable offline report for purchase-aware and continuous-policy experiments."""

import base64
from html import escape
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .study_report import bootstrap_session_means
from .study_files import stage_complete, finish_stage, save_table
from .ranking_settings import top_k

REPORT_METRICS = ("ndcg_at_k","click_ndcg","gain_ndcg","recall_at_k","purchase_recall","combined_ild")


def expanded_intervals(metrics, replicates=2000, seed=4285):
    """Paired session bootstrap with separate denominators for purchases and clicks."""
    sessions=metrics.drop_duplicates("query_id").set_index("query_id").session_key
    result=[]
    for metric in REPORT_METRICS:
        pivot=metrics.pivot(index="query_id",columns="policy_id",values=metric)
        for policy in pivot:
            values=pivot[policy].dropna()
            distribution=bootstrap_session_means(values,sessions.reindex(values.index),seed=seed,replicates=replicates)
            pair=pivot[[policy]].rename(columns={policy:"policy"}).join(pivot.relevance.rename("baseline")).dropna()
            delta=pair.policy-pair.baseline
            change=bootstrap_session_means(delta,sessions.reindex(pair.index),seed=seed,replicates=replicates)
            result.append(dict(
                policy_id=policy,metric=metric,queries=len(values),paired_queries=len(pair),
                estimate=values.mean(),lower=np.nanquantile(distribution,.025) if len(values) else np.nan,
                upper=np.nanquantile(distribution,.975) if len(values) else np.nan,
                delta=delta.mean(),delta_lower=np.nanquantile(change,.025) if len(pair) else np.nan,
                delta_upper=np.nanquantile(change,.975) if len(pair) else np.nan,
            ))
    return pd.DataFrame(result)


def write_expanded_report(config,folder):
    """Export all comparisons and trajectories with explicit white-background styling."""
    if not stage_complete(folder,"confirmation_evaluation"):
        raise ValueError("Complete final evaluation before reporting.")
    if stage_complete(folder,"report"):
        return folder/"report.html"
    print("[7/7] Save report, uncertainty intervals and session trajectories",flush=True)
    k=top_k(config)
    metrics=pd.read_parquet(folder/"confirmation_policies.parquet")
    summary=pd.read_csv(folder/"confirmation_metrics.csv")
    frozen=json.loads((folder/"frozen_choices.json").read_text())
    shortlist=json.loads((folder/"shortlist.json").read_text())
    objective=json.loads((folder/"objective.json").read_text())
    audit=json.loads((folder/"data_audit.json").read_text())
    intervals=expanded_intervals(metrics, config["bootstrap_replicates"], config["seed"])
    save_table(intervals,folder/"confidence_intervals.csv")
    policy_ids=summary.policy_id.tolist()
    names=[p.replace("__"," / ") for p in policy_ids]
    # Policy labels remain visible without crowding a horizontal chart.
    fig,axes=plt.subplots(1,3,figsize=(16,max(5,len(names)*.38)))
    for ax,metric,title in zip(axes,("ndcg_at_k","purchase_recall","combined_ild"),
                              (f"Primary NDCG@{k}",f"Purchase Recall@{k}",f"Combined ILD@{k}")):
        values=intervals[intervals.metric.eq(metric)].set_index("policy_id").reindex(policy_ids)
        y=np.arange(len(names))
        ax.barh(y,values.estimate,color="#277d8c")
        ax.errorbar(values.estimate,y,xerr=np.vstack([
            np.maximum(0,values.estimate-values.lower),np.maximum(0,values.upper-values.estimate)]),
            fmt="none",ecolor="#17212b",capsize=3)
        ax.set_yticks(y,names if ax is axes[0] else [""]*len(names))
        ax.set_title(title); ax.invert_yaxis(); ax.set_xlim(left=0)
    fig.suptitle("Fresh confirmation results with 95% session-bootstrap intervals")
    fig.tight_layout(); fig.savefig(folder/"confirmation.png",dpi=170); plt.close(fig)

    grid=pd.read_csv(folder/"validation_grid.csv")
    fig,ax=plt.subplots(figsize=(9,5))
    for family,group in grid.groupby("family"):
        ax.scatter(group.ndcg_at_k,group.combined_ild,label=family,alpha=.65)
    base=grid[grid.policy_id.eq("relevance")].iloc[0]
    ax.axvline(base.ndcg_at_k*.95,color="gray",linestyle="--")
    ax.set(xlabel=f"Validation primary NDCG@{k}",ylabel=f"Combined ILD@{k}",title="Relevance and diversity trade-off")
    ax.legend(); fig.tight_layout(); fig.savefig(folder/"tradeoff.png",dpi=170); plt.close(fig)

    # Actual query-time points show whether the policy responds to changing history.
    info=pd.read_parquet(folder/"confirmation_features"/"queries.parquet")
    traces=metrics.merge(info[["query_id","query_timeframe_ms"]],on="query_id",validate="many_to_one")
    save_table(traces[["session_key","query_id","query_timeframe_ms","policy_id","history_unique_items",
                       "history_ild","alpha"]],folder/"session_trajectories.csv")
    base_context=traces[traces.policy_id.eq("relevance")].sort_values("query_timeframe_ms")
    grouped=base_context.groupby("session_key").history_ild.agg(["count","mean","first","last"])
    sessions=[]
    measurable=grouped[grouped["count"]>=2]
    if len(measurable):
        sessions += [measurable["mean"].idxmin(), measurable["mean"].idxmax(),
                     (measurable["last"]-measurable["first"]).idxmin()]
    sparse=grouped[grouped["count"].eq(0)]
    if len(sparse):
        sessions.append(sparse.index[0])
    sessions=list(dict.fromkeys(sessions))[:4]
    if not sessions:
        sessions=info.session_key.drop_duplicates().head(3).tolist()
    fig,axes=plt.subplots(max(1,len(sessions)),2,figsize=(12,max(4,len(sessions)*3)),squeeze=False)
    for row,session in enumerate(sessions):
        group=traces[traces.session_key.eq(session)]
        for policy,p in group.groupby("policy_id"):
            if policy=="original_order": continue
            p=p.sort_values("query_timeframe_ms")
            x=np.arange(1,len(p)+1)
            axes[row,0].plot(x,p.history_ild,marker=".",label=policy)
            axes[row,1].plot(x,p.alpha,marker=".",label=policy)
        axes[row,0].set(title=f"{session} — observed session variety",ylabel="History ILD",xlabel="Query order")
        axes[row,1].set(title="Relevance weight at each query",ylabel="Alpha",xlabel="Query order",ylim=(-.05,1.05))
    axes[0,1].legend(fontsize=6,loc="best")
    fig.tight_layout(); fig.savefig(folder/"trajectories.png",dpi=170); plt.close(fig)

    def table(frame):
        return frame.to_html(index=False,float_format=lambda x:f"{x:.4f}",escape=True)
    def image(name):
        return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((folder/name).read_bytes()).decode()+'">'
    winner=summary.set_index("policy_id").loc[frozen["winner"]]
    baseline=summary.set_index("policy_id").loc["relevance"]
    examples=[]
    deltas=metrics.pivot(index="query_id",columns="policy_id",values="ndcg_at_k")
    changes=(deltas[frozen["winner"]]-deltas.relevance).dropna().sort_values()
    for qid in list(dict.fromkeys(changes.head(2).index.tolist()+changes.tail(2).index.tolist())):
        examples.append(table(metrics[metrics.query_id.eq(qid)][[
            "query_id","policy_id","ndcg_at_k","purchase_recall","history_ild","alpha",
            "topk_item_ids","purchased_item_ids",
        ]]))
    model_table=pd.read_csv(folder/"model_comparison.csv")
    # Pilot scores are a fixed development reference. Objectives and feature
    # extensions are shown explicitly so changes are not attributed solely to size.
    pilot_report_path="results/anonymous_rankers/20261104T192951Z-321c6432/validation_metrics.csv"
    from pathlib import Path
    pilot=table(pd.read_csv(pilot_report_path).query("segment == 'all'")) if Path(pilot_report_path).exists() else "<p>Pilot reference unavailable.</p>"
    body=f"""
<h1>CP4285 expanded anonymous session study</h1>
<p>Frozen at {escape(frozen['frozen_at_utc'])}. The selected policy is
<strong>{escape(frozen['winner'])}</strong>. Its confirmation primary NDCG@{k} is
{winner.ndcg_at_k:.4f}, versus {baseline.ndcg_at_k:.4f} for relevance-only.
Its combined ILD is {winner.combined_ild:.4f}, versus {baseline.combined_ild:.4f}.</p>
<h2>Question and hypothesis</h2>
<p>Does observed session variety help choose the relevance-diversity balance?
Focused interactions should retain more relevance weight. Cross-category browsing
may support more diversity. Recent interactions may indicate a shift in intent.</p>
<h2>Execution flow</h2>
<pre>main.py expanded_main()
  prepare_expanded() → full training, fixed development sets, fresh confirmation reservation
  build_feature_cache() → disk-backed query batches and memory maps
  train_expanded_models() → click and purchase-aware LambdaRank variants
  tune_expanded() → fixed, quota, threshold, linear, exponential and history-decay ablations
  select_expanded() → choose from the validation shortlist
  freeze_expanded() → save input, model and choice fingerprints
  prepare_confirmation() → materialize only the reserved final sessions
  evaluate_expanded_confirmation() → evaluate all frozen comparisons
  write_expanded_report() → statistics, trajectories, examples and report</pre>
<h2>Data and purchase attribution</h2>
{table(pd.DataFrame(audit['splits']).T.reset_index(names='split'))}
{table(pd.DataFrame(audit['purchases']).T.reset_index(names='split'))}
<p>Each purchase is attributed to the unique latest preceding query in its daily
session that contained the purchased product. Equal latest timestamps remain
ambiguous. Repeated eligible earlier queries are audited as attribution sensitivity
cases. Unmatched purchases are retained in the audit. All features use strictly
earlier events. Candidate lists preserve their original order.</p>
<h2>Relevance objectives and model comparisons</h2>
<p>The primary endpoint was declared from training and validation purchase support
before model scores were examined. It is <strong>{escape(objective['primary'])}</strong>.
Click and purchase-aware metrics remain separate. Purchase grades 0,1,2 map to
gains 0,1,3. A purchased item receives its highest grade once.</p>
{table(model_table)}
<p>The click variants use the original feature schema. The purchase/session variant
also includes earlier purchase counts and training-only purchase popularity.
These feature and label differences are explicit experimental choices.</p>
<h3>Pilot development reference (K=10)</h3>{pilot}
<p>The pilot table uses K=10. The current experiment uses K={k}; compare numeric
scores across these tables only when the cutoffs match.</p>
<h2>Adaptive formulas</h2>
<pre>score = alpha * normalized relevance + (1-alpha) * mean distance to selected items
linear alpha = min + (max-min)*(1-H)
exponential alpha = min + (max-min)*(exp(-beta*H)-exp(-beta))/(1-exp(-beta))
weighted history H = sum(w_i*w_j*distance_ij) / sum(w_i*w_j)
linear time weights = max(0,1-age_minutes/window_minutes)
exponential time weights = exp(-ln(2)*age_minutes/half_life_minutes)</pre>
<p>Only known distinct-product pairs contribute to history H. Histories contain at
most the configured 100 earlier events. Sparse or zero-effective-weight history
uses the validation-chosen default. Every policy is recomputed from that query's
prefix. More interactions alone do not force a larger diversity weight.</p>
<h2>Validation and selection</h2>
<p>Validation first chooses no-decay settings. Time-decay variants then hold those
settings fixed to isolate recency effects. Selection compares the saved shortlist.
The choice rule maximizes combined ILD while allowing at most 5% relative loss
in primary NDCG and no ILD loss against relevance-only.</p>
{image('tradeoff.png')}
<pre>{escape(json.dumps(shortlist['policies'],indent=2))}</pre>
{table(pd.read_csv(folder/'selection_metrics.csv'))}
<h2>Fresh confirmation results at K={k}</h2>
{table(summary)}
{image('confirmation.png')}
<p>NDCG discounts gain by log2(rank+1) and divides by the ideal candidate ordering.
Recall divides retrieved relevant candidates by all relevant candidates in the
query. Each endpoint averages eligible queries equally. Purchase recall is
conditional on an attributed purchase in the candidate set. Candidate matching
coverage and unmatched purchases are separate audit outcomes.</p>
<h2>Uncertainty</h2>
<p>Paired 95% percentile intervals resample whole sessions {config["bootstrap_replicates"]:,} times with seed
{config["seed"]}, preserving query-weighted means. Delta is against relevance-only.
A delta interval spanning zero leaves the effect direction uncertain.</p>
{table(intervals)}
<h2>Within-session behaviour</h2>
<p>These actual examples are selected by low or high mean measurable history ILD,
the largest decrease in observed ILD, and sparse history where available. Lines show
observed diversity and relevance weight at each query. The full trajectory CSV
includes every query and frozen policy.</p>
{image('trajectories.png')}
<h2>Descriptive recommendation examples</h2>
<p>Examples show the smallest and largest primary-NDCG changes for the selected
policy against relevance-only. Product IDs are anonymized.</p>
{''.join(examples)}
<h2>Limitations</h2>
<p>Clicks and purchases are logged outcomes affected by the original ordering.
Purchase attribution is a last-touch heuristic. The metadata distance measures
lexical and category differences. Session boundaries follow source session ID and
date. Sparse purchases can produce wide confidence intervals. These offline
results measure ranking and list variety. Actual CTR and conversion effects
require online user experiments.</p>
<h2>Feature construction resources</h2>
<pre>{escape(json.dumps(json.loads((folder/"features_info.json").read_text()),indent=2))}</pre>
<p>The feature timer covers loading and constructing cached development features.
Peak observed RSS is sampled at batch boundaries. Model training durations are
shown in the model comparison. Per-query distance and reranking times are in the
metric tables.</p>
<h2>Reproduction</h2>
<pre>uv run main.py expand-resume --study-run {escape(str(folder))}</pre>
<p>Run configuration, code hashes, frozen choices, per-query results, attribution
audits and compressed feature batches accompany this report.</p>
"""
    html="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>CP4285 expanded study</title><style>
:root{color-scheme:only light}html,body{background:#fff!important;color:#17212b!important}
body{font:17px/1.7 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px 60px}
h1,h2,h3{line-height:1.3}h2{margin-top:45px;border-bottom:2px solid #dce3eb;padding-bottom:10px}
table{display:block;max-width:100%;overflow-x:auto;border-collapse:collapse;font-size:14px;margin:24px 0}
th,td{padding:11px;border:1px solid #d4dde7;text-align:left}thead{background:#edf2f7}
tbody tr:nth-child(even){background:#f7f9fc}img{max-width:100%;height:auto;margin:24px 0;background:#fff}
pre{background:#f5f7fa;padding:20px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}
</style><body>"""+body+"</body></html>"
    temporary=folder/"report.html.tmp";temporary.write_text(html,encoding="utf-8");temporary.replace(folder/"report.html")
    finish_stage(folder,"report",["report.html","confidence_intervals.csv","session_trajectories.csv",
                                 "confirmation.png","tradeoff.png","trajectories.png"])
    print(f"Expanded report: {(folder/'report.html').resolve()}",flush=True)
    return folder/"report.html"
