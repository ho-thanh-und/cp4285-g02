"""Evidence tables, paired session bootstrap, required figures and final report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .policies import cluster_bootstrap
from .benchmark_data import OUT, ROOT, load, write_json
from .benchmark_experiment import subset, score_queries

METRICS=['ndcg10','recall10_calibrated','ild10']


def relative(value, reference):
    return np.divide(100*(value-reference),reference,out=np.full_like(np.asarray(value,dtype=float),np.nan),where=np.asarray(reference)!=0)


def slices(meta):
    n=meta.history_count.to_numpy();positive=meta.positives.to_numpy()>0
    masks={f'prefix_{label}':mask&positive for label,mask in [
        ('0',n==0),('1',n==1),('2',n==2),('3',n==3),('4',n==4),('5+',n>=5)]}
    masks.update(known=meta.known.to_numpy()&positive,anonymous=(~meta.known.to_numpy())&positive,
        known_profile=meta.has_user_profile.to_numpy()&positive,
        zero_click=~positive,anomalous=meta.anomalous.to_numpy(),
        single_item=(meta.candidate_count.to_numpy()==1)&positive,
        single_category=(meta.category_count.to_numpy()==1)&positive,
        multi_category=(meta.category_count.to_numpy()>1)&positive,
        contradiction=(meta.contradiction.to_numpy()>=.8)&(n>=2)&positive,
        no_contradiction=(meta.contradiction.to_numpy()<.8)&(n>=2)&positive)
    return masks


def table_markdown(frame, columns, formats=None):
    formats=formats or {}
    rows=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame.to_dict('records'):
        rows.append('| '+' | '.join(formats.get(c,lambda x:str(x))(row[c]) for c in columns)+' |')
    return '\n'.join(rows)


def summarize(write_documents=False):
    frozen=json.loads((OUT/'frozen.json').read_text())
    receipt=json.loads((OUT/'confirmation_receipt.json').read_text())
    cache=np.load(OUT/'confirmation_metrics.npz');values=cache['values'].astype(np.float64);names=cache['names'].tolist()
    meta=pd.read_parquet(OUT/'confirmation_metadata.parquet')
    positive=meta.positives.to_numpy()>0
    means=values[positive].mean(axis=0)
    print('Bootstrapping 1,000 paired raw-session blocks...',flush=True)
    boot=cluster_bootstrap(values[positive],meta.session_id.to_numpy()[positive],1000)
    rows=[];comparisons=[]
    for j,name in enumerate(names):
        row=dict(method=name,queries=int(positive.sum()))
        for m,metric in enumerate(METRICS):
            row[metric]=means[j,m]
            row[metric+'_low'],row[metric+'_high']=np.quantile(boot[:,j,m],[.025,.975])
        rows.append(row)
        for baseline in ['original_serp','relevance','fixed_top30','continuous_none']:
            b=names.index(baseline)
            for m,metric in enumerate(METRICS):
                absdelta=boot[:,j,m]-boot[:,b,m]
                reldelta=relative(boot[:,j,m],boot[:,b,m])
                comparisons.append(dict(method=name,baseline=baseline,metric=metric,score=means[j,m],
                    baseline_score=means[b,m],absolute_delta=means[j,m]-means[b,m],relative_percent=relative(means[j,m],means[b,m]).item(),
                    absolute_ci_low=np.quantile(absdelta,.025),absolute_ci_high=np.quantile(absdelta,.975),
                    relative_ci_low=np.nanquantile(reldelta,.025),relative_ci_high=np.nanquantile(reldelta,.975)))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'confirmation_summary.csv',index=False)
    differences=pd.DataFrame(comparisons);differences.to_csv(OUT/'confirmation_differences.csv',index=False)
    sliced=[]
    for label,mask in slices(meta).items():
        for j,name in enumerate(names):
            mean=values[mask,j].mean(axis=0) if mask.any() else np.full(3,np.nan)
            sliced.append(dict(segment=label,method=name,queries=int(mask.sum()),**dict(zip(METRICS,mean))))
    sliced=pd.DataFrame(sliced);sliced.to_csv(OUT/'confirmation_segments.csv',index=False)
    triggers=meta[(meta.history_count>=frozen['selected']['tau']) & (meta.contradiction>=.8) & (meta.history_diversity>=.8)]
    triggers=triggers.sort_values(['date','timeframe','query_id']).drop_duplicates('episode')
    triggers[['query_id','session_id','episode','date','timeframe','history_count','history_diversity','contradiction']].to_csv(OUT/'clarification_candidates.csv',index=False)
    # H3a is a paired, matched-training identity ablation on KNOWN queries only.
    n=meta.history_count.to_numpy()
    identity=[]
    for label,mask in [('0',n==0),('1',n==1),('2',n==2),('3',n==3),('4',n==4),('5+',n>=5),('summarised',n>=frozen['selected']['tau'])]:
        mask=mask&meta.known.to_numpy()&positive
        entry=dict(prefix=label,queries=int(mask.sum()),known_profiles=int(meta.loc[mask,'has_user_profile'].sum()))
        if mask.any():
            pair=values[mask][:,[names.index('relevance'),names.index('anonymous')],:2]
            means_pair=pair.mean(axis=0)
            draws=cluster_bootstrap(pair,meta.loc[mask,'session_id'].to_numpy(),1000)
            retention=100*draws[:,1]/draws[:,0]
            for m,metric in enumerate(METRICS[:2]):
                entry[metric+'_known']=means_pair[0,m]
                entry[metric+'_anonymous']=means_pair[1,m]
                entry[metric+'_retention_percent']=100*means_pair[1,m]/means_pair[0,m]
                entry[metric+'_retention_low'],entry[metric+'_retention_high']=np.nanquantile(retention[:,m],[.025,.975])
        identity.append(entry)
    identity=pd.DataFrame(identity);identity.to_csv(OUT/'anonymity.csv',index=False)
    # Preference-shift diagnostic uses all trained validation delta/tau models,
    # without selecting against the confirmation outcomes.
    val=subset(load('validation'))
    dynamics=[]
    for record in pd.read_csv(OUT/'validation_search.csv').to_dict('records'):
        if not record['name'].startswith('dynamics_'):continue
        scored=score_queries(np.load(OUT/'predictions'/f'{record["name"]}.npy'),val)
        for label,mask in slices(val['meta']).items():
            if label not in ['contradiction','no_contradiction','zero_click']:continue
            score=scored[mask].mean(axis=0) if mask.any() else np.full(3,np.nan)
            dynamics.append(dict(tau=record['tau'],delta=record['delta'],segment=label,queries=int(mask.sum()),ndcg10=score[0],recall10_calibrated=score[1]))
    pd.DataFrame(dynamics).to_csv(OUT/'contradiction_dynamics.csv',index=False)
    figures(summary,differences,identity)
    hypothesis_report(summary,differences,identity,frozen,receipt,write_document=write_documents)
    if write_documents:presentation_notes(summary,differences,frozen,receipt)
    inventory(meta,receipt,documents=write_documents)
    print(summary[['method']+METRICS].to_string(index=False),flush=True)
    print('Numerical summaries, requirement inventory and five research figures written.',flush=True)


def report():
    """Explicit opt-in entry point for document generation."""
    summarize(write_documents=True)


def figures(summary,differences,identity):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    output=OUT/'figures';output.mkdir(exist_ok=True)
    policy=pd.read_csv(OUT/'policy_validation_search.csv')
    baseline=policy[(policy.kind=='fixed')&(policy.floor=='none')&(policy.weight==1.)].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for floor,marker in [('none','o'),('top30','s'),('score03','^')]:
        p=policy[(policy.kind=='fixed')&(policy.floor==floor)]
        for ax,metric in zip(axes,['ndcg10','recall10']):
            ax.plot(relative(p[metric].to_numpy(),baseline[metric]),relative(p.ild10.to_numpy(),baseline.ild10),marker+'-',label=f'Fixed, {floor}',alpha=.65)
    d=differences[differences.baseline=='relevance']
    labels={'fixed_top30':'Static, top 30','fixed_score03':'Static, floor 0.3',
        'threshold_top30':'Median threshold','continuous_top30':'Continuous','learned_top30':'Learned fallback'}
    for name in ['fixed_top30','fixed_score03','threshold_top30','continuous_top30','learned_top30']:
        for ax,metric in zip(axes,METRICS[:2]):
            x=d[(d.method==name)&(d.metric==metric)].relative_percent.iloc[0]
            y=d[(d.method==name)&(d.metric=='ild10')].relative_percent.iloc[0]
            ax.scatter(x,y,s=55,label=f'{labels[name]} (confirmation)',zorder=4)
    for ax,label in zip(axes,['NDCG@10','calibrated Recall@10']):
        ax.axvline(-1,color='black',linestyle='--',linewidth=.8)
        ax.set(xlabel=f'Relative {label} difference (%)',ylabel='Relative token ILD@10 difference (%)')
        ax.legend(fontsize=7)
    fig.suptitle('1. Relevance/diversity trade-off: validation curves, frozen confirmation points')
    fig.tight_layout();fig.savefig(output/'01_pareto_full.png',dpi=180)
    for ax in axes:ax.set(xlim=(-1.5,.15),ylim=(-.1,2.25))
    fig.savefig(output/'01_pareto.png',dpi=180);plt.close(fig)
    dynamics=pd.read_csv(OUT/'validation_search.csv');dynamics=dynamics[dynamics.name.str.startswith('dynamics_')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for tau,group in dynamics.groupby('tau'):
        for ax,metric in zip(axes,['ndcg10','recall10']):
            ax.plot(group.delta,group[metric],'o-',label=f'tau={tau:g}')
    for ax,label in zip(axes,['NDCG@10','calibrated Recall@10']):
        ax.set(xlabel='Exponential event discount delta',ylabel=label);ax.legend()
    fig.suptitle('2. Summarisation threshold and decay (validation only)')
    fig.tight_layout();fig.savefig(output/'02_dynamics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));p=identity[identity.prefix!='summarised'].copy();x=np.arange(len(p))
    for metric,label in zip(METRICS[:2],['NDCG@10','Recall@10']):
        y=p[metric+'_retention_percent'].to_numpy()-100
        lo=p[metric+'_retention_low'].to_numpy()-100;hi=p[metric+'_retention_high'].to_numpy()-100
        ax.errorbar(x,y,yerr=np.vstack([np.maximum(0,y-lo),np.maximum(0,hi-y)]),marker='o',capsize=3,label=label)
    ax.axhline(-5,color='black',ls='--',label='95% retention reference')
    ax.set(xticks=x,xticklabels=[f'{r.prefix}\n(n={r.queries})' for r in p.itertuples()],xlabel='Prior event count, known-user queries only',ylabel='Anonymous vs identity-aware relative difference (%)',title='3. Cost of masking user identity; paired 95% session intervals')
    ax.legend();fig.tight_layout();fig.savefig(output/'03_anonymity.png',dpi=180);plt.close(fig)
    scale=pd.read_csv(OUT/'scalability.csv')
    score=summary.set_index('method')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    ild_axis=axes[0].twinx()
    axes[0].plot(scale.fraction*100,[score.loc[n,'ndcg10'] for n in scale.name],'o-',color='#234c71')
    ild_axis.plot(scale.fraction*100,[score.loc[n,'ild10'] for n in scale.name],'s--',color='#288576')
    axes[0].set_ylabel('Confirmation NDCG@10',color='#234c71')
    ild_axis.set_ylabel('Confirmation token ILD@10',color='#288576')
    axes[1].plot(scale.fraction*100,scale.training_seconds,'o-',color='#bb6c27')
    axes[0].set(xlabel='Whole training-session fraction (%)')
    axes[1].set(xlabel='Whole training-session fraction (%)',ylabel='Training + validation setup time (seconds)')
    fig.suptitle('4. Supervised training scale; fixed historical archive and common confirmation set')
    fig.tight_layout();fig.savefig(output/'04_scalability.png',dpi=180);plt.close(fig)
    timing=json.loads((OUT/'confirmation_timing.json').read_text())
    chosen=json.loads((OUT/'frozen.json').read_text())['selected']['name']
    latency=timing['prediction_ms_per_query'][chosen]
    methods={'LambdaMART':('relevance',latency),
        'Fixed MMR':('fixed_top30',latency+timing['mmr_ms_per_query']['fixed_top30']),
        'Adaptive MMR':('continuous_top30',latency+timing['mmr_ms_per_query']['continuous_top30'])}
    baseline_timing=json.loads((OUT/'baseline_latency.json').read_text())['components']
    methods['Popularity']=('global_popularity',baseline_timing['global_popularity']['mean_ms'])
    methods['Session-kNN']=('session_knn',baseline_timing['session_knn']['mean_ms'])
    serving_path=OUT/'end_to_end_latency.json'
    if serving_path.exists():
        serving=json.loads(serving_path.read_text())['components']
        methods={label:(method,serving[key]['mean_ms']) for label,method,key in [
            ('Popularity','global_popularity','popularity'),('Session-kNN','session_knn','session_knn'),
            ('LambdaMART','relevance','relevance'),('Fixed MMR','fixed_top30','fixed_top30'),
            ('Adaptive MMR','continuous_top30','continuous_top30')]}
    fig,ax=plt.subplots(figsize=(8,5))
    for label,(method,ms) in methods.items():
        offset=(6,-15) if label=='Fixed MMR' else (6,8)
        ax.scatter(ms,score.loc[method,'ndcg10'],s=85);ax.annotate(label,(ms,score.loc[method,'ndcg10']),xytext=offset,textcoords='offset points')
    ax.margins(x=.25,y=.15)
    ax.set_xscale('log')
    ax.set(xlabel='Warm local inference (ms/query, log scale)',ylabel='Confirmation NDCG@10',
        title='5. End-to-end query inference; archive construction and I/O excluded' if serving_path.exists() else '5. Component latency; tree features and token matrix construction excluded')
    fig.tight_layout();fig.savefig(output/'05_latency.png',dpi=180);plt.close(fig)


def hypothesis_report(summary,differences,identity,frozen,receipt,write_document=False):
    selected=frozen['selected'];score=summary.set_index('method');d=differences[differences.baseline=='relevance']
    raw=json.loads((ROOT/'complete.json').read_text())
    confirmation_meta=pd.read_parquet(OUT/'confirmation_metadata.parquet')
    h1=[]
    for name in ['fixed_top30','fixed_score03','threshold_top30','continuous_top30','learned_top30']:
        part=d[d.method==name].set_index('metric')
        fixed_pair=differences[(differences.method==name)&(differences.baseline=='fixed_top30')].set_index('metric')
        meets=bool((part.loc[METRICS[:2],'relative_percent']>=-1).all() and part.loc['ild10','absolute_ci_low']>0)
        h1.append(dict(method=name,meets_point_relevance_budget_and_positive_ild_ci=meets,
            beats_fixed_ild_with_positive_paired_95_ci=bool(fixed_pair.loc['ild10','absolute_ci_low']>0),
            relevance_noninferiority_1_percent_supported_by_marginal_CIs=bool((part.loc[METRICS[:2],'relative_ci_low']>=-1).all()),
            ndcg_relative_percent=part.loc['ndcg10','relative_percent'],
            recall_relative_percent=part.loc['recall10_calibrated','relative_percent'],
            ild_relative_percent=part.loc['ild10','relative_percent']))
    write_json(OUT/'hypothesis_checks.json',dict(H1=h1,
        adaptive_beats_fixed_diversity_in_top30_comparison=any(r['beats_fixed_ild_with_positive_paired_95_ci'] for r in h1 if r['method'] in ['threshold_top30','continuous_top30','learned_top30']),
        H1_scope='Budget assessed on point estimates; positive ILD paired interval. This is not a joint noninferiority test or a multiplicity-adjusted test that adaptive beats fixed.',
        H2=dict(best_validation_tau=selected['tau'],best_validation_delta=selected['delta'],
            point_estimate_matches_predicted_region=selected['tau']>=3 and .7<=selected['delta']<=.9),
        H3a=identity[identity.prefix=='summarised'].to_dict('records'),
        H3b='Matched naive/exposure diagnostics are in confirmation_summary.csv; exposure weighting does not identify causal off-policy value.',
        RQ2='Abrupt token shifts are measured offline; interactive clarification benefit cannot be identified without interactive outcomes.'))
    if not write_document:return
    cols=['method']+METRICS
    main=summary[summary.method.isin(['original_serp','global_popularity','category_popularity','session_knn','relevance','anonymous','fixed_top30','fixed_score03','threshold_top30','continuous_top30','learned_top30'])]
    static=d[d.method=='fixed_score03'].set_index('metric')
    graded_name=f'dynamics_t{selected["tau"]}_d{selected["delta"]}'
    exposure_difference=100*(score.loc[graded_name,'ndcg10']/score.loc['naive','ndcg10']-1)
    identity_summary=identity[identity.prefix=='summarised'].iloc[0]
    text=f'''# Session benchmark: session-adaptive re-ranking on Diginetica

This run implements the supplied project experiment plan in a new output directory. Model selection used validation only. Confirmation excluded every raw sessionId in all three previously inspected test/confirmation reservations. All configurations and policy hashes were frozen before the new confirmation scores were calculated.

## Selected model and main results

The selected relevance model is **{selected['name']}**, a **{selected['leaves']}-leaf LambdaMART** trained on **{selected['fraction']:.0%}** of eligible training sub-sessions, with **tau={selected['tau']}**, **delta={selected['delta']}**, unseen-negative weight **{selected['unseen']}**, binary targets **{selected['binary']}**, filtered training **{selected['filtered']}**, and augmented training **{selected['augment']}**. This is the best validation configuration among the models actually tested, not a claim of a global optimum.

Confirmation covers **{receipt['queries']:,} queries / {receipt['candidates']:,} candidates**, including **{receipt['positive_queries']:,} queries with an observed clicked candidate**. The main relevance table conditions on those positive queries. Zero-click queries are retained and their lexical diversity is reported separately; absent clicks are not declared irrelevant ground truth.

Across all {raw['total_queries']:,} raw query rows, anonymity is {raw['anonymous_fraction']:.2%}. The reserved confirmation population has only {confirmation_meta.known.mean():.2%} known-user queries. The global 37.72% known-user figure therefore does not describe this later released-label evaluation period. Sampling distributions at all four fractions are recorded in `dataset_populations.csv`.

{table_markdown(main,cols,{m:lambda x:f'{x:.6f}' for m in METRICS})}

Among the comparable frozen policy families, **static MMR with lambda 0.75 and normalized relevance floor 0.3** had the highest full-validation ILD. On confirmation it changes NDCG by **{static.loc['ndcg10','relative_percent']:+.2f}%**, calibrated Recall by **{static.loc['recall10_calibrated','relative_percent']:+.2f}%**, and ILD by **{static.loc['ild10','relative_percent']:+.2f}%** against relevance-only ranking. Its NDCG interval is **{static.loc['ndcg10','relative_ci_low']:+.2f}% to {static.loc['ndcg10','relative_ci_high']:+.2f}%**, so the strict 1% loss bound is not established by that interval. The more conservative lambda-0.9 top-30 policy gives a smaller diversity gain with a smaller relevance loss. Adaptive diversity did not improve ILD over the static top-30 control. The learned controller failed its internal validation constraint and fell back to relevance-only ranking.

The hypothesized decay benefit is not supported by the selected delta=1 point. The matched graded/exposure-weighted diagnostic changes confirmation NDCG by **{exposure_difference:+.2f}%** versus the same-scale binary naive ranker. In the summarised known-user slice, the anonymous control retains **{identity_summary['ndcg10_retention_percent']:.2f}%** of identity-aware NDCG, paired interval **{identity_summary['ndcg10_retention_low']:.2f}% to {identity_summary['ndcg10_retention_high']:.2f}%**, across **{identity_summary['queries']} queries** with **{identity_summary['known_profiles']} usable persistent profiles**. These findings support a simple relevance model and static diversification more strongly than the proposed adaptive and exposure-calibrated improvements.

Every method has raw scores, session bootstrap confidence intervals, absolute differences and relative percentage differences against both original SERP and relevance-only scoring in `confirmation_summary.csv` and `confirmation_differences.csv`. The bootstrap uses 1,000 paired resamples of raw session IDs, grouping correlated sub-sessions together. Intervals are exploratory and unadjusted for multiple comparisons.

## Hypotheses

H1 is assessed against a **1% relative tolerance for both NDCG and calibrated Recall**, independently of the lexical-diversity gain. See `hypothesis_checks.json` for the frozen controllers' confirmation checks. The relevance tolerance is a validation selection rule, not a guarantee on new samples. A larger Jaccard ILD is expected when optimizing Jaccard and does not establish more useful recommendations. A claim that adaptive beats static requires a paired comparison directly against static as well as relevance-only; the report does not infer this from two separately significant comparisons.

H2's full **4 x 5 threshold/decay grid** is in `validation_search.csv` and Figure 2. Each configuration is independently fitted, using only one chosen decay feature. Histories have 1/2/4 view/click/purchase weights. Below tau, historical similarity features shrink toward query-token context. Cold queries use query/context/popularity information. The selected point above is evidence about this sample; a hypothesized tau or delta is not forced to win.

H3a uses **matched retraining with every identity feature masked**, restricting the retention analysis to known-user confirmation queries. `anonymity.csv` gives 0/1/2/3/4/5+ event buckets, available user-profile counts and paired retention intervals, plus the summarised regime. Comparing all traffic would dilute the effect with already-anonymous queries.

H3b compares matched naive binary targets with graded targets and SkipAbove unseen weights 0/.2 in the diagnostic model rows. View attribution is weak: a future same-episode candidate view before the next query / 20-minute boundary becomes a gain-1 training target. An associated click has gain 2, upgraded to gain 4 only when that clicked item is subsequently purchased in the same interval. Primary evaluation still uses observed distinct query-click targets so that all methods share ground truth. This is an exposure-weighting experiment, **not unbiased off-policy evaluation**: logged propensities and randomized interventions are unavailable.

## Follow-up questions

RQ1 uses nested **10/25/50/100% whole training-session subsets**, stratified by calendar week, event-length bucket, known/anonymous identity and click presence. Every scale uses the same validation episodes and the same reserved confirmation queries. The historical popularity/user/kNN archive is held fixed before 31 March: this isolates supervised training scale and does not measure ingesting four fractions of all raw events. Feature preparation time is recorded separately; the document's “under 60 seconds” expectation is not assumed or asserted.

RQ2 defines contradiction as high token distance between the two most recent distinct observed items; it is a measurable shift proxy, not recovered user intent. `contradiction_dynamics.csv` compares passive decay across those validation slices. An interactive prompt's gain cannot be estimated from a dataset without clarification responses. The uncertainty signal can nominate a prompt trigger for a future controlled study, but this experiment cannot find a proven optimal prompt point.

RQ3 reports zero-click, >100-click episode, single-item, single/multiple-category and abrupt-shift segments. Filtered and token-similar context-substitution training models are fitted and scored with unchanged labels. Augmentation appends half-weight alternate contexts that substitute the candidate most token-similar to the last observed item, replacing last-item similarity and conservatively mixing the weighted summary. It is a robustness perturbation; it does not invent clicks or prove realistic user behavior. Segment counts are always shown, including empty segments.

## Session and leakage controls

Sub-sessions split on date changes or inter-event gaps >1,200,000 recorded timeframe units across queries and all interaction types. The history window is strictly earlier than the query, no farther back than 1,200,000, and in the same sub-session. Equal timestamps, future events, current-query click labels and query duration are excluded from features. The 20-minute interpretation follows the specification; numeric distributions alone do not prove the timeframe unit. Persistent-user profiles, popularity and session-kNN neighbors use the fixed pre-April archive, strictly before all supervised examples. Train, validation and confirmation have disjoint episodes and embargoed dates. Raw sessionId can recur on another date under the formal sub-session rule; the confirmation exclusion and uncertainty bootstrap additionally operate on raw IDs.

The dataset's hidden-label (`is.test`) rows are excluded from supervised scoring rather than treated as zero-click records. Candidate lists are deduplicated in supplied display order and retain all unique original candidates; no negative candidate subsampling is used. Missing catalog/token metadata stops preparation for audit.

## Figures, case studies and reproduction

The five figures are in `figures/01_pareto.png` through `05_latency.png`. Validation trade-off curves and frozen confirmation points are labeled separately. Case studies in `confirmation_examples.json` expose actual anonymized item IDs and recorded targets, without inventing readable product names. `end_to_end_latency.json` measures warm local inference on 1,000 common queries, including event lookup, feature construction, model scoring, token distances and MMR. Feature reconstruction is checked against the prepared cache. Archive construction, disk reads and network transport are excluded. Separate component timings remain available.

`paired_session_cases.json` adds early/later query pairs from the same episode, with the actual item tokens. `zero_click_token_proxy_summary.csv` reports a secondary lexical-relevance proxy when a no-click query has an observed later candidate view. Queries without such a target remain unjudged. `clarification_candidates.csv` nominates the first qualifying uncertainty trigger in each episode using prefix features only, pending an interactive study.

Checked behavioral references, the unsupported early-preference percentage claim, and the exact-paper access limitation are documented in [LITERATURE_NOTES.md](../../LITERATURE_NOTES.md).

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u -m session_rerank.benchmark_data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u -m session_rerank.benchmark_experiment all
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-recbole/bin/python -m pytest -q
```

Completed stages reuse their recorded outputs; confirmation is not silently rescored. Raw CSV hashes, query ID hashes, protocol hash, model hashes and frozen controller hash are recorded. Earlier experiments remain unchanged and are not compared numerically across different populations as though their scores were interchangeable.

## Work requiring additional evidence

Real interactive clarification and causal off-policy reward require new interaction/propensity data. Semantic shopper relevance requires readable product attributes or human judgments. External replication requires another data source or later outcome period. Presentation rehearsal and team approval remain human activities. See `requirement_status.csv` for the complete requirement-level status and limitations.
'''
    (OUT/'REPORT.md').write_text(text)


def inventory(meta,receipt,documents=False):
    items=[
        ('M1','20-minute causal multi-event session extractor','complete','All raw events; strict query-time window and date/gap episode boundaries'),
        ('M1','Nested stratified 10/25/50/100 session subsets','complete','Whole sub-sessions; fixed historical archive; training scale measured on common validation/confirmation'),
        ('M1','K=10 calibrated evaluator and delta tables','complete','Both anchor baselines, paired 95% intervals; zero-click separately'),
        ('M2','Original SERP and 1/2/4 popularity baselines','complete','Global and category-conditioned archive statistics'),
        ('M2','Session-kNN','complete','Historical episode/item incidence, cosine nearest 100 sessions'),
        ('M2','Naive vs exposure-calibrated rankers','complete','Binary and graded targets; unseen-negative weights 0/.2; offline heuristic'),
        ('M2','Fixed lambda sweep and relevance floors','complete','All six specified weights and none/top30/normalized .3 floors'),
        ('M3','Tau x delta sweep and early shrinkage','complete','All 20 independently fitted configurations'),
        ('M3','Threshold/continuous/learned controllers','complete','Threshold heuristic is not an exact verified reproduction of the cited paper; learned depth-2 controller uses internal validation holdout'),
        ('M3','Exact cited-paper threshold-rule reproduction','requires_full_text','The 2021 full algorithm is behind publisher/institutional access controls; project threshold heuristic is separately labeled'),
        ('M4a','Known-user anonymity ablation','complete','Matched retraining; known queries only, 0/1/2/3/4/5+ buckets and paired retention intervals'),
        ('M4a','Problematic segments/filtering/augmentation','complete','Empty slices disclosed; no synthetic click labels'),
        ('M4a','Contradiction/passive-decay analysis','complete','Token-shift proxy and validation decay sensitivity'),
        ('M4a','Interactive clarification beats passive browsing','requires_new_data','No prompt responses or intervention outcomes in Diginetica'),
        ('M4b','10/25/50/100 training and frozen confirmation','complete',f'{receipt["queries"]} newly reserved confirmation queries, previous raw session IDs excluded'),
        ('M4b','Session bootstrap and five graphs','complete','1,000 raw-session cluster resamples; measured model/MMR inference components'),
        ('M4b','End-to-end local inference latency','complete','Popularity, actual session-kNN, tree and MMR measured on 1,000 common queries, including features/distances; reconstructed features checked against prepared cache'),
        ('M5','Real ranking case studies','complete','confirmation_examples.json, paired_session_cases.json and evidence CSVs'),
        ('M5','Final report','complete' if documents else 'outside_current_scope','User requested project work without report/poster preparation; existing drafts are not regenerated'),
        ('M5','Cognitive/temporal literature','complete','Checked citations and limits in LITERATURE_NOTES.md; unsupported quantitative early-preference claim omitted'),
        ('M6','Research figure package','complete','Five measured experiment figures'),
        ('M6','Final presentation and submission','outside_current_scope','User requested project work; no presentation creation or external submission is authorized'),
        ('H3b','Identified causal off-policy reward','requires_new_data','Logged propensities or a randomized exposure study are absent'),
    ]
    pd.DataFrame(items,columns=['milestone','requirement','status','evidence_or_remaining']).to_csv(OUT/'requirement_status.csv',index=False)


def presentation_notes(summary,differences,frozen,receipt):
    selected=frozen['selected'];score=summary.set_index('method')
    delta=differences[(differences.method=='relevance')&(differences.baseline=='original_serp')].set_index('metric')
    notes=f'''# Presentation copy and evidence

These are editable slide notes and evidence links. Native PowerPoint layout/export remains to be completed and checked.

## 1. Session-adaptive product re-ranking

Diginetica search candidate lists. Team G2. Session benchmark, 31 October 2026.

## 2. Research hypotheses

- H1: Session adaptation adds token variety while keeping both relevance metrics within 1% of relevance-only ranking.
- H2: A minimum history threshold and recent-event weighting improve observed-click ranking.
- H3: Session context recovers most of the identity-aware score, and exposure-calibrated feedback improves ranking.

Speaker note: These are hypotheses tested by the experiment, rather than conclusions assumed by the introduction. Behavioral motivation and primary-source links are in `../../LITERATURE_NOTES.md`.

## 3. Causal session context

- Start another sub-session when the date changes or an inter-event gap exceeds 1,200,000 recorded timeframe units.
- Use strictly earlier views, clicks and purchases inside the active lookback window, with engagement weights 1, 2 and 4.
- Exclude current-query labels, equal/future timestamps and query duration from features.

Speaker note: The 20-minute interpretation follows the specification. The numeric data alone do not prove the timeframe unit. Source: `config/session_benchmark_protocol.json` and the extractor tests.

## 4. Validation and confirmation

- Sweep all 20 threshold/decay pairs and the declared training variants.
- Retrain the selected configuration at 10%, 25%, 50% and 100% of eligible training sessions.
- Freeze models and controllers before scoring {receipt['queries']:,} reserved confirmation queries.

Speaker note: All earlier inspected raw session IDs are excluded. The main relevance table includes {receipt['positive_queries']:,} queries with an observed clicked candidate. No-click and hidden-label records are different cases. Source: `dataset_populations.csv`, `confirmation_receipt.json` and `frozen.json`.

## 5. Selected relevance model

{selected['leaves']}-leaf LambdaMART, training fraction {selected['fraction']:.0%}, tau {selected['tau']}, delta {selected['delta']}.

NDCG@10 **{score.loc['relevance','ndcg10']:.5f}**, calibrated Recall@10 **{score.loc['relevance','recall10_calibrated']:.5f}**.

Relative NDCG difference from original SERP **{delta.loc['ndcg10','relative_percent']:+.2f}%**, with paired 95% interval **{delta.loc['ndcg10','relative_ci_low']:+.2f}% to {delta.loc['ndcg10','relative_ci_high']:+.2f}%**.

Speaker note: State whether the selected model uses binary or graded feedback from its frozen configuration. It is the best validation choice in the tested search. Avoid comparing this score to earlier experiments with different confirmation populations.

## 6. Relevance and token variety

Figure: [Relative Pareto comparison](figures/01_pareto.png)

Speaker note: Separate validation sweep curves from frozen confirmation points. Use `hypothesis_checks.json` and direct differences against `fixed_top30` to determine whether adaptive ranking improves on static ranking. A diversity increase alone does not establish H1.

## 7. Session threshold and temporal discounting

Figure: [All 20 threshold/decay fits](figures/02_dynamics.png)

Speaker note: Report the observed selected point even if it disagrees with the predicted threshold/decay region. The curves are validation results. `contradiction_dynamics.csv` gives the abrupt token-shift diagnostic.

## 8. Cost of masking user identity

Figure: [Matched identity ablation](figures/03_anonymity.png)

Speaker note: Show known-user queries only, with usable persistent-profile counts. Read retention and confidence intervals from `anonymity.csv`. Anonymous traffic would dilute this comparison.

## 9. Training scale and compute

Figures: [Training scale](figures/04_scalability.png) and [Measured inference components](figures/05_latency.png)

Speaker note: The historical archive stays fixed across training fractions. All scale models use a common confirmation population. The latency chart includes warm in-memory history lookup, feature construction and token distances. It excludes archive construction, disk reads and network transport.

## 10. Robustness and a real ranking example

- Compare naive, exposure-weighted, filtered and augmented diagnostic rows on the same queries.
- Show an actual query's candidate IDs and targets from `confirmation_examples.json`.
- Use anonymized item IDs and token overlap rather than inventing readable product names.

Speaker note: `confirmation_segments.csv` reports zero-click, anomalous, single-item/category and abrupt-shift slices. Empty slices are evidence about availability. Substitution augments historical context and preserves observed target labels.

## 11. Findings and remaining evidence

- State which hypotheses the measured comparisons support, using the paired intervals.
- Clarification prompts need interaction outcomes, and causal off-policy estimates need an identified exposure model.
- Human semantic judgments would help establish whether token variety is useful to shoppers.

Speaker note: The threshold controller is a project heuristic, pending verification against the subscription-only full paper. Literature sources and the exact access limitation are in `../../LITERATURE_NOTES.md`. The final claim should concern observed-click ranking and lexical variety.
'''
    (OUT/'PRESENTATION_NOTES.md').write_text(notes)
