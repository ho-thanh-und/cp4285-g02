"""Validation-only policy selection, plus paired cluster bootstrap reporting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def file_hash(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def policy_indices(policy, meta, weights):
    count=meta.history_count.to_numpy()
    diversity=meta.history_diversity.to_numpy()
    kind=policy['kind']
    chosen=np.ones(len(meta))
    if kind=='fixed':
        chosen[:]=policy['weight']
    elif kind=='history_gate':
        chosen[count>=policy['min_history']]=policy['weight']
    elif kind=='diversity_threshold':
        eligible=(count>=policy['min_history']) & np.isfinite(diversity)
        chosen[eligible]=np.where(diversity[eligible]>=policy['threshold'],policy['broad_weight'],policy['narrow_weight'])
    elif kind=='progression':
        eligible=count>=2
        chosen[eligible]=1-(1-policy['start_weight'])*np.exp(-(count[eligible]-2)/policy['tau'])
    else:
        raise ValueError(kind)
    return np.abs(chosen[:,None]-weights[None,:]).argmin(axis=1)


def apply_policy(policy,meta,weights,metrics):
    indices=policy_indices(policy,meta,weights)
    return metrics[np.arange(len(meta)),indices]


def choose(candidates,meta,weights,metrics,budget=.02):
    floor=metrics[:,0,0].mean()*(1-budget)
    best=None
    score=(-np.inf,-np.inf)
    for policy in candidates:
        values=apply_policy(policy,meta,weights,metrics)
        mean=values.mean(axis=0)
        if mean[0]+1e-12>=floor and (mean[2],mean[0])>score:
            best=dict(policy)
            best['validation_ndcg']=float(mean[0])
            best['validation_ild']=float(mean[2])
            score=(mean[2],mean[0])
    if best is None:
        raise RuntimeError('No feasible policy; candidate families must include relevance-only fallback')
    return best


def freeze():
    if Path('results/frozen_policies.json').exists():
        raise RuntimeError('Policies already frozen. Do not change selection after inspecting test results.')
    cache=np.load('results/cache/validation.npz')
    metrics,weights=cache['metrics'],cache['weights']
    meta=pd.read_parquet('results/cache/validation_metadata.parquet')
    fixed=[dict(kind='fixed',weight=float(w)) for w in weights]
    gates=[dict(kind='history_gate',weight=float(w),min_history=n) for n in [2,3,5] for w in weights]
    thresholds=[]
    for n in [2,3,5]:
        observed=meta.loc[meta.history_count>=n,'history_diversity'].dropna()
        if observed.empty:
            continue
        for quantile in [.25,.5,.75]:
            threshold=float(observed.quantile(quantile))
            for narrow in weights:
                for broad in weights:
                    if narrow>=broad:
                        thresholds.append(dict(kind='diversity_threshold',min_history=n,threshold=threshold,
                                               quantile=quantile,narrow_weight=float(narrow),broad_weight=float(broad)))
    progression=[dict(kind='progression',start_weight=float(w),tau=tau) for w in weights for tau in [1,3,5]]
    families={'fixed':fixed,'history_gate':gates,'adaptive_diversity':thresholds,'progression':progression}
    selected={name:choose(policies,meta,weights,metrics) for name,policies in families.items()}
    payload=dict(selection='Maximum validation mean ILD@10 subject to <=2% relative mean NDCG@10 loss against the same relevance ranker. Point-estimate constraint, not a noninferiority guarantee.',
                 budget=.02,candidate_counts={k:len(v) for k,v in families.items()},policies=selected,
                 model_sha256=file_hash('results/model.txt'),validation_cache_sha256=file_hash('results/cache/validation.npz'),
                 protocol_sha256=file_hash('config/protocol.json'))
    Path('results/frozen_policies.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)


def cluster_bootstrap(values,clusters,repetitions=1000):
    _,inverse=np.unique(clusters,return_inverse=True)
    n_clusters=inverse.max()+1
    counts=np.bincount(inverse,minlength=n_clusters)
    sums=np.zeros((n_clusters,)+values.shape[1:])
    np.add.at(sums,inverse,values)
    rng=np.random.default_rng(4285)
    bootstrap=np.empty((repetitions,)+values.shape[1:])
    for b in range(repetitions):
        draw=rng.integers(n_clusters,size=n_clusters)
        bootstrap[b]=sums[draw].sum(axis=0)/counts[draw].sum()
    return bootstrap


def report(name):
    cache=np.load(f'results/cache/{name}.npz')
    meta=pd.read_parquet(f'results/cache/{name}_metadata.parquet')
    metrics,weights=cache['metrics'],cache['weights']
    frozen=json.loads(Path('results/frozen_policies.json').read_text())
    policies=frozen['policies']
    values={'original_order':cache['baselines'][:,0], 'popularity':cache['baselines'][:,1],
            'session_repeat_popularity':cache['baselines'][:,2], 'relevance_ranker':metrics[:,0]}
    values.update({k:apply_policy(p,meta,weights,metrics) for k,p in policies.items()})
    # Seeded control: permute the diversity signal only within each history-count stratum.
    shuffled=meta.copy()
    rng=np.random.default_rng(4285)
    for count in shuffled.history_count.unique():
        index=shuffled.index[shuffled.history_count==count]
        shuffled.loc[index,'history_diversity']=rng.permutation(shuffled.loc[index,'history_diversity'].to_numpy())
    values['shuffled_diversity_control']=apply_policy(policies['adaptive_diversity'],shuffled,weights,metrics)
    names=list(values)
    array=np.stack(list(values.values()),axis=1)
    bootstrap=cluster_bootstrap(array,meta.session_id.to_numpy())
    rows=[]
    for j,method in enumerate(names):
        mean=array[:,j].mean(axis=0)
        lo,hi=np.quantile(bootstrap[:,j],[.025,.975],axis=0)
        row=dict(method=method,queries=len(meta))
        for k,metric in enumerate(['ndcg10','recall10','ild10']):
            row[metric]=mean[k]
            row[metric+'_ci_low']=lo[k]
            row[metric+'_ci_high']=hi[k]
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'results/{name}_summary.csv',index=False)
    comparisons=[]
    for reference in ['relevance_ranker','fixed','history_gate']:
        ref=names.index(reference)
        for j,method in enumerate(names):
            if j==ref: continue
            delta=array[:,j].mean(axis=0)-array[:,ref].mean(axis=0)
            draws=bootstrap[:,j]-bootstrap[:,ref]
            relative=100*(bootstrap[:,j]/bootstrap[:,ref]-1)
            lo,hi=np.quantile(draws,[.025,.975],axis=0)
            rlo,rhi=np.quantile(relative,[.025,.975],axis=0)
            for k,metric in enumerate(['ndcg10','recall10','ild10']):
                comparisons.append(dict(method=method,reference=reference,metric=metric,absolute_difference=delta[k],
                    relative_difference_percent=100*delta[k]/array[:,ref,k].mean(),ci_low=lo[k],ci_high=hi[k],relative_ci_low=rlo[k],relative_ci_high=rhi[k]))
    pd.DataFrame(comparisons).to_csv(f'results/{name}_paired_differences.csv',index=False)
    slices={
        'history_0':meta.history_count==0,'history_1':meta.history_count==1,
        'history_2_4':meta.history_count.between(2,4),'history_5_plus':meta.history_count>=5,
        'single_category':meta.category_count==1,'multiple_categories':meta.category_count>1,
        'earlier_dates':meta.date<('2016-05-24' if name=='test' else '2016-05-08'),
        'later_dates':meta.date>=('2016-05-24' if name=='test' else '2016-05-08'),
    }
    sliced=[]
    for label,mask in slices.items():
        for method,v in values.items():
            if mask.any():
                mean=v[mask].mean(axis=0)
                sliced.append(dict(slice=label,method=method,queries=int(mask.sum()),ndcg10=mean[0],recall10=mean[1],ild10=mean[2]))
    pd.DataFrame(sliced).to_csv(f'results/{name}_slices.csv',index=False)
    Path(f'results/{name}_history_audit.json').write_text(json.dumps(dict(queries=len(meta),
        history_count_distribution={int(k):int(v) for k,v in meta.history_count.value_counts().sort_index().items()},
        fraction_with_two_prior_distinct_clicked_items=float((meta.history_count>=2).mean())),indent=2)+'\n')
    plot(name,meta,weights,metrics,values)
    print(pd.DataFrame(rows)[['method','queries','ndcg10','recall10','ild10']].to_string(index=False),flush=True)


def plot(name,meta,weights,metrics,values):
    Path('results/figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,6))
    means=metrics.mean(axis=0)
    ax.plot(means[:,0],means[:,2],'o-',color='#8c98a4',label='Fixed weight sweep')
    for w,x,y in zip(weights,means[:,0],means[:,2]):
        if w in [1.,.9,.8,.6]:
            ax.annotate(f'weight {w:g}',(x,y),xytext=(5,6),textcoords='offset points',fontsize=8)
    palette={'relevance_ranker':'#203b55','fixed':'#eb9a28','history_gate':'#54a59b','adaptive_diversity':'#b84669','progression':'#7064ad'}
    for method,color in palette.items():
        if method=='history_gate' and np.allclose(values[method],values['adaptive_diversity']):
            continue
        x,_,y=values[method].mean(axis=0)
        label=method.replace('_',' ')
        if method=='adaptive_diversity' and np.allclose(values[method],values['history_gate']):
            label+=' (= history gate)'
        ax.scatter(x,y,s=100,color=color,label=label,zorder=4)
    ax.axvline(values['relevance_ranker'][:,0].mean()*.98,color='#203b55',linestyle='--',alpha=.5,label='2% relative loss reference')
    reference=values['relevance_ranker'][:,0].mean()
    ax.set(xlim=(reference*.94,reference*1.015),ylim=(means[0,2]-.003,means[weights==.6,2][0]+.005),
           xlabel='NDCG@10 (higher is better)',ylabel='Token Jaccard ILD@10 (higher is more varied)',
           title=f'{name.capitalize()}: trade-off near the relevance baseline')
    ax.legend(fontsize=9,loc='best')
    fig.tight_layout()
    fig.savefig(f'results/figures/{name}_tradeoff.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    categories=['0','1','2–4','5+']
    masks=[meta.history_count==0,meta.history_count==1,meta.history_count.between(2,4),meta.history_count>=5]
    axes[0].bar(categories,[int(mask.sum()) for mask in masks],color='#203b55')
    axes[0].set(xlabel='Prior distinct clicked items (capped at 10)',ylabel='Queries',title='How much session history is available?')
    for method in ['fixed','adaptive_diversity','history_gate']:
        if method=='history_gate' and np.allclose(values[method],values['adaptive_diversity']):
            continue
        delta=values[method][:,2]-values['relevance_ranker'][:,2]
        label=method.replace('_',' ')
        if method=='adaptive_diversity' and np.allclose(values[method],values['history_gate']):
            label+=' (= history gate)'
        axes[1].plot(categories,[float(delta[mask].mean()) if mask.any() else np.nan for mask in masks],'o-',label=label)
    axes[1].set(xlabel='Prior distinct clicked items',ylabel='Absolute ILD@10 gain',title='Where does diversification help?')
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'results/figures/{name}_history.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','report'])
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args()
    if args.action=='freeze': freeze()
    else: report(args.split)


if __name__=='__main__':
    main()
