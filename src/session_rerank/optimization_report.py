"""Freeze validation-selected models/policies, then run one confirmation analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .benchmark import Catalog
from .core import normalize,greedy_rerank,ranking_metrics
from .optimization_data import load_features
from .policies import file_hash,choose,apply_policy,cluster_bootstrap
from .recbole_model import ndcg_by_query

OUT=Path('results/optimization')
WEIGHTS=np.array([1.,.99,.97,.95,.9,.85,.8,.75,.7,.6,0.])
METRICS=['ndcg10','recall10','jaccard_ild10','cosine_ild10']


def hybrid_scores(base,neural,data,weight):
    result=np.array(base,copy=True)
    for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        # Neural scores add information only when a query has a click prefix.
        if data['meta'].history_count.iloc[j]>0:
            result[a:b]=weight*normalize(base[a:b])+(1-weight)*normalize(neural[a:b])
    return result


def all_metrics(order,relevance,jaccard,cosine):
    a,b,c=ranking_metrics(order,relevance,jaccard)
    k=len(order)
    d=float(cosine[np.ix_(order,order)].sum()/(k*(k-1))) if k>1 else 0.
    return a,b,c,d


def evaluate(data,scores,rerank_scores=None):
    catalog=Catalog()
    values={name:np.empty((len(data['meta']),4)) for name in scores}
    sweep=np.empty((len(data['meta']),len(WEIGHTS),4)) if rerank_scores is not None else None
    top_examples=[]
    for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        items=data['items'][a:b].tolist()
        similarity=catalog.similarity(items)
        jac=1-similarity
        np.fill_diagonal(jac,0)
        sizes=catalog.sizes[[catalog.row[x] for x in items]]
        intersection=similarity*(sizes[:,None]+sizes[None,:])/(1+similarity)
        denominator=np.sqrt(sizes[:,None]*sizes[None,:])
        cos=1-np.divide(intersection,denominator,out=np.zeros_like(intersection),where=denominator>0)
        np.fill_diagonal(cos,0)
        for name,score in scores.items():
            order=np.argsort(-score[a:b],kind='stable')[:10]
            values[name][j]=all_metrics(order,data['y'][a:b],jac,cos)
        if rerank_scores is not None:
            for k,weight in enumerate(WEIGHTS):
                order=greedy_rerank(rerank_scores[a:b],jac,float(weight))
                sweep[j,k]=all_metrics(order,data['y'][a:b],jac,cos)
        if len(top_examples)<12 and data['meta'].history_count.iloc[j]>=2 and rerank_scores is not None:
            top_examples.append(dict(query_id=int(data['meta'].query_id.iloc[j]),
                history_ids=data['prefixes'][j][data['prefixes'][j]>0].tolist(),
                candidate_count=len(items),clicked_ids=[x for x,y in zip(items,data['y'][a:b]) if y],
                top10_by_weight={str(w):[items[x] for x in greedy_rerank(rerank_scores[a:b],jac,float(w))] for w in WEIGHTS}))
        if (j+1)%2000==0:print(f'Evaluated {j+1:,} / {len(data["meta"]):,} full candidate lists',flush=True)
    return values,sweep,top_examples


def policy_search(data,metrics):
    meta=data['meta']
    candidates={
        'fixed':[dict(kind='fixed',weight=float(w)) for w in WEIGHTS],
        'history_gate':[dict(kind='history_gate',weight=float(w),min_history=n) for w in WEIGHTS for n in [2,3,5]],
        'adaptive_diversity':[],
        'progression':[dict(kind='progression',start_weight=float(w),tau=tau) for w in WEIGHTS for tau in [1,3,5]],
    }
    for n in [2,3,5]:
        observed=meta.loc[meta.history_count>=n,'history_diversity'].dropna()
        if observed.empty:continue
        for q in [.25,.5,.75]:
            for narrow in WEIGHTS:
                for broad in WEIGHTS:
                    if narrow>=broad:
                        candidates['adaptive_diversity'].append(dict(kind='diversity_threshold',min_history=n,
                            threshold=float(observed.quantile(q)),quantile=q,narrow_weight=float(narrow),broad_weight=float(broad)))
    return {name:choose(options,meta,WEIGHTS,metrics) for name,options in candidates.items()}


def freeze():
    if (OUT/'frozen_selection.json').exists():raise RuntimeError('Selection already frozen')
    data=load_features('validation')
    models={}
    scores={}
    original=lgb.Booster(model_file='results/model.txt')
    scores['supplied_order']=-np.arange(len(data['y']),dtype=np.float32)
    scores['recent_popularity']=np.asarray(data['X'][:,13])
    scores['random_order']=np.random.default_rng(4285).random(len(data['y']),dtype=np.float32)
    scores['original_ranker']=original.predict(data['X'][:,:12],num_threads=4)
    models['original_ranker']=dict(path='results/model.txt',sha256=file_hash('results/model.txt'),features=12,variant='base')
    for path in sorted(OUT.glob('*.json')):
        info=json.loads(path.read_text())
        if 'validation_ndcg10' not in info or 'name' not in info:continue
        name=info['name']
        model_path=OUT/f'{name}.txt'
        scores[name]=np.load(OUT/f'{name}_validation.npy')
        models[name]=dict(path=str(model_path),sha256=file_hash(model_path),features=info['features'],variant=info.get('variant','base'))
    if not (OUT/'gru4rec_validation.npy').exists():raise RuntimeError('Finish RecBole training before model selection')
    neural=np.load(OUT/'gru4rec_validation.npy')
    scores['recbole_gru4rec']=neural
    records=[dict(model=name,validation_ndcg10=float(ndcg_by_query(value,data['y'],data['offsets']).mean())) for name,value in scores.items()]
    tree_names=list(models)
    fusion=[]
    for name in tree_names:
        for weight in [0.,.25,.5,.75,.9,.95,1.]:
            candidate=hybrid_scores(scores[name],neural,data,weight) if weight<1 else scores[name]
            score=float(ndcg_by_query(candidate,data['y'],data['offsets']).mean())
            fusion.append(dict(base=name,tree_weight=weight,validation_ndcg10=score))
    selected=max(fusion,key=lambda x:(x['validation_ndcg10'],x['tree_weight']))
    chosen=hybrid_scores(scores[selected['base']],neural,data,selected['tree_weight']) if selected['tree_weight']<1 else scores[selected['base']]
    scores['selected_relevance']=chosen
    values,sweep,_=evaluate(data,scores,chosen)
    policies=policy_search(data,sweep)
    for name,policy in policies.items():values[name]=apply_policy(policy,data['meta'],WEIGHTS,sweep)
    save_report('validation',data,values,sweep)
    np.savez_compressed(OUT/'validation_sweep.npz',metrics=sweep,weights=WEIGHTS)
    pd.DataFrame(records).to_csv(OUT/'validation_models.csv',index=False)
    pd.DataFrame(fusion).to_csv(OUT/'validation_fusion_search.csv',index=False)
    payload=dict(selected=selected,models=models,policies=policies,
        recbole_checkpoint_sha256=file_hash(OUT/'recbole_gru4rec.pth'),
        protocol_sha256=file_hash('config/optimization_protocol.json'),
        recbole_config_sha256=file_hash('config/recbole_gru4rec.yaml'),
        weights=WEIGHTS.tolist(),confirmation_opened=False,
        selection='Highest validation NDCG@10; fusion only on nonempty click histories; policies maximize validation Jaccard ILD subject to 2% relative NDCG loss')
    (OUT/'frozen_selection.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)


def confirmation():
    if (OUT/'confirmation_per_query.npz').exists():raise RuntimeError('Confirmation already evaluated; reuse the saved report')
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    if frozen['protocol_sha256']!=file_hash('config/optimization_protocol.json') or frozen['recbole_config_sha256']!=file_hash('config/recbole_gru4rec.yaml'):
        raise RuntimeError('Protocol/config changed after selection freeze')
    if frozen['recbole_checkpoint_sha256']!=file_hash(OUT/'recbole_gru4rec.pth'):
        raise RuntimeError('RecBole checkpoint changed after selection freeze')
    neural_path=OUT/'gru4rec_confirmation.npy'
    if not neural_path.exists():
        # Keep PyTorch inference out of a process that has activated LightGBM's
        # OpenMP runtime. The two macOS wheel runtimes can otherwise deadlock.
        subprocess.run([sys.executable,'-m','session_rerank.recbole_model','score',
                        '--split','confirmation'],check=True)
    data=load_features('confirmation')
    scores={'supplied_order':-np.arange(len(data['y']),dtype=np.float32),
            'recent_popularity':np.asarray(data['X'][:,13]),
            'random_order':np.random.default_rng(4285).random(len(data['y']),dtype=np.float32)}
    for name,info in frozen['models'].items():
        if info['sha256']!=file_hash(info['path']):raise RuntimeError(f'Model {name} changed')
        features=load_features('confirmation',info['variant'])['X'][:,:info['features']]
        scores[name]=lgb.Booster(model_file=info['path']).predict(features,num_threads=4)
    scores['recbole_gru4rec']=np.load(neural_path)
    selected=frozen['selected']
    chosen=hybrid_scores(scores[selected['base']],scores['recbole_gru4rec'],data,selected['tree_weight']) if selected['tree_weight']<1 else scores[selected['base']]
    scores['selected_relevance']=chosen
    values,sweep,examples=evaluate(data,scores,chosen)
    for name,policy in frozen['policies'].items():values[name]=apply_policy(policy,data['meta'],WEIGHTS,sweep)
    names=list(values)
    np.savez_compressed(OUT/'confirmation_per_query.npz',values=np.stack(list(values.values()),axis=1),names=names,sweep=sweep,weights=WEIGHTS)
    data['meta'].to_parquet(OUT/'confirmation_metadata.parquet',index=False)
    (OUT/'confirmation_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    save_report('confirmation',data,values,sweep)
    (OUT/'confirmation_receipt.json').write_text(json.dumps(dict(frozen_selection_sha256=file_hash(OUT/'frozen_selection.json'),
        confirmation_ids_sha256=json.loads(Path('config/optimization_protocol.json').read_text())['confirmation_query_ids_sha256'],
        queries=len(data['meta']),candidate_rows=int(data['offsets'][-1]),no_post_confirmation_tuning=True),indent=2)+'\n')


def save_report(split,data,values,sweep):
    names=list(values)
    array=np.stack(list(values.values()),axis=1)
    bootstrap=cluster_bootstrap(array,data['meta'].session_id.to_numpy())
    rows=[]
    for j,name in enumerate(names):
        mean=array[:,j].mean(axis=0)
        lo,hi=np.quantile(bootstrap[:,j],[.025,.975],axis=0)
        record=dict(method=name,queries=len(data['meta']))
        for k,metric in enumerate(METRICS):
            record.update({metric:mean[k],metric+'_ci_low':lo[k],metric+'_ci_high':hi[k]})
        rows.append(record)
    pd.DataFrame(rows).to_csv(OUT/f'{split}_summary.csv',index=False)
    comparisons=[]
    for reference in ['original_ranker','selected_relevance','fixed']:
        if reference not in names:continue
        ref=names.index(reference)
        for j,name in enumerate(names):
            if name==reference:continue
            delta=array[:,j].mean(axis=0)-array[:,ref].mean(axis=0)
            lo,hi=np.quantile(bootstrap[:,j]-bootstrap[:,ref],[.025,.975],axis=0)
            rlo,rhi=np.quantile(100*(bootstrap[:,j]/bootstrap[:,ref]-1),[.025,.975],axis=0)
            for k,metric in enumerate(METRICS):
                comparisons.append(dict(method=name,reference=reference,metric=metric,absolute_difference=delta[k],
                    relative_difference_percent=100*delta[k]/array[:,ref,k].mean(),ci_low=lo[k],ci_high=hi[k],
                    relative_ci_low=rlo[k],relative_ci_high=rhi[k]))
    pd.DataFrame(comparisons).to_csv(OUT/f'{split}_paired_differences.csv',index=False)
    meta=data['meta']
    sizes=np.diff(data['offsets'])
    masks={'no_click_history':meta.history_count==0,'one_prior_click':meta.history_count==1,
           'two_or_more_prior_clicks':meta.history_count>=2,'single_category':meta.category_count==1,
           'multiple_categories':meta.category_count>1,
           'earlier_dates':meta.date<('2016-05-24' if split=='confirmation' else '2016-05-08'),
           'later_dates':meta.date>=('2016-05-24' if split=='confirmation' else '2016-05-08')}
    masks.update({'candidates_up_to_10':sizes<=10,'candidates_11_to_50':(sizes>10)&(sizes<=50),
                  'candidates_51_to_200':sizes>50})
    sliced=[]
    for label,mask in masks.items():
        for name,value in values.items():
            if mask.any():sliced.append(dict(slice=label,method=name,queries=int(mask.sum()),**dict(zip(METRICS,value[mask].mean(axis=0)))))
    pd.DataFrame(sliced).to_csv(OUT/f'{split}_slices.csv',index=False)
    plot(split,values,sweep)
    print(pd.DataFrame(rows)[['method','ndcg10','recall10','jaccard_ild10','cosine_ild10']].to_string(index=False),flush=True)


def plot(split,values,sweep):
    directory=OUT/'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    order=[x for x in ['original_ranker','baseline_20k','enhanced_20k','enhanced_60k','enhanced_60k_63','views_20k','views_60k','views_60k_63','recbole_gru4rec','selected_relevance'] if x in values]
    fig,ax=plt.subplots(figsize=(10,6))
    colors=['#bdc5cc']*len(order)
    colors[-1]='#197a78'
    ax.barh([x.replace('_',' ') for x in order],[values[x][:,0].mean() for x in order],color=colors)
    ax.invert_yaxis()
    ax.set(xlabel='NDCG@10 on the same full candidate lists',title=f'{split.capitalize()}: relevance model comparison')
    ax.set_xlim(left=0)
    for j,name in enumerate(order):ax.text(values[name][:,0].mean()+.001,j,f'{values[name][:,0].mean():.4f}',va='center')
    ax.set_xlim(right=max(values[x][:,0].mean() for x in order)*1.15)
    fig.tight_layout();fig.savefig(directory/f'{split}_relevance.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    means=sweep.mean(axis=0)
    ax.plot(means[:,0],means[:,2],'o-',color='#a5afb8',label='Fixed weight sweep')
    for name,color in [('selected_relevance','#203b55'),('fixed','#d58b28'),('adaptive_diversity','#ae3d6a'),('history_gate','#197a78')]:
        if name not in values:continue
        if name=='history_gate' and np.allclose(values[name],values['adaptive_diversity']):continue
        mean=values[name].mean(axis=0)
        label=name.replace('_',' ')
        if name=='adaptive_diversity' and np.allclose(values[name],values['history_gate']):label+=' (= history gate)'
        ax.scatter(mean[0],mean[2],s=80,label=label,color=color,zorder=3)
    reference=values['selected_relevance'][:,0].mean()
    ax.axvline(reference*.98,linestyle='--',color='#203b55',alpha=.5,label='2% relative loss reference')
    ax.set(xlim=(reference*.94,reference*1.015),
           ylim=(values['selected_relevance'][:,2].mean()-.003,means[WEIGHTS>0,2].max()+.005),
           xlabel='NDCG@10',ylabel='Token Jaccard ILD@10',title=f'{split.capitalize()}: diversification near the relevance baseline')
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(directory/f'{split}_diversity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','confirmation','report'])
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='confirmation':confirmation()
    else:
        with np.load(OUT/'confirmation_per_query.npz') as cache:
            values={str(name):cache['values'][:,i] for i,name in enumerate(cache['names'])}
            save_report('confirmation',load_features('confirmation'),values,cache['sweep'])


if __name__=='__main__':main()
