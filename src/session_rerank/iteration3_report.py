"""Nominate on old validation, select once on fresh development, confirm once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from .iteration3 import ROOT,OUT,load_data,build_split,hash_ids
from .recbole_model import ndcg_by_query
from .optimization_report import hybrid_scores,evaluate,policy_search
from .policies import file_hash,apply_policy


def neural(variant,split):
    path=OUT/f'{variant}_{split}.npy'
    if not path.exists():
        subprocess.run([sys.executable,'-m','session_rerank.neural_search','score','--variant',variant,'--split',split],check=True)
    return np.load(path)


def tree_prediction(name,data,split):
    if name.startswith('previous_'):
        original=name.removeprefix('previous_')
        model=lgb.Booster(model_file=f'results/optimization/{original}.txt')
        return model.predict(data['X'][:,:model.num_feature()],num_threads=4)
    info=json.loads((OUT/f'{name}.json').read_text())
    return lgb.Booster(model_file=str(OUT/f'{name}.txt')).predict(load_data(split,'collab')['X'][:,info['columns']],num_threads=4)


def score_descriptor(descriptor,split,data,cache):
    key=json.dumps(descriptor,sort_keys=True)
    if key in cache:return cache[key]
    kind=descriptor['kind']
    if kind=='tree':score=tree_prediction(descriptor['name'],data,split)
    elif kind=='neural':score=neural(descriptor['name'],split)
    elif kind=='previous':
        base=tree_prediction('previous_views_60k_63',data,split)
        score=hybrid_scores(base,neural('bpr',split),data,.75)
    elif kind=='fusion':
        base=score_descriptor(dict(kind='tree',name=descriptor['tree']),split,data,cache)
        seq=neural(descriptor['neural'],split)
        score=hybrid_scores(base,seq,data,descriptor['weight'])
    else:raise ValueError(kind)
    cache[key]=score
    return score


def fingerprints():
    paths=['config/iteration3_protocol.json','config/recbole_gru_ce.yaml','config/recbole_sas_ce.yaml',
           'results/optimization/recbole_gru4rec.pth','results/optimization/views_60k.txt','results/optimization/views_60k_63.txt']
    paths += [str(p) for p in OUT.glob('*.pth')]
    paths += [str(p) for p in OUT.glob('*.txt')]
    return {p:file_hash(p) for p in sorted(paths)}


def select():
    if (OUT/'frozen_selection.json').exists():raise RuntimeError('Already frozen')
    # Every searched model must have completed before opening new selection outcomes.
    for name in ['collab31','collab63','anonymous31','gru_ce','sas_ce']:
        if not (OUT/f'{name}.json').exists():raise RuntimeError(f'Finish {name} first')
    validation=load_data('validation')
    cache={}
    trees=['previous_views_60k','previous_views_60k_63','collab31','collab63','anonymous31']
    records=[]
    def assess(descriptor):
        s=score_descriptor(descriptor,'validation',validation,cache)
        value=float(ndcg_by_query(s,validation['y'],validation['offsets']).mean())
        records.append(dict(descriptor=descriptor,validation_ndcg10=value))
        return records[-1]
    tree_results=[assess(dict(kind='tree',name=name)) for name in trees]
    neural_results=[assess(dict(kind='neural',name=name)) for name in ['gru_ce','sas_ce']]
    previous=assess(dict(kind='previous'))
    best_tree=max(tree_results,key=lambda x:x['validation_ndcg10'])
    fusions=[]
    for seq in ['gru_ce','sas_ce']:
        for weight in [.25,.5,.75,.9,.95]:
            fusions.append(assess(dict(kind='fusion',tree=best_tree['descriptor']['name'],neural=seq,weight=weight)))
    nominees={'previous_hybrid':previous['descriptor'],
              'best_tree':best_tree['descriptor'],
              'best_neural':max(neural_results,key=lambda x:x['validation_ndcg10'])['descriptor'],
              'best_fusion':max(fusions,key=lambda x:x['validation_ndcg10'])['descriptor'],
              'anonymous_control':dict(kind='tree',name='anonymous31')}
    (OUT/'validation_search.json').write_text(json.dumps(records,indent=2)+'\n')
    # Persist fitted candidates before any fresh selection scores are computed.
    hashes=fingerprints()
    (OUT/'candidate_manifest.json').write_text(json.dumps(dict(nominees=nominees,hashes=hashes),indent=2)+'\n')
    build_split('selection')
    data=load_data('selection')
    cache={}
    values={name:score_descriptor(desc,'selection',data,cache) for name,desc in nominees.items()}
    means={name:float(ndcg_by_query(score,data['y'],data['offsets']).mean()) for name,score in values.items()}
    pd.DataFrame([dict(candidate=name,ndcg10=value,queries=len(data['meta'])) for name,value in means.items()]).to_csv(OUT/'selection_summary.csv',index=False)
    selected=max(means,key=means.get)
    # Policy tuning stays on the original development sample.
    chosen_validation=score_descriptor(nominees[selected],'validation',validation,{})
    _,sweep,_=evaluate(validation,{'selected_relevance':chosen_validation},chosen_validation)
    policies=policy_search(validation,sweep)
    frozen=dict(selected_name=selected,selected=nominees[selected],nominees=nominees,
                original_validation_scores=records,fresh_selection_scores=means,policies=policies,
                hashes=hashes,selection_queries=len(data['meta']),
                protocol_sha256=file_hash('config/iteration3_protocol.json'))
    (OUT/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(frozen,indent=2),flush=True)


def verify(frozen):
    for path,digest in frozen['hashes'].items():
        if file_hash(path)!=digest:raise RuntimeError(f'Frozen input changed: {path}')


def confirm():
    if (OUT/'confirmation_per_query.npz').exists():raise RuntimeError('Confirmation already evaluated')
    frozen=json.loads((OUT/'frozen_selection.json').read_text());verify(frozen)
    q=pd.read_parquet(ROOT/'confirmation_ids.parquet')
    protocol=json.loads(Path('config/iteration3_protocol.json').read_text())
    assert hash_ids(q.queryId)==protocol['confirmation_query_ids_sha256']
    used=set(pd.read_parquet('data/processed/test.parquet').sessionId)|set(pd.read_parquet('data/processed_v2/confirmation_ids.parquet').sessionId)
    assert not set(q.sessionId)&used
    build_split('confirmation')
    data=load_data('confirmation');cache={}
    scores={name:score_descriptor(desc,'confirmation',data,cache) for name,desc in frozen['nominees'].items()}
    # Original v1 reference and frozen v2 winner are evaluated on the same queries.
    scores['original_ranker']=lgb.Booster(model_file='results/model.txt').predict(data['X'][:,:12],num_threads=4)
    scores['selected_relevance']=scores[frozen['selected_name']]
    values,sweep,examples=evaluate(data,scores,scores['selected_relevance'])
    from .optimization_report import WEIGHTS
    for name,policy in frozen['policies'].items():values[name]=apply_policy(policy,data['meta'],WEIGHTS,sweep)
    np.savez_compressed(OUT/'confirmation_per_query.npz',values=np.stack(list(values.values()),axis=1),names=list(values),sweep=sweep,weights=WEIGHTS)
    (OUT/'confirmation_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    report(values,sweep,data)
    (OUT/'confirmation_receipt.json').write_text(json.dumps(dict(frozen_selection_sha256=file_hash(OUT/'frozen_selection.json'),
        confirmation_ids_sha256=protocol['confirmation_query_ids_sha256'],queries=len(data['meta']),candidate_rows=int(data['offsets'][-1]),no_post_confirmation_tuning=True),indent=2)+'\n')


def report(values=None,sweep=None,data=None):
    import session_rerank.optimization_report as reporting
    reporting.OUT=OUT
    if values is None:
        data=load_data('confirmation')
        with np.load(OUT/'confirmation_per_query.npz') as cache:
            values={str(name):cache['values'][:,i] for i,name in enumerate(cache['names'])}
            sweep=cache['sweep']
    reporting.save_report('confirmation',data,values,sweep)
    names=list(values)
    array=np.stack(list(values.values()),axis=1)
    from .policies import cluster_bootstrap
    draws=cluster_bootstrap(array,data['meta'].session_id.to_numpy())
    reference=names.index('previous_hybrid');selected=names.index('selected_relevance')
    difference=array[:,selected].mean(axis=0)-array[:,reference].mean(axis=0)
    relative=100*(draws[:,selected]/draws[:,reference]-1)
    lo,hi=np.quantile(draws[:,selected]-draws[:,reference],[.025,.975],axis=0)
    rlo,rhi=np.quantile(relative,[.025,.975],axis=0)
    comparisons=[dict(metric=metric,absolute_difference=difference[k],relative_percent=100*difference[k]/array[:,reference,k].mean(),
        ci_low=lo[k],ci_high=hi[k],relative_ci_low=rlo[k],relative_ci_high=rhi[k]) for k,metric in enumerate(reporting.METRICS)]
    pd.DataFrame(comparisons).to_csv(OUT/'improvement_vs_previous.csv',index=False)
    figures(values)


def figures(values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    rows=pd.read_csv(OUT/'confirmation_summary.csv').set_index('method')
    order=['original_ranker','previous_hybrid','best_neural','anonymous_control','best_tree','selected_relevance']
    labels=['Original reference','Previous hybrid','Best RecBole CE model','Anonymous-user control','Collaborative tree','Selected hybrid']
    mean=rows.loc[order,'ndcg10'].to_numpy()
    lo=rows.loc[order,'ndcg10_ci_low'].to_numpy();hi=rows.loc[order,'ndcg10_ci_high'].to_numpy()
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(labels,mean,xerr=np.stack([mean-lo,hi-mean]),
        color=['#b6c1c9']*5+['#197a78'],error_kw={'elinewidth':1,'capsize':3})
    ax.invert_yaxis();ax.set_xlim(0,max(hi)*1.15)
    for j,value in enumerate(mean):ax.text(hi[j]+.003,j,f'{value:.4f}',va='center')
    ax.set(xlabel='NDCG@10 (95% intervals from session resampling)',
           title=f'New confirmation: same {int(rows.iloc[0].queries):,} full-candidate queries')
    fig.tight_layout();fig.savefig(directory/'confirmation_models.png',dpi=180);plt.close(fig)
    path=OUT/'monthly_identity_coverage.csv'
    if path.exists():
        frame=pd.read_csv(path)
        fig,ax=plt.subplots(figsize=(8,4))
        labels=[text[5:] if text!='2016-06' else '06 (1 day)' for text in frame.month]
        ax.plot(labels,100*frame.anonymous_fraction,'o-',color='#197a78')
        ax.set(ylim=(-3,103),xlabel='Month in 2016',ylabel='Missing user IDs (%)',
               title='Identifier availability changes across released-label queries')
        fig.tight_layout();fig.savefig(directory/'identity_coverage.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['select','confirm','report'])
    action=parser.parse_args().action
    {'select':select,'confirm':confirm,'report':report}[action]()


if __name__=='__main__':main()
