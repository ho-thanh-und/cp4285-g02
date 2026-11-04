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
from .optimization_report import hybrid_scores
from .policies import file_hash


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
    """Legacy selection command, using the same comparison and policy stages."""
    from .pipeline.comparison import compare_models
    from .pipeline.diversification import choose_diversification
    from .pipeline.evaluation import freeze_choices
    if (OUT / 'frozen_selection.json').exists():
        raise RuntimeError('Already frozen')
    comparison = compare_models()
    freeze_choices(comparison, choose_diversification(comparison))


def verify(frozen):
    for path,digest in frozen['hashes'].items():
        if file_hash(path)!=digest:raise RuntimeError(f'Frozen input changed: {path}')


def confirm():
    """Legacy confirmation command; cached outcomes can regenerate reports."""
    from .pipeline.state import ModelComparison, read_json
    from .pipeline.evaluation import freeze_and_evaluate
    from .pipeline.reporting import save_results
    frozen = read_json(OUT / 'frozen_selection.json')
    results = freeze_and_evaluate(ModelComparison.from_frozen(frozen), frozen['policies'])
    save_results(results)


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
