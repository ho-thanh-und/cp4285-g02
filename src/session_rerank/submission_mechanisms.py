"""Post-confirmation diagnostic mechanisms, explicitly not new confirmation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .policies import cluster_bootstrap
from .protocol_core import LAMBDAS
from .benchmark_data import load, OUT as BENCHMARK_OUT
from .benchmark_experiment import subset,policy_indices

OUT=Path('results/submission/mechanisms')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((REV/'frozen.json').read_text())
    confirmation=np.load(REV/'confirmation_metrics.npz')
    values=confirmation['values'].astype(np.float64);names=confirmation['names'].tolist()
    meta=pd.read_parquet(REV/'confirmation_metadata.parquet')
    positive=meta.positives.to_numpy()>0;n=meta.history_count.to_numpy()
    val=subset(load('validation'));vm=val['meta'];vp=vm.positives.to_numpy()>0
    cache=np.load(REV/'validation_policy_cache.npz')['values'].astype(np.float64)
    availability=[]
    for cohort,m,mask in [('validation',vm,vp),('confirmation',meta,positive)]:
        for threshold in [1,2,3,5]:
            eligible=(m.history_count.to_numpy()>=threshold)&mask
            availability.append(dict(cohort=cohort,tau=threshold,positive_queries=int(mask.sum()),
                eligible_queries=int(eligible.sum()),eligible_fraction=float(eligible.sum()/mask.sum()),
                nonzero_diversity_fraction=float((m.loc[eligible,'history_diversity']>0).mean())))
    pd.DataFrame(availability).to_csv(OUT/'adaptation_availability.csv',index=False)
    rows=[]
    for label,mask in [('all_positive',positive),('cold',positive&(n==0)),('prefix_1_4',positive&(n>0)&(n<5)),
        ('prefix_5_plus',positive&(n>=5)),('known_profile_5_plus',positive&(n>=5)&meta.has_user_profile.to_numpy())]:
        if not mask.any():continue
        for model in ['relevance','anonymous','fixed_top30','fixed_score03','gate_top30','threshold_top30','continuous_top30']:
            j=names.index(model);reference=names.index('relevance')
            mean=values[mask,j].mean(axis=0);base=values[mask,reference].mean(axis=0)
            draws=cluster_bootstrap(values[mask][:,[reference,j]],meta.loc[mask,'session_id'].to_numpy(),1000)
            relative=100*(draws[:,1]/draws[:,0]-1)
            record=dict(segment=label,model=model,queries=int(mask.sum()),ndcg10=mean[0],recall10=mean[1],ild10=mean[2])
            for k,metric in enumerate(['ndcg','recall','ild']):
                record[metric+'_relative_percent']=100*(mean[k]/base[k]-1)
                record[metric+'_relative_low'],record[metric+'_relative_high']=np.nanquantile(relative[:,k],[.025,.975])
            rows.append(record)
    pd.DataFrame(rows).to_csv(OUT/'conditional_policy_effects.csv',index=False)
    # Shuffle within exact event counts: this isolates diversity from prefix length.
    permutations=[];rng=np.random.default_rng(4285)
    for name in ['threshold_top30','continuous_top30']:
        p=frozen['policies'][name];idx=policy_indices(p,vm)
        observed=cache[np.arange(len(vm)),p['floor_index'],idx][vp].mean(axis=0)
        for repetition in range(200):
            shuffled=vm.copy()
            for count,indices in vm.groupby('history_count').groups.items():
                shuffled.loc[indices,'history_diversity']=rng.permutation(vm.loc[indices,'history_diversity'].to_numpy())
            chosen=policy_indices(p,shuffled)
            mean=cache[np.arange(len(vm)),p['floor_index'],chosen][vp].mean(axis=0)
            permutations.append(dict(policy=name,permutation=repetition,ndcg10=mean[0],recall10=mean[1],ild10=mean[2],
                observed_ndcg10=observed[0],observed_ild10=observed[2]))
    pd.DataFrame(permutations).to_csv(OUT/'within_length_shuffle.csv',index=False)
    # Accessed labels are only used to diagnose an upper bound, never a policy.
    baseline=cache[:,1,-1];active=vp&(vm.history_count.to_numpy()>=5)
    mild=np.flatnonzero(np.array(LAMBDAS)==.9)[0]
    effect=cache[:,1,mild,0]-baseline[:,0]
    correlation=spearmanr(vm.loc[active,'history_diversity'],effect[active])
    feasible=(cache[:,1,:,0]>=baseline[:,0,None])&(cache[:,1,:,1]>=baseline[:,1,None])
    oracle=np.where(feasible,cache[:,1,:,2],-np.inf).argmax(axis=1)
    oracle_ild=cache[np.arange(len(vm)),1,oracle,2]
    result=dict(analysis_status='Exploratory diagnostics after confirmation; no model or deployment policy selected from these outcomes',
        no_cold_adaptation='The continuous/gate policies use lambda=1 at n=0, while static MMR can diversify cold queries',
        history_diversity_relevance_effect_spearman=dict(queries=int(active.sum()),rho=float(correlation.statistic),p_value=float(correlation.pvalue)),
        oracle_label_dependent_non_deployable_ild_gain=float(oracle_ild[vp].mean()-baseline[vp,2].mean()),
        caveat='The oracle reads target outcomes and is only a diagnostic ceiling. It is never included as a deployable model or a selected confirmation result.',
        anonymization_caveat='Known-ID retention is weak evidence of compensation when the identity-aware model has no usable old profile. Profile-present slices are reported separately.')
    (OUT/'mechanism_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
