import numpy as np
import pandas as pd
import pytest
import json
import hashlib
import sys
from types import SimpleNamespace
from session_rerank.protocol_core import (
    causal_indices, calibrated_metrics, exposure_weights, floor_order, nested_fractions,
)
from session_rerank.benchmark_data import NAMES, assign_episodes
from session_rerank.benchmark_experiment import feature_view, choose_feasible, policy_indices


@pytest.mark.parametrize('stage',['all','checks'])
def test_experiment_commands_do_not_generate_documents(stage,monkeypatch):
    from session_rerank import benchmark_experiment as experiment
    called=[]
    monkeypatch.setattr(experiment,'development',lambda:called.append('development'))
    monkeypatch.setattr(experiment,'confirmation',lambda:called.append('confirmation'))
    for module,function in [('protocol_audit','audit'),('latency_benchmark','benchmark'),
            ('proxy_analysis','proxy_report'),('case_studies','case_studies'),('serving_benchmark','benchmark')]:
        monkeypatch.setitem(sys.modules,'session_rerank.'+module,
            SimpleNamespace(**{function:lambda name=module:called.append(name)}))
    monkeypatch.setitem(sys.modules,'session_rerank.report_generation',
        SimpleNamespace(summarize=lambda write_documents:pytest.fail('Numerical experiment command generated a document')
            if write_documents else called.append('summary')))
    experiment.main([stage])
    expected=['protocol_audit','latency_benchmark','proxy_analysis','case_studies','serving_benchmark','summary']
    assert called==(['development','confirmation']+expected if stage=='all' else expected)


@pytest.mark.parametrize('corruption',[None,'metrics','model','query_order','population'])
def test_confirmation_cache_reuse_checks_frozen_provenance(tmp_path,monkeypatch,corruption):
    from session_rerank import benchmark_experiment as experiment
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(experiment,'OUT',tmp_path)
    (tmp_path/'models').mkdir();model=tmp_path/'models/example.txt';model.write_text('frozen model')
    record=dict(name='example',model_sha256=sha(model))
    frozen=dict(selected=record,matched_anonymous=record,diagnostic_models=[],scale_models=[])
    freeze=tmp_path/'frozen.json';freeze.write_text(json.dumps(frozen))
    metrics=tmp_path/'confirmation_metrics.npz';np.savez(metrics,values=np.zeros((2,1,3)))
    meta=pd.DataFrame(dict(query_id=[11,12],candidate_count=[2,1],positives=[1,0]))
    meta.to_parquet(tmp_path/'confirmation_metadata.parquet',index=False)
    receipt=dict(frozen_sha256=sha(freeze),metrics_sha256=sha(metrics),queries=2,candidates=3,positive_queries=1,
        query_ids_sha256=hashlib.sha256(meta.query_id.to_numpy().tobytes()).hexdigest())
    (tmp_path/'confirmation_receipt.json').write_text(json.dumps(receipt))
    if corruption=='metrics':metrics.write_bytes(b'changed metrics')
    if corruption=='model':model.write_text('changed model')
    if corruption=='query_order':meta=meta.iloc[::-1]
    if corruption=='population':meta.loc[0,'candidate_count']=7
    if corruption in ['query_order','population']:meta.to_parquet(tmp_path/'confirmation_metadata.parquet',index=False)
    if corruption is None:experiment.verify_confirmation_receipt(frozen)
    else:
        with pytest.raises(RuntimeError):experiment.verify_confirmation_receipt(frozen)


def test_strict_causal_window_and_episode_start():
    times=np.array([0,1,1_200_000,1_200_001,1_200_002])
    assert causal_indices(times,1_200_001).tolist()==[1,2]
    assert causal_indices(times,1_200_001,start=2).tolist()==[2]
    assert causal_indices(times,0).tolist()==[]


def test_joint_timeline_splits_date_and_inactivity_but_queries_bridge_item_gaps():
    queries=pd.DataFrame(dict(sessionId=[1,1,1,1,2],eventdate=['2016-05-01']*3+['2016-05-02','2016-05-01'],
        timeframe=[0,1_000_000,4_000_001,0,0]))
    events=pd.DataFrame(dict(sessionId=[1,1],eventdate=['2016-05-01']*2,timeframe=[100,2_000_000]))
    assign_episodes(queries,events)
    assert queries.episode.iloc[0]==queries.episode.iloc[1]==events.episode.iloc[0]==events.episode.iloc[1]
    assert queries.episode.nunique()==4


def test_decision_surface_recall_and_ndcg_with_more_than_ten_positives():
    values=calibrated_metrics(np.arange(10),np.ones(14))
    np.testing.assert_allclose(values[:2],[1,1])
    assert calibrated_metrics([0],np.zeros(1)).tolist()==[0,0,0]


def test_skip_above_unseen_and_ambiguous_negative_weights():
    np.testing.assert_allclose(exposure_weights([0,1,0,1,0],[1,0,0,0,0],.2),[.5,1,1,1,.2])
    np.testing.assert_allclose(exposure_weights([0,0],unseen=0),[0,0])


@pytest.mark.parametrize('floor',['top30','score03'])
def test_floor_cannot_promote_ineligible_item_and_backfills_short_pool(floor):
    scores=np.r_[np.linspace(1,.7,10),np.zeros(25)]
    distances=np.ones((35,35))-np.eye(35)
    distances[:10,:10]=0
    order=floor_order(scores,distances,weight=0,floor=floor)
    assert len(order)==10 and len(set(order))==10
    if floor=='top30':assert max(order)<30
    if floor=='score03':assert max(order)<10
    np.testing.assert_array_equal(floor_order(np.ones(4),np.zeros((4,4)),.5,'score03'),np.arange(4))


def test_all_floors_at_full_relevance_preserve_stable_original_ties():
    for floor in ['none','top30','score03']:
        assert floor_order([.8,.9,.8],np.ones((3,3)),1.,floor).tolist()==[1,0,2]


def test_nested_sampling_is_reproducible_and_preserves_episode_units():
    frame=pd.DataFrame(dict(episode=np.arange(100),split='train',week='2016-12',length_bucket='5+',known=False,positive=True))
    a=nested_fractions(frame);b=nested_fractions(frame)
    np.testing.assert_array_equal(a.fraction_rank,b.fraction_rank)
    selected=[set(a.loc[a.fraction_rank<f,'episode']) for f in [.1,.25,.5,1.]]
    assert [len(x) for x in selected]==[10,25,50,100]
    assert all(x<=y for x,y in zip(selected[:-1],selected[1:]))


def test_model_only_sees_selected_decay_and_anonymity_masks_all_identity_features():
    x=np.zeros((2,len(NAMES)),dtype=np.float32)
    x[:,5]=.2;x[:,8:10]=.8;x[:,10]=[1,3];x[:,13:16]=1
    x[:,NAMES.index('decay_0.7')]=.6
    view=feature_view(x,3,.7,anonymous=True)
    assert view.shape==(2,18)
    np.testing.assert_allclose(view[:,17],[.2+(.6-.2)/3,.6])
    assert not view[:,13:16].any()
    assert not x[:,13:16].min()==0, 'Source cache must not be mutated'


def test_policy_selection_enforces_both_relevance_constraints():
    meta=pd.DataFrame(dict(history_count=[3,3],history_diversity=[.8,.8]))
    cache=np.zeros((2,1,6,3));cache[:,:,:,0:2]=1;cache[:,:,:,2]=.5
    cache[:,0,0]=[1,.97,.99]  # Good NDCG but forbidden Recall loss.
    cache[:,0,-1]=[1,1,.5]
    policies=[dict(kind='fixed',weight=w,tau=3,floor_index=0) for w in [0.,1.]]
    result=choose_feasible(policies,meta,cache,np.ones(2,dtype=bool),np.array([1,1,.5]))
    assert result['weight']==1.


def test_threshold_and_continuous_policies_are_sensitive_to_history_diversity():
    meta=pd.DataFrame(dict(history_count=[0,5,5],history_diversity=[0,.1,.9]))
    indices=policy_indices(dict(kind='threshold',weight=0.,tau=3,threshold=.5),meta)
    assert indices.tolist()==[5,5,0]
    assert policy_indices(dict(kind='continuous',weight=0.,tau=3),meta)[1] > policy_indices(dict(kind='continuous',weight=0.,tau=3),meta)[2]
