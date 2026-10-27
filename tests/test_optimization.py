import importlib.util
import numpy as np
import pytest
from session_rerank.optimization_data import available_day,day_index,window_counts
from session_rerank.policies import cluster_bootstrap


def test_aggregates_observe_embargo_and_split_cap():
    assert available_day('2016-05-01')==day_index('2016-04-29')
    assert available_day('2016-05-14','2016-04-29')==day_index('2016-04-29')
    assert available_day('2016-05-16','2016-05-14')==day_index('2016-05-14')


def test_rolling_statistics_cannot_include_future_counts():
    daily=np.array([[1,0],[2,1],[4,0],[1000,1000]],dtype=np.float32)
    cumulative=daily.cumsum(axis=0)
    np.testing.assert_array_equal(window_counts(cumulative,2,2),[6,1])
    np.testing.assert_array_equal(window_counts(cumulative,1),[3,1])
    np.testing.assert_array_equal(window_counts(cumulative,-1),[0,0])


def test_paired_cluster_bootstrap_preserves_query_weighting_and_differences():
    values=np.array([[1.,3.],[1.,3.],[7.,9.]])
    draws=cluster_bootstrap(values,np.array([10,10,20]),repetitions=100)
    assert set(draws[:,0])=={1.,3.,7.}
    np.testing.assert_allclose(draws[:,1]-draws[:,0],2.)


@pytest.mark.skipif(importlib.util.find_spec('recbole') is None,reason='RecBole environment required')
def test_recbole_prefix_mapping_preserves_chronology_and_handles_unseen_items():
    from session_rerank.recbole_model import make_sequences
    sequence,lengths,usable=make_sequences(np.array([[11,99,22,0],[99,0,0,0],[0,0,0,0]]),{'11':1,'22':2})
    assert sequence.tolist()==[[1,2,0,0],[0,0,0,0],[0,0,0,0]]
    assert lengths.tolist()==[2,1,1]
    assert usable.tolist()==[True,False,False]


@pytest.mark.skipif(importlib.util.find_spec('recbole') is None,reason='RecBole environment required')
def test_same_task_ndcg_matches_hand_calculation():
    from session_rerank.recbole_model import ndcg_by_query
    result=ndcg_by_query(np.array([.1,.9,.8,.2]),np.array([1,0,1,0]),np.array([0,2,4]))
    np.testing.assert_allclose(result,[1/np.log2(3),1])


@pytest.mark.skipif(importlib.util.find_spec('recbole') is None,reason='RecBole environment required')
@pytest.mark.parametrize('model_name',['GRU4Rec','SASRec'])
def test_recbole_candidate_adapter_matches_official_predict_and_empty_history_fallback(model_name):
    import torch
    from recbole.config import Config
    from recbole.data.interaction import Interaction
    from recbole.utils import get_model
    from session_rerank.recbole_model import candidate_scores
    cfg=Config(model=model_name,dataset='diginetica_session',config_file_list=['config/recbole_gru4rec.yaml'])
    class Shape:
        def num(self,field):return 4
    torch.manual_seed(42)
    model=get_model(model_name)(cfg,Shape()).eval()
    data=dict(prefixes=np.array([[11,22,0,0,0,0,0,0,0,0],[0]*10]),
              offsets=np.array([0,2,4]),items=np.array([11,33,11,33]),X=np.zeros((4,14),dtype=np.float32))
    data['X'][2:,13]=[.2,.7]
    actual=candidate_scores(model,{'11':1,'22':2,'33':3},data)
    expected=model.predict(Interaction({'item_id_list':torch.tensor([[1,2]+[0]*8]*2),
        'item_length':torch.tensor([2,2]),'item_id':torch.tensor([1,3])})).detach().numpy()
    np.testing.assert_allclose(actual[:2],expected,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(actual[2:],[.2,.7])
