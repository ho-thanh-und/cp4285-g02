import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from session_rerank.policies import policy_indices, choose


def test_adaptive_policy_falls_back_with_insufficient_history():
    meta=pd.DataFrame({'history_count':[0,1,2,4],'history_diversity':[np.nan,np.nan,.1,.9]})
    p=dict(kind='diversity_threshold',min_history=2,threshold=.5,narrow_weight=.9,broad_weight=.6)
    weights=np.array([1,.9,.6])
    assert policy_indices(p,meta,weights).tolist()==[0,0,1,2]


def test_policy_selection_rejects_diversity_gain_beyond_relevance_budget():
    meta=pd.DataFrame({'history_count':[0,0],'history_diversity':[np.nan,np.nan]})
    metrics=np.array([[[.5,.5,.4],[.45,.5,.9]],[[.5,.5,.4],[.45,.5,.9]]])
    result=choose([dict(kind='fixed',weight=1),dict(kind='fixed',weight=.6)],meta,np.array([1,.6]),metrics)
    assert result['weight']==1


@pytest.mark.skipif(not Path('data/processed/train.parquet').exists(),reason='Downloaded data needed')
def test_real_data_splits_are_disjoint_chronological_and_avoid_hidden_labels():
    protocol=json.loads(Path('config/protocol.json').read_text())
    frames=[pd.read_parquet(f'data/processed/{name}.parquet') for name in ['train','validation','test']]
    assert frames[0].eventdate.max()<frames[1].eventdate.min()<frames[2].eventdate.min()
    assert frames[1].eventdate.max()<frames[2].eventdate.min()
    all_ids=[]
    for frame in frames:
        assert not frame['is.test'].any()
        assert not frame.eventdate.isin(protocol['embargo_dates']).any()
        all_ids.extend(frame.queryId)
    assert len(all_ids)==len(set(all_ids))
    audit=json.loads(Path('results/data_audit.json').read_text())
    assert audit['click_delay_quantiles']['1.0']<24*60*60*1000
