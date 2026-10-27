import numpy as np
import pandas as pd
from scipy import sparse
from session_rerank.iteration3 import context_cutoff,collaborative_scores,profile_map
from session_rerank.optimization_data import day_index,known_item_rows


def test_all_training_context_cutoffs_precede_query_by_two_days():
    for day in pd.date_range('2016-04-01','2016-04-29'):
        text=day.strftime('%Y-%m-%d')
        assert day_index(context_cutoff('train',text))<=day_index(text)-2
    assert context_cutoff('selection','2016-05-14')=='2016-04-29'
    assert context_cutoff('confirmation','2016-05-16')=='2016-05-14'


def test_unknown_historical_item_never_updates_last_catalog_row():
    positions,rows=known_item_rows([1,2,3,99,-1],np.array([-1,0,-1,1]))
    assert positions.tolist()==[0,2]
    assert rows.tolist()==[0,1]


def test_profile_uses_latest_distinct_items_and_excludes_anonymous_identity():
    source=pd.DataFrame({'userId':[7,7,7,np.nan],'itemId':[11,22,11,99]})
    assert profile_map(source,'userId')=={'7':[11,22]}


def test_collaborative_scores_use_only_supplied_prefix():
    class Catalog:
        row={11:0,22:1,33:2}
    graph=sparse.csr_matrix([[0,.8,.2],[.8,0,.4],[.2,.4,0]])
    np.testing.assert_allclose(collaborative_scores(graph,[22,33],[11],Catalog()),[[.8],[.2]])
    np.testing.assert_array_equal(collaborative_scores(graph,[22,33],[],Catalog()),[[0],[0]])
