import numpy as np
from session_rerank.popularity_diagnostics import head_items,uniform_head_share,target_strata


def test_head_uses_archive_popularity_and_deterministic_ties():
    np.testing.assert_array_equal(head_items([40,20,30,10],[1,7,7,0],.5),[20,30])


def test_uniform_opportunity_respects_different_list_lengths():
    # One unavoidable head recommendation plus ten tail recommendations.
    assert np.isclose(uniform_head_share([1,0],[1,100]),1/11)
    assert not np.isclose(uniform_head_share([1,0],[1,100]),1/101)


def test_target_strata_are_disjoint_and_handle_multiple_clicks():
    masks=target_strata([2,0,1,0],[2,2,2,0])
    assert {name:np.flatnonzero(m).tolist() for name,m in masks.items()}=={
        'head_only':[0],'tail_only':[1],'mixed':[2],'no_click':[3]}
    assert np.all(np.sum(list(masks.values()),axis=0)==1)
