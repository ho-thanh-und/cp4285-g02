import numpy as np
from session_rerank.retailrocket_replication import encode_categories,candidate_pool,PropertyCatalog
from session_rerank.independent_audit import raw_metrics


def test_retailrocket_category_zero_is_a_real_category_not_the_unknown_pool():
    categories=encode_categories({10:0,11:9})
    assert categories=={10:1,11:10}
    pools={1:[10,20],10:[11,21]};global_pool=[99,98]
    assert candidate_pool(10,categories,pools,global_pool)==(1,[10,20])
    assert candidate_pool(12345,categories,pools,global_pool)==(0,global_pool)
    assert candidate_pool(None,categories,pools,global_pool)==(0,global_pool)


def test_retrieval_does_not_inject_an_unknown_ground_truth_target():
    _,pool=candidate_pool(10,{10:1},{1:[10,20]},[99,98])
    assert 12345 not in pool
    assert raw_metrics(pool,{12345})[0]==(0.,0.)


def test_property_jaccard_matches_hand_calculation():
    c=PropertyCatalog({10:{'a','b'},20:{'b','c'},30:{'d'}},{10:1,20:1,30:2})
    np.testing.assert_allclose(c.similarity([10],[20,30]),[[1/3,0]],atol=1e-7)
