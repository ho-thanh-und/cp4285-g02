"""Check that K and the two independent decay rates change the measured behavior."""

from math import log

import numpy as np
import pandas as pd
import pytest

from session_rerank.adaptive_policies import (
    expanded_order, ndcg, relevance_metrics, relevance_weight, weighted_history_context,
)
from session_rerank.diversity import ProductDiversity
from session_rerank.ranking_settings import (
    top_k, reranking_settings, configured_policy, history_decay_minutes,
)


def test_top_k_above_ten_changes_retrieval_and_metric_cutoffs():
    gains = np.zeros(20)
    gains[11] = 1
    order = np.arange(20)
    assert ndcg(order, gains, k=5) == 0
    assert ndcg(order, gains, k=12) == pytest.approx(1/np.log2(13))
    assert relevance_metrics(order, gains, np.zeros(20), k=5)['recall_at_k'] == 0
    assert relevance_metrics(order, gains, np.zeros(20), k=12)['recall_at_k'] == 1
    distances = np.ones((20, 20)) - np.eye(20)
    for policy in (dict(family='relevance'), dict(family='fixed', weight=.6),
                   dict(family='quota', keep=4),
                   dict(family='linear', alpha_min=.2, alpha_max=1., default_weight=.8)):
        selected = expanded_order(np.arange(20), distances, policy, .5, k=12)
        assert len(selected) == 12 and len(set(selected)) == 12


def test_relevance_decay_rate_has_effect_for_linear_and_exponential():
    for family in ('linear', 'exponential'):
        slow = configured_policy({'reranking': {'decay_function_type': family, 'decay_function_rate': .5}})
        fast = configured_policy({'reranking': {'decay_function_type': family, 'decay_function_rate': 3.}})
        assert relevance_weight(fast, .5) < relevance_weight(slow, .5)
        assert relevance_weight(fast, np.nan) == .8
    fixed = configured_policy({'reranking': {'decay_function_type': 'fixed', 'fixed_alpha': .4}})
    assert relevance_weight(fixed, 0) == relevance_weight(fixed, 1) == .4


def test_history_rate_uses_minutes_and_minimum_items_triggers_fallback():
    catalog = ProductDiversity(pd.DataFrame(dict(item_id=[1,2,3], category_id=[1,1,2], name_tokens=['1','1','2'])))
    queries = pd.DataFrame({'query_id': [1]})
    history = pd.DataFrame(dict(query_id=[1,1,1], item_id=[1,2,3],
                                event_type=['click','view','click'], age_ms=[60000,60000,1800000]))
    def context(rate, minimum=2, metric='combined'):
        settings = reranking_settings({'reranking': {'history_decay_type':'exponential',
                                                     'history_decay_rate_per_minute':rate}})
        return weighted_history_context(queries, history, catalog, 'exponential', history_decay_minutes(settings),
                                        metric=metric, minimum_items=minimum).history_ild.iloc[0]
    assert context(log(2)/5) < context(log(2)/30)
    assert np.isnan(context(log(2)/5, minimum=4))
    settings = reranking_settings({'reranking': {'history_decay_type':'linear', 'history_decay_rate_per_minute':.2}})
    assert history_decay_minutes(settings) == 5


@pytest.mark.parametrize('update', [
    {'decay_function_type':'unknown'}, {'decay_function_rate':0},
    {'history_decay_type':'unknown'}, {'history_decay_rate_per_minute':-1},
    {'alpha_min':.9, 'alpha_max':.2}, {'fallback_alpha':2}, {'history_max_events':0},
    {'minimum_history_items':1}, {'diversity_metric':'unknown'}, {'typo':5},
])
def test_bad_ranking_controls_are_rejected(update):
    with pytest.raises(ValueError):
        reranking_settings({'reranking': update})


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_bad_k_is_rejected(value):
    with pytest.raises(ValueError, match='top_k'):
        top_k({'top_k': value})


def test_missing_metadata_does_not_satisfy_minimum_usable_history():
    catalog = ProductDiversity(pd.DataFrame(dict(item_id=[1,2,3], category_id=[1,1,None], name_tokens=['1','1',''])))
    queries = pd.DataFrame({'query_id': [1]})
    history = pd.DataFrame(dict(query_id=[1,1,1], item_id=[1,2,3], event_type=['view']*3, age_ms=[1000]*3))
    result = weighted_history_context(queries, history, catalog, minimum_items=3).iloc[0]
    assert result.history_unique_items == 3 and result.history_usable_items == 2
    assert np.isnan(result.history_ild)
