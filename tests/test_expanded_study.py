"""Check purchase attribution, memory-mapped features, decay math and the final-test boundary."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from test_samples import raw_experiment
from session_rerank.samples import prepare_samples
from session_rerank.expanded_data import (
    attribute_purchases, build_feature_cache, prepare_confirmation, prepare_expanded,
    read_candidates, query_features,
)
from session_rerank.adaptive_policies import (
    relevance_weight, weighted_history_context, relevance_metrics,
)
from session_rerank.diversity import ProductDiversity
from session_rerank.expanded_experiment import (
    open_expanded_run, tune_expanded, select_expanded, freeze_expanded,
    evaluate_expanded_confirmation,
)
from session_rerank.expanded_report import write_expanded_report
from session_rerank import expanded_training
from session_rerank.study_files import stage_complete


def test_purchase_attribution_uses_unique_latest_preceding_candidate():
    db=sqlite3.connect(":memory:")
    db.executescript("""
    CREATE TABLE queries(query_id,session_key,timeframe,items);
    CREATE TABLE events(event_id,session_key,timeframe,item_id,kind);
    CREATE TABLE attribution(event_id,query_id,item_id,status,eligible_queries);
    INSERT INTO queries VALUES(1,'a',10,'2,3'),(2,'a',20,'2,4'),
      (3,'a',30,'2,5'),(4,'a',30,'2,6'),(5,'b',5,'2,3');
    INSERT INTO events VALUES(1,'a',25,2,'purchase'),(2,'a',30,5,'purchase'),
      (3,'a',35,2,'purchase'),(4,'a',8,2,'purchase');
    """)
    attribute_purchases(db)
    rows=db.execute("SELECT query_id,status,eligible_queries FROM attribution ORDER BY event_id").fetchall()
    assert rows==[(2,"matched",2),(None,"no_prior_candidate",0),
                 (None,"ambiguous_time",4),(None,"no_prior_candidate",0)]


@pytest.mark.parametrize("family",["linear","exponential"])
def test_continuous_weights_have_shared_endpoints_and_monotonicity(family):
    policy=dict(family=family,alpha_min=.2,alpha_max=.9,beta=3.,default_weight=.6)
    assert relevance_weight(policy,0)==pytest.approx(.9)
    assert relevance_weight(policy,1)==pytest.approx(.2)
    assert relevance_weight(policy,np.nan)==.6
    weights=[relevance_weight(policy,h) for h in np.linspace(0,1,101)]
    assert np.all(np.diff(weights)<=0)
    if family=="exponential":
        policy["beta"]=1e-10
        assert relevance_weight(policy,.5)==pytest.approx(.55)


def test_time_decay_and_missing_history():
    catalog=ProductDiversity(pd.DataFrame(dict(item_id=[1,2,3],category_id=[1,1,2],
                                               name_tokens=["1","1","2"])))
    queries=pd.DataFrame(dict(query_id=[1,2]))
    # Two recent similar items and one old different item.
    history=pd.DataFrame(dict(query_id=[1,1,1],item_id=[1,2,3],
        event_type=["view","click","view"],age_ms=[60000,60000,1800000]))
    plain=weighted_history_context(queries,history,catalog).set_index("query_id")
    recent=weighted_history_context(queries,history,catalog,"exponential",5).set_index("query_id")
    linear=weighted_history_context(queries,history,catalog,"linear",5).set_index("query_id")
    assert plain.loc[1,"history_ild"]==pytest.approx(2/3)
    assert recent.loc[1,"history_ild"] < plain.loc[1,"history_ild"]
    assert linear.loc[1,"history_ild"]==0
    assert np.isnan(plain.loc[2,"history_ild"])
    history["age_ms"]=3600000
    assert np.isnan(weighted_history_context(queries,history,catalog,"linear",5).history_ild.iloc[0])
    history.loc[0,"age_ms"]=0
    with pytest.raises(ValueError,match="strictly"):
        weighted_history_context(queries,history,catalog)


def test_purchase_gain_is_used_once_and_metrics_have_separate_populations():
    result=relevance_metrics(np.array([0,2,1]),[1,1,0],[0,1,0])
    assert result["gain_ndcg"]==pytest.approx(2.5/(3+1/np.log2(3)))
    assert result["click_ndcg"]==pytest.approx(1.5/(1+1/np.log2(3)))
    assert result["purchase_recall"]==1
    result=relevance_metrics(np.array([0]),[False],[False])
    assert np.isnan(result["gain_ndcg"]) and np.isnan(result["purchase_recall"])


def test_feature_prefix_excludes_current_future_and_own_outcomes():
    catalog={1:(2.,1,frozenset([1])),2:(3.,2,frozenset([2]))}
    query=(10,"s","train","2016-04-01",100,"1",1,"1,2")
    events=[(50,"view",1,None),(80,"purchase",2,None),(100,"click",2,9),
            (110,"click",1,10)]
    popularity={k:{} for k in ("click","view","purchase")}
    from collections import Counter
    popularity={k:Counter() for k in popularity}
    items,features,history,truncated=query_features(query,events,catalog,popularity,100)
    assert features[:,7].tolist()==[0,0]
    assert features[:,11].tolist()==[1,0]
    assert features[:,16].tolist()==[0,1]
    assert features[:,17].tolist()==[1,1]
    assert len(history)==2 and truncated==0


def test_expanded_pipeline_with_real_models_and_fresh_confirmation(raw_experiment,monkeypatch):
    _,protocol=raw_experiment
    original=json.loads(protocol.read_text())
    original["splits"]["confirmation"]["target_queries"]=1
    protocol.write_text(json.dumps(original))
    pilot=prepare_samples(protocol)
    monkeypatch.chdir(protocol.parent)
    config=json.loads((__import__("pathlib").Path(__file__).parents[1]/"config/expanded_study.json").read_text())
    config.update(pilot_directory=str(pilot),output_directory=str(protocol.parent/"expanded"),
                  confirmation_target_queries=1,query_batch_size=2,
                  purchase_primary_min_train_queries=1,purchase_primary_min_validation_queries=1,
                  history_windows_minutes=[5],alpha_bounds=[[.2,1.]],exponential_beta=[1.],
                  purchase_gain_sensitivity=[], top_k=2)
    config["training"].update(num_leaves=[3], min_data_in_leaf=[1], num_boost_round=3,
                              early_stopping_rounds=2, log_evaluation_period=0)
    config_file=protocol.parent/"expanded.json"
    config_file.write_text(json.dumps(config))
    config,folder=open_expanded_run(config_file)
    with pytest.raises(ValueError,match="Freeze"):
        prepare_confirmation(config,folder)
    prepare_expanded(config,folder)
    fresh=pd.read_parquet(folder/"confirmation_ids.parquet")
    previous=pd.read_parquet(pilot/"confirmation_ids.parquet")
    assert not set(fresh.session_key)&set(previous.session_key)
    build_feature_cache(config,folder)
    train=read_candidates(folder,"train")
    assert train.query_id.nunique()==5
    matrix=np.load(folder/"train_features"/"X.npy",mmap_mode="r")
    assert matrix.shape==(len(train),18)
    assert train.purchased.any() and (train.grade==2).any()
    # Actual training is small here, but uses the normal grouped LightGBM path.
    expanded_training.train_expanded_models(config,folder)
    assert json.loads((folder/"objective.json").read_text())["primary"]=="gain_ndcg"
    tune_expanded(config,folder)
    select_expanded(config,folder)
    assert not (folder/"confirmation_opened.json").exists()
    frozen=freeze_expanded(folder)
    prepare_confirmation(config,folder)
    build_feature_cache(config,folder,confirmation=True)
    evaluate_expanded_confirmation(config,folder)
    report=write_expanded_report(config,folder)
    assert report.exists() and "background:#fff" in report.read_text()
    assert "NDCG@2" in report.read_text()
    final = pd.read_parquet(folder/"confirmation_policies.parquet")
    assert final.top_k.eq(2).all()
    assert final.topk_item_ids.map(lambda value: len(json.loads(value))).eq(2).all()
    assert stage_complete(folder,"report")
    assert stage_complete(folder,"training")
    assert json.loads((folder/"frozen_choices.json").read_text())==frozen
    # A completed run resumes without changing its final-test reservation or results.
    config2,folder2=open_expanded_run(resume=folder)
    assert config2==config and folder2==folder
    expanded_training.train_expanded_models(config,folder)
    evaluate_expanded_confirmation(config,folder)
