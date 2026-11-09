"""Check the website serving code and API on a small synthetic hand-off folder."""

import json
import threading
import urllib.error
import urllib.request

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import torch

from session_rerank.api import clean_event, make_server
from session_rerank.expanded_data import query_features
from session_rerank.handoff import top_ids
from session_rerank.sequence_models import build_model
from session_rerank.serving import Recommender, distinct_in_order
from test_sequence_models import SEQUENCE

PAGE = dict(top_k=4, relevance_slots=2, pool_size=6, fixed_alphas=[0.8])


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    folder = tmp_path_factory.mktemp("handoff")
    items = np.arange(101, 141)
    products = pd.DataFrame(dict(item_id=items, price_log2=(items % 5).astype(float), category_id=items % 4,
                                 name_tokens=[f"{i % 7},{i % 5 + 10}" for i in items]))
    products.to_parquet(folder / "catalog.parquet", index=False)
    (folder / "home").mkdir()
    torch.manual_seed(0)
    vocabulary = items[:30]
    model = build_model("sasrec", len(vocabulary), SEQUENCE)
    torch.save(model.state_dict(), folder / "home" / "model.pt")
    np.save(folder / "home" / "vocabulary_item_ids.npy", vocabulary)
    np.save(folder / "home" / "fallback_item_ids.npy", items[::-1])
    np.save(folder / "home" / "popular_item_ids.npy", items[5:])
    (folder / "home" / "settings.json").write_text(json.dumps(dict(kind="sasrec", policy="six_four_sasrec",
                                                                   page=PAGE, model=SEQUENCE)))
    (folder / "item").mkdir()
    table = pd.DataFrame({f"r{i}": np.roll(items, -i) for i in range(1, 5)})
    table.loc[0, "r4"] = -1
    table.insert(0, "item_id", items)
    table.to_parquet(folder / "item" / "you_may_also_like.parquet", index=False)
    (folder / "category").mkdir()
    rng = np.random.default_rng(0)
    booster = lgb.train(dict(objective="regression", verbosity=-1, min_data_in_leaf=5),
                        lgb.Dataset(rng.normal(size=(300, 20)), rng.normal(size=300)), num_boost_round=5)
    booster.save_model(str(folder / "category" / "model.txt"))
    stats = pd.DataFrame(dict(item_id=items, prior_clicks=items % 9, prior_views=0, prior_purchases=items % 2,
                              window_click_rate=(items % 3) / 10, window_purchase_rate=(items % 2) / 100))
    stats.to_parquet(folder / "category" / "product_stats.parquet", index=False)
    (folder / "category" / "settings.json").write_text(json.dumps(dict(
        model="lightgbm_purchase_session", history_limit=100, base_features=18, feature_indices=list(range(20)))))
    (folder / "manifest.json").write_text(json.dumps(dict(created_utc="test", pages=dict(
        home=dict(policy="six_four_sasrec"), item=dict(policy="fixed_0.8_20"),
        category=dict(model="lightgbm_purchase_session")))))
    return folder, booster, stats


@pytest.fixture(scope="module")
def recommender(handoff):
    return Recommender(handoff[0])


def test_top_ids_keep_catalog_order_for_ties():
    assert top_ids(np.array([1.0, 3.0, 3.0, 0.0]), np.array([10, 11, 12, 13]), 3).tolist() == [11, 12, 10]
    assert distinct_in_order([5, 3, 5, 7, 3]) == [5, 3, 7]


def test_home_cold_start_then_sasrec_with_six_four(recommender):
    cold = recommender.home([])
    assert cold["items"] == list(range(106, 110)) and cold["diverse_positions"] == []
    assert recommender.home([999999])["items"] == cold["items"]  # unknown products count as no click
    served = recommender.home([103, 107, 103])
    assert len(served["items"]) == 4 and len(set(served["items"])) == 4
    assert not {103, 107} & set(served["items"])  # already clicked products are left out
    assert served["source"] == "six_four_sasrec" and served["diverse_positions"] == [3, 4]
    # The first two (relevance) positions are SASRec's two best products.
    from session_rerank.sequence_models import ranked
    rows = recommender.item_index.loc[[103, 107]].tolist()
    members, _, personal = ranked(recommender.model, rows, recommender.vocab, recommender.ids_to_rows,
                                  recommender.fallback_rows, SEQUENCE, 6, recommender.weights)
    assert personal and served["items"][:2] == recommender.catalog_ids[members[:2]].tolist()


def test_home_with_only_products_unknown_to_sasrec_uses_its_fallback(recommender):
    served = recommender.home([135])  # in the catalog, not in SASRec's vocabulary
    assert served["source"].startswith("popularity") and 135 not in served["items"]


def test_item_page_lookup(recommender):
    assert recommender.item(101)["items"] == [102, 103, 104]  # -1 padding dropped
    assert recommender.item(110)["items"] == [111, 112, 113, 114]
    with pytest.raises(KeyError):
        recommender.item(5)


def test_category_page_orders_by_the_booster_on_the_evaluated_features(handoff, recommender):
    folder, booster, stats = handoff
    clicks = [dict(item_id=105, time_ms=1000), dict(item_id=109, time_ms=5000)]
    served = recommender.category(1, clicks=clicks, now_ms=9000)
    products = [i for i in range(101, 141) if i % 4 == 1]
    assert sorted(served["items"]) == products and served["used_session"]
    # Rebuild the features with the evaluation's own function and score them directly.
    catalog = {i: (float(i % 5), i % 4, frozenset({i % 7, i % 5 + 10})) for i in range(101, 141)}
    from collections import Counter
    popularity = {kind: Counter(dict(zip(stats.item_id, stats[f"prior_{kind}s"])))
                  for kind in ("click", "view", "purchase")}
    events = [(1000, "click", 105, None), (5000, "click", 109, None)]
    _, rows, _, _ = query_features((-1, "x", "serve", "", 9000, None, 1, ",".join(map(str, products))), events,
                                   catalog, popularity, 100)
    extra = stats.set_index("item_id").loc[products, ["window_click_rate", "window_purchase_rate"]].to_numpy()
    expected = booster.predict(np.hstack([rows[:, :18], extra]))
    assert np.allclose(served["scores"], np.sort(expected)[::-1], atol=1e-6)
    assert served["items"][0] == products[int(np.argmax(expected))]
    assert len(recommender.category(1, limit=3)["items"]) == 3
    assert recommender.category(99)["items"] == []


def test_log_events_keep_only_whitelisted_fields():
    event = clean_event(dict(session_id="abc", event="click", page="home", time_ms=10, item_id=5, position=2,
                             shown=[5, 6], ip="1.2.3.4", user_agent="x"))
    assert "ip" not in event and "user_agent" not in event and event["position"] == 2 and "logged_date" in event
    for bad in (dict(event="click", page="home", time_ms=1), dict(session_id="a", event="view", page="home",
                                                                     time_ms=1),
                dict(session_id="a", event="click", page="home", time_ms="1"),
                dict(session_id="a", event="click", page="home", time_ms=1, shown=["x"])):
        with pytest.raises(ValueError):
            clean_event(bad)


def call(port, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_api_end_to_end(recommender, tmp_path):
    log = tmp_path / "interactions.jsonl"
    server = make_server(recommender, port=0, log_path=log)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        assert call(port, "/health")[1]["status"] == "ok"
        status, home = call(port, "/recommend/home", dict(clicks=[dict(item_id=107, time_ms=5), dict(item_id=103,
                                                                                                    time_ms=1)]))
        assert status == 200 and home == recommender.home([103, 107])  # clicks put in time order
        assert call(port, "/recommend/item/110")[1]["items"] == [111, 112, 113, 114]
        assert call(port, "/recommend/item/5")[0] == 404
        status, category = call(port, "/recommend/category", dict(category_id=1, limit=2))
        assert status == 200 and len(category["items"]) == 2
        assert call(port, "/recommend/category", dict())[0] == 400
        assert call(port, "/log", dict(session_id="s1", event="impression", page="home", time_ms=0,
                                       shown=home["items"], list_source=home["source"], secret="x"))[0] == 200
        assert call(port, "/log", dict(session_id="s1", event="teleport", page="home", time_ms=0))[0] == 400
        assert call(port, "/nowhere")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 1 and "secret" not in lines[0] and lines[0]["shown"] == home["items"]


def test_cli_dispatches_handoff_and_requires_a_folder_for_serve(monkeypatch):
    import main
    calls = []
    monkeypatch.setattr("session_rerank.handoff.run_handoff", lambda *args: calls.append(args))
    main.cli(["handoff"])
    assert len(calls) == 1
    with pytest.raises(SystemExit):
        main.cli(["serve"])
