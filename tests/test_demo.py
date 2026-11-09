"""Clew uses the existing recommender; no production predictions are fabricated."""
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from test_serving import handoff, recommender
from session_rerank.demo import bundle_issues, make_demo_server

@contextmanager
def running(model=None):
    server = make_demo_server(model, port=0, issues=["home/model.pt missing"])
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()

def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()

def test_setup_never_pretends_models_are_ready():
    with running() as port:
        assert b"Clew" in get(port, "/")[1]
        status = json.loads(get(port, "/demo/status")[1])
        assert not status["ready"] and "home/model.pt missing" in status["issues"]
        assert get(port, "/recommend/item/101")[0] == 503
        assert get(port, "/../pyproject.toml")[0] == 404

def test_catalog_maps_real_ids_and_routes_keep_model_output(recommender):
    with running(recommender) as port:
        products = json.loads(get(port, "/demo/products?ids=105,109")[1])["products"]
        assert [p["item_id"] for p in products] == [105, 109]
        assert [p["category_id"] for p in products] == [1, 1]
        assert get(port, "/demo/products?ids=999999")[0] == 404
        assert json.loads(get(port, "/recommend/item/101")[1]) == recommender.item(101)
        categories = json.loads(get(port, "/demo/categories")[1])["categories"]
        assert sum(c["count"] for c in categories) == 40

def test_incomplete_bundle_lists_exact_missing_files(tmp_path):
    issues = bundle_issues(tmp_path)
    assert "home/model.pt missing" in issues
    assert "category/model.txt missing" in issues
    assert "item/you_may_also_like.parquet missing" in issues


def test_bundle_integrity_rejects_changed_files_and_failed_verification(tmp_path):
    import hashlib
    from session_rerank.demo import REQUIRED

    files = {}
    for name in REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        if name not in ("manifest.json", "verification.json"):
            files[name] = {"sha256": hashlib.sha256(b"fixture").hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    (tmp_path / "verification.json").write_text(json.dumps({"passed": True}))
    assert bundle_issues(tmp_path) == []
    (tmp_path / "home/model.pt").write_bytes(b"changed")
    assert bundle_issues(tmp_path) == ["home/model.pt: checksum missing or different"]
    (tmp_path / "verification.json").write_text(json.dumps({"passed": False}))
    assert "Saved export verification did not pass" in bundle_issues(tmp_path)


def test_clew_rejects_bad_input_without_losing_the_connection(recommender):
    import http.client
    from test_serving import call

    with running(recommender) as port:
        for body in ([], {"clicks": [None]}, {"clicks": [{"item_id": True, "time_ms": 0}]},
                     {"clicks": [{"item_id": 105, "time_ms": -1}]}):
            assert call(port, "/recommend/home", body)[0] == 400
        assert call(port, "/recommend/category", {"category_id": 1, "limit": -1})[0] == 400
        assert call(port, "/recommend/category", {"category_id": 1, "now_ms": "later"})[0] == 400
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("POST", "/recommend/home", headers={"Content-Length": "-1"})
            assert connection.getresponse().status == 400
        finally:
            connection.close()
        assert call(port, "/recommend/home", {"clicks": []})[0] == 200


def test_clew_navigation_keeps_one_history_and_preserves_service_order(recommender):
    from test_serving import call

    with running(recommender) as port:
        assert call(port, "/recommend/home", {"clicks": []})[1] == recommender.home([])
        # Choosing a category adds no product click.
        assert call(port, "/recommend/category", {"category_id": 1, "clicks": []})[1] == recommender.category(1)
        clicks = [{"item_id": 105, "time_ms": 1000}, {"item_id": 109, "time_ms": 5000}]
        for click in clicks:
            assert call(port, f"/recommend/item/{click['item_id']}")[1] == recommender.item(click["item_id"])
        assert call(port, "/recommend/home", {"clicks": clicks})[1] == recommender.home([105, 109])
        body = {"category_id": 1, "clicks": clicks, "now_ms": 9000, "limit": 40}
        assert call(port, "/recommend/category", body)[1] == recommender.category(1, clicks, 9000, limit=40)
        # A reset sends empty history; a separate visitor never inherits these clicks.
        assert call(port, "/recommend/home", {"clicks": []})[1] == recommender.home([])

def test_clew_serves_only_its_assets_and_disables_visit_logging(recommender):
    from test_serving import call

    with running(recommender) as port:
        for path, content_type in (("/", "text/html"), ("/clew.js", "text/javascript"),
                                   ("/clew.css", "text/css")):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == content_type
                assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert "default-src 'self'" in response.headers["Content-Security-Policy"]
                assert "Access-Control-Allow-Origin" not in response.headers
                assert response.read()
        assert get(port, "/../HANDOFF.md")[0] == 404
        assert get(port, "/demo/products?ids=invalid")[0] == 400
        assert call(port, "/log", {})[0] == 404


def test_display_catalog_validation_and_fallback(recommender, tmp_path, monkeypatch):
    import hashlib
    import pytest
    import session_rerank.demo as demo

    web = tmp_path / "web"
    (web / "catalog").mkdir(parents=True)
    (web / "assets/products").mkdir(parents=True)
    (web / "assets/products/catalog.png").write_bytes(b"image")
    fingerprint = hashlib.sha256((recommender.folder / "manifest.json").read_bytes()).hexdigest()
    (web / "catalog/categories.json").write_text(json.dumps({
        "manifest_sha256": fingerprint,
        "categories": [{"category_id": 1, "name": "Electronics", "featured": True, "cover": 105}],
    }))
    product = {"item_id": 105, "category_id": 1, "name": "Headphones", "price": 39, "image": 0}
    catalog = {"manifest_sha256": fingerprint, "products": [product]}
    (web / "catalog/products.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(demo, "WEB", web)
    monkeypatch.setitem(recommender.manifest, "created_utc", "real")
    with running(recommender) as port:
        items = json.loads(get(port, "/demo/products?ids=105,109")[1])["products"]
        assert items[0]["name"] == "Headphones"
        assert items[1] == {"item_id": 109, "category_id": 1}
        assert get(port, "/assets/products/../../catalog/products.json")[0] == 404
    product["category_id"] = 999
    (web / "catalog/products.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="Invalid display metadata"):
        demo.make_demo_server(recommender, port=0)
