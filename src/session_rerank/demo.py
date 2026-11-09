"""Local Clew UI around the existing, frozen recommender service."""
import hashlib
import json
import math
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .api import make_handler

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
REQUIRED = (
    "manifest.json", "verification.json", "catalog.parquet",
    "home/model.pt", "home/settings.json", "home/vocabulary_item_ids.npy",
    "home/fallback_item_ids.npy", "home/popular_item_ids.npy",
    "item/you_may_also_like.parquet", "category/model.txt",
    "category/settings.json", "category/product_stats.parquet",
)

def bundle_issues(folder):
    folder = Path(folder)
    missing = [f"{name} missing" for name in REQUIRED if not (folder / name).is_file()]
    if missing:
        return missing
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        verification = json.loads((folder / "verification.json").read_text())
        issues = [] if verification.get("passed") is True else ["Saved export verification did not pass"]
        for name in REQUIRED:
            if name in ("manifest.json", "verification.json"):
                continue
            expected = manifest.get("files", {}).get(name, {}).get("sha256")
            actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            if not expected or actual != expected:
                issues.append(f"{name}: checksum missing or different")
        return issues
    except (ValueError, TypeError, AttributeError, OSError) as error:
        return [f"Cannot verify bundle: {error}"]

def make_demo_server(recommender=None, port=8000, issues=()):
    base = make_handler(recommender)
    categories, products = [], {}
    if recommender is not None and recommender.manifest.get("created_utc") != "test":
        catalog = json.loads((WEB / "catalog/products.json").read_text())
        category_data = json.loads((WEB / "catalog/categories.json").read_text())
        fingerprint = hashlib.sha256((recommender.folder / "manifest.json").read_bytes()).hexdigest()
        if catalog["manifest_sha256"] != fingerprint or category_data["manifest_sha256"] != fingerprint:
            raise ValueError("Display catalog does not match the selected handoff")
        categories = category_data["categories"]
        if len({c["category_id"] for c in categories}) != len(categories):
            raise ValueError("Duplicate display category")
        for product in catalog["products"]:
            item = product["item_id"]
            if (item in products or item not in recommender.catalog
                or product["category_id"] != recommender.catalog[item][1]
                or not isinstance(product["price"], (int, float))
                or not math.isfinite(product["price"]) or product["price"] < 0
                or product["image"] not in range(16)):
                raise ValueError(f"Invalid display metadata for product {item}")
            products[item] = product
        if not (WEB / "assets/products/catalog.png").is_file():
            raise ValueError("Display catalog image is missing")

    class Handler(base):
        def send_header(self, keyword, value):
            # Clew serves its UI and API together; other sites need no access.
            if not keyword.lower().startswith("access-control-"):
                super().send_header(keyword, value)

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            super().end_headers()

        def do_GET(self):
            url = urlsplit(self.path)
            assets = {"/": ("index.html", "text/html"), "/clew.css": ("clew.css", "text/css"),
                      "/clew.js": ("clew.js", "text/javascript")}
            if url.path == "/assets/products/catalog.png":
                data = (WEB / "assets/products/catalog.png").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if url.path in assets:
                name, kind = assets[url.path]
                data = (WEB / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", kind + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if url.path == "/demo/status":
                self.send(200, dict(ready=recommender is not None, issues=list(issues),
                                    export=recommender.manifest.get("created_utc") if recommender else None))
                return
            if recommender is None:
                self.send(503 if url.path.startswith(("/demo/", "/recommend/", "/health")) else 404,
                          dict(error="The final model package is not loaded. See the setup screen."))
                return
            try:
                if url.path == "/demo/categories":
                    display = {c["category_id"]: c for c in categories}
                    result = [dict(category_id=c, count=len(ids), **{k: v for k, v in display.get(c, {}).items() if k != "category_id"})
                              for c, ids in sorted(recommender.products_by_category.items())]
                    self.send(200, dict(categories=result))
                elif url.path == "/demo/products":
                    values = parse_qs(url.query).get("ids", [""])[0].split(",")
                    if len(values) > 200:
                        raise ValueError("Request at most 200 products")
                    product_list = []
                    for value in values:
                        item = int(value)
                        price, category, _ = recommender.catalog[item]
                        product_list.append(products.get(item, dict(item_id=item, category_id=category)))
                    self.send(200, dict(products=product_list))
                else:
                    super().do_GET()
            except KeyError:
                self.send(404, dict(error="Product not in the catalog"))
            except ValueError as error:
                self.send(400, dict(error=str(error)))

        def do_POST(self):
            if self.path not in ("/recommend/home", "/recommend/category"):
                self.send(404, dict(error="unknown path"))
            elif recommender is None:
                self.send(503, dict(error="The final model package is not loaded."))
            else:
                super().do_POST()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)

def run(folder=None, port=8000):
    choices = sorted((ROOT / "results" / "handoff").glob("*/manifest.json"))
    folder = Path(folder) if folder else (choices[-1].parent if choices else
                                         ROOT / "results" / "handoff" / "20261108T203934Z")
    issues = bundle_issues(folder)
    recommender = None
    if not issues:
        from .serving import Recommender
        recommender = Recommender(folder)
    else:
        print(f"Model package: {folder}\n" + "\n".join(issues), flush=True)
    server = make_demo_server(recommender, port, issues)
    print(f"Clew: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
