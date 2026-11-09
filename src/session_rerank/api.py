"""A small JSON API for the website, using only the Python standard library.

    uv run main.py serve --handoff results/handoff/<stamp> --port 8000

Endpoints (JSON in and out; browsers on another origin are allowed):
  GET  /health                  versions of the frozen choices
  POST /recommend/home          {"clicks": [{"item_id": 123, "time_ms": 0}, ...]}
  GET  /recommend/item/<id>     "You may also like" for one product
  POST /recommend/category      {"category_id": 7, "clicks": [...], "now_ms": 90000, "limit": 50,
                                 "products": [optional list of product IDs to order]}
  POST /log                     one interaction event (HANDOFF.md), appended to the log file

Privacy (ETHICS.md §5): the server never records IP addresses or request lines,
and /log keeps only the whitelisted fields below; anything else is dropped.
"""

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG_FIELDS = {
    "session_id": str,        # random per visit, made by the website; no account or cross-visit ID
    "event": str,             # "impression", "click" or "purchase"
    "page": str,              # "home", "item" or "category"
    "time_ms": int,           # milliseconds since the session started
    "item_id": int,           # the clicked or bought product
    "shown": list,            # product IDs shown, in display order (impressions)
    "position": int,          # 1-based position of the clicked product in the shown list
    "context_item_id": int,   # the product whose page is open (item page)
    "category_id": int,       # the category page
    "list_source": str,       # e.g. "six_four_sasrec", from the recommendation response
}
EVENTS, PAGES = {"impression", "click", "purchase"}, {"home", "item", "category"}
MAX_BODY = 1 << 20


def clean_event(event):
    """Keep only the whitelisted fields with the right types; refuse unusable events."""
    if not isinstance(event, dict):
        raise ValueError("an event must be a JSON object")
    kept = {}
    for name, kind in LOG_FIELDS.items():
        if name not in event or event[name] is None:
            continue
        value = event[name]
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an integer")
        if kind is str and (not isinstance(value, str) or len(value) > 64):
            raise ValueError(f"{name} must be a short string")
        if kind is list and (not isinstance(value, list) or len(value) > 500
                             or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
            raise ValueError(f"{name} must be a list of product IDs")
        kept[name] = value
    for name in ("session_id", "event", "page", "time_ms"):
        if name not in kept:
            raise ValueError(f"{name} is required")
    if kept["event"] not in EVENTS or kept["page"] not in PAGES:
        raise ValueError(f"event must be one of {sorted(EVENTS)} and page one of {sorted(PAGES)}")
    kept["logged_date"] = datetime.now(timezone.utc).date().isoformat()  # the day only, no clock time
    return kept


def make_server(recommender, port=8000, log_path=Path("logs/interactions.jsonl"), host="127.0.0.1"):
    lock = threading.Lock()
    log_path = Path(log_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # no request lines or client addresses in any log
            pass

        def send(self, status, body):
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_OPTIONS(self):
            self.send(204, {})

        def do_GET(self):
            try:
                if self.path == "/health":
                    self.send(200, dict(status="ok", pages=recommender.manifest["pages"],
                                        export=recommender.manifest.get("created_utc")))
                elif self.path.startswith("/recommend/item/"):
                    self.send(200, recommender.item(int(self.path.rsplit("/", 1)[1])))
                else:
                    self.send(404, dict(error="unknown path"))
            except KeyError:
                self.send(404, dict(error="product not in the catalog"))
            except ValueError as error:
                self.send(400, dict(error=str(error)))

        def do_POST(self):
            try:
                body = self.body()
                if self.path == "/recommend/home":
                    clicks = sorted(body.get("clicks", []), key=lambda c: c.get("time_ms", 0))
                    self.send(200, recommender.home([int(c["item_id"]) for c in clicks]))
                elif self.path == "/recommend/category":
                    self.send(200, recommender.category(int(body["category_id"]), body.get("clicks", []),
                                                       body.get("now_ms"), body.get("products"), body.get("limit")))
                elif self.path == "/log":
                    event = clean_event(body)
                    with lock:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(log_path, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(event) + "\n")
                    self.send(200, dict(logged=True))
                else:
                    self.send(404, dict(error="unknown path"))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                self.send(400, dict(error=f"bad request: {error}"))

    return ThreadingHTTPServer((host, port), Handler)


def serve(handoff, port=8000, log_path=Path("logs/interactions.jsonl"), host="127.0.0.1"):
    from .serving import Recommender
    print(f"[serve] Loading {handoff}", flush=True)
    server = make_server(Recommender(handoff), port, log_path, host)
    print(f"[serve] http://{host}:{server.server_address[1]}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
