# Website hand-off

How to run the frozen recommenders behind the demo website. The models are the
ones confirmed on the final test (`REPORT.md` §5.4, `EXPERIMENT_LOG.md` §6l).

| Page | What the website shows | Model |
|---|---|---|
| Home page | 10 products. Before the first click: the most-clicked products of April. After it: personalised from the session's clicks; positions 7–10 are the "diverse" products | SASRec + 6 + 4 |
| Category page | The category's products, best first | LightGBM `purchase_session` |
| Item page, "You may also like" | 10 products related to the one being viewed | item-kNN + fixed alpha 0.8 (precomputed) |

## 1. Export the models (once)

```bash
uv run main.py handoff
```

This writes `results/handoff/<time stamp>/` (about 27 MB; `results/` is not in
Git; the current export is `results/handoff/20261108T203934Z`) and then **verifies** it against the saved final-test runs:

- home page: the served lists equal the saved SASRec + 6 + 4 lists;
- item page: every served list has the saved NDCG@10;
- category page: every category query's served order has the saved NDCG@10.

The export stops with an error if any check fails. The current export passed
all three: 2,000 of 2,000 checked home-page cases give the identical list, all
20,000 item pages and all 952 category queries give the identical NDCG@10. The result is in
`verification.json` inside the folder; `manifest.json` lists every file with its
SHA-256 hash, the code commit and the frozen choices.

| File | Contents |
|---|---|
| `catalog.parquet` | `item_id`, `category_id`, `price_log2`, `name_tokens` for all 184,047 products |
| `home/model.pt`, `home/settings.json` | SASRec weights; model and 6 + 4 settings |
| `home/vocabulary_item_ids.npy` | Products SASRec knows |
| `home/fallback_item_ids.npy`, `home/popular_item_ids.npy` | SASRec's fallback order; cold-start popularity |
| `item/you_may_also_like.parquet` | `item_id`, `r1` … `r10` for every product (`-1` = no product) |
| `category/model.txt`, `category/settings.json` | LightGBM booster and its feature columns |
| `category/product_stats.parquet` | The frozen popularity inputs of the category ranker |

## 2. Start the API

```bash
uv run main.py serve --handoff results/handoff/<time stamp> --port 8000
```

It loads in about 6 seconds and answers on `http://127.0.0.1:8000`. It uses only
the Python standard library (no web framework) and allows calls from a website
on another port.

| Call | Body or path | Returns |
|---|---|---|
| `GET /health` | – | `status`, the frozen choices, export time |
| `POST /recommend/home` | `{"clicks": [{"item_id": 123, "time_ms": 0}, ...]}` | `items` (10 IDs), `source`, `diverse_positions` |
| `GET /recommend/item/<item_id>` | – | `items` (10 IDs) |
| `POST /recommend/category` | `{"category_id": 7, "clicks": [...], "now_ms": 90000, "limit": 50}` | `items` best first, `scores` |
| `POST /log` | one event (Section 4) | `{"logged": true}` |

- `time_ms` is milliseconds since the visit started, as in the dataset. Send all
  clicks of the visit so far, in any order; repeats are ignored.
- `/recommend/category` orders every catalog product of the category by
  default; pass `"products": [...]` to order your own list instead.
- An unknown product gives `404`; a malformed request gives `400` with a
  message.

Example:

```bash
curl -X POST http://127.0.0.1:8000/recommend/home -H "Content-Type: application/json" -d "{\"clicks\": [{\"item_id\": 9654, \"time_ms\": 0}]}"
```

From Python without the API (run from the repository root with
`uv run python`, so the project code and its settings are found):

```python
from session_rerank.serving import Recommender
recommender = Recommender("results/handoff/<time stamp>")
recommender.home([9654])                     # after one click
recommender.item(9654)                       # "You may also like"
recommender.category(1010, clicks=[{"item_id": 9654, "time_ms": 0}], now_ms=60000, limit=20)
```

## 3. Showing the lists

- **Home page:** show `items` in order. `diverse_positions` (7–10) are the
  products chosen for variety; the page may label them, e.g. "Something
  different".
- **Item page:** show `items` under "You may also like".
- **Category page:** show `items` in order. `limit` only shortens the list
  (there is no offset): fetch enough products and page through them on the
  website. The model
  learned that shoppers often click a product again, so a product clicked
  earlier in the visit usually moves to the top (e.g. product 9654 in category
  1010). Keep this, or drop already-clicked products from the display; the
  offline results include the behaviour.
- **Names:** the catalog has only anonymised name tokens. The demo's names and
  pictures are AI-generated and fictional (`ETHICS.md` §7): label them as such,
  do not use real brand names, and keep a map from `item_id` to the generated
  name. All models keep using `item_id`.

## 4. Logging interactions (privacy by design)

`POST /log` appends one JSON line to `logs/interactions.jsonl` and keeps **only**
these fields; anything else is dropped, and the server never records IP
addresses or request lines (`ETHICS.md` §5):

| Field | Type | Meaning |
|---|---|---|
| `session_id` | string, required | Random ID made by the website for this visit; not stored in a lasting cookie, no account |
| `event` | required | `impression`, `click` or `purchase` |
| `page` | required | `home`, `item` or `category` |
| `time_ms` | integer, required | Milliseconds since the visit started |
| `shown` | list of IDs | The products shown, in display order (impressions) |
| `item_id` | integer | The clicked or bought product |
| `position` | integer | 1-based position of the clicked product in `shown` |
| `context_item_id` | integer | The product page that is open (item page) |
| `category_id` | integer | The category page |
| `list_source` | string | The `source` field of the recommendation response |

The server adds only the date (`logged_date`), not the time of day. Log an
`impression` each time a list is shown and a `click` when a product is clicked;
this gives the positions needed to evaluate the pages online and to correct for
position bias later. Show visitors a short notice that clicks are recorded for a
course project, and delete the log after grading.

## 5. Speed (laptop CPU, measured on the real export)

| Call | Time |
|---|---:|
| Load the hand-off folder | about 6 s |
| Home page, cold start / after a click | 24 ms / 17 ms |
| Item page | 3 ms |
| Category page (category 1010) / largest category (3,851 products) | 11–13 ms / 35 ms |

## 6. Limits to keep in mind

- The category ranker was evaluated on the site's lists of up to 200 products;
  ranking whole categories (median 48, up to 3,851 products) was not evaluated
  offline.
- Products clicked by nobody before May are recommended rarely: the models learn
  from past clicks (`REPORT.md` §6–7).
- Text search is not supported.
- The models are frozen as of the final test; retraining needs the full
  pipeline (`EXPERIMENT_LOG.md` §10).
