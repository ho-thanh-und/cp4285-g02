# Clew

Start the local demo from the project root:

```bash
.venv/bin/python scripts/run_demo.py --handoff results/handoff/20261109T100233Z --port 8000
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

To choose the home page's four discovery places from SASRec's top 100 instead of its top 20 (the website setting chosen on 2026-11-09; the tested setting is 20, see EXPERIMENT_LOG.md §6p), add `--home-pool 100`. Places 1–6 do not change; the home lists are then labelled `six_four_sasrec_pool100` in responses and logs, and `/demo/status` reports the pool.

To let a re-clicked product count as the shopper's latest interest, add `--click-order last`. SASRec then reads the visit's products in order of their last click instead of their first. The first-click order is how SASRec was trained and tested. The last-click order was chosen for the website after the validation and selection weeks (EXPERIMENT_LOG.md §6r). Home lists are labelled with `_last`, and `/demo/status` reports the order.

The website setting used for testing:

```bash
.venv/bin/python scripts/run_demo.py --handoff results/handoff/20261109T100233Z --home-pool 100 --click-order last --log
```

To record visits for an online comparison, add `--log`. It is off by default. With it, the page shows a notice in the footer and posts anonymous `impression` and `click` events (random visit ID, page, shown product IDs, clicked product and position, list source) to `/log`. The server keeps only the whitelisted fields in HANDOFF.md §4 and appends them to `logs/interactions.jsonl`, which git ignores. Delete the log after grading. Without `--log`, `/log` answers 404 as before.

Open http://127.0.0.1:8000. Stop the server with Ctrl+C in its terminal before restarting it. Models and display metadata load at startup. Refreshing the browser alone does not reload them.

The selected handoff manifest SHA-256 is `487579de3202734d90c23efb6cf0a21c452160971f1b16783cacdf10f6ebe9b3`. The saved package verification passed. Cold home uses saved popularity; personalized home uses SASRec's top 20 and 6+4 selection; category uses LightGBM; related items use the saved item-kNN list with alpha 0.8. No recommendation is filtered for display.

Browse a category, open a product, follow a related product, and return home. Start new session clears the tab's clicks and starts a new visit ID. Back and Forward navigate without adding clicks. Reloading keeps the visit: clicks are kept in the tab's `sessionStorage` and end with the tab. Tabs are independent.

Every product card shows its category in the picture's top-left corner. There are 1,217 categories: four are featured in the menu, and the rest are named "<Theme> collection <id>". Variety should be judged by these labels, not by the product names or pictures, which are illustrative. On the home page, the four discovery places (6 + 4) are labelled "Something different" in the picture's bottom-left corner, with a one-line explanation. Products already clicked in this visit are labelled "Viewed". The category ranker moves them up, as it was trained to; see HANDOFF.md. Category pages show 20 products at a time, up to the API's top 500, with a "Showing X of Y products" line. An unknown product shows "Product not found". The page says "Loading Clew…" until the first data arrives.

The fictional SGD prices, names and unbranded product images are display metadata, not recovered Diginetica identities. The local contact sheet was generated with OpenAI imagegen; its prompt and attribution are in `catalog/asset-sources.json`. Sixteen pictures are reused across product families; numeric suffixes distinguish individual catalog records. `scripts/prepare_demo_catalog.py` reproduces the metadata and coverage record using the selected handoff, without training.

The four featured anonymous categories were assigned distinct fictional themes after inspecting their token frequencies and exported related-item connections:

| Original category | Fictional theme | Most frequent anonymous tokens | Same-category related links |
| --- | --- | --- | --- |
| 47 | Electronics | 48552, 43569, 491069 | 15,753 |
| 780 | Kitchen appliances | 14784, 71, 54484 | 2,122 |
| 1096 | Clothing | 13, 25, 34 | 9,449 |
| 842 | Home furnishings | 21648, 32250, 1315 | 35,415 |

These associations are fictional; token IDs do not reveal product meaning. The inspected related lists were predominantly within their original category. Supporting categories keep one stable category identity and are assigned a theme from the most frequent sampled incoming connections. `catalog/coverage.json` records the cold home, category top 20, three two-click visits, SASRec vocabulary membership, top-20/final-10 category spreads, and 6+4 positions. The sampled top-20 lists were each concentrated in one category; variety slots do not guarantee different categories. The 120 seed products and 598 one-hop targets keep their stored display metadata, and every other categorized product in the export gets its name, picture and price computed on request from its category's theme: every one of the export's categories has a theme in `catalog/categories.json`, so all 184,047 products display fully.

A product without a category (none in this export) would still open with its product ID, image unavailable and price unavailable; recommendation order is unchanged. Whole-category browsing is a demonstration extension, while the reported category experiment used logged candidate lists. Images and prices are illustrative.

Run focused checks:

```bash
.venv/bin/python -m pytest tests/test_demo.py tests/test_serving.py -q
node --check web/clew.js
```

A headless Chrome walkthrough covered desktop and 375 px layouts, keyboard focus and product activation, home → category → item → related item → home, reset, Back/Forward, and related-item failure and retry. The mobile check found no horizontal overflow.
