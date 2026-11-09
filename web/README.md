# Clew

Start the local demo from the project root:

```bash
.venv/bin/python scripts/run_demo.py --handoff results/handoff/20261109T100233Z --port 8000
```

Open http://127.0.0.1:8000. Stop the server with Ctrl+C in its terminal before restarting it. Models and display metadata load at startup. Refreshing the browser alone does not reload them.

The selected handoff manifest SHA-256 is `487579de3202734d90c23efb6cf0a21c452160971f1b16783cacdf10f6ebe9b3`. The saved package verification passed. Cold home uses saved popularity; personalized home uses SASRec's top 20 and 6+4 selection; category uses LightGBM; related items use the saved item-kNN list with alpha 0.8. No recommendation is filtered for display.

Browse a category, open a product, follow a related product, and return home. Start new session clears the tab's clicks. Back and Forward navigate without adding clicks; reloading the page starts a fresh visit. Tabs are independent.

The fictional SGD prices, names and unbranded product images are display metadata, not recovered Diginetica identities. The local contact sheet was generated with OpenAI imagegen; its prompt and attribution are in `catalog/asset-sources.json`. Sixteen pictures are reused across product families; numeric suffixes distinguish individual catalog records. `scripts/prepare_demo_catalog.py` reproduces the metadata and coverage record using the selected handoff, without training.

The four featured anonymous categories were assigned distinct fictional themes after inspecting their token frequencies and exported related-item connections:

| Original category | Fictional theme | Most frequent anonymous tokens | Same-category related links |
| --- | --- | --- | --- |
| 47 | Electronics | 48552, 43569, 491069 | 15,753 |
| 780 | Kitchen appliances | 14784, 71, 54484 | 2,122 |
| 1096 | Clothing | 13, 25, 34 | 9,449 |
| 842 | Home furnishings | 21648, 32250, 1315 | 35,415 |

These associations are fictional; token IDs do not reveal product meaning. The inspected related lists were predominantly within their original category. Supporting categories keep one stable category identity and are assigned a theme from the most frequent sampled incoming connections. `catalog/coverage.json` records the cold home, category top 20, three two-click visits, SASRec vocabulary membership, top-20/final-10 category spreads, and 6+4 positions. The sampled top-20 lists were each concentrated in one category; variety slots do not guarantee different categories. The 120 seed products and 598 one-hop targets have display metadata; 6,764 products are prepared in total.

Coverage is finite. An unexpected unprepared item still opens with its product ID, image unavailable and price unavailable; recommendation order is unchanged. Whole-category browsing is a demonstration extension, while the reported category experiment used logged candidate lists. Images and prices are illustrative.

Run focused checks:

```bash
.venv/bin/python -m pytest tests/test_demo.py tests/test_serving.py -q
node --check web/clew.js
```

A headless Chrome walkthrough covered desktop and 375 px layouts, keyboard focus and product activation, home → category → item → related item → home, reset, Back/Forward, and related-item failure and retry. The mobile check found no horizontal overflow.
