"""Serve the frozen recommendations of the three website pages.

Reads a hand-off folder written by `uv run main.py handoff` (handoff.py) and
returns the lists the frozen choices produce (EXPERIMENT_LOG.md §7; confirmed
on the final test, §6l):

  home page      Before the first click: the most-clicked products of April
                 (popularity). After it: SASRec reads the session's clicked
                 products and ranks the catalog; 6 + 4 then keeps its six most
                 relevant products and adds four mutually diverse ones from its
                 top 20. Products the shopper already clicked are left out.
  item page      "You may also like": item-kNN on April co-clicks with fixed
                 alpha 0.8 over the 20 nearest neighbours (filled with popular
                 products of the category when there are fewer). Precomputed for
                 every catalog product.
  category page  LightGBM (purchase_session, rank_xendcg) orders the category's
                 products from product, popularity and session features. It works
                 without any click (cold start) and uses the session's clicks
                 once there are some.

Every list comes from the same functions that produced the evaluated lists, and
the export checks this against the saved confirmation-week results.

Session events are (item_id, time_ms) clicks, optionally with kind "purchase";
time_ms is milliseconds since the session started, as in the dataset.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def distinct_in_order(item_ids):
    """Clicked products in order of first click, repeats removed (as in the evaluation)."""
    seen, ordered = set(), []
    for item in item_ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class Recommender:
    """All three pages from one hand-off folder. Loading takes a few seconds; calls are fast."""

    def __init__(self, folder):
        import lightgbm as lgb
        import torch
        from .core import tokens
        from .diversity import ProductDiversity
        from .sequence_models import build_model, item_weights

        folder = Path(folder)
        self.folder = folder
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        products = pd.read_parquet(folder / "catalog.parquet")
        self.catalog_ids = products.item_id.to_numpy()
        self.item_index = pd.Series(np.arange(len(products)), index=self.catalog_ids)
        self.category_of = products.set_index("item_id").category_id
        self.diversity = ProductDiversity(products)

        # Home page.
        home = json.loads((folder / "home" / "settings.json").read_text(encoding="utf-8"))
        self.page, self.model_settings, self.home_policy = home["page"], home["model"], home["policy"]
        self.popular_ids = np.load(folder / "home" / "popular_item_ids.npy")
        vocab_rows = self.item_index.loc[np.load(folder / "home" / "vocabulary_item_ids.npy")].to_numpy()
        self.vocab = pd.Series(np.arange(1, len(vocab_rows) + 1), index=vocab_rows)
        self.ids_to_rows = pd.Series(vocab_rows, index=np.arange(1, len(vocab_rows) + 1))
        self.fallback_rows = self.item_index.loc[np.load(folder / "home" / "fallback_item_ids.npy")].to_numpy()
        self.kind = home["kind"]
        if not self.home_policy.endswith(self.kind):
            raise ValueError(f"Home policy {self.home_policy} is not a {self.kind} policy")
        self.model = build_model(self.kind, len(self.vocab), self.model_settings)
        self.model.load_state_dict(torch.load(folder / "home" / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.weights = item_weights(self.model)

        # Item page: one row of recommended product IDs per catalog product, in catalog order.
        table = pd.read_parquet(folder / "item" / "you_may_also_like.parquet")
        if not np.array_equal(table.item_id.to_numpy(), self.catalog_ids):
            raise ValueError("The 'You may also like' table is not in catalog order")
        self.ymal = table.drop(columns="item_id").to_numpy()

        # Category page.
        category = json.loads((folder / "category" / "settings.json").read_text(encoding="utf-8"))
        self.category_settings = category
        self.booster = lgb.Booster(model_file=str(folder / "category" / "model.txt"))
        stats = pd.read_parquet(folder / "category" / "product_stats.parquet")
        if not np.array_equal(stats.item_id.to_numpy(), self.catalog_ids):
            raise ValueError("The category product statistics are not in catalog order")
        self.popularity = {kind: Counter(dict(zip(stats.item_id.to_numpy(), stats[f"prior_{kind}s"].to_numpy())))
                           for kind in ("click", "view", "purchase")}
        self.window_rates = stats[["window_click_rate", "window_purchase_rate"]].to_numpy(np.float32)
        self.catalog = {int(r.item_id): (r.price_log2, None if pd.isna(r.category_id) else int(r.category_id),
                                         tokens(r.name_tokens)) for r in products.itertuples(index=False)}
        self.products_by_category = {int(c): group.item_id.tolist()
                                     for c, group in products.dropna(subset=["category_id"]).groupby("category_id")}

    # ---------------------------------------------------------------- home ---

    def home(self, clicked_item_ids=()):
        """Ten products for the home page given the session's clicked products so far."""
        from .sequence_models import diverse_lists, ranked
        k = self.page["top_k"]
        known = [int(i) for i in distinct_in_order(clicked_item_ids) if int(i) in self.item_index.index]
        if not known:
            items = [int(i) for i in self.popular_ids[:k]]
            return dict(page="home", items=items, source="popularity (no click yet)", diverse_positions=[])
        prefix = self.item_index.loc[known].tolist()
        pool = max(k, self.page["pool_size"])
        members, scores, personal = ranked(self.model, prefix, self.vocab, self.ids_to_rows, self.fallback_rows,
                                           self.model_settings, pool, self.weights)
        if self.home_policy == self.kind:
            top = members[:k]  # relevance only
        else:
            lists = diverse_lists(members, scores, self.diversity, self.catalog_ids, self.page)
            top = lists[self.home_policy[:-len(self.kind) - 1]]  # e.g. six_four_sasrec -> six_four
        slots = self.page["relevance_slots"]
        return dict(page="home", items=self.catalog_ids[top].astype(int).tolist(),
                    source=self.home_policy if personal else "popularity (no clicked product known to SASRec)",
                    diverse_positions=list(range(slots + 1, len(top) + 1)) if "six_four" in self.home_policy else [])

    # ---------------------------------------------------------------- item ---

    def item(self, item_id):
        """'You may also like' for one product (KeyError for a product outside the catalog)."""
        row = self.item_index[int(item_id)]
        items = [int(i) for i in self.ymal[row] if i >= 0]
        return dict(page="item", item_id=int(item_id), items=items, source=self.manifest["pages"]["item"]["policy"],
                    diverse_positions=[])

    # ------------------------------------------------------------ category ---

    def category_scores(self, category_id, products, clicks=(), now_ms=None):
        """LightGBM scores of `products` on a category page, from the session's clicks so far."""
        from .expanded_data import query_features
        events = sorted((int(e["time_ms"]), e.get("kind", "click"), int(e["item_id"]), None) for e in clicks)
        now = int(now_ms) if now_ms is not None else (events[-1][0] + 1 if events else 0)
        query = (-1, "website", "serve", "", now, None, int(category_id), ",".join(str(int(p)) for p in products))
        items, rows, _, _ = query_features(query, events, self.catalog, self.popularity,
                                           self.category_settings["history_limit"])
        width = self.category_settings["base_features"]
        extra = np.zeros((len(items), 2), dtype=np.float32)
        index = self.item_index.reindex(items).to_numpy()
        known = ~np.isnan(index)
        extra[known] = self.window_rates[index[known].astype(int)]
        matrix = np.hstack([rows[:, :width], extra])
        return np.asarray(items), self.booster.predict(matrix[:, self.category_settings["feature_indices"]])

    def category(self, category_id, clicks=(), now_ms=None, products=None, limit=None):
        """The category's products, best first. `products` defaults to every catalog product in it."""
        products = self.products_by_category.get(int(category_id), []) if products is None else list(products)
        if not products:
            return dict(page="category", category_id=int(category_id), items=[], scores=[], source="empty category")
        items, scores = self.category_scores(category_id, products, clicks, now_ms)
        order = np.argsort(-scores, kind="stable")[:limit]
        return dict(page="category", category_id=int(category_id), items=items[order].astype(int).tolist(),
                    scores=[round(float(s), 6) for s in scores[order]], source=self.category_settings["model"],
                    used_session=bool(clicks))
