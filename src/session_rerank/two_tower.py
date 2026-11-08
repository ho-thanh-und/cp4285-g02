"""Two-tower first-stage filter, trained with rank-aware click weights.

Why the rank matters. The site showed each query's products in a fixed order,
and shoppers look at the top more often than the bottom. In the 10,000-query
training sample rank 1 is clicked 3.1% of the time, rank 10 2.2% and rank 50
0.9%. We model the chance that a shopper looked at rank r as p(r) = r ** -eta
(so p(1) = 1) and fit eta to those click rates.

Rank-aware pair weights. Each training pair is (clicked product, other product):
  * clicked at rank r          weight r ** eta    A deep click counts more: the
                                                  shopper passed better-placed
                                                  products to choose it.
  * shown but skipped, rank r  weight r ** -eta   Skipping rank 1 is strong
                                                  evidence against it; skipping
                                                  rank 150 may mean it was never
                                                  seen, so the penalty is small.
  * never shown                fixed weight       The logs say nothing about
                                                  these; they teach the model
                                                  the wider pool it filters.
Pair loss = w_click * w_other * softplus(score_other - score_click).

Towers. The product tower embeds anonymized name tokens, category, price band
and popularity. The query tower embeds the page category (0 for a text search),
search tokens, earlier clicked/viewed products of the session (through the
product tower) and the query's position in its session. Score = scaled cosine.

Popularity without leakage. Clicks and views per product per day are counted
from the full raw files over the training window (all shoppers, not only the
sampled sessions). A training query on day d sees only days before d, as daily
rates, so its own clicks never inflate its products. Validation queries come
after the window and see the whole window. The popularity baseline uses the
same whole-window counts, so the comparison is fair.

Use as a filter. For a category page (95% of queries) the pool is every catalog
product in that category; for a text search it is the whole catalog. Keep the
top K1. Clicks exist almost only for products the site showed, so recall of the
clicked products is the measure; a retrieved product that was never shown
cannot be judged either way.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .expanded_data import read_candidates
from .study_files import save_table, write_json

TEXT_SEARCH = 0  # Query category used by the dataset for text searches.


# ---------------------------------------------------------------- settings ---

def read_two_tower_settings(path):
    """Reject impossible values before reading any data."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("embedding_dim", "hidden_dim", "max_name_tokens", "max_query_tokens",
                 "max_history_items", "max_position", "epochs", "batch_size", "threads",
                 "propensity_fit_max_rank"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("shown_negatives_per_positive", "unshown_negatives_per_positive"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if settings["shown_negatives_per_positive"] + settings["unshown_negatives_per_positive"] < 1:
        raise ValueError("Use at least one negative product per click")
    for name in ("learning_rate", "max_positive_weight", "unshown_negative_weight"):
        if not settings[name] > 0:
            raise ValueError(f"{name} must be positive")
    if settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    if not isinstance(settings.get("use_popularity"), bool):
        raise ValueError("use_popularity must be true or false")
    if not isinstance(settings.get("raw_directory"), str):
        raise ValueError("raw_directory must name the folder with the raw CSV files")
    eta = settings["propensity_eta"]
    if eta != "estimate" and (isinstance(eta, bool) or not isinstance(eta, (int, float)) or eta < 0):
        raise ValueError('propensity_eta must be "estimate" or a nonnegative number')
    k1 = settings["k1_values"]
    if not k1 or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in k1):
        raise ValueError("k1_values must be positive integers")
    return settings


# ------------------------------------------------------------ rank weights ---

def estimate_eta(candidates, max_rank):
    """Fit click rate ~ rank ** -eta on ranks 1..max_rank of the training lists."""
    shown = candidates[candidates.original_rank <= max_rank]
    rate = shown.groupby("original_rank").clicked.mean()
    rate = rate[rate > 0]
    if len(rate) < 2:
        return 0.0
    slope = np.polyfit(np.log(rate.index.to_numpy(float)), np.log(rate.to_numpy(float)), 1)[0]
    return float(max(0.0, -slope))


def click_weight(rank, eta, cap):
    """Bonus for a click at this rank: 1 at rank 1, larger deeper, capped."""
    return np.minimum(np.asarray(rank, float) ** eta, cap)


def other_weight(rank, eta, unshown_weight):
    """Penalty weight for a non-clicked product; NaN rank means never shown."""
    rank = np.asarray(rank, float)
    shown = np.isfinite(rank)
    return np.where(shown, np.where(shown, rank, 1.0) ** -eta, unshown_weight)


# ------------------------------------------------------------------- data ---

def _token_matrix(strings, vocab, width):
    """Comma-separated token strings -> padded index matrix; unknown tokens dropped."""
    matrix = np.zeros((len(strings), width), dtype=np.int64)
    for row, value in enumerate(strings):
        if isinstance(value, str) and value:
            indices = [vocab[t] for t in value.split(",") if t in vocab][:width]
            matrix[row, :len(indices)] = indices
    return matrix


POPULARITY_FEATURES = 3  # log daily click rate, log daily view rate, counts available.


def daily_activity(raw_directory, item_index, start, days):
    """Clicks and views per product per day of the training window, from raw CSVs.

    Returns two arrays [days, products]. Clicks take their date from their query.
    Products missing from the catalog and dates outside the window are ignored.
    """
    raw = Path(raw_directory)
    dates = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "eventdate"])
    clicks = pd.read_csv(raw / "train-clicks.csv", sep=";", usecols=["queryId", "itemId"])
    clicks = clicks.merge(dates, on="queryId")
    views = pd.read_csv(raw / "train-item-views.csv", sep=";", usecols=["itemId", "eventdate"])
    counts = []
    for frame in (clicks, views):
        offset = (pd.to_datetime(frame.eventdate) - pd.Timestamp(start)).dt.days.to_numpy()
        keep = (offset >= 0) & (offset < days) & frame.itemId.isin(item_index.index).to_numpy()
        matrix = np.zeros((days, len(item_index)), dtype=np.float32)
        np.add.at(matrix, (offset[keep], item_index.loc[frame.itemId[keep]].to_numpy()), 1)
        counts.append(matrix)
    return counts


def popularity_features(clicks, views):
    """Features [days + 1, products, 3]; row d uses only days before d."""
    days = clicks.shape[0]
    before_clicks = np.vstack([np.zeros((1, clicks.shape[1]), np.float32), clicks.cumsum(0)])
    before_views = np.vstack([np.zeros((1, views.shape[1]), np.float32), views.cumsum(0)])
    observed = np.arange(days + 1, dtype=np.float32)[:, None]
    per_day = np.maximum(observed, 1)
    return np.stack([np.log1p(before_clicks / per_day), np.log1p(before_views / per_day),
                     np.broadcast_to(observed > 0, before_clicks.shape).astype(np.float32)], axis=-1)


class RunData:
    """Products, queries, session histories and logged lists from one trained run.

    Only training-split queries and events build the vocabulary and popularity.
    Histories use click/view events strictly before each query in its session.
    """

    def __init__(self, run_folder, settings):
        run_folder = Path(run_folder)
        self.run_folder = run_folder
        products = pd.read_parquet(run_folder / "products.parquet")
        self.item_ids = products.item_id.to_numpy(np.int64)
        self.item_index = pd.Series(np.arange(len(products)), index=self.item_ids)
        with sqlite3.connect(run_folder / "development.sqlite") as db:
            queries = pd.read_sql("SELECT query_id, session_key, split, eventdate, timeframe, query_tokens, "
                                  "category FROM queries", db)
            events = pd.read_sql("SELECT session_key, kind, timeframe, query_id, item_id FROM events "
                                 "WHERE kind IN ('click', 'view') AND item_id IS NOT NULL", db)
        events = events[events.item_id.isin(self.item_index.index)].copy()
        events["item"] = self.item_index.loc[events.item_id.astype(np.int64)].to_numpy()

        train_tokens = queries.loc[queries.split.eq("train"), "query_tokens"].dropna()
        vocabulary = {t for s in products.name_tokens.dropna() for t in s.split(",") if t}
        vocabulary |= {t for s in train_tokens for t in str(s).split(",") if t}
        vocab = {token: index for index, token in enumerate(sorted(vocabulary), start=1)}
        self.vocab_size = len(vocab) + 1
        self.names = _token_matrix(products.name_tokens.tolist(), vocab, settings["max_name_tokens"])
        self.item_category = products.category_id.fillna(0).to_numpy(np.int64)
        self.item_price = products.price_log2.clip(lower=0).to_numpy(np.int64)

        queries = queries.sort_values(["session_key", "timeframe", "query_id"]).reset_index(drop=True)
        queries["position"] = (queries.groupby("session_key").cumcount() + 1).clip(upper=settings["max_position"])
        queries["category"] = queries.category.fillna(TEXT_SEARCH).astype(np.int64)
        self.queries = queries.set_index("query_id")
        self.query_row = pd.Series(np.arange(len(queries)), index=queries.query_id)
        self.query_tokens = _token_matrix(queries.query_tokens.tolist(), vocab, settings["max_query_tokens"])
        self.query_category = queries.category.to_numpy(np.int64)
        self.query_position = queries.position.to_numpy(np.int64)
        self.history = self._histories(queries, events, settings["max_history_items"])
        self.categories = int(max(self.item_category.max(), self.query_category.max())) + 1
        self.prices = int(self.item_price.max()) + 1

        # Pools to filter from: products of each category, or the whole catalog.
        order = np.argsort(self.item_category, kind="stable")
        starts = np.searchsorted(self.item_category[order], np.arange(self.categories + 1))
        self._category_items = order
        self._category_starts = starts
        # The training window runs from the first to the last training query date.
        train_dates = pd.to_datetime(queries.loc[queries.split.eq("train"), "eventdate"])
        start, days = train_dates.min(), int((train_dates.max() - train_dates.min()).days) + 1
        offset = (pd.to_datetime(queries.eventdate) - start).dt.days.to_numpy()
        # Training queries see days before their own; later splits see the whole window.
        self.query_day = np.where(queries.split.eq("train"), np.clip(offset, 0, days), days).astype(np.int64)
        self.window = dict(start=str(start.date()), days=days)
        self.use_popularity = settings["use_popularity"]
        clicks, views = daily_activity(settings["raw_directory"], self.item_index, start, days)
        self.popularity_inputs = popularity_features(clicks, views)
        # Baseline ranking: all clicks and views of the whole training window.
        self.popularity = (clicks.sum(0) + views.sum(0)).astype(float)
        self.window_clicks = int(clicks.sum())

        candidates = {split: read_candidates(run_folder, f"{split}") for split in ("train", "validation")}
        for frame in candidates.values():
            frame["item"] = self.item_index.reindex(frame.item_id.to_numpy()).to_numpy()
        self.candidates = candidates

    @staticmethod
    def _histories(queries, events, limit):
        """Most recent distinct clicked/viewed products before each query, padded with -1."""
        joined = queries[["query_id", "session_key", "timeframe"]].merge(
            events.rename(columns={"timeframe": "event_time", "query_id": "event_query"}), on="session_key")
        earlier = joined[(joined.event_time < joined.timeframe) & (joined.event_query != joined.query_id)]
        earlier = (earlier.sort_values("event_time", ascending=False)
                   .drop_duplicates(["query_id", "item"]).groupby("query_id").head(limit))
        history = np.full((len(queries), limit), -1, dtype=np.int64)
        rows = pd.Series(np.arange(len(queries)), index=queries.query_id)
        for qid, items in earlier.groupby("query_id", sort=False)["item"]:
            values = items.to_numpy()
            history[rows[qid], :len(values)] = values
        return history

    def pool(self, category):
        """Catalog rows a query filters from: its category, or everything for text search."""
        if category == TEXT_SEARCH or category >= self.categories:
            return np.arange(len(self.item_ids))
        return self._category_items[self._category_starts[category]:self._category_starts[category + 1]]


def training_examples(data):
    """One row per clicked logged product: query row, product row and its rank."""
    frame = data.candidates["train"].dropna(subset=["item"])
    clicks = frame[frame.clicked]
    return pd.DataFrame(dict(query=data.query_row.loc[clicks.query_id].to_numpy(),
                             item=clicks["item"].astype(np.int64).to_numpy(),
                             rank=clicks.original_rank.to_numpy(np.int64)))


def logged_lists(data, split):
    """Per query: logged product rows, ranks and click flags, in displayed order."""
    frame = data.candidates[split].dropna(subset=["item"]).sort_values(["query_id", "original_rank"])
    return {int(qid): (g["item"].astype(np.int64).to_numpy(), g.original_rank.to_numpy(np.int64),
                       g.clicked.to_numpy(bool))
            for qid, g in frame.groupby("query_id", sort=False)}


def sample_negatives(data, examples, lists, settings, eta, rng):
    """Shown-skipped and never-shown products for each click, with their weights.

    Returns item rows [B, S+U] and weights [B, S+U]; empty slots have weight 0.
    """
    shown_n, unshown_n = settings["shown_negatives_per_positive"], settings["unshown_negatives_per_positive"]
    items = np.zeros((len(examples), shown_n + unshown_n), dtype=np.int64)
    weights = np.zeros_like(items, dtype=float)
    query_ids = data.query_row.index.to_numpy()
    for row, (query, _, _) in enumerate(examples):
        logged, ranks, clicked = lists[int(query_ids[query])]
        skipped = np.flatnonzero(~clicked)
        if shown_n and len(skipped):
            pick = rng.choice(skipped, shown_n)
            items[row, :shown_n] = logged[pick]
            weights[row, :shown_n] = other_weight(ranks[pick], eta, settings["unshown_negative_weight"])
        if unshown_n:
            pool = data.pool(int(data.query_category[query]))
            unseen = np.setdiff1d(pool, logged, assume_unique=False)
            if len(unseen):
                items[row, shown_n:] = rng.choice(unseen, unshown_n)
                weights[row, shown_n:] = settings["unshown_negative_weight"]
    return items, weights


# ------------------------------------------------------------------ model ---

def build_model(data, settings):
    """Create the two towers. Torch is imported here so other stages stay light."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    dim, hidden = settings["embedding_dim"], settings["hidden_dim"]

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(data.vocab_size, dim, padding_idx=0)
            self.categories = nn.Embedding(data.categories, dim)  # Shared by pages and products.
            self.prices = nn.Embedding(data.prices, dim)
            self.positions = nn.Embedding(settings["max_position"] + 1, dim)
            extra = POPULARITY_FEATURES if data.use_popularity else 0
            self.item_mlp = nn.Sequential(nn.Linear(3 * dim + extra, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.query_mlp = nn.Sequential(nn.Linear(4 * dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

        def bag(self, tokens):
            mask = (tokens > 0).float().unsqueeze(-1)
            return (self.tokens(tokens) * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        def items(self, rows, days):
            """Product vectors; popularity is read as of each row's query day."""
            rows, days = torch.as_tensor(rows), torch.as_tensor(days)
            parts = [self.bag(torch.as_tensor(data.names)[rows]),
                     self.categories(torch.as_tensor(data.item_category)[rows]),
                     self.prices(torch.as_tensor(data.item_price)[rows])]
            if data.use_popularity:
                parts.append(torch.as_tensor(data.popularity_inputs)[days, rows])
            return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)

        def queries(self, rows):
            rows = torch.as_tensor(rows)
            history = torch.as_tensor(data.history)[rows]
            mask = (history >= 0).float().unsqueeze(-1)
            day = torch.as_tensor(data.query_day)[rows].unsqueeze(1).expand_as(history)
            vectors = self.items(history.clamp(min=0).reshape(-1), day.reshape(-1)).reshape(*history.shape, dim)
            session = (vectors * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            has_history = (mask.sum(1) > 0).float()
            x = torch.cat([self.bag(torch.as_tensor(data.query_tokens)[rows]),
                           self.categories(torch.as_tensor(data.query_category)[rows]),
                           self.positions(torch.as_tensor(data.query_position)[rows]),
                           session, has_history], dim=-1)
            return F.normalize(self.query_mlp(x), dim=-1)

        def query_days(self, rows):
            return torch.as_tensor(data.query_day)[torch.as_tensor(rows)]

        def scale(self):
            return self.log_scale.clamp(0.0, 4.0).exp()

    return TwoTower()


def pair_loss(model, queries, positives, negatives, w_click, w_other):
    """Weighted pairwise loss: push each click above its paired products."""
    import torch
    import torch.nn.functional as F
    q = model.queries(queries)
    day = model.query_days(queries)
    positive = (q * model.items(positives, day)).sum(-1) * model.scale()
    flat = torch.as_tensor(negatives).reshape(-1)
    other = model.items(flat, day.unsqueeze(1).expand(*negatives.shape).reshape(-1)).reshape(*negatives.shape, -1)
    negative = (q.unsqueeze(1) * other).sum(-1) * model.scale()
    weights = torch.as_tensor(w_click, dtype=torch.float32).unsqueeze(1) * torch.as_tensor(w_other, dtype=torch.float32)
    losses = F.softplus(negative - positive.unsqueeze(1))
    return (weights * losses).sum() / weights.sum().clamp(min=1e-9)


# ------------------------------------------------------------- evaluation ---

def item_embeddings(model, data, chunk=8192):
    """Product vectors with whole-window popularity, as used after training ends."""
    import torch
    day = data.window["days"]
    with torch.no_grad():
        return np.concatenate([
            model.items(np.arange(start, end), np.full(end - start, day)).numpy()
            for start in range(0, len(data.item_ids), chunk)
            for end in [min(start + chunk, len(data.item_ids))]])


def evaluate_retrieval(model, data, lists, k_values, items=None):
    """Recall of clicked products when keeping the top K1, per query and method.

    Pools: "logged" reorders only the products the site showed; "full" filters
    the category (or the catalog for text search). Baselines use the same pool:
    the site's own order, training popularity, and the random-choice expectation.
    """
    import torch
    items = item_embeddings(model, data) if items is None else items
    query_ids = [qid for qid, (_, _, clicked) in lists.items() if clicked.any()]
    with torch.no_grad():
        vectors = model.queries(data.query_row.loc[query_ids].to_numpy()).numpy()
    rows = []
    for qid, vector in zip(query_ids, vectors):
        logged, ranks, clicked = lists[qid]
        category = int(data.query_category[data.query_row[qid]])
        clicked_items = set(logged[clicked].tolist())
        full = data.pool(category)
        base = dict(query_id=qid, query_type="text" if category == TEXT_SEARCH else "category",
                    position=int(data.query_position[data.query_row[qid]]), pool_size=len(full),
                    clicked=len(clicked_items), clicked_in_pool=len(clicked_items & set(full.tolist())))
        logged_scores = items[logged] @ vector
        full_scores = items[full] @ vector
        orders = {
            ("logged", "two_tower"): logged[np.argsort(-logged_scores, kind="stable")],
            ("logged", "site_order"): logged,
            ("logged", "popularity"): logged[np.argsort(-data.popularity[logged], kind="stable")],
            ("full", "two_tower"): full[np.argsort(-full_scores, kind="stable")],
            ("full", "site_order"): logged,
            ("full", "popularity"): full[np.argsort(-data.popularity[full], kind="stable")],
        }
        logged_set = set(logged.tolist())
        for k in k_values:
            for (pool, method), order in orders.items():
                top = order[:k]
                rows.append(base | dict(pool=pool, method=method, k1=k,
                                        recall=len(clicked_items & set(top.tolist())) / len(clicked_items),
                                        shown_share=np.mean([item in logged_set for item in top]) if len(top) else np.nan))
            for pool, size in (("logged", len(logged)), ("full", len(full))):
                rows.append(base | dict(pool=pool, method="random", k1=k, recall=min(1.0, k / size), shown_share=np.nan))
    return pd.DataFrame(rows)


def _click_ndcg(top, clicked_items, k):
    """Binary click NDCG@k; never-shown products count as not clicked."""
    gains = np.array([item in clicked_items for item in top[:k]], dtype=float)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(k, len(clicked_items))].sum()
    return float(gains @ discounts[:len(gains)] / ideal) if ideal > 0 else np.nan


def evaluate_top_lists(model, data, lists, items, k=10, seed=0):
    """NDCG@k and ILD@k of each method's own top-k list, before any LightGBM reranking.

    This shows what a shopper would see if the filter's order were used directly.
    Never-shown products cannot have clicks, so NDCG understates full-pool lists;
    unshown_share reports how much of each list that affects.
    """
    import torch
    from .diversity import ProductDiversity
    catalog = ProductDiversity(pd.read_parquet(data.run_folder / "products.parquet"))
    rng = np.random.default_rng(seed)
    query_ids = [qid for qid, (_, _, clicked) in lists.items() if clicked.any()]
    with torch.no_grad():
        vectors = model.queries(data.query_row.loc[query_ids].to_numpy()).numpy()
    rows = []
    for qid, vector in zip(query_ids, vectors):
        logged, _, clicked = lists[qid]
        row = data.query_row[qid]
        category = int(data.query_category[row])
        clicked_items = set(logged[clicked].tolist())
        full = data.pool(category)
        lists_by_method = {
            ("logged", "site_order"): logged[:k],
            ("logged", "popularity"): logged[np.argsort(-data.popularity[logged], kind="stable")[:k]],
            ("logged", "two_tower"): logged[np.argsort(-(items[logged] @ vector), kind="stable")[:k]],
            ("logged", "random"): rng.choice(logged, min(k, len(logged)), replace=False),
            ("full", "popularity"): full[np.argsort(-data.popularity[full], kind="stable")[:k]],
            ("full", "two_tower"): full[np.argsort(-(items[full] @ vector), kind="stable")[:k]],
            ("full", "random"): rng.choice(full, min(k, len(full)), replace=False),
        }
        logged_set = set(logged.tolist())
        for (pool, method), top in lists_by_method.items():
            ids = data.item_ids[top]
            for item in ids:
                if item not in catalog.name_tokens:
                    catalog.name_tokens[item], catalog.categories[item] = frozenset(), None
            distances = catalog.distances(ids)
            pairs = np.triu_indices(len(top), 1)
            diversity = {}
            for name, matrix in distances.items():
                values = matrix[pairs]
                known = values[np.isfinite(values)]
                diversity[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
            rows.append(dict(query_id=qid, query_type="text" if category == TEXT_SEARCH else "category",
                             position=int(data.query_position[row]), pool=pool, method=method, k=k,
                             click_ndcg=_click_ndcg(top, clicked_items, k),
                             unshown_share=float(np.mean([item not in logged_set for item in top])),
                             **diversity))
    return pd.DataFrame(rows)


def summarize_top_lists(measured):
    groups = [measured.assign(segment="all"),
              measured.assign(segment=np.where(measured.position.eq(1), "first_query", "later_query"))]
    data = pd.concat(groups, ignore_index=True)
    return (data.groupby(["segment", "pool", "method", "k"], sort=False)
            [["click_ndcg", "token_ild", "category_ild", "combined_ild", "unshown_share"]]
            .mean().join(data.groupby(["segment", "pool", "method", "k"], sort=False).size().rename("queries"))
            .reset_index())


def summarize_retrieval(measured):
    keys = ["pool", "method", "k1"]
    groups = [measured.assign(segment="all"),
              measured.assign(segment="query_" + measured.query_type),
              measured.assign(segment=np.where(measured.position.eq(1), "first_query", "later_query"))]
    data = pd.concat(groups, ignore_index=True)
    return (data.groupby(["segment"] + keys, sort=False)
            .agg(recall=("recall", "mean"), shown_share=("shown_share", "mean"), queries=("query_id", "size"),
                 mean_pool_size=("pool_size", "mean"))
            .reset_index())


def plot_recall(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = summary[summary.segment.eq("all")]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, pool, title in ((axes[0], "logged", "Reorder the site's logged list"),
                              (axes[1], "full", "Filter the category / catalog")):
        for method, frame in data[data.pool.eq(pool)].groupby("method"):
            axis.plot(frame.k1, frame.recall, "-o", label=method, ms=4)
        axis.set_title(title)
        axis.set_xlabel("K1 (products kept)")
        axis.set_ylabel("recall of clicked products")
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "recall_at_k1.png", dpi=150)
    plt.close(figure)


# ------------------------------------------------------------------- run ---

def train_two_tower(run_folder, settings_path=Path("config/two_tower.json"), output_root=None):
    """Fit the towers on training clicks, choose the epoch on validation, save everything."""
    import torch
    settings = read_two_tower_settings(settings_path)
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    rng = np.random.default_rng(settings["seed"])
    run_folder = Path(run_folder)
    if not (run_folder / "development.sqlite").exists():
        raise ValueError(f"{run_folder} is not a prepared learning-curve run")
    print("[1/3] Read products, queries, session histories and logged lists", flush=True)
    data = RunData(run_folder, settings)
    eta = (estimate_eta(data.candidates["train"], settings["propensity_fit_max_rank"])
           if settings["propensity_eta"] == "estimate" else float(settings["propensity_eta"]))
    examples = training_examples(data)
    train_lists, validation_lists = logged_lists(data, "train"), logged_lists(data, "validation")
    if examples.empty:
        raise ValueError("No clicked training products to learn from")
    print(f"Rank effect eta = {eta:.3f}: click at rank 10 weighs "
          f"{click_weight(10, eta, settings['max_positive_weight']):.2f}, skip at rank 1 weighs 1.00, "
          f"skip at rank 10 weighs {other_weight(10, eta, 0):.2f}", flush=True)

    model = build_model(data, settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    selection_k = 100 if 100 in settings["k1_values"] else max(settings["k1_values"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{run_folder.name}"
    folder.mkdir(parents=True)
    write_json(folder / "two_tower_settings.json", dict(
        trained_run=str(run_folder), settings=settings, eta=eta, training_clicks=len(examples),
        popularity_window=data.window, popularity_window_clicks=data.window_clicks,
        selection="epoch with the highest validation full-pool recall at K1 = %d" % selection_k))

    print(f"[2/3] Train on {len(examples):,} clicks for {settings['epochs']} epochs", flush=True)
    log, best = [], (-1.0, None, 0)
    values = examples.to_numpy()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = rng.permutation(len(values))
        total, batches = 0.0, 0
        for start in range(0, len(order), settings["batch_size"]):
            batch = values[order[start:start + settings["batch_size"]]]
            negatives, w_other = sample_negatives(data, batch, train_lists, settings, eta, rng)
            w_click = click_weight(batch[:, 2], eta, settings["max_positive_weight"])
            loss = pair_loss(model, batch[:, 0], batch[:, 1], negatives, w_click, w_other)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + float(loss), batches + 1
        model.eval()
        measured = evaluate_retrieval(model, data, validation_lists, [selection_k])
        recall = measured[measured.pool.eq("full") & measured.method.eq("two_tower")].recall.mean()
        log.append(dict(epoch=epoch, train_loss=total / batches, validation_full_recall=recall, k1=selection_k))
        print(f"Epoch {epoch}: loss {total / batches:.4f}, validation recall@{selection_k} {recall:.4f}", flush=True)
        if recall > best[0]:
            best = (recall, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    model.eval()

    print(f"[3/3] Evaluate the best epoch ({best[2]}) on validation", flush=True)
    items = item_embeddings(model, data)
    measured = evaluate_retrieval(model, data, validation_lists, settings["k1_values"], items)
    summary = summarize_retrieval(measured)
    top_lists = evaluate_top_lists(model, data, validation_lists, items, seed=settings["seed"])
    save_table(top_lists, folder / "validation_top10.parquet")
    save_table(summarize_top_lists(top_lists), folder / "top10_summary.csv")
    torch.save(model.state_dict(), folder / "two_tower.pt")
    np.save(folder / "item_embeddings.npy", items)
    np.save(folder / "item_ids.npy", data.item_ids)
    save_table(pd.DataFrame(log), folder / "training_log.csv")
    save_table(measured, folder / "validation_retrieval.parquet")
    save_table(summary, folder / "retrieval_summary.csv")
    plot_recall(summary, folder)
    print(f"Two-tower filter saved to {folder}", flush=True)
    return folder
