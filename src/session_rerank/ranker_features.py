"""Turn the prepared anonymous tables into one feature row per shown product."""

from collections import Counter

import numpy as np
import pandas as pd

from .core import jaccard_distance, tokens

PRODUCT_FEATURES = [
    "price_log2", "category_id", "query_category_match", "query_text_missing",
    "query_token_overlap", "prior_product_clicks", "prior_product_views",
]
SESSION_FEATURES = [
    "session_clicks", "session_views", "session_history_missing",
    "session_item_clicks", "session_item_views",
    "session_item_age_ms", "session_category_match", "session_token_similarity",
]


def add_popularity(rows, train_events):
    """Count activity on earlier training days, then freeze counts for validation."""
    counts = {kind: Counter() for kind in ("click", "view")}
    train = rows[rows.split.eq("train")]
    validation = rows[rows.split.eq("validation")]
    for kind, counter in counts.items():
        column = f"prior_product_{kind}s"
        rows[column] = 0
        events_by_date = {
            date: events.item_id.dropna()
            for date, events in train_events[train_events.event_type.eq(kind)].groupby("eventdate")
        }
        # Score a whole day before adding its events so its clicks cannot leak.
        for date, day_rows in train.groupby("eventdate", sort=True):
            rows.loc[day_rows.index, column] = day_rows.item_id.map(counter).fillna(0)
            counter.update(events_by_date.get(date, ()))
        rows.loc[validation.index, column] = validation.item_id.map(counter).fillna(0)


def add_session_features(rows, histories, product_tokens, product_categories):
    """Summarize only the earlier clicks and views already allowed for each query."""
    clicks_and_views = histories[histories.event_type.isin(["click", "view"])].dropna(subset=["item_id"])
    groups = dict(tuple(clicks_and_views.groupby("query_id", sort=False)))
    features = pd.DataFrame(0.0, index=rows.index, columns=SESSION_FEATURES)
    features["session_item_age_ms"] = np.nan
    features["session_history_missing"] = 1.0
    empty = frozenset()
    for query_id, candidates in rows.groupby("query_id", sort=False):
        history = groups.get(query_id)
        if history is None:
            continue
        click_counts = Counter(history.loc[history.event_type.eq("click"), "item_id"])
        view_counts = Counter(history.loc[history.event_type.eq("view"), "item_id"])
        total_clicks = sum(click_counts.values())
        total_views = sum(view_counts.values())
        age_by_item = history.groupby("item_id").age_ms.min()
        history_categories = {
            product_categories[item] for item in age_by_item.index if item in product_categories
        }
        history_tokens = {
            product_tokens[item] for item in age_by_item.index if product_tokens.get(item)
        }

        features.loc[candidates.index, SESSION_FEATURES] = [
            (
                total_clicks, total_views, 0.0,
                click_counts[item], view_counts[item], age_by_item.get(item, np.nan),
                int(product_categories.get(item) in history_categories),
                max((1 - jaccard_distance(product_tokens.get(item, empty), past)
                     for past in history_tokens), default=0.0),
            )
            for item in candidates.item_id
        ]
    return rows.join(features)


def build_ranker_features(folder):
    """Load train and validation only. Keep click labels separate from model inputs."""
    visible = [("split", "in", ["train", "validation"])]
    candidates = pd.read_parquet(folder / "candidates.parquet", filters=visible)
    queries = pd.read_parquet(folder / "queries.parquet", filters=visible)
    products = pd.read_parquet(folder / "products.parquet")
    histories = pd.read_parquet(folder / "histories.parquet", filters=visible)
    train_events = pd.read_parquet(folder / "events.parquet", filters=[("split", "==", "train")])

    rows = candidates.merge(queries, on=["query_id", "split"], validate="many_to_one")
    rows = rows.merge(products.drop(columns="name_tokens"), on="item_id", how="left", validate="many_to_one")
    rows = rows.sort_values(["query_id", "original_rank"]).reset_index(drop=True)
    product_tokens = products.set_index("item_id").name_tokens.map(tokens).to_dict()
    product_categories = products.dropna(subset=["category_id"]).set_index("item_id").category_id.to_dict()
    query_tokens = queries.set_index("query_id").query_tokens.map(tokens).to_dict()

    rows["query_category_match"] = rows.category_id.eq(rows.query_category_id).fillna(False).astype(int)
    rows["query_text_missing"] = rows.query_id.map(lambda query: int(not query_tokens[query]))
    rows["query_token_overlap"] = [
        1 - jaccard_distance(query_tokens[query], product_tokens[item])
        if query_tokens[query] and product_tokens.get(item) else np.nan
        for query, item in zip(rows.query_id, rows.item_id)
    ]
    # Category IDs identify groups rather than quantities. LightGBM reads this as categorical.
    rows["category_id"] = rows.category_id.astype("category")
    add_popularity(rows, train_events)
    return add_session_features(rows, histories, product_tokens, product_categories)
