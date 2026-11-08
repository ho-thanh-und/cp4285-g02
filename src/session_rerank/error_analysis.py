"""Error analysis of the frozen models on the selection data (no new models).

For every page it compares the frozen method with a reference on identical
cases and breaks the outcome down by properties of the case, so the report can
name where each model fails and why.

Category pages: frozen LightGBM ranker vs the popularity sort, per query
(win / tie / loss on click NDCG@10), by session history, how deep the clicked
product sat in the site's list, the clicked product's April popularity
(including products with no April clicks), clicks per query, position in the
session and category; plus the worst losses.

Item page and home page: the evaluation cases are rebuilt exactly (their
sampling is deterministic and checked against the saved results) and each case
gets: whether any later-clicked product was reachable in the frozen 20-product
pool (a ceiling for any reranking), whether later clicks stayed in the same
category, the number of co-click neighbours, and product popularity.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .home_page import click_sequences, evaluation_cases, knn_scores, read_home_settings, top_excluding
from .item_knn import cosine_similarity, read_knn_settings, session_baskets
from .ranker_phase2 import window_activity
from .study_files import save_table, write_json
from .ymal import item_pages, read_ymal_settings

POPULARITY_BINS = [-1, 0, 9, 99, np.inf]
POPULARITY_LABELS = ["0 (no April clicks)", "1-9", "10-99", "100+"]


def outcome_table(frame, column, model="model", reference="reference", metric="ndcg"):
    """Per bucket: cases, both methods' means, mean difference and win/tie/loss shares."""
    diff = frame[f"{metric}_{model}"] - frame[f"{metric}_{reference}"]
    data = frame.assign(_diff=diff, _win=diff > 1e-12, _loss=diff < -1e-12)
    table = data.groupby(column, observed=True, sort=True).agg(
        cases=("_diff", "size"), **{f"{metric}_{model}": (f"{metric}_{model}", "mean"),
                                    f"{metric}_{reference}": (f"{metric}_{reference}", "mean")},
        difference=("_diff", "mean"), win_share=("_win", "mean"), loss_share=("_loss", "mean"))
    table["tie_share"] = 1 - table.win_share - table.loss_share
    table["case_share"] = table.cases / table.cases.sum()
    return table.reset_index().rename(columns={column: "bucket"}).assign(dimension=column)


def popularity_bucket(values):
    return pd.cut(values, POPULARITY_BINS, labels=POPULARITY_LABELS)


# ----------------------------------------------------------- category pages ---

def category_pages(settings, popularity, item_index):
    run = Path(settings["category_run"])
    per_query = pd.read_parquet(run / "selection_per_query.parquet")
    candidates = pd.read_parquet(run / "selection_candidates.parquet")
    with sqlite3.connect(run / "selection.sqlite") as db:
        queries = pd.read_sql("SELECT query_id, session_key, timeframe, category FROM queries", db)
    queries["position"] = queries.sort_values(["session_key", "timeframe", "query_id"]).groupby("session_key").cumcount() + 1
    model = per_query[per_query.method.eq(settings["category_model"])]
    reference = per_query[per_query.method.eq("popularity_sort")]
    frame = (model[["query_id", "click_ndcg", "query_type", "has_history"]]
             .merge(reference[["query_id", "click_ndcg"]], on="query_id", suffixes=("_model", "_reference"))
             .rename(columns={"click_ndcg_model": "ndcg_model", "click_ndcg_reference": "ndcg_reference"}))
    frame = frame[frame.query_type.eq("category") & frame.ndcg_model.notna()]
    rows = item_index.reindex(candidates.item_id.to_numpy()).to_numpy()
    candidates["april_clicks"] = np.where(np.isnan(rows), 0, popularity[np.nan_to_num(rows).astype(int)])
    candidates["popularity_rank"] = candidates.groupby("query_id").april_clicks.rank(ascending=False, method="first")
    clicked = candidates[candidates.clicked]
    per_click = clicked.groupby("query_id").agg(
        first_clicked_rank=("original_rank", "min"), clicks=("item_id", "size"),
        most_popular_click=("april_clicks", "max"), any_unseen_click=("april_clicks", lambda s: bool((s == 0).any())),
        best_popularity_rank=("popularity_rank", "min"))
    frame = frame.merge(per_click.reset_index(), on="query_id").merge(queries[["query_id", "position", "category"]],
                                                                     on="query_id")
    frame["history"] = np.where(frame.has_history, "with history", "no history")
    frame["clicked_depth"] = pd.cut(frame.first_clicked_rank, [0, 10, 50, 100, np.inf],
                                    labels=["rank 1-10", "11-50", "51-100", "101+"])
    frame["clicked_popularity"] = popularity_bucket(frame.most_popular_click)
    frame["clicks_in_query"] = pd.cut(frame.clicks, [0, 1, 2, np.inf], labels=["1", "2", "3+"])
    frame["position_in_session"] = pd.cut(frame.position, [0, 1, 2, np.inf], labels=["1st", "2nd", "3rd+"])
    frame["unseen_clicked_product"] = np.where(frame.any_unseen_click, "a clicked product had no April clicks",
                                               "all clicked products had April clicks")
    tables = pd.concat([outcome_table(frame, column) for column in
                        ("history", "clicked_depth", "clicked_popularity", "clicks_in_query",
                         "position_in_session", "unseen_clicked_product")], ignore_index=True)
    by_category = outcome_table(frame, "category")
    by_category = by_category[by_category.cases >= 15].sort_values("difference")
    examples = (frame.assign(diff=frame.ndcg_model - frame.ndcg_reference).nsmallest(settings["examples"], "diff")
                [["query_id", "history", "ndcg_model", "ndcg_reference", "first_clicked_rank", "clicks",
                  "most_popular_click", "best_popularity_rank", "category"]])
    overall = dict(queries=len(frame), win=float((frame.ndcg_model > frame.ndcg_reference + 1e-12).mean()),
                   loss=float((frame.ndcg_model < frame.ndcg_reference - 1e-12).mean()),
                   both_zero=float(((frame.ndcg_model == 0) & (frame.ndcg_reference == 0)).mean()),
                   unseen_click_share=float(frame.any_unseen_click.mean()))
    return tables, by_category, examples, overall


# ------------------------------------------------- item page and home page ---

def similarity_and_popularity(settings, item_index, products):
    knn = read_knn_settings(settings["knn_config"])
    session, item, _ = session_baskets(settings["raw_directory"], "2016-04-01", "2016-04-30", item_index,
                                       knn["max_items_per_session"])
    similarity, _ = cosine_similarity(session, item, products, knn["shrinkage"])
    return similarity.tocsr()


def item_page(settings, similarity, popularity, item_index, categories):
    config = read_ymal_settings(settings["ymal_config"])
    window = config["window"]
    pages, _ = item_pages(config["raw_directory"], window["start"], window["end_exclusive"], item_index,
                          config["max_seeds"], config["seed"])
    per_page = pd.read_parquet(Path(settings["ymal_run"]) / "per_page.parquet")
    frozen = per_page[per_page.method.eq(config["frozen_policy"])].sort_values("page").reset_index(drop=True)
    reference = per_page[per_page.method.eq("category_popularity")].sort_values("page").reset_index(drop=True)
    catalog_ids = item_index.index.to_numpy()
    if len(frozen) != len(pages) or not np.array_equal(frozen.item.to_numpy(),
                                                       catalog_ids[[p[1] for p in pages]]):
        raise ValueError("Rebuilt item pages do not match the saved selection results")
    pool_size = int(config["frozen_policy"].rsplit("_", 1)[1])
    rows = []
    for (_, seed, later), neighbours in zip(pages, frozen.neighbours):
        row = similarity[seed]
        top = row.indices[np.argsort(-row.data, kind="stable")[:pool_size]]
        same = np.mean([categories[t] == categories[seed] for t in later])
        rows.append(dict(neighbours=int(neighbours), seed_april_clicks=popularity[seed],
                         later_same_category_share=same,
                         reachable=bool(set(later) & set(row.indices.tolist())),
                         in_pool=bool(set(later) & set(top.tolist()))))
    frame = pd.DataFrame(rows)
    frame["ndcg_model"], frame["ndcg_reference"] = frozen.ndcg, reference.ndcg
    frame["hit_model"], frame["hit_reference"] = frozen.hit, reference.hit
    frame["co_click_neighbours"] = pd.cut(frame.neighbours, [-1, 0, 4, 19, np.inf],
                                          labels=["0", "1-4", "5-19", "20+"])
    frame["item_popularity"] = popularity_bucket(frame.seed_april_clicks)
    frame["later_clicks_category"] = pd.cut(frame.later_same_category_share, [-0.01, 0, 0.999, 1],
                                            labels=["all other categories", "mixed", "all same category"])
    frame["later_click_in_pool"] = np.where(frame.in_pool, "a later click is in the 20-product pool",
                                            "no later click in the pool")
    tables = pd.concat([outcome_table(frame, column) for column in
                        ("co_click_neighbours", "item_popularity", "later_clicks_category", "later_click_in_pool")]
                       + [outcome_table(frame, column, metric="hit") for column in
                          ("co_click_neighbours", "later_click_in_pool")], ignore_index=True)
    overall = dict(pages=len(frame), in_pool=float(frame.in_pool.mean()), reachable=float(frame.reachable.mean()),
                   no_neighbours=float((frame.neighbours == 0).mean()),
                   later_all_same_category=float((frame.later_same_category_share == 1).mean()),
                   hit_model=float(frame.hit_model.mean()), hit_reference=float(frame.hit_reference.mean()))
    return tables, overall


def home_page(settings, similarity, popularity, item_index, categories):
    config = read_home_settings(settings["home_config"])
    window = config["window"]
    week = click_sequences(config["raw_directory"], window["start"], window["end_exclusive"], item_index)
    cases, _ = evaluation_cases(week, config["max_cases"], config["seed"])
    per_case = pd.read_parquet(Path(settings["home_run"]) / "per_case.parquet")
    frozen = per_case[per_case.method.eq(config["frozen_policy"])].sort_values("case").reset_index(drop=True)
    neural = per_case[per_case.method.eq("two_tower")].sort_values("case").reset_index(drop=True)
    if len(frozen) != len(cases) or not np.array_equal(frozen.clicked_so_far.to_numpy(),
                                                       [len(c[1]) for c in cases]):
        raise ValueError("Rebuilt home-page cases do not match the saved selection results")
    lookup = SimpleNamespace(similarity=similarity, popularity=popularity)
    rows = []
    for _, prefix, later in cases:
        pool = top_excluding(knn_scores(lookup, prefix), set(prefix), config["pool_size"], scores=True)
        last = prefix[-1]
        rows.append(dict(last_april_clicks=popularity[last],
                         later_same_category_share=np.mean([categories[t] == categories[last] for t in later]),
                         in_pool=bool(set(later) & set(pool.tolist())),
                         has_neighbours=bool(similarity[prefix].nnz)))
    frame = pd.DataFrame(rows)
    frame["clicked_so_far"] = frozen.clicked_so_far
    frame["ndcg_model"], frame["ndcg_reference"] = frozen.ndcg, neural.ndcg
    frame["hit_model"], frame["hit_reference"] = frozen.hit, neural.hit
    frame["clicks_so_far"] = pd.cut(frame.clicked_so_far, [0, 1, 2, 5, np.inf], labels=["1", "2", "3-5", "6+"])
    frame["last_click_popularity"] = popularity_bucket(frame.last_april_clicks)
    frame["later_clicks_category"] = pd.cut(frame.later_same_category_share, [-0.01, 0, 0.999, 1],
                                            labels=["all other categories", "mixed", "all same category"])
    frame["later_click_in_pool"] = np.where(frame.in_pool, "a later click is in the 20-product pool",
                                            "no later click in the pool")
    tables = pd.concat([outcome_table(frame, column) for column in
                        ("clicks_so_far", "last_click_popularity", "later_clicks_category", "later_click_in_pool")],
                       ignore_index=True)
    overall = dict(cases=len(frame), in_pool=float(frame.in_pool.mean()),
                   no_neighbours=float((~frame.has_neighbours).mean()),
                   later_all_same_category=float((frame.later_same_category_share == 1).mean()),
                   ndcg_model=float(frame.ndcg_model.mean()), ndcg_two_tower=float(frame.ndcg_reference.mean()))
    return tables, overall


def run_error_analysis(settings_path=Path("config/error_analysis.json"), output_root=None):
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    products = pd.read_parquet(Path(settings["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    categories = products.category_id.fillna(-1).to_numpy(np.int64)
    print("[errors] April popularity and co-click similarity", flush=True)
    clicks, _ = window_activity(settings["raw_directory"], item_index, pd.Timestamp("2016-04-01"), 29)
    popularity = clicks.sum(0).astype(float)
    similarity = similarity_and_popularity(settings, item_index, len(products))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / stamp
    folder.mkdir(parents=True)
    print("[errors] Category pages", flush=True)
    category_tables, by_category, examples, category_overall = category_pages(settings, popularity, item_index)
    print("[errors] Item page", flush=True)
    item_tables, item_overall = item_page(settings, similarity, popularity, item_index, categories)
    print("[errors] Home page", flush=True)
    home_tables, home_overall = home_page(settings, similarity, popularity, item_index, categories)
    save_table(category_tables, folder / "category_pages_breakdown.csv")
    save_table(by_category, folder / "category_pages_by_category.csv")
    save_table(examples, folder / "category_pages_worst_losses.csv")
    save_table(item_tables, folder / "item_page_breakdown.csv")
    save_table(home_tables, folder / "home_page_breakdown.csv")
    write_json(folder / "overview.json", dict(settings=settings, category_pages=category_overall,
                                              item_page=item_overall, home_page=home_overall))
    print(f"[errors] Saved to {folder}", flush=True)
    return folder
