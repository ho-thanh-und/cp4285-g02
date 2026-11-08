"""Exposure and popularity bias of the recommendation lists (ethics section).

Which products get shown, and how evenly? For each page, the selection-week
lists of every evaluated method are rebuilt with the same code that scored
them. Each rebuilt list's NDCG@10 must equal the saved value, so the lists
measured here are exactly the lists that were evaluated. Every slot of a list
counts once (no position weighting).

Measures per page and method:
  * coverage: distinct products shown / products that could be shown (the
    whole catalog on the home and item pages; the products the site listed
    on category pages, which only reorder that list)
  * Gini coefficient of exposure over those products (0 = every product shown
    equally often, 1 = all exposure on one product)
  * share of slots on the popular head (the most-clicked products that together
    received `head_click_share` of April clicks), on the rest of the products
    with April clicks (tail), and on products with no April clicks
  * median April clicks of the products shown
Reference row "shopper_clicks": the products shoppers actually clicked later
(the hits). head_share_ratio = a method's head share / the shoppers' head
share; above 1 means the method concentrates on popular products more than
shoppers' own choices do.
"""

import gc
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .ranker_phase2 import window_activity
from .study_files import save_table, write_json

HEAD, TAIL, NO_CLICKS = 0, 1, 2
PAGES = ("home", "item", "category")


def read_exposure_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not settings["pages"] or set(settings["pages"]) - set(PAGES):
        raise ValueError(f"pages must be a non-empty subset of {PAGES}")
    if not 0 < settings["head_click_share"] < 1:
        raise ValueError("head_click_share must be between 0 and 1")
    if settings["popularity_window"]["end_exclusive"] > "2016-05-01":
        raise ValueError("popularity_window must end before the evaluation weeks")
    return settings


def gini(counts):
    """Gini coefficient of non-negative counts (0 = equal, towards 1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float(2 * (np.arange(1, n + 1) @ x) / (n * total) - (n + 1) / n)


def popularity_groups(popularity, head_click_share):
    """HEAD for the most-clicked products holding head_click_share of all clicks, TAIL for
    the other clicked products, NO_CLICKS for the rest."""
    order = np.argsort(-popularity, kind="stable")
    cumulative = np.cumsum(popularity[order]) / popularity.sum()
    head = int(np.searchsorted(cumulative, head_click_share)) + 1
    groups = np.where(popularity > 0, TAIL, NO_CLICKS)
    groups[order[:head]] = HEAD
    return groups


def exposure_row(shown, eligible, popularity, groups):
    """shown: product rows of every slot; rows 0..eligible-1 are the products that could be shown."""
    shown = np.asarray(shown, dtype=int)
    counts = np.bincount(shown, minlength=eligible)
    group = groups[shown]
    return dict(slots=len(shown), distinct_products=int((counts > 0).sum()), eligible_products=int(eligible),
                coverage=float((counts > 0).mean()), gini=gini(counts),
                head_share=float((group == HEAD).mean()), tail_share=float((group == TAIL).mean()),
                no_click_share=float((group == NO_CLICKS).mean()),
                median_april_clicks=float(np.median(popularity[shown])))


def page_summary(page, lists, clicked, eligible, popularity, groups, frozen):
    """One row per method plus the shoppers' own later clicks as the reference."""
    rows = [dict(page=page, method=method, **exposure_row(np.concatenate(tops), eligible, popularity, groups))
            for method, tops in lists.items()]
    rows.append(dict(page=page, method="shopper_clicks",
                     **exposure_row(np.concatenate(clicked), eligible, popularity, groups)))
    frame = pd.DataFrame(rows)
    frame["head_share_ratio"] = frame.head_share / frame.head_share.iloc[-1]
    frame["frozen"] = frame.method.eq(frozen)
    return frame


def list_ndcg(top, targets):
    """NDCG of one list against a set of later clicks, as in ymal.list_metrics."""
    hits = np.array([t in targets for t in top], dtype=float)
    discounts = 1 / np.log2(np.arange(2, len(top) + 2))
    return float(hits @ discounts / discounts[:min(len(top), len(targets))].sum())


def check_reproduced(lists, targets, saved, order_column, page):
    """Every rebuilt list must give the NDCG saved by the evaluation run."""
    for method, tops in lists.items():
        expected = saved[saved.method.eq(method)].sort_values(order_column).ndcg.to_numpy()
        measured = np.array([list_ndcg(top, t) for top, t in zip(tops, targets)])
        if len(expected) != len(measured) or not np.allclose(measured, expected):
            raise ValueError(f"Rebuilt {page} lists of {method} do not reproduce the saved NDCG")


# ------------------------------------------------------------------ pages ---

def home_lists(settings):
    from .home_page import (HomeData, case_lists, click_sequences, evaluation_cases, item_vectors,
                            load_frozen_model, read_home_settings)
    from .item_knn import read_knn_settings
    home = read_home_settings(settings["home_config"])
    data = HomeData(home, read_knn_settings(home["knn_config"]))
    model = load_frozen_model(data, home)
    window = home["window"]
    week = click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], data.item_index)
    cases, _ = evaluation_cases(week, home["max_cases"], home["seed"])
    vectors = item_vectors(model, data)
    lists, targets = defaultdict(list), []
    for number, (_, prefix, later) in enumerate(cases, start=1):
        for method, top in case_lists(data, model, vectors, home, prefix).items():
            lists[method].append(np.asarray(top, dtype=int))
        targets.append(later)
        if number % 5000 == 0:
            print(f"[exposure] home page {number:,}/{len(cases):,}", flush=True)
    check_reproduced(lists, targets, pd.read_parquet(Path(settings["home_run"]) / "per_case.parquet"), "case", "home")
    return dict(lists), [np.fromiter(t, dtype=int) for t in targets], home["frozen_policy"], home["trained_run"]


def item_lists(settings):
    from .item_knn import read_knn_settings
    from .ymal import YmalData, item_pages, page_lists, read_ymal_settings
    ymal = read_ymal_settings(settings["ymal_config"])
    data = YmalData(ymal, read_knn_settings(ymal["knn_config"]))
    window = ymal["window"]
    pages, _ = item_pages(ymal["raw_directory"], window["start"], window["end_exclusive"], data.item_index,
                          ymal["max_seeds"], ymal["seed"])
    lists, targets = defaultdict(list), []
    for _, seed_item, later in pages:
        for method, (top, _) in page_lists(data, ymal, seed_item).items():
            lists[method].append(np.asarray(top, dtype=int))
        targets.append(set(later))
    check_reproduced(lists, targets, pd.read_parquet(Path(settings["ymal_run"]) / "per_page.parquet"), "page", "item")
    return dict(lists), [np.fromiter(t, dtype=int) for t in targets], ymal["frozen_policy"], ymal["trained_run"]


def category_lists(settings):
    """Top-k item IDs per category-page query for every method of the held-out run."""
    import lightgbm as lgb
    from .event_ablation import MODELS, per_query_metrics
    from .heldout import POPULARITY_COLUMNS, frozen_models, read_heldout_settings, split_features
    from .item_knn import KnnData, read_knn_settings
    from .ranker_phase2 import Phase2Data, read_phase2_settings
    from .ranking_settings import top_k
    heldout = read_heldout_settings(settings["heldout_config"])
    phase2_settings = read_phase2_settings(heldout["phase2_config"])
    variant, models = frozen_models(heldout["phase2_run"], phase2_settings)
    run = Path(settings["heldout_run"])
    frozen = "lightgbm_" + json.loads((run / "selection_choice.json").read_text())["chosen"]
    data = Phase2Data(heldout["trained_run"], phase2_settings)
    data.matrices.clear()
    gc.collect()
    matrix, candidates, query_table, click_table = split_features(data, run / f"{heldout['split']}.sqlite")
    saved = pd.read_parquet(run / f"{heldout['split']}_candidates.parquet")
    if not (np.array_equal(saved.item_id, candidates.item_id) and np.array_equal(saved.query_id, candidates.query_id)):
        raise ValueError("Rebuilt category-page candidates differ from the saved held-out run")
    knn = KnnData(heldout["trained_run"], read_knn_settings(heldout["knn_config"]))
    knn.use_queries(query_table, click_table)
    methods = {"site_order": -candidates.original_rank.to_numpy(float),
               "popularity_sort": matrix[:, POPULARITY_COLUMNS[0]].astype(float),
               "item_knn": knn.scores(candidates)[0]}
    for model in models:
        indices = list(MODELS[model][1]) + (POPULARITY_COLUMNS if variant["popularity"] else [])
        booster = lgb.Booster(model_file=str(run / f"{model}.txt"))
        methods[f"lightgbm_{model}"] = booster.predict(matrix[:, indices])
    config = data.inputs.config
    k = top_k(config)
    per_query = pd.read_parquet(run / f"{heldout['split']}_per_query.parquet")
    category = set(query_table.index[query_table.category.fillna(0).ne(0)])
    lists = {}
    for method, score in methods.items():
        measured = per_query_metrics(candidates, score, config["label_gains"], k).set_index("query_id").click_ndcg
        expected = per_query[per_query.method.eq(method)].set_index("query_id").click_ndcg
        if not np.allclose(measured.reindex(expected.index), expected, equal_nan=True):
            raise ValueError(f"Rebuilt category-page scores of {method} do not reproduce the saved NDCG")
        frame = candidates.assign(score=score)
        lists[method] = [group.item_id.to_numpy()[np.argsort(-group.score.to_numpy(), kind="stable")[:k]]
                         for qid, group in frame.groupby("query_id", sort=False) if qid in category]
    clicked = [group.item_id.to_numpy() for qid, group in candidates[candidates.clicked].groupby("query_id", sort=False)
               if qid in category]
    return lists, clicked, frozen, candidates[candidates.query_id.isin(category)].item_id.unique(), heldout["trained_run"]


# -------------------------------------------------------------------- run ---

def run_exposure(settings_path=Path("config/exposure.json"), output_root=None):
    settings = read_exposure_settings(settings_path)
    products = pd.read_parquet(Path(settings["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    window = settings["popularity_window"]
    start = pd.Timestamp(window["start"])
    print("[exposure] April clicks per product", flush=True)
    clicks, _ = window_activity(settings["raw_directory"], item_index, start,
                                int((pd.Timestamp(window["end_exclusive"]) - start).days))
    popularity = clicks.sum(0).astype(float)
    groups = popularity_groups(popularity, settings["head_click_share"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / stamp
    folder.mkdir(parents=True)
    frames = []
    for page in settings["pages"]:
        print(f"[exposure] Rebuild the {page} page lists", flush=True)
        if page == "category":
            lists, clicked, frozen, listed, trained_run = category_lists(settings)
            # Category pages only reorder the site's list: index the products it listed.
            rows = item_index.reindex(listed).to_numpy()
            known = ~np.isnan(rows)
            local_popularity = np.where(known, popularity[np.nan_to_num(rows).astype(int)], 0.0)
            local_groups = np.where(known, groups[np.nan_to_num(rows).astype(int)], NO_CLICKS)
            local = pd.Series(np.arange(len(listed)), index=listed)
            lists = {m: [local.loc[t].to_numpy() for t in tops] for m, tops in lists.items()}
            clicked = [local.loc[t].to_numpy() for t in clicked]
            frames.append(page_summary(page, lists, clicked, len(listed), local_popularity, local_groups, frozen))
        else:
            lists, clicked, frozen, trained_run = home_lists(settings) if page == "home" else item_lists(settings)
            frames.append(page_summary(page, lists, clicked, len(products), popularity, groups, frozen))
        if trained_run != settings["trained_run"]:
            raise ValueError(f"The {page} run used another catalog ({trained_run})")
        print(f"[exposure] {page} page lists reproduce the saved NDCG", flush=True)
    summary = pd.concat(frames, ignore_index=True)
    save_table(summary, folder / "summary.csv")
    write_json(folder / "exposure.json", dict(
        settings=settings, catalog_products=len(products), april_clicks=float(popularity.sum()),
        head_products=int((groups == HEAD).sum()), tail_products=int((groups == TAIL).sum()),
        no_click_products=int((groups == NO_CLICKS).sum())))
    print(f"[exposure] Saved to {folder}", flush=True)
    return folder
