"""Item page "You may also like": item-kNN recommendations with 6 + 4 diversity.

Product rules (EXPERIMENT_LOG.md §7, §8.1): clicks only, and products that were
not clicked are not penalised.

Item pages. Every product a shopper clicked in an eligible session (no hidden-
label query) of one week opens an item page. That page is evaluable when the
shopper clicked at least one *different* product later in the same session;
those later clicks are the hits. Pages without a later click are skipped rather
than scored as failures. All eligible sessions of the week are used, not only
the sampled queries; the similarity comes from April only, so nothing leaks.
Only the validation week (May 1–7) and the selection week (May 9–14) may be
used; confirmation stays sealed.

Candidate pool for an item. Its top co-click neighbours (item-kNN cosine from
April, §6c), up to K1; if fewer exist, the pool is filled with the most popular
products of the item's category, then of the catalog. Relevance = similarity
(fill products rank below every neighbour, in popularity order).

Methods on identical item pages:
  * category_popularity  most popular products of the item's category (cold baseline)
  * relevance_K1         top 10 of the pool by relevance
  * fixed_<alpha>_K1     greedy relevance/variety balance with a fixed alpha
  * six_four_K1          six most relevant, then four mutually diverse from the rest of the pool

Metrics: Recall@10, hit rate, MRR@10, NDCG@10 (all against later clicks);
ILD@10 (token, category, combined), distinct categories, share of products
outside the item's category, and tail-to-head distance (mean distance from
positions 7–10 to positions 1–6), which shows whether the diverse tail differs
from the relevant head rather than only within itself.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_policies import diverse_tail_order
from .core import greedy_rerank
from .diversity import ProductDiversity
from .event_ablation import session_bootstrap
from .item_knn import cosine_similarity, read_knn_settings, session_baskets
from .study_files import save_table, write_json

ALLOWED_WINDOWS = {"validation": ("2016-05-01", "2016-05-08"), "selection": ("2016-05-09", "2016-05-15")}
RELEVANCE = ["recall", "hit", "mrr", "ndcg"]
DIVERSITY = ["combined_ild", "token_ild", "category_ild", "distinct_categories", "outside_category_share",
             "tail_to_head_distance"]


def read_ymal_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    window = settings["window"]
    if ALLOWED_WINDOWS.get(window["name"]) != (window["start"], window["end_exclusive"]):
        raise ValueError(f"window must be one of {ALLOWED_WINDOWS}; confirmation stays sealed")
    if not 0 < settings["relevance_slots"] < settings["top_k"]:
        raise ValueError("relevance_slots must be between 1 and top_k - 1")
    if any(k1 < settings["top_k"] for k1 in settings["pool_sizes"]) or not settings["pool_sizes"]:
        raise ValueError("pool_sizes must be at least top_k")
    if any(not 0 <= a <= 1 for a in settings["fixed_alphas"]):
        raise ValueError("fixed_alphas must be in [0, 1]")
    if window["name"] == "selection":
        # The held-out week may only be opened with a policy frozen on validation.
        frozen = settings.get("frozen_policy")
        names = {"category_popularity"} | {f"{kind}_{k1}" for k1 in settings["pool_sizes"]
                                           for kind in ["relevance", "six_four"]
                                           + [f"fixed_{a:g}" for a in settings["fixed_alphas"]]}
        if frozen not in names:
            raise ValueError(f"The selection week needs frozen_policy, one of {sorted(names)}")
    return settings


def item_pages(raw_directory, start, end_exclusive, item_index, max_seeds, seed):
    """(session, item row, later-clicked item rows) for evaluable item pages of a week."""
    raw = Path(raw_directory)
    queries = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "sessionId", "eventdate", "is.test"])
    queries = queries[(queries.eventdate >= start) & (queries.eventdate < end_exclusive)]
    hidden = queries.groupby(["sessionId", "eventdate"])["is.test"].transform("any")
    queries = queries[~hidden]
    clicks = pd.read_csv(raw / "train-clicks.csv", sep=";").merge(queries, on="queryId")
    clicks = clicks[clicks.itemId.isin(item_index.index)]
    first = (clicks.groupby(["sessionId", "eventdate", "itemId"], sort=False).timeframe.min().reset_index()
             .sort_values(["sessionId", "eventdate", "timeframe"]))
    pages = []
    for (session, date), group in first.groupby(["sessionId", "eventdate"], sort=False):
        items = group.itemId.to_numpy()
        times = group.timeframe.to_numpy()
        for position, item in enumerate(items):
            later = set(items[times > times[position]]) - {item}
            if later:
                pages.append((f"{session}|{date}", int(item_index[item]),
                              [int(item_index[i]) for i in sorted(later)]))
    total = len(pages)
    if total > max_seeds:
        keep = np.sort(np.random.default_rng(seed).choice(total, max_seeds, replace=False))
        pages = [pages[i] for i in keep]
    return pages, dict(eligible_sessions=int(first.groupby(["sessionId", "eventdate"]).ngroups),
                       evaluable_pages=total, evaluated_pages=len(pages), window_clicks=int(len(clicks)))


def candidate_pool(similarity, item, pool_size, fill_order):
    """Top co-click neighbours, filled with popularity; returns rows and relevance scores."""
    row = similarity[item]
    order = np.argsort(-row.data, kind="stable")[:pool_size]
    members = row.indices[order].tolist()
    scores = row.data[order].astype(float).tolist()
    seen = set(members) | {item}
    floor = (min(scores) if scores else 1.0) * 0.5
    for candidate in fill_order:
        if len(members) >= pool_size:
            break
        if candidate not in seen:
            members.append(int(candidate))
            seen.add(candidate)
            floor *= 0.999
            scores.append(floor)
    return np.asarray(members, dtype=int), np.asarray(scores), len(order)


def list_metrics(top, targets, item, catalog_ids, categories, diversity, relevance_slots):
    """Relevance against later clicks and diversity of one recommendation list."""
    k = len(top)
    hits = np.array([t in targets for t in top])
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(k, len(targets))].sum()
    first = np.flatnonzero(hits)
    ids = catalog_ids[top].tolist()
    for product in ids:
        if product not in diversity.name_tokens:
            diversity.name_tokens[product], diversity.categories[product] = frozenset(), None
    matrices = diversity.distances(ids)
    pairs = np.triu_indices(k, 1)
    result = dict(recall=hits.sum() / len(targets), hit=float(hits.any()),
                  mrr=float(1 / (first[0] + 1)) if len(first) else 0.0, ndcg=float(hits @ discounts / ideal),
                  distinct_categories=len(set(categories[top])),
                  outside_category_share=float(np.mean(categories[top] != categories[item])))
    for name, matrix in matrices.items():
        values = matrix[pairs]
        known = values[np.isfinite(values)]
        result[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
    cross = matrices["combined"][:relevance_slots, relevance_slots:]
    known = cross[np.isfinite(cross)]
    result["tail_to_head_distance"] = float(known.mean()) if len(known) else np.nan
    return result


class YmalData:
    """Catalog, April co-click similarity and popularity orders shared by every item page."""

    def __init__(self, settings, knn_settings):
        products = pd.read_parquet(Path(settings["trained_run"]) / "products.parquet")
        self.item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
        self.catalog_ids = products.item_id.to_numpy()
        self.categories = products.category_id.fillna(-1).to_numpy(np.int64)
        self.diversity = ProductDiversity(products)
        session, item, clicks = session_baskets(settings["raw_directory"], "2016-04-01", "2016-04-30",
                                                self.item_index, knn_settings["max_items_per_session"])
        similarity, _ = cosine_similarity(session, item, len(products), knn_settings["shrinkage"])
        self.similarity = similarity.tocsr()
        popularity = np.bincount(self.item_index.loc[clicks.itemId].to_numpy(),
                                 minlength=len(products)).astype(float)
        self.global_order = np.argsort(-popularity, kind="stable")
        self.by_category = {c: self.global_order[self.categories[self.global_order] == c]
                            for c in np.unique(self.categories)}


def page_lists(data, settings, seed_item):
    """Every method's top-k product rows for one item page, with its co-click neighbour count."""
    k, slots = settings["top_k"], settings["relevance_slots"]
    by_category, global_order, diversity = data.by_category, data.global_order, data.diversity
    category = data.categories[seed_item]
    fill = np.concatenate([by_category[category], global_order[:200]])
    category_top = np.array([p for p in by_category[category] if p != seed_item][:k]
                            or [p for p in global_order[:k + 1] if p != seed_item][:k])
    lists = {"category_popularity": (category_top, 0)}
    for pool_size in settings["pool_sizes"]:
        members, scores, neighbours = candidate_pool(data.similarity, seed_item, pool_size, fill)
        ids = data.catalog_ids[members].tolist()
        for product in ids:
            if product not in diversity.name_tokens:
                diversity.name_tokens[product], diversity.categories[product] = frozenset(), None
        reward = np.nan_to_num(diversity.distances(ids)["combined"], nan=0.0)
        orders = {f"relevance_{pool_size}": np.argsort(-scores, kind="stable")[:k],
                  f"six_four_{pool_size}": diverse_tail_order(scores, reward, slots, k)}
        for alpha in settings["fixed_alphas"]:
            orders[f"fixed_{alpha:g}_{pool_size}"] = greedy_rerank(scores, reward, alpha, k)
        for method, order in orders.items():
            lists[method] = (members[np.asarray(order)], neighbours)
    return lists


def run_ymal(settings_path=Path("config/ymal.json"), output_root=None):
    settings = read_ymal_settings(settings_path)
    knn_settings = read_knn_settings(settings["knn_config"])
    window = settings["window"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{window['name']}"
    folder.mkdir(parents=True)
    if window["name"] == "selection":
        # Freeze before any data of the held-out week is read.
        from .heldout import git_state
        write_json(folder / "frozen_choices.json", dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), code=git_state(Path.cwd()),
            page="item page, You may also like", policy=settings["frozen_policy"],
            reason=settings.get("frozen_reason"), similarity=knn_settings | dict(compare_with=[]),
            relevance_slots=settings["relevance_slots"], top_k=settings["top_k"],
            metrics=["Recall@10", "hit rate", "MRR@10", "NDCG@10", "ILD@10", "tail-to-head distance"],
            heldout_week=window))
        print(f"[ymal] Frozen policy {settings['frozen_policy']} written before opening the week", flush=True)
    print("[ymal] Co-click similarity from April raw clicks", flush=True)
    data = YmalData(settings, knn_settings)
    item_index, catalog_ids, categories = data.item_index, data.catalog_ids, data.categories
    diversity, similarity = data.diversity, data.similarity
    print(f"[ymal] Item pages of the {window['name']} week {window['start']} to before {window['end_exclusive']}",
          flush=True)
    pages, page_stats = item_pages(settings["raw_directory"], window["start"], window["end_exclusive"], item_index,
                                   settings["max_seeds"], settings["seed"])
    slots = settings["relevance_slots"]
    rows = []
    for number, (session_key, seed_item, later) in enumerate(pages, start=1):
        targets = set(later)
        base = dict(session_key=session_key, item=int(catalog_ids[seed_item]), targets=len(targets))
        for method, (top, neighbours) in page_lists(data, settings, seed_item).items():
            rows.append(base | dict(method=method, neighbours=neighbours,
                                    **list_metrics(top, targets, seed_item, catalog_ids, categories,
                                                   diversity, slots)))
        if number % 2000 == 0 or number == len(pages):
            print(f"[ymal] {number:,}/{len(pages):,} item pages", flush=True)
    per_page = pd.DataFrame(rows)
    per_page["page"] = per_page.groupby("method").cumcount()
    per_page["has_neighbours"] = per_page.groupby("page").neighbours.transform("max") > 0
    summary = pd.concat([
        frame.groupby("method", sort=False)[RELEVANCE + DIVERSITY].mean()
        .join(frame.groupby("method", sort=False).size().rename("pages")).reset_index().assign(segment=name)
        for name, frame in (("all_pages", per_page), ("with_neighbours", per_page[per_page.has_neighbours]))],
        ignore_index=True)
    rng = np.random.default_rng(settings["seed"])
    differences = []
    for reference in ("category_popularity", f"relevance_{settings['pool_sizes'][-1]}"):
        base = per_page[per_page.method.eq(reference)].set_index("page")
        for method, frame in per_page.groupby("method", sort=False):
            if method == reference:
                continue
            frame = frame.set_index("page").reindex(base.index)
            for metric in RELEVANCE + ["combined_ild", "category_ild", "tail_to_head_distance"]:
                diff = frame[metric] - base[metric]
                low, high = session_bootstrap(diff, base.session_key, settings["bootstrap_replicates"], rng)
                differences.append(dict(method=method, reference=reference, metric=metric,
                                        value=float(frame[metric].mean()), reference_value=float(base[metric].mean()),
                                        difference=float(diff.mean()), ci_low=low, ci_high=high))
    write_json(folder / "ymal_settings.json", dict(
        settings=settings, knn=knn_settings, pages=page_stats,
        pages_with_neighbours=int(per_page.loc[per_page.method.eq("category_popularity"), "has_neighbours"].sum()),
        similarity_pairs=int(similarity.nnz)))
    save_table(per_page, folder / "per_page.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(pd.DataFrame(differences), folder / "differences.csv")
    print(f"[ymal] Saved to {folder}", flush=True)
    return folder
