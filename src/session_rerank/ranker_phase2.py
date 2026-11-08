"""Phase 2: improve the category-page relevance ranker.

Agreed data choices (EXPERIMENT_LOG.md §7) apply to every variant: features use
clicks only (no product views), all training sessions are used, and text-search
queries are left out of training and of early stopping. Category pages are the
product surface this ranker serves, so they are the primary evaluation segment.

Variants, each retrained with the run's full grid of leaf settings:
  * popularity        two extra features: daily click and purchase rates of the
                      product over the whole training window, from the full raw
                      files (all shoppers, not only the sampled sessions). A
                      training query on day d sees only days before d.
  * position_debias   LightGBM's unbiased LambdaRank: each training row carries
                      its displayed-rank bucket, and LightGBM learns how much
                      each bucket inflates clicks. Scoring ignores position.
  * objective         "lambdarank" or "rank_xendcg" (a different listwise loss;
                      LightGBM applies position correction to lambdarank only).
  * learning_rate     a lower rate grows more, smaller trees before stopping.

References on the same validation queries: the site's original order, and the
logged list sorted by whole-window click popularity (which matched LightGBM in
Experiment B). Phase 2's goal is a ranker that beats the popularity sort on
category pages with an interval that excludes zero.
"""

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .event_ablation import (
    MODELS, RunInputs, fit_grid, per_query_metrics, session_bootstrap, training_queries,
)
from .expanded_data import FEATURES, PURCHASE_FEATURES, read_candidates
from .expanded_training import training_settings
from .ranking_settings import top_k
from .study_files import save_table, write_json
from .two_tower import popularity_features

POPULARITY_FEATURES = ["window_click_rate", "window_purchase_rate"]
ALL_FEATURES = FEATURES + PURCHASE_FEATURES + POPULARITY_FEATURES
HISTORY_MISSING = FEATURES.index("session_history_missing")


def read_phase2_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if settings["event_variant"] not in ("all_events", "clicks_only", "unexplained_views"):
        raise ValueError("event_variant must be all_events, clicks_only or unexplained_views")
    if not settings["models"] or set(settings["models"]) - set(MODELS):
        raise ValueError(f"models must come from {sorted(MODELS)}")
    variants = settings["variants"]
    ids = [v["id"] for v in variants]
    if not variants or len(set(ids)) != len(ids):
        raise ValueError("variants need unique ids")
    for variant in variants:
        if variant["objective"] not in ("lambdarank", "rank_xendcg"):
            raise ValueError("objective must be lambdarank or rank_xendcg")
        if not 0 < variant["learning_rate"] <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
    edges = settings["position_bucket_edges"]
    if edges != sorted(set(edges)) or edges[0] != 1:
        raise ValueError("position_bucket_edges must increase and start at 1")
    return settings


def window_activity(raw_directory, item_index, start, days):
    """Clicks and purchases per product per day of the training window, from raw CSVs."""
    raw = Path(raw_directory)
    dates = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "eventdate"])
    clicks = pd.read_csv(raw / "train-clicks.csv", sep=";", usecols=["queryId", "itemId"]).merge(dates, on="queryId")
    purchases = pd.read_csv(raw / "train-purchases.csv", sep=";", usecols=["itemId", "eventdate"])
    counts = []
    for frame in (clicks, purchases):
        offset = (pd.to_datetime(frame.eventdate) - pd.Timestamp(start)).dt.days.to_numpy()
        keep = (offset >= 0) & (offset < days) & frame.itemId.isin(item_index.index).to_numpy()
        matrix = np.zeros((days, len(item_index)), dtype=np.float32)
        np.add.at(matrix, (offset[keep], item_index.loc[frame.itemId[keep]].to_numpy()), 1)
        counts.append(matrix)
    return counts


def position_buckets(ranks, edges):
    """Bucket displayed ranks (1-based) into 0-based ids, e.g. 1,2,…,6-10,11-20,…"""
    return np.searchsorted(np.asarray(edges), np.asarray(ranks), side="right") - 1


class Phase2Data:
    """Clicks-only features plus window popularity, positions and query segments."""

    def __init__(self, trained_run, settings):
        self.inputs = RunInputs(trained_run)
        folder = self.inputs.folder
        print(f"[features] Rebuild features with {settings['event_variant']}", flush=True)
        matrices, _ = self.inputs.features(settings["event_variant"], 60_000, extra_columns=len(POPULARITY_FEATURES))
        products = pd.read_parquet(folder / "products.parquet")
        item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
        train_dates = pd.to_datetime(pd.Series([q[3] for q in self.inputs.queries["train"]]))
        start, days = train_dates.min(), int((train_dates.max() - train_dates.min()).days) + 1
        self.window = dict(start=str(start.date()), days=days)
        print(f"[features] Count raw clicks and purchases, {self.window['start']} + {days} days", flush=True)
        clicks, purchases = window_activity(settings["raw_directory"], item_index, start, days)
        rates = np.ascontiguousarray(popularity_features(clicks, purchases)[..., :2])
        del clicks, purchases
        self.matrices, self.meta, self.query_type, self.ranks = {}, {}, {}, {}
        for split, (matrix, items) in matrices.items():
            queries = self.inputs.queries[split]
            meta = pd.read_parquet(folder / f"{split}_features" / "queries.parquet")
            if meta.query_id.tolist() != [q[0] for q in queries]:
                raise ValueError(f"{split} query order differs from the feature cache")
            # Only two columns are needed; the full candidate table is large at 50k+ queries.
            candidates = pd.concat([pd.read_parquet(path, columns=["item_id", "original_rank"]) for path in
                                    sorted((folder / f"{split}_features").glob("candidates_*.parquet"))],
                                   ignore_index=True)
            if not np.array_equal(candidates.item_id.to_numpy(), items):
                raise ValueError(f"{split} candidate order differs from the feature cache")
            offset = (pd.to_datetime(pd.Series([q[3] for q in queries])) - start).dt.days.to_numpy()
            query_day = np.clip(offset, 0, days) if split == "train" else np.full(len(queries), days)
            row_day = np.repeat(query_day, meta.candidates.to_numpy())
            index = item_index.reindex(items).to_numpy()
            known = np.flatnonzero(~np.isnan(index))
            # Fill the two reserved popularity columns in place (unknown products stay 0).
            width = len(FEATURES) + len(PURCHASE_FEATURES)
            matrix[known, width:] = rates[row_day[known], index[known].astype(int)]
            self.matrices[split] = matrix
            self.meta[split] = meta
            self.query_type[split] = np.array(["text" if (q[6] or 0) == 0 else "category" for q in queries])
            self.ranks[split] = candidates.original_rank.to_numpy(np.int32)
            del candidates, row_day, index, known
        valid = self.matrices["validation"]
        starts = np.r_[0, np.cumsum(self.meta["validation"].candidates.to_numpy())[:-1]]
        self.has_history = pd.Series(valid[starts, HISTORY_MISSING] == 0, index=self.meta["validation"].query_id)
        self.validation_type = pd.Series(self.query_type["validation"], index=self.meta["validation"].query_id)
        del matrices
        gc.collect()


def tag(frame, data, **labels):
    return frame.assign(query_type=frame.query_id.map(data.validation_type).to_numpy(),
                        has_history=frame.query_id.map(data.has_history).to_numpy(), **labels)


def compare(per_query, settings):
    """Every variant minus a reference (base and popularity sort), same model and statistic."""
    rng = np.random.default_rng(settings["seed"])
    rows = []
    segments = {
        "category_pages": lambda f: f.query_type.eq("category"),
        "category_with_history": lambda f: f.query_type.eq("category") & f.has_history,
        "category_no_history": lambda f: f.query_type.eq("category") & ~f.has_history.astype(bool),
        "all_queries": lambda f: pd.Series(True, index=f.index),
        "text_search": lambda f: f.query_type.eq("text"),
    }
    references = per_query[per_query.model.eq("reference")]
    for (model, statistic), frame in per_query[~per_query.model.eq("reference")].groupby(["model", "statistic"], sort=False):
        baselines = {"base": frame[frame.variant.eq(settings["variants"][0]["id"])],
                     "popularity_sort": references[references.variant.eq("popularity_sort")
                                                   & references.statistic.eq(statistic)]}
        for reference_name, base in baselines.items():
            base = base.set_index("query_id")
            for variant, other in frame.groupby("variant", sort=False):
                other = other.set_index("query_id").reindex(base.index)
                for segment, select in segments.items():
                    mask = select(base).to_numpy(bool)
                    diff = (other.click_ndcg - base.click_ndcg)[mask]
                    low, high = session_bootstrap(diff, base.session_key[mask], settings["bootstrap_replicates"], rng)
                    rows.append(dict(model=model, statistic=statistic, variant=variant, reference=reference_name,
                                     segment=segment, value=float(other.click_ndcg[mask].mean()),
                                     reference_value=float(base.click_ndcg[mask].mean()),
                                     difference=float(diff.mean()), ci_low=low, ci_high=high,
                                     queries=int(diff.notna().sum())))
    return pd.DataFrame(rows)


def run_phase2(trained_run, settings_path=Path("config/ranker_phase2.json"), output_root=None):
    settings = read_phase2_settings(settings_path)
    data = Phase2Data(trained_run, settings)
    config = data.inputs.config
    base_training = training_settings(config)
    k = top_k(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{Path(trained_run).name}"
    folder.mkdir(parents=True)
    write_json(folder / "phase2_settings.json", dict(trained_run=str(trained_run), settings=settings,
                                                     training=base_training, popularity_window=data.window))
    validation = read_candidates(data.inputs.folder, "validation")
    rows, gains, audit = [], [], []
    # References: the site's order and the logged list sorted by window click popularity.
    valid_matrix = data.matrices["validation"]
    for name, score in (("site_order", -data.ranks["validation"].astype(float)),
                        ("popularity_sort", valid_matrix[:, ALL_FEATURES.index("window_click_rate")].astype(float))):
        measured = per_query_metrics(validation, score, config["label_gains"], k)
        for statistic in ("selected_trial", "grid_mean"):
            rows.append(tag(measured, data, model="reference", variant=name, statistic=statistic))
    exclude_text = settings["exclude_text_search_from_training"]
    train_category = data.query_type["train"] == "category"
    valid_category = data.query_type["validation"] == "category"
    buckets = position_buckets(data.ranks["train"], settings["position_bucket_edges"])
    for variant in settings["variants"]:
        training = base_training | dict(learning_rate=variant["learning_rate"])
        for model in settings["models"]:
            objective, indices = MODELS[model]
            indices = list(indices) + ([len(FEATURES) + len(PURCHASE_FEATURES), len(FEATURES) + len(PURCHASE_FEATURES) + 1]
                                       if variant["popularity"] else [])
            mask = training_queries(data.meta["train"], objective, "all_sessions", 1)
            if exclude_text:
                mask = mask & train_category
            print(f"[train] {variant['id']} / {model}: {int(mask.sum()):,} training queries, "
                  f"{len(indices)} features", flush=True)
            trials = fit_grid(model, objective, indices, data.matrices["train"], valid_matrix, data.inputs,
                              training, mask, k, rank_objective=variant["objective"],
                              train_positions=buckets if variant["position_debias"] else None,
                              feature_names=ALL_FEATURES,
                              valid_query_mask=valid_category if exclude_text else None)
            best = max(trials, key=lambda trial: trial["queries"].loc[
                trial["queries"].query_id.map(data.validation_type).eq("category").to_numpy(), "click_ndcg"].mean())
            grid_mean = (pd.concat([t["queries"] for t in trials]).groupby(["query_id", "session_key"], sort=False)
                         [["click_ndcg", "gain_ndcg", "recall_at_k", "purchase_recall", "mrr"]].mean().reset_index())
            rows.append(tag(best["queries"], data, model=model, variant=variant["id"], statistic="selected_trial"))
            rows.append(tag(grid_mean, data, model=model, variant=variant["id"], statistic="grid_mean"))
            gains.extend(dict(variant=variant["id"], model=model, feature=f, gain=g) for f, g in best["gain"].items())
            audit.append(dict(variant=variant["id"], model=model, training_queries=int(mask.sum()),
                              selected_leaves=best["leaves"], selected_min_rows=best["min_rows"],
                              selected_iterations=best["iterations"],
                              trials=[dict(leaves=t["leaves"], min_rows=t["min_rows"], iterations=t["iterations"],
                                           click_ndcg_all=round(t["click_ndcg"], 5)) for t in trials]))
    per_query = pd.concat(rows, ignore_index=True)
    summary = (pd.concat([per_query.assign(segment="category_pages")[per_query.query_type.eq("category")],
                          per_query.assign(segment="all_queries")])
               .groupby(["segment", "model", "variant", "statistic"], sort=False)
               .agg(click_ndcg=("click_ndcg", "mean"), gain_ndcg=("gain_ndcg", "mean"),
                    recall_at_k=("recall_at_k", "mean"), mrr=("mrr", "mean"), queries=("click_ndcg", "count")).reset_index())
    save_table(per_query, folder / "validation_per_query.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(compare(per_query, settings), folder / "differences.csv")
    save_table(pd.DataFrame(gains), folder / "feature_gain.csv")
    write_json(folder / "audit.json", audit)
    print(f"Phase 2 saved to {folder}", flush=True)
    return folder
