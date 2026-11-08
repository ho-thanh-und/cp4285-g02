"""Phase 1: which events and which training sessions should the ranker learn from?

Event rules decide which product views enter the features (session history and
product popularity). Clicks and purchases are always kept.
  * all_events         every click and view (the current pipeline)
  * clicks_only        no views at all
  * unexplained_views  drop a view when the same product was clicked in the same
                       session up to N seconds earlier. 70% of clicks are followed
                       by a view of that product within seconds, so those views
                       repeat the click; the remaining views record other browsing.

Training sets decide which sessions provide LambdaRank query groups.
  * all_sessions          every training session (the current pipeline)
  * multi_click_sessions  only sessions with at least two clicks in total
Validation always uses every validation query, so the comparison is fair.

Every combination retrains the session models with the run's full grid of
leaf settings. Because the grid alone moves validation NDCG by about 0.01, the
summary reports both the validation-selected trial and the mean over the grid.
The rebuilt "all_events" features must equal the run's cached features exactly
before anything is trained; this checks that the rebuild copies the pipeline.
"""

import gc
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .adaptive_policies import ndcg, relevance_metrics
from .core import tokens
from .expanded_data import FEATURES, PURCHASE_FEATURES, query_features, read_candidates
from .expanded_training import training_settings
from .ranking_settings import reranking_settings, top_k
from .study_files import save_table, write_json

MODELS = {
    "click_session": ("click", list(range(len(FEATURES)))),
    "purchase_session": ("purchase", list(range(len(FEATURES) + len(PURCHASE_FEATURES)))),
}
EVENT_VARIANTS = ("all_events", "clicks_only", "unexplained_views")
TRAINING_SETS = ("all_sessions", "multi_click_sessions")
HISTORY_MISSING = FEATURES.index("session_history_missing")


def read_ablation_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not settings["event_variants"] or set(settings["event_variants"]) - set(EVENT_VARIANTS):
        raise ValueError(f"event_variants must come from {EVENT_VARIANTS}")
    if not settings["training_sets"] or set(settings["training_sets"]) - set(TRAINING_SETS):
        raise ValueError(f"training_sets must come from {TRAINING_SETS}")
    if not settings["models"] or set(settings["models"]) - set(MODELS):
        raise ValueError(f"models must come from {sorted(MODELS)}")
    if settings["explained_view_window_seconds"] <= 0 or settings["minimum_session_clicks"] < 1:
        raise ValueError("The view window and minimum session clicks must be positive")
    if settings["bootstrap_replicates"] < 1:
        raise ValueError("bootstrap_replicates must be positive")
    return settings


# ------------------------------------------------------------- event rules ---

def filter_events(events, variant, window_ms):
    """Apply one event rule to the run's events table (clicks/purchases untouched)."""
    if variant == "all_events":
        return events
    is_view = events.kind.eq("view")
    if variant == "clicks_only":
        return events[~is_view]
    clicks = events.loc[events.kind.eq("click"), ["session_key", "item_id", "timeframe"]]
    views = events.loc[is_view, ["session_key", "item_id", "timeframe"]].reset_index()
    pairs = views.merge(clicks, on=["session_key", "item_id"], suffixes=("", "_click"))
    gap = pairs.timeframe - pairs.timeframe_click
    explained = pairs.loc[(gap >= 0) & (gap <= window_ms), "index"].unique()
    return events.drop(index=explained)


def popularity_snapshots(events, train_sessions, training_dates):
    """Counts before each training date, and final counts, exactly as the cache builds them."""
    train_events = events[events.session_key.isin(train_sessions) & events.kind.isin(["click", "view", "purchase"])]
    daily = defaultdict(list)
    grouped = train_events.groupby(["eventdate", "kind", "item_id"]).size()
    for (date, kind, item), count in grouped.items():
        daily[date].append((kind, int(item), int(count)))
    counters = {kind: Counter() for kind in ("click", "view", "purchase")}
    snapshots = {}
    for date in sorted(set(daily) | set(training_dates)):
        snapshots[date] = {kind: counter.copy() for kind, counter in counters.items()}
        for kind, item, count in daily[date]:
            counters[kind][item] += count
    return snapshots, counters


class RunInputs:
    """Queries, events and catalog of one learning-curve run, read once."""

    def __init__(self, folder):
        folder = Path(folder)
        self.folder = folder
        config = json.loads((folder / "expanded_run.json").read_text())["config"]
        self.config = config
        self.history_limit = (reranking_settings(config)["history_max_events"] if "reranking" in config else 100)
        products = pd.read_parquet(folder / "products.parquet")
        self.catalog = {int(r.item_id): (r.price_log2, None if pd.isna(r.category_id) else int(r.category_id),
                                         tokens(r.name_tokens)) for r in products.itertuples(index=False)}
        with sqlite3.connect(folder / "development.sqlite") as db:
            self.queries = {split: db.execute("""SELECT * FROM queries WHERE split=?
                                ORDER BY eventdate,session_key,timeframe,query_id""", (split,)).fetchall()
                            for split in ("train", "validation")}
            self.events = pd.read_sql("""SELECT event_id,session_key,eventdate,kind,timeframe,query_id,item_id
                                         FROM events ORDER BY timeframe,event_id""", db)
        self.train_sessions = {q[1] for q in self.queries["train"]}
        self.training_dates = sorted({q[3] for q in self.queries["train"]})

    def features(self, variant, window_ms, extra_columns=0):
        """Rebuild the train and validation feature matrices under one event rule.

        Each matrix is allocated once at its final size (no stacking copies), so
        peak memory stays near the size of the data at 50k+ queries.
        extra_columns leaves zero-filled columns at the end for callers to fill.
        """
        events = filter_events(self.events, variant, window_ms)
        snapshots, counters = popularity_snapshots(events, self.train_sessions, self.training_dates)
        by_session = defaultdict(list)
        for row in events[["session_key", "timeframe", "kind", "item_id", "query_id"]].itertuples(index=False):
            item = None if pd.isna(row.item_id) else int(row.item_id)
            query = None if pd.isna(row.query_id) else int(row.query_id)
            by_session[row.session_key].append((row.timeframe, row.kind, item, query))
        matrices = {}
        width = len(FEATURES) + len(PURCHASE_FEATURES)
        for split, queries in self.queries.items():
            total = int(pd.read_parquet(self.folder / f"{split}_features" / "queries.parquet",
                                        columns=["candidates"]).candidates.sum())
            matrix = np.zeros((total, width + extra_columns), dtype=np.float32)
            items = np.empty(total, dtype=np.int64)
            offset = 0
            for query in queries:
                popularity = snapshots.get(query[3], counters) if split == "train" else counters
                found, block, _, _ = query_features(query, by_session[query[1]], self.catalog, popularity,
                                                    self.history_limit)
                matrix[offset:offset + len(found), :width] = block
                items[offset:offset + len(found)] = found
                offset += len(found)
            if offset != total:
                raise ValueError(f"{split}: rebuilt {offset:,} candidate rows, cache has {total:,}")
            matrices[split] = (matrix, items)
        return matrices, dict(views_kept=int(events.kind.eq("view").sum()),
                              views_total=int(self.events.kind.eq("view").sum()))


# ---------------------------------------------------------------- training ---

def training_queries(metadata, objective, training_set, minimum_clicks):
    """Query mask for LambdaRank groups: positive groups, optionally multi-click sessions only."""
    positive = metadata.clicks.gt(0) if objective == "click" else (metadata.clicks + metadata.purchases).gt(0)
    if training_set == "all_sessions":
        return positive.to_numpy()
    session_clicks = metadata.groupby("session_key").clicks.transform("sum")
    return (positive & session_clicks.ge(minimum_clicks)).to_numpy()


def lightgbm_dataset(matrix, labels, metadata, query_mask, feature_indices, seed, threads, max_bin, reference=None,
                     *, feature_names=None, positions=None):
    """LightGBM data for the selected query groups.

    feature_names covers every matrix column (defaults to the cached features).
    positions, one integer per row, enables LightGBM's position-bias correction.
    """
    rows = np.repeat(query_mask, metadata.candidates.to_numpy())
    names = feature_names or (FEATURES + PURCHASE_FEATURES)
    selected = np.ascontiguousarray(matrix[np.ix_(np.flatnonzero(rows), feature_indices)], dtype=np.float32)
    dataset = lgb.Dataset(selected, label=np.asarray(labels)[rows], group=metadata.loc[query_mask, "candidates"].tolist(),
                          feature_name=[names[i] for i in feature_indices],
                          categorical_feature=[1], reference=reference, free_raw_data=True,
                          position=None if positions is None else np.asarray(positions)[rows],
                          params=dict(feature_pre_filter=False, data_random_seed=seed, num_threads=threads,
                                      max_bin=max_bin))
    dataset.construct()
    return dataset


def reciprocal_rank(order, clicked):
    """MRR@k term: 1 / position of the first clicked product in the top k, else 0.

    NaN when the query has no clicked candidate, matching NDCG's population.
    """
    clicked = np.asarray(clicked, bool)
    if not clicked.any():
        return np.nan
    hits = np.flatnonzero(clicked[np.asarray(order, int)])
    return float(1 / (hits[0] + 1)) if len(hits) else 0.0


def per_query_metrics(validation, prediction, label_gains, k):
    """Click/gain NDCG@k, recall@k, purchase recall and MRR@k for each query."""
    frame = validation.assign(score=prediction)
    rows = []
    for qid, group in frame.groupby("query_id", sort=False):
        order = np.argsort(-group.score.to_numpy(), kind="stable")[:k]
        rows.append(dict(query_id=qid, session_key=group.session_key.iloc[0],
                         **relevance_metrics(order, group.clicked, group.purchased, label_gains, k),
                         mrr=reciprocal_rank(order, group.clicked)))
    return pd.DataFrame(rows)


def fit_grid(name, objective, feature_indices, train_matrix, valid_matrix, inputs, settings, query_mask, k,
             *, rank_objective="lambdarank", train_positions=None, feature_names=None, valid_query_mask=None):
    """Fit every grid setting; return per-query validation metrics for each trial.

    rank_objective is "lambdarank" or "rank_xendcg". train_positions (one bucket
    per training row) turns on position-bias correction. valid_query_mask limits
    the early-stopping groups (default: every positive validation query); the
    returned metrics always cover every validation query.
    """
    config, folder = inputs.config, inputs.folder
    train_meta = pd.read_parquet(folder / "train_features" / "queries.parquet")
    valid_meta = pd.read_parquet(folder / "validation_features" / "queries.parquet")
    label_file = "click_labels.npy" if objective == "click" else "grades.npy"
    train_labels = np.load(folder / "train_features" / label_file)
    valid_labels = np.load(folder / "validation_features" / label_file)
    valid_mask = training_queries(valid_meta, objective, "all_sessions", 1)
    if valid_query_mask is not None:
        valid_mask = valid_mask & valid_query_mask
    seed, threads = config["seed"], config["threads"]
    train = lightgbm_dataset(train_matrix, train_labels, train_meta, query_mask, feature_indices, seed, threads,
                             settings["max_bin"], feature_names=feature_names, positions=train_positions)
    valid = lightgbm_dataset(valid_matrix, valid_labels, valid_meta, valid_mask, feature_indices, seed, threads,
                             settings["max_bin"], reference=train, feature_names=feature_names)
    fit_gains = [0, 1] if objective == "click" else config["label_gains"]
    validation = read_candidates(folder, "validation")

    def custom_metric(prediction, data):
        labels = data.get_label().astype(int)
        gains = np.asarray(fit_gains)[labels]
        offsets = np.r_[0, np.cumsum(data.get_group())]
        values = [ndcg(np.argsort(-prediction[a:b], kind="stable")[:k], gains[a:b], k)
                  for a, b in zip(offsets[:-1], offsets[1:])]
        return f"rank_ndcg_at_{k}", float(np.nanmean(values)), True

    results = []
    for leaves, minimum in product(settings["num_leaves"], settings["min_data_in_leaf"]):
        model = lgb.train(dict(
            objective=rank_objective, metric="None", label_gain=fit_gains,
            learning_rate=settings["learning_rate"], num_leaves=leaves, min_data_in_leaf=minimum,
            max_depth=settings["max_depth"], max_bin=settings["max_bin"],
            lambda_l1=settings["lambda_l1"], lambda_l2=settings["lambda_l2"],
            feature_fraction=settings["feature_fraction"], bagging_fraction=settings["bagging_fraction"],
            bagging_freq=settings["bagging_freq"], lambdarank_truncation_level=settings["lambdarank_truncation_level"],
            feature_pre_filter=False, deterministic=True, force_col_wise=True, seed=seed,
            num_threads=threads, verbosity=-1,
        ), train, num_boost_round=settings["num_boost_round"], valid_sets=[valid], feval=custom_metric,
            callbacks=[lgb.early_stopping(settings["early_stopping_rounds"], verbose=False)])
        prediction = model.predict(valid_matrix[:, feature_indices], num_threads=threads)
        measured = per_query_metrics(validation, prediction, config["label_gains"], k)
        results.append(dict(leaves=leaves, min_rows=minimum, iterations=model.best_iteration,
                            click_ndcg=float(measured.click_ndcg.mean()), queries=measured,
                            gain=dict(zip(model.feature_name(), model.feature_importance("gain")))))
        print(f"  {name} leaves={leaves} min_rows={minimum}: click NDCG@{k} "
              f"{measured.click_ndcg.mean():.4f}, {model.best_iteration} trees", flush=True)
    del train, valid
    gc.collect()
    return results


# -------------------------------------------------------------- comparison ---

def session_bootstrap(diff, sessions, replicates, rng):
    """95% interval of a per-query mean difference, resampling whole sessions."""
    frame = pd.DataFrame(dict(diff=diff.to_numpy(), session=sessions.to_numpy())).dropna()
    if frame.empty:
        return np.nan, np.nan
    sums = frame.groupby("session")["diff"].agg(["sum", "count"])
    draws = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=replicates)
    boot = (draws @ sums["sum"].to_numpy()) / (draws @ sums["count"].to_numpy())
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def compare(per_query, settings):
    """Each variant minus the current pipeline (all_events, all_sessions), same model."""
    rng = np.random.default_rng(settings["seed"])
    rows = []
    for (model, statistic), frame in per_query.groupby(["model", "statistic"], sort=False):
        base = frame[frame.event_variant.eq("all_events") & frame.training_set.eq("all_sessions")].set_index("query_id")
        for (variant, training_set), other in frame.groupby(["event_variant", "training_set"], sort=False):
            other = other.set_index("query_id").reindex(base.index)
            for segment, mask in (("all", np.ones(len(base), bool)), ("with_history", base.has_history.to_numpy()),
                                  ("no_history", ~base.has_history.to_numpy())):
                for metric in ("click_ndcg", "gain_ndcg"):
                    diff = (other[metric] - base[metric])[mask]
                    low, high = session_bootstrap(diff, base.session_key[mask], settings["bootstrap_replicates"], rng)
                    rows.append(dict(model=model, statistic=statistic, event_variant=variant, training_set=training_set,
                                     segment=segment, metric=metric, value=float(other[metric][mask].mean()),
                                     baseline=float(base[metric][mask].mean()), difference=float(diff.mean()),
                                     ci_low=low, ci_high=high, queries=int(diff.notna().sum())))
    return pd.DataFrame(rows)


def run_event_ablation(trained_run, settings_path=Path("config/event_ablation.json"), output_root=None):
    settings = read_ablation_settings(settings_path)
    inputs = RunInputs(trained_run)
    training = training_settings(inputs.config)
    k = top_k(inputs.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{Path(trained_run).name}"
    folder.mkdir(parents=True)
    write_json(folder / "ablation_settings.json", dict(trained_run=str(trained_run), settings=settings, training=training))
    window_ms = settings["explained_view_window_seconds"] * 1000
    train_meta = pd.read_parquet(inputs.folder / "train_features" / "queries.parquet")
    expected_items = {split: read_candidates(inputs.folder, split).item_id.to_numpy() for split in ("train", "validation")}
    valid_cached = np.load(inputs.folder / "validation_features" / "X.npy")
    has_history = None
    rows, gains, audit = [], [], []
    for variant in settings["event_variants"]:
        print(f"[features] Rebuild train/validation features with {variant}", flush=True)
        matrices, counts = inputs.features(variant, window_ms)
        for split, (_, items) in matrices.items():
            if not np.array_equal(items, expected_items[split]):
                raise ValueError(f"Rebuilt {split} candidates differ from the run's cached order")
        if variant == "all_events":
            for split in ("train", "validation"):
                cached = np.load(inputs.folder / f"{split}_features" / "X.npy", mmap_mode="r")
                if not np.allclose(matrices[split][0][:, :cached.shape[1]], cached, equal_nan=True):
                    raise ValueError(f"Rebuilt all_events {split} features differ from the cached features")
            print("[check] Rebuilt all_events features equal the cached features", flush=True)
        # History segment is fixed by the current pipeline's features for every variant.
        if has_history is None:
            reference = matrices["validation"][0] if variant == "all_events" else valid_cached
            starts = np.r_[0, np.cumsum(pd.read_parquet(inputs.folder / "validation_features" / "queries.parquet")
                                        .candidates.to_numpy())[:-1]]
            has_history = pd.Series(reference[starts, HISTORY_MISSING] == 0,
                                    index=[q[0] for q in inputs.queries["validation"]])
        audit.append(dict(event_variant=variant, **counts))
        for training_set in settings["training_sets"]:
            for model in settings["models"]:
                objective, indices = MODELS[model]
                mask = training_queries(train_meta, objective, training_set, settings["minimum_session_clicks"])
                print(f"[train] {variant} / {training_set} / {model}: {int(mask.sum()):,} training queries", flush=True)
                trials = fit_grid(model, objective, indices, matrices["train"][0], matrices["validation"][0],
                                  inputs, training, mask, k)
                best = max(trials, key=lambda trial: trial["click_ndcg"])
                selected = best["queries"].assign(statistic="selected_trial")
                grid_mean = (pd.concat([t["queries"] for t in trials]).groupby(["query_id", "session_key"], sort=False)
                             [["click_ndcg", "gain_ndcg", "recall_at_k", "purchase_recall", "mrr"]].mean().reset_index()
                             .assign(statistic="grid_mean"))
                for frame in (selected, grid_mean):
                    rows.append(frame.assign(event_variant=variant, training_set=training_set, model=model,
                                             training_queries=int(mask.sum()),
                                             has_history=frame.query_id.map(has_history).to_numpy()))
                gains.extend(dict(event_variant=variant, training_set=training_set, model=model,
                                  feature=feature, gain=value) for feature, value in best["gain"].items())
                audit.append(dict(event_variant=variant, training_set=training_set, model=model,
                                  training_queries=int(mask.sum()), selected_leaves=best["leaves"],
                                  selected_min_rows=best["min_rows"], selected_iterations=best["iterations"],
                                  trial_ndcg=[round(t["click_ndcg"], 5) for t in trials]))
        del matrices
        gc.collect()
    per_query = pd.concat(rows, ignore_index=True)
    summary = (per_query.groupby(["model", "statistic", "event_variant", "training_set"], sort=False)
               .agg(click_ndcg=("click_ndcg", "mean"), gain_ndcg=("gain_ndcg", "mean"),
                    recall_at_k=("recall_at_k", "mean"), mrr=("mrr", "mean"), training_queries=("training_queries", "first"),
                    queries=("query_id", "size")).reset_index())
    save_table(per_query, folder / "validation_per_query.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(compare(per_query, settings), folder / "difference_vs_current.csv")
    save_table(pd.DataFrame(gains), folder / "feature_gain.csv")
    write_json(folder / "audit.json", json.loads(pd.DataFrame(audit).to_json(orient="records")))
    print(f"Event ablation saved to {folder}", flush=True)
    return folder
