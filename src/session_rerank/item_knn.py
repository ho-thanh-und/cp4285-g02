"""Classical collaborative-filtering baseline: item-kNN on session co-clicks.

Similarity. Over the training window (April 1–29), every visible session's
clicked products form one "basket" (full raw files, all shoppers; at most the
`max_items_per_session` most recent distinct products per session). Two products
are similar when the same sessions click both:

    sim(i, j) = co_clicks(i, j) / (sqrt(clicks_sessions(i) * clicks_sessions(j)) + shrinkage)

This is the classic item-to-item collaborative filter ("shoppers who clicked
this also clicked that"); it is also the planned "You may also like" model.

Ranking a category page. A candidate's score is the sum of its similarity to the
products this shopper clicked earlier in the same session (clicks only, strictly
before the query). Ties, and shoppers without earlier clicks, fall back to
window click popularity and then the site's order, the standard fallback for
neighbourhood baselines. Without history the ranking therefore equals the
popularity sort; the two differ only for queries with session history.

Evaluation uses the same per-query metrics as the rankers (click NDCG@10,
recall@10, MRR@10) and can join per-query results of saved Phase 2 runs, so all
methods are compared on identical queries with session-bootstrap intervals.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .event_ablation import per_query_metrics, session_bootstrap
from .expanded_data import read_candidates
from .ranking_settings import top_k
from .study_files import save_table, write_json

METRICS = ["click_ndcg", "recall_at_k", "mrr"]


def read_knn_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if settings["max_items_per_session"] < 2:
        raise ValueError("max_items_per_session must be at least 2 for co-clicks to exist")
    if settings["shrinkage"] < 0:
        raise ValueError("shrinkage must be nonnegative")
    for entry in settings["compare_with"]:
        if set(entry) != {"label", "per_query", "variant", "model", "statistic"}:
            raise ValueError("compare_with entries need label, per_query, variant, model and statistic")
    return settings


def session_baskets(raw_directory, start, end_exclusive, item_index, max_items):
    """(session row, product column) pairs of distinct clicked products in the window."""
    raw = Path(raw_directory)
    queries = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "sessionId", "eventdate"])
    queries = queries[(queries.eventdate >= start) & (queries.eventdate < end_exclusive)]
    clicks = pd.read_csv(raw / "train-clicks.csv", sep=";").merge(queries, on="queryId")
    clicks = clicks[clicks.itemId.isin(item_index.index)]
    latest = (clicks.groupby(["sessionId", "eventdate", "itemId"], sort=False).timeframe.max()
              .reset_index().sort_values("timeframe", ascending=False))
    latest = latest.groupby(["sessionId", "eventdate"], sort=False).head(max_items)
    session = latest.groupby(["sessionId", "eventdate"], sort=False).ngroup().to_numpy()
    return session, item_index.loc[latest.itemId].to_numpy(), clicks


def cosine_similarity(session, item, products, shrinkage):
    """Sparse product-by-product co-click cosine similarity with a zero diagonal."""
    basket = sparse.csr_matrix((np.ones(len(item), np.float32), (session, item)),
                               shape=(int(session.max()) + 1 if len(session) else 0, products))
    counts = sparse.csr_matrix(basket.T @ basket)
    counts.setdiag(0)
    counts.eliminate_zeros()
    support = np.asarray(basket.sum(axis=0)).ravel()
    coo = counts.tocoo()
    values = coo.data / (np.sqrt(support[coo.row] * support[coo.col]) + shrinkage)
    similarity = sparse.csr_matrix((values.astype(np.float32), (coo.row, coo.col)), shape=counts.shape)
    return similarity, support


class KnnData:
    """Similarity, popularity and validation inputs for one prepared run folder."""

    def __init__(self, trained_run, settings):
        folder = Path(trained_run)
        self.folder = folder
        self.config = json.loads((folder / "expanded_run.json").read_text())["config"]
        products = pd.read_parquet(folder / "products.parquet")
        self.item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
        with sqlite3.connect(folder / "development.sqlite") as db:
            queries = pd.read_sql("SELECT query_id, session_key, split, eventdate, timeframe, category FROM queries", db)
            events = pd.read_sql("""SELECT e.session_key, e.timeframe, e.item_id, e.query_id FROM events e
                                    WHERE e.kind='click' AND e.session_key IN
                                    (SELECT session_key FROM queries WHERE split='validation')""", db)
        dates = sorted(queries.loc[queries.split.eq("train"), "eventdate"])
        self.window = dict(start=dates[0], end_exclusive=str((pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).date()))
        print(f"[knn] Co-click baskets from raw clicks {self.window['start']} to before {self.window['end_exclusive']}",
              flush=True)
        session, item, clicks = session_baskets(settings["raw_directory"], self.window["start"],
                                                self.window["end_exclusive"], self.item_index,
                                                settings["max_items_per_session"])
        self.similarity, self.support = cosine_similarity(session, item, len(products), settings["shrinkage"])
        known = clicks.itemId.isin(self.item_index.index)
        self.popularity = np.bincount(self.item_index.loc[clicks.itemId[known]].to_numpy(),
                                      minlength=len(products)).astype(float)
        self.stats = dict(window=self.window, sessions=int(session.max()) + 1 if len(session) else 0,
                          multi_click_sessions=int((np.bincount(session) >= 2).sum()) if len(session) else 0,
                          products_with_clicks=int((self.support > 0).sum()),
                          similarity_pairs=int(self.similarity.nnz), window_clicks=int(len(clicks)))
        self.validation_queries = queries[queries.split.eq("validation")].set_index("query_id")
        self.validation_clicks = events

    def use_queries(self, queries, clicks):
        """Score another split: queries indexed by query_id (session_key, timeframe,
        category) and that split's click events (session_key, timeframe, item_id,
        query_id). The similarity and popularity stay those of the training window."""
        self.validation_queries = queries
        self.validation_clicks = clicks

    def history(self, query_id):
        """Distinct products clicked earlier in the query's session (catalog rows)."""
        query = self.validation_queries.loc[query_id]
        clicks = self.validation_clicks
        earlier = clicks[clicks.session_key.eq(query.session_key) & (clicks.timeframe < query.timeframe)
                         & ~clicks.query_id.eq(query_id)]
        items = earlier.item_id.dropna().astype(np.int64)
        items = items[items.isin(self.item_index.index)].unique()
        return self.item_index.loc[items].to_numpy()

    def scores(self, candidates):
        """kNN score with popularity tie-break for every validation candidate row."""
        score = np.zeros(len(candidates))
        neighbours = np.zeros(len(candidates), dtype=int)
        rows = self.item_index.reindex(candidates.item_id.to_numpy()).to_numpy()
        tiebreak = self.popularity / (self.popularity.max() + 1) * 1e-7
        history_of = {}
        for qid, group in candidates.groupby("query_id", sort=False):
            history = self.history(qid)
            history_of[qid] = len(history)
            index = group.index.to_numpy()
            known = ~np.isnan(rows[index])
            members = rows[index][known].astype(int)
            if len(history):
                block = self.similarity[members][:, history]
                score[index[known]] = np.asarray(block.sum(axis=1)).ravel()
                neighbours[index[known]] = np.diff(block.indptr)
            score[index[known]] += tiebreak[members]
        return score, neighbours, history_of


def run_item_knn(trained_run, settings_path=Path("config/item_knn.json"), output_root=None):
    settings = read_knn_settings(settings_path)
    data = KnnData(trained_run, settings)
    k = top_k(data.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{Path(trained_run).name}"
    folder.mkdir(parents=True)
    validation = read_candidates(data.folder, "validation")
    print(f"[knn] Rank {validation.query_id.nunique():,} validation queries", flush=True)
    knn, neighbours, history = data.scores(validation)
    rows_index = data.item_index.reindex(validation.item_id.to_numpy()).to_numpy()
    popularity = np.where(np.isnan(rows_index), 0, data.popularity[np.nan_to_num(rows_index).astype(int)])
    methods = {"site_order": -validation.original_rank.to_numpy(float),
               "popularity_sort": popularity, "item_knn": knn}
    query_type = data.validation_queries.category.fillna(0).map(lambda c: "text" if c == 0 else "category")
    rows = []
    for name, score in methods.items():
        measured = per_query_metrics(validation, score, data.config["label_gains"], k)
        rows.append(measured.assign(method=name))
    for entry in settings["compare_with"]:
        other = pd.read_parquet(entry["per_query"])
        other = other[other.variant.eq(entry["variant"]) & other.model.eq(entry["model"])
                      & other.statistic.eq(entry["statistic"])]
        if other.empty or "mrr" not in other:
            raise ValueError(f"{entry['label']}: no matching rows with MRR in {entry['per_query']}")
        rows.append(other[["query_id", "session_key"] + METRICS].assign(method=entry["label"]))
    per_query = pd.concat(rows, ignore_index=True)
    per_query["query_type"] = per_query.query_id.map(query_type).to_numpy()
    per_query["has_history"] = per_query.query_id.map(lambda q: history.get(q, 0) > 0).to_numpy()
    segments = {"category_pages": per_query.query_type.eq("category"),
                "category_with_history": per_query.query_type.eq("category") & per_query.has_history,
                "category_no_history": per_query.query_type.eq("category") & ~per_query.has_history,
                "all_queries": pd.Series(True, index=per_query.index)}
    summary = pd.concat([per_query[mask].groupby("method", sort=False)[METRICS].mean()
                         .join(per_query[mask].groupby("method", sort=False).click_ndcg.count().rename("queries"))
                         .reset_index().assign(segment=name) for name, mask in segments.items()], ignore_index=True)
    rng = np.random.default_rng(settings["seed"])
    differences = []
    for reference in ("popularity_sort", "item_knn"):
        base = per_query[per_query.method.eq(reference)].set_index("query_id")
        for method, frame in per_query.groupby("method", sort=False):
            if method == reference:
                continue
            frame = frame.set_index("query_id").reindex(base.index)
            for name in segments:
                mask = segments[name][per_query.method.eq(reference)].to_numpy()
                for metric in METRICS:
                    diff = (frame[metric] - base[metric])[mask]
                    low, high = session_bootstrap(diff, base.session_key[mask], settings["bootstrap_replicates"], rng)
                    differences.append(dict(method=method, reference=reference, segment=name, metric=metric,
                                            value=float(frame[metric][mask].mean()),
                                            reference_value=float(base[metric][mask].mean()),
                                            difference=float(diff.mean()), ci_low=low, ci_high=high,
                                            queries=int(diff.notna().sum())))
    candidates_with_neighbour = validation.assign(neighbours=neighbours).groupby("query_id").neighbours.apply(
        lambda n: bool((n > 0).any()))
    stats = data.stats | dict(
        validation_queries=int(validation.query_id.nunique()),
        queries_with_history=int(sum(v > 0 for v in history.values())),
        queries_with_history_and_a_coclicked_candidate=int(sum(
            candidates_with_neighbour.get(q, False) for q, n in history.items() if n > 0)),
    )
    write_json(folder / "knn_settings.json", dict(trained_run=str(trained_run), settings=settings, stats=stats))
    save_table(per_query, folder / "validation_per_query.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(pd.DataFrame(differences), folder / "differences.csv")
    print(f"Item-kNN saved to {folder}", flush=True)
    return folder
