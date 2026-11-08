"""End-to-end validation: filter K1 products, score them with LightGBM, rerank to a top 10.

Stage 1, filter. For each validation query, choose the products to consider:
  * logged         the site's own logged list (the existing pipeline)
  * two_tower_K1   the K1 best products of the category (or catalog for a text
                   search) according to the saved two-tower filter
  * popularity_K1  the K1 most popular products of the same pool, as a baseline
  * *_shown        the same two filters keeping only products the site showed.
                   Never-shown products have no click labels; dropping them
                   before cutting the top 10 measures what the filter loses
                   without counting unjudged products as failures.
Stage 2, score. Build the usual LightGBM features for every product in the pool
(query-time session history, popularity, category/token matches) and score
them with a saved learning-curve model. None of these features needs the
product to have been shown.
Stage 3, rerank. Apply each policy (relevance only, fixed alpha, exponential
session-adaptive alpha, 6 + 4 diverse tail) to build the top 10.

Metrics use the same definitions throughout. Click NDCG@10 and click recall@10
divide by all products the shopper clicked, so a pool that misses clicked
products is penalised. Products the site never showed cannot have clicks; the
unshown share reports how much of each top 10 is affected. ILD@10 is reported
for token, category and the 50/50 combined distance.
"""

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .adaptive_policies import expanded_order, relevance_weight, weighted_history_context
from .core import tokens
from .diversity import ProductDiversity
from .expanded_data import query_features, read_histories
from .study_files import save_table, write_json
from .two_tower import (
    RunData, TEXT_SEARCH, _click_ndcg, build_model, item_embeddings,
)


def read_pipeline_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not settings["k1_values"] or any(not isinstance(k, int) or k < settings["top_k"] for k in settings["k1_values"]):
        raise ValueError("k1_values must be integers no smaller than top_k")
    if not settings["models"] or not settings["policies"]:
        raise ValueError("List at least one model and one policy")
    ids = [p["id"] for p in settings["policies"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Policy ids must be unique")
    for policy in settings["policies"]:
        if policy["family"] not in ("relevance", "fixed", "exponential", "linear", "diverse_tail"):
            raise ValueError(f"Unsupported policy family: {policy['family']}")
    return settings


def load_tower(tower_folder, trained_run):
    """Rebuild the saved two-tower model and its data view of this trained run."""
    import torch
    saved = json.loads((tower_folder / "two_tower_settings.json").read_text())
    if Path(saved["trained_run"]).resolve() != Path(trained_run).resolve():
        raise ValueError(f"The two-tower filter was trained on {saved['trained_run']}, not {trained_run}")
    data = RunData(trained_run, saved["settings"])
    model = build_model(data, saved["settings"])
    model.load_state_dict(torch.load(tower_folder / "two_tower.pt"))
    model.eval()
    return data, model


def end_of_training_popularity(db):
    """The click/view/purchase counts LightGBM features use for validation queries."""
    counters = {kind: Counter() for kind in ("click", "view", "purchase")}
    for kind, item, count in db.execute("""SELECT kind,item_id,count(*) FROM events
            WHERE kind IN ('click','view','purchase') AND session_key IN
            (SELECT session_key FROM queries WHERE split='train') GROUP BY kind,item_id"""):
        counters[kind][item] += count
    return counters


def validation_inputs(db):
    """Query rows, session events and click/purchase labels for validation queries."""
    queries = db.execute("""SELECT * FROM queries WHERE split='validation'
                            ORDER BY eventdate,session_key,timeframe,query_id""").fetchall()
    sessions = sorted({q[1] for q in queries})
    marks = ",".join("?" for _ in sessions)
    events = defaultdict(list)
    for row in db.execute(f"""SELECT session_key,timeframe,kind,item_id,query_id FROM events
            WHERE session_key IN ({marks}) ORDER BY timeframe,event_id""", sessions):
        events[row[0]].append(row[1:])
    clicks = defaultdict(set)
    for qid, item in db.execute("""SELECT e.query_id,e.item_id FROM events e JOIN queries q ON q.query_id=e.query_id
            WHERE q.split='validation' AND e.kind='click' AND e.timeframe>q.timeframe"""):
        clicks[qid].add(item)
    return queries, events, clicks


def list_metrics(top, clicked, logged_set, distances, k):
    """Click NDCG/recall against all clicks, ILD of the list, and its unshown share."""
    metrics = dict(click_ndcg=_click_ndcg(top, clicked, k),
                   click_recall=len(clicked & set(top.tolist())) / len(clicked) if clicked else np.nan,
                   unshown_share=float(np.mean([item not in logged_set for item in top])) if len(top) else np.nan)
    pairs = np.triu_indices(len(top), 1)
    for name, matrix in distances.items():
        values = matrix[pairs]
        known = values[np.isfinite(values)]
        metrics[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
    return metrics


def measure_pipeline(trained_run, tower_folder, settings):
    trained_run, tower_folder = Path(trained_run), Path(tower_folder)
    config = json.loads((trained_run / "expanded_run.json").read_text())["config"]
    k = settings["top_k"]
    print("[1/3] Load the two-tower filter, LightGBM models and validation sessions", flush=True)
    data, tower = load_tower(tower_folder, trained_run)
    items_vectors = item_embeddings(tower, data)
    products = pd.read_parquet(trained_run / "products.parquet")
    catalog = {int(r.item_id): (r.price_log2, None if pd.isna(r.category_id) else int(r.category_id),
                                tokens(r.name_tokens)) for r in products.itertuples(index=False)}
    diversity = ProductDiversity(products)
    boosters = {}
    for model in settings["models"]:
        details = json.loads((trained_run / f"{model}_trials.json").read_text())["result"]
        boosters[model] = (lgb.Booster(model_file=str(trained_run / f"{model}.txt")), details["feature_indices"])
    with sqlite3.connect(trained_run / "development.sqlite") as db:
        popularity = end_of_training_popularity(db)
        queries, events, clicks = validation_inputs(db)
    history_limit = config.get("reranking", {}).get("history_max_events", 100)
    query_table = pd.read_parquet(trained_run / "validation_features" / "queries.parquet")
    context = weighted_history_context(query_table, read_histories(trained_run, "validation"), diversity,
                                       "none", metric="combined", minimum_items=2).set_index("query_id")
    import torch
    with torch.no_grad():
        query_vectors = tower.queries(data.query_row.loc[[q[0] for q in queries]].to_numpy()).numpy()
    max_k1 = max(settings["k1_values"])

    print(f"[2/3] Filter, score and rerank {len(queries):,} validation queries", flush=True)
    rows = []
    for number, (query, vector) in enumerate(zip(queries, query_vectors), start=1):
        qid, session, _, _, _, _, category, item_string = query
        logged = [int(i) for i in item_string.split(",") if i]
        logged_set = set(logged)
        clicked = clicks.get(qid, set())
        pool = data.pool(int(category) if category is not None else TEXT_SEARCH)
        tower_rank = data.item_ids[pool[np.argsort(-(items_vectors[pool] @ vector), kind="stable")[:max_k1]]]
        popular_rank = data.item_ids[pool[np.argsort(-data.popularity[pool], kind="stable")[:max_k1]]]
        union = list(dict.fromkeys(logged + tower_rank.tolist() + popular_rank.tolist()))
        position = {item: index for index, item in enumerate(union)}
        candidate_query = query[:7] + (",".join(map(str, union)),)
        _, features, _, _ = query_features(candidate_query, events[session], catalog, popularity, history_limit)
        for item in union:
            if item not in diversity.name_tokens:
                diversity.name_tokens[item], diversity.categories[item] = frozenset(), None
        distances = diversity.distances(union)
        pools = {("logged", 0): logged}  # k1 = 0 means the whole logged list, whatever its length.
        for k1 in settings["k1_values"]:
            pools[("two_tower", k1)] = tower_rank[:k1].tolist()
            pools[("popularity", k1)] = popular_rank[:k1].tolist()
            # Fair-judgement variants: the same filter, keeping only products the
            # site showed (unjudged products are dropped before the top 10 is cut).
            pools[("two_tower_shown", k1)] = [i for i in tower_rank[:k1].tolist() if i in logged_set]
            pools[("popularity_shown", k1)] = [i for i in popular_rank[:k1].tolist() if i in logged_set]
        history_ild = float(context.loc[qid, "history_ild"]) if qid in context.index else np.nan
        for model, (booster, indices) in boosters.items():
            scores = booster.predict(features[:, indices], num_threads=config["threads"])
            for (pool_name, k1), members in pools.items():
                if not members:
                    continue
                index = np.array([position[item] for item in members])
                reward = np.nan_to_num(distances["combined"][np.ix_(index, index)], nan=0.)
                for policy in settings["policies"]:
                    order = expanded_order(scores[index], reward, policy, history_ild, k)
                    top = np.asarray(members)[order]
                    sub = {name: matrix[np.ix_(index[order], index[order])] for name, matrix in distances.items()}
                    rows.append(dict(
                        query_id=qid, session_key=session, query_type="text" if category in (None, TEXT_SEARCH) else "category",
                        pool=pool_name, k1=k1, model=model, policy_id=policy["id"], pool_size=len(members),
                        top_items=json.dumps(top.tolist()),
                        alpha=relevance_weight(policy, history_ild), clicked=len(clicked),
                        pool_click_coverage=len(clicked & set(members)) / len(clicked) if clicked else np.nan,
                        **list_metrics(top, clicked, logged_set, sub, k),
                    ))
        if number % 100 == 0 or number == len(queries):
            print(f"Pipeline: {number:,}/{len(queries):,} queries", flush=True)
    return pd.DataFrame(rows)


METRICS = ["click_ndcg", "click_recall", "combined_ild", "token_ild", "category_ild", "unshown_share",
           "pool_click_coverage", "pool_size"]


def summarize(measured):
    keys = ["model", "pool", "k1", "policy_id"]
    means = measured.groupby(keys, sort=False)[METRICS].mean()
    counts = measured.groupby(keys, sort=False).agg(queries=("query_id", "size"), click_queries=("click_ndcg", "count"))
    return means.join(counts).reset_index()


def differences_vs_logged(measured, replicates, seed):
    """Each filtered pool minus the logged pool, same model and policy; session bootstrap."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model, policy), frame in measured.groupby(["model", "policy_id"], sort=False):
        base = frame[frame.pool.eq("logged")].set_index("query_id")
        sessions = base.session_key.unique()
        draws = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions)), size=replicates)
        for (pool, k1), other in frame[~frame.pool.eq("logged")].groupby(["pool", "k1"], sort=False):
            other = other.set_index("query_id").reindex(base.index)
            for metric in ("click_ndcg", "combined_ild", "token_ild"):
                diff = (other[metric] - base[metric]).dropna()
                by_session = diff.groupby(base.loc[diff.index, "session_key"]).agg(["sum", "count"]).reindex(sessions).fillna(0)
                boot = (draws @ by_session["sum"].to_numpy()) / (draws @ by_session["count"].to_numpy())
                rows.append(dict(model=model, policy_id=policy, pool=pool, k1=k1, metric=metric,
                                 difference=float(diff.mean()), ci_low=float(np.quantile(boot, .025)),
                                 ci_high=float(np.quantile(boot, .975)), queries=len(diff)))
    return pd.DataFrame(rows)


def plot_tradeoff(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(summary.model.unique())
    figure, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    markers = {"relevance": "o", "fixed_0.8": "s", "exponential_1": "^", "six_four": "D"}
    filtered = sorted(k for k in summary.k1.unique() if k > 0)
    shown_k1 = 100 if 100 in filtered else filtered[len(filtered) // 2]
    for axis, model in zip(axes.flat, models):
        # One K1 keeps the plot readable; every K1 is in summary.csv.
        data = summary[summary.model.eq(model) & summary.k1.isin([0, shown_k1])]
        for (pool, k1), frame in data.groupby(["pool", "k1"], sort=False):
            label = "logged list" if pool == "logged" else f"{pool} K1={k1}"
            line = axis.plot(frame.combined_ild, frame.click_ndcg, "-", alpha=.4)[0]
            for _, point in frame.iterrows():
                axis.scatter(point.combined_ild, point.click_ndcg, color=line.get_color(),
                             marker=markers.get(point.policy_id, "o"), label=label if point.policy_id == frame.policy_id.iloc[0] else None)
        axis.set_title(f"{model}: filter -> LightGBM -> rerank")
        axis.set_xlabel("combined ILD@10")
        axis.set_ylabel("click NDCG@10")
        axis.grid(alpha=.3)
        axis.legend(fontsize=7)
    figure.text(.01, .01, "markers: o relevance, s fixed 0.8, ^ exponential, D six-four", fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "ndcg_vs_ild.png", dpi=150)
    plt.close(figure)


def run_filter_pipeline(trained_run, tower_folder, settings_path=Path("config/filter_pipeline.json"), output_root=None):
    settings = read_pipeline_settings(settings_path)
    trained_run, tower_folder = Path(trained_run), Path(tower_folder)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{trained_run.name}"
    folder.mkdir(parents=True)
    write_json(folder / "pipeline_settings.json", dict(trained_run=str(trained_run), two_tower=str(tower_folder),
                                                       settings=settings))
    measured = measure_pipeline(trained_run, tower_folder, settings)
    print("[3/3] Summarize and save", flush=True)
    summary = summarize(measured)
    save_table(measured, folder / "validation_pipeline.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(differences_vs_logged(measured, settings["bootstrap_replicates"], settings["seed"]),
               folder / "difference_vs_logged.csv")
    plot_tradeoff(summary, folder)
    print(f"Filter pipeline saved to {folder}", flush=True)
    return folder
