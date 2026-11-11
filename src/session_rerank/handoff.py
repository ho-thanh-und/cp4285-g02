"""Export the frozen models for the website, then prove the export reproduces the final test.

`uv run main.py handoff` (config/handoff.json) writes one folder:

  catalog.parquet                    item_id, category_id, price_log2, name_tokens
  home/model.pt, settings.json       SASRec weights and settings, the 6 + 4 page settings
  home/vocabulary_item_ids.npy       products SASRec knows (row i = SASRec id i + 1)
  home/fallback_item_ids.npy         SASRec's popularity fallback (February–April clicks)
  home/popular_item_ids.npy          cold start before the first click (April clicks)
  item/you_may_also_like.parquet     item_id, r1..r10 for every catalog product (-1 = none)
  category/model.txt, settings.json  frozen LightGBM booster, feature columns
  category/product_stats.parquet     frozen popularity inputs of the category ranker
  manifest.json                      sources, policies, code commit, file hashes
  verification.json                  the checks below

Verification uses the saved confirmation-week runs (results/confirmation):
  home page      every checked case: the served list equals the saved SASRec + 6 + 4 list
  item page      every page: NDCG@10 of the served list equals the saved value
  category page  every category query: NDCG@10 of the served order equals the saved value
The export fails if any check fails.
"""

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .study_files import write_json

K_COLUMNS = [f"r{i}" for i in range(1, 11)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def top_ids(counts, catalog_ids, count):
    """Most counted products first, ties in catalog order (the evaluation's stable argsort)."""
    return catalog_ids[np.argsort(-counts, kind="stable")[:count]].astype(np.int64)


# ----------------------------------------------------------------- export ---

def export_home(settings, catalog_ids, item_index, folder):
    from .home_page import click_sequences, read_home_settings
    from .ranker_phase2 import window_activity
    from .sequence_models import read_sequence_settings
    home = read_home_settings(settings["home_config"])
    sequence, _ = read_sequence_settings(settings["sequence_config"])
    if len(sequence["models"]) != 1 or not sequence.get("frozen_policy"):
        raise ValueError("sequence_config must name one model and its frozen_policy")
    kind, policy = sequence["models"][0], sequence["frozen_policy"]
    frozen = Path(sequence["frozen_run"])
    saved = json.loads((frozen / "sequence_settings.json").read_text())
    model_settings = saved["settings"]
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(frozen / f"{kind}.pt", folder / "model.pt")
    vocab_rows = np.load(frozen / "vocabulary_rows.npy")
    np.save(folder / "vocabulary_item_ids.npy", catalog_ids[vocab_rows].astype(np.int64))
    # SASRec's own fallback, built as in run_sequence_models: sessions clicking each product.
    window = model_settings["training_window"]
    sequences = click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], item_index)
    clicks = np.zeros(len(catalog_ids))
    for _, _, items in sequences:
        np.add.at(clicks, items, 1)
    np.save(folder / "fallback_item_ids.npy", top_ids(clicks, catalog_ids, 2000))
    # Cold start, as HomeData.global_order: April clicks of the home page's training window.
    start = pd.Timestamp(home["training_window"]["start"])
    days = int((pd.Timestamp(home["training_window"]["end_exclusive"]) - start).days)
    april, _ = window_activity(home["raw_directory"], item_index, start, days)
    np.save(folder / "popular_item_ids.npy", top_ids(april.sum(0).astype(float), catalog_ids, 2000))
    page = {key: home[key] for key in ("top_k", "relevance_slots", "pool_size", "fixed_alphas")}
    write_json(folder / "settings.json", dict(kind=kind, policy=policy, page=page, model=model_settings,
                                              model_run=frozen.as_posix(), best_epoch=saved["training"].get(
                                                  kind, {}).get("best_epoch")))
    return dict(policy=policy, model_run=frozen.as_posix(), cold_start="popularity: April clicks")


def export_item(settings, catalog_ids, folder, progress=20000):
    from .item_knn import read_knn_settings
    from .ymal import YmalData, page_lists, read_ymal_settings
    ymal = read_ymal_settings(settings["ymal_config"])
    policy = ymal["frozen_policy"]  # e.g. fixed_0.8_20: fixed alpha 0.8 over a pool of 20
    if policy == "category_popularity":
        pool, alphas = ymal["pool_sizes"][0], ymal["fixed_alphas"]
    else:
        pool = int(policy.rsplit("_", 1)[1])
        alphas = [float(policy.split("_")[1])] if policy.startswith("fixed_") else ymal["fixed_alphas"]
    # Only the frozen pool size and alpha are needed; the frozen list does not depend on the others.
    reduced = ymal | dict(pool_sizes=[pool], fixed_alphas=alphas)
    data = YmalData(reduced, read_knn_settings(ymal["knn_config"]))
    if not np.array_equal(data.catalog_ids, catalog_ids):
        raise ValueError("The item page uses another catalog")
    table = np.full((len(catalog_ids), ymal["top_k"]), -1, dtype=np.int64)
    for row in range(len(catalog_ids)):
        top = page_lists(data, reduced, row)[policy][0]
        table[row, :len(top)] = catalog_ids[top]
        if (row + 1) % progress == 0:
            print(f"[handoff] 'You may also like' {row + 1:,}/{len(catalog_ids):,}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(table, columns=K_COLUMNS[:ymal["top_k"]])
    frame.insert(0, "item_id", catalog_ids.astype(np.int64))
    frame.to_parquet(folder / "you_may_also_like.parquet", index=False)
    return dict(policy=policy, products=int(len(catalog_ids)))


def category_inputs(settings):
    """The frozen ranker's popularity inputs, built exactly as Phase2Data and split_features build them."""
    from .event_ablation import RunInputs, filter_events, popularity_snapshots
    from .heldout import read_heldout_settings
    from .ranker_phase2 import read_phase2_settings, window_activity
    from .two_tower import popularity_features
    heldout = read_heldout_settings(settings["heldout_config"])
    phase2 = read_phase2_settings(heldout["phase2_config"])
    inputs = RunInputs(heldout["trained_run"])
    products = pd.read_parquet(Path(heldout["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    train_events = filter_events(inputs.events, "clicks_only", 60_000)
    _, counters = popularity_snapshots(train_events, inputs.train_sessions, inputs.training_dates)
    train_dates = pd.to_datetime(pd.Series([q[3] for q in inputs.queries["train"]]))
    start, days = train_dates.min(), int((train_dates.max() - train_dates.min()).days) + 1
    clicks, purchases = window_activity(phase2["raw_directory"], item_index, start, days)
    rates = np.ascontiguousarray(popularity_features(clicks, purchases)[..., :2])
    return heldout, phase2, inputs, counters, rates[days]


def export_category(settings, catalog_ids, folder):
    from .event_ablation import MODELS
    from .expanded_data import FEATURES, PURCHASE_FEATURES
    from .heldout import POPULARITY_COLUMNS, frozen_models
    from .ranker_phase2 import ALL_FEATURES
    heldout, phase2, inputs, counters, rates = category_inputs(settings)
    selection = Path(settings["heldout_selection_run"])
    chosen = json.loads((selection / "selection_choice.json").read_text())["chosen"]
    variant, models = frozen_models(heldout["phase2_run"], phase2)
    indices = list(MODELS[chosen][1]) + (POPULARITY_COLUMNS if variant["popularity"] else [])
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selection / f"{chosen}.txt", folder / "model.txt")
    stats = pd.DataFrame(dict(item_id=catalog_ids.astype(np.int64)))
    for kind in ("click", "view", "purchase"):
        stats[f"prior_{kind}s"] = [counters[kind][int(item)] for item in catalog_ids]
    stats["window_click_rate"], stats["window_purchase_rate"] = rates[:, 0], rates[:, 1]
    stats.to_parquet(folder / "product_stats.parquet", index=False)
    write_json(folder / "settings.json", dict(
        model=f"lightgbm_{chosen}", chosen_on=selection.as_posix(), variant=variant, trees=models[chosen],
        history_limit=int(inputs.history_limit), base_features=len(FEATURES) + len(PURCHASE_FEATURES),
        feature_indices=indices, feature_names=[ALL_FEATURES[i] for i in indices],
        event_rule="clicks only (views ignored), as in training"))
    return dict(model=f"lightgbm_{chosen}", chosen_on=selection.as_posix())


# ----------------------------------------------------------------- verify ---

def verify_home(recommender, confirmation, runs, every):
    from .home_page import click_sequences, evaluation_cases, read_home_settings
    home = read_home_settings(confirmation["settings"]["home_config"])
    window = home["window"]
    week = click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], recommender.item_index)
    cases, _ = evaluation_cases(week, home["max_cases"], home["seed"])
    shown = pd.read_parquet(Path(runs["home_sequence"]) / "lists.parquet")
    shown = shown[shown.method.eq(recommender.home_policy)].sort_values(["case", "position"])
    saved = {case: group.item_id.tolist() for case, group in shown.groupby("case")}
    checked = list(range(0, len(cases), every))
    different = [case for case in checked
                 if recommender.home(recommender.catalog_ids[cases[case][1]].tolist())["items"] != saved[case]]
    return dict(cases=len(checked), of=len(cases), identical=len(checked) - len(different), different=different[:20])


def verify_item(recommender, confirmation, runs):
    from .exposure import list_ndcg
    from .ymal import item_pages, read_ymal_settings
    ymal = read_ymal_settings(confirmation["settings"]["ymal_config"])
    window = ymal["window"]
    pages, _ = item_pages(ymal["raw_directory"], window["start"], window["end_exclusive"], recommender.item_index,
                          ymal["max_seeds"], ymal["seed"])
    saved = pd.read_parquet(Path(runs["item"]) / "per_page.parquet")
    saved = saved[saved.method.eq(ymal["frozen_policy"])].sort_values("page").ndcg.to_numpy()
    measured = np.array([list_ndcg(recommender.item(recommender.catalog_ids[seed])["items"],
                                   set(recommender.catalog_ids[later].tolist())) for _, seed, later in pages])
    same = np.isclose(measured, saved)
    return dict(pages=len(pages), identical_ndcg=int(same.sum()), mean_ndcg=float(measured.mean()),
                saved_mean_ndcg=float(saved.mean()))


def verify_category(recommender, runs, settings):
    from .event_ablation import per_query_metrics
    from .ranking_settings import top_k
    run = Path(runs["category"])
    config = json.loads((Path(settings["trained_run"]) / "expanded_run.json").read_text())["config"]
    candidates = pd.read_parquet(run / "confirmation_candidates.parquet")
    with sqlite3.connect(run / "confirmation.sqlite") as db:
        queries = pd.read_sql("SELECT query_id, session_key, timeframe, category FROM queries", db)
        events = pd.read_sql("""SELECT session_key, kind, timeframe, query_id, item_id FROM events
                                WHERE kind IN ('click', 'purchase') ORDER BY timeframe, event_id""", db)
    category = queries[queries.category.fillna(0).ne(0)].set_index("query_id")
    by_session = {key: group for key, group in events.groupby("session_key")}
    scores = np.full(len(candidates), np.nan)
    for qid, group in candidates[candidates.query_id.isin(category.index)].groupby("query_id", sort=False):
        query = category.loc[qid]
        history = by_session.get(query.session_key, events.iloc[:0])
        history = history[(history.timeframe < query.timeframe) & ~history.query_id.eq(qid)]
        clicks = [dict(item_id=int(r.item_id), time_ms=int(r.timeframe), kind=r.kind) for r in history.itertuples()]
        _, values = recommender.category_scores(int(query.category), group.item_id.tolist(), clicks,
                                                int(query.timeframe))
        scores[group.index] = values
    mask = candidates.query_id.isin(category.index).to_numpy()
    measured = per_query_metrics(candidates[mask], scores[mask], config["label_gains"], top_k(config))
    saved = pd.read_parquet(run / "confirmation_per_query.parquet")
    saved = saved[saved.method.eq(recommender.category_settings["model"])].set_index("query_id").click_ndcg
    measured = measured.set_index("query_id").click_ndcg.reindex(saved.index[saved.index.isin(category.index)])
    expected = saved.reindex(measured.index)
    same = np.isclose(measured.to_numpy(), expected.to_numpy(), equal_nan=True)
    return dict(queries=int(len(measured)), identical_ndcg=int(same.sum()),
                mean_ndcg=float(np.nanmean(measured)), saved_mean_ndcg=float(np.nanmean(expected)))


# -------------------------------------------------------------------- run ---

def run_handoff(settings_path=Path("config/handoff.json"), output_root=None):
    from .heldout import git_state
    from .serving import Recommender
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    if settings.get("reuse_export"):
        # Check an earlier export again (e.g. after fixing a check) without rebuilding it.
        folder = Path(settings["reuse_export"])
        print(f"[handoff] Re-verifying {folder}", flush=True)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = Path(output_root or settings["output_directory"]) / stamp
        folder.mkdir(parents=True)
        products = pd.read_parquet(Path(settings["trained_run"]) / "products.parquet")
        catalog_ids = products.item_id.to_numpy()
        item_index = pd.Series(np.arange(len(products)), index=catalog_ids)
        products.to_parquet(folder / "catalog.parquet", index=False)
        print("[handoff] Home page: SASRec, fallback and cold-start popularity", flush=True)
        pages = dict(home=export_home(settings, catalog_ids, item_index, folder / "home"))
        print("[handoff] Category pages: LightGBM and its frozen product inputs", flush=True)
        pages["category"] = export_category(settings, catalog_ids, folder / "category")
        print("[handoff] Item page: 'You may also like' for every product", flush=True)
        pages["item"] = export_item(settings, catalog_ids, folder / "item")
        files = {path.relative_to(folder).as_posix(): dict(bytes=path.stat().st_size, sha256=sha256(path))
                 for path in sorted(folder.rglob("*")) if path.is_file()}
        write_json(folder / "manifest.json", dict(
            created_utc=datetime.now(timezone.utc).isoformat(), code=git_state(Path.cwd()), settings=settings,
            catalog_products=int(len(catalog_ids)), pages=pages, files=files))

    print("[handoff] Verify against the saved confirmation-week runs", flush=True)
    confirmation = Path(settings["verify"]["confirmation_run"])
    frozen = json.loads((confirmation / "frozen_choices.json").read_text())
    runs = json.loads((confirmation / "runs.json").read_text())
    recommender = Recommender(folder)
    checks = dict(home=verify_home(recommender, frozen, runs, settings["verify"]["home_every_nth_case"]),
                  item=verify_item(recommender, frozen, runs),
                  category=verify_category(recommender, runs, settings))
    passed = (checks["home"]["identical"] == checks["home"]["cases"]
              and checks["item"]["identical_ndcg"] == checks["item"]["pages"]
              and checks["category"]["identical_ndcg"] == checks["category"]["queries"])
    write_json(folder / "verification.json", dict(passed=passed, confirmation_run=confirmation.as_posix(), **checks))
    print(json.dumps(checks, indent=2), flush=True)
    if not passed:
        raise ValueError(f"The hand-off does not reproduce the confirmation week; see {folder / 'verification.json'}")
    print(f"[handoff] Saved and verified: {folder}", flush=True)
    return folder
