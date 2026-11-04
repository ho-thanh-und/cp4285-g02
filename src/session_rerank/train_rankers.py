"""Fit and compare two LambdaRank models on the anonymous session pilot."""

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd

from .ranker_features import PRODUCT_FEATURES, SESSION_FEATURES, build_ranker_features

K = 10
MODEL_SETTINGS = dict(
    objective="lambdarank", n_estimators=500, learning_rate=0.05,
    lambdarank_truncation_level=13, random_state=4285, deterministic=True,
    force_col_wise=True, n_jobs=4, verbosity=-1,
)
EARLY_STOPPING = 30
GRID = tuple(product((15, 31), (100, 300)))


def fit_ranker(train, validation, features):
    """Try the same four settings for each model and keep the best validation fit."""
    # LightGBM takes group sizes in row order. Each group is one complete query list.
    train_groups = train.groupby("query_id", sort=False).size().tolist()
    validation_groups = validation.groupby("query_id", sort=False).size().tolist()
    best = None
    trials = []
    for leaves, min_rows in GRID:
        model = lgb.LGBMRanker(**MODEL_SETTINGS, num_leaves=leaves, min_child_samples=min_rows)
        model.fit(
            train[features], train.clicked.astype(int), group=train_groups,
            eval_X=validation[features], eval_y=validation.clicked.astype(int),
            eval_group=[validation_groups], eval_at=[K],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        ndcg = model.best_score_["valid_0"][f"ndcg@{K}"]
        trials.append(dict(num_leaves=leaves, min_child_samples=min_rows,
                           best_iteration=model.best_iteration_, lightgbm_ndcg_at_10=ndcg))
        if best is None or ndcg > best.best_score_["valid_0"][f"ndcg@{K}"]:
            best = model
    return best, trials


def evaluate_validation(predictions, score_columns):
    """Average query-level ranking metrics over the same labelled queries for each model."""
    per_query = []
    discounts = 1 / np.log2(np.arange(K) + 2)
    for query_id, candidates in predictions.groupby("query_id", sort=False):
        clicked = candidates.clicked.to_numpy(dtype=bool)
        positive_count = int(clicked.sum())
        if not positive_count:
            continue
        for model, column in score_columns.items():
            # Candidates are in original rank order, so equal scores keep that order.
            order = np.argsort(-candidates[column].to_numpy(), kind="stable")[:K]
            hits = clicked[order]
            ndcg = np.dot(hits, discounts[:len(hits)]) / discounts[:min(positive_count, K)].sum()
            per_query.append(dict(
                query_id=query_id, session_key=candidates.session_key.iloc[0], model=model,
                has_history=candidates.has_history.iloc[0],
                ndcg_at_10=ndcg, recall_at_10=hits.sum() / positive_count,
            ))
    metrics = pd.DataFrame(per_query)
    summary = pd.concat([
        metrics.assign(segment="all"),
        metrics.assign(segment=np.where(metrics.has_history, "with_history", "without_history")),
    ]).groupby(["segment", "model"], sort=False, as_index=False).agg(
        queries=("query_id", "size"), ndcg_at_10=("ndcg_at_10", "mean"),
        recall_at_10=("recall_at_10", "mean"),
    )
    return metrics, summary


def train_rankers(dataset_folder, output_root=Path("results/anonymous_rankers")):
    """Save the two fitted rankers, validation comparison and candidate scores."""
    print("[2/6] Build features and train the anonymous LightGBM rankers", flush=True)
    rows = build_ranker_features(dataset_folder)
    train = rows[rows.split.eq("train")]
    validation = rows[rows.split.eq("validation")]
    # With no observed click there is no positive candidate to rank.
    fit_train = train[train.groupby("query_id").clicked.transform("any")]
    fit_validation = validation[validation.groupby("query_id").clicked.transform("any")]
    if fit_train.empty or fit_validation.empty:
        raise ValueError("Training and validation each need queries with a recorded candidate click.")

    columns = ["query_id", "session_key", "split", "item_id", "original_rank", "clicked"]
    predictions = validation[columns].copy()
    predictions["has_history"] = validation.session_history_missing.eq(0)
    predictions["original_score"] = -predictions.original_rank
    score_columns = {"original_order": "original_score"}
    models = {
        "query_product": PRODUCT_FEATURES,
        "query_product_session": PRODUCT_FEATURES + SESSION_FEATURES,
    }
    manifest = dict(
        sample_output_sha256=json.loads((dataset_folder / "manifest.json").read_text())["output_sha256"],
        lightgbm_version=lgb.__version__, cutoff=K, settings=MODEL_SETTINGS,
        early_stopping_rounds=EARLY_STOPPING,
        train_queries_used=int(fit_train.query_id.nunique()),
        train_queries_without_click=int(train.query_id.nunique() - fit_train.query_id.nunique()),
        validation_queries_without_click=int(validation.query_id.nunique() - fit_validation.query_id.nunique()),
        models={},
    )

    # Save each run separately and publish it only after all its files are ready.
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{stamp}-{uuid4().hex[:8]}"
    with TemporaryDirectory(dir=output_root) as temporary:
        folder = Path(temporary)
        for name, features in models.items():
            print(f"Fit {name} ({len(GRID)} settings)", flush=True)
            model, trials = fit_ranker(fit_train, fit_validation, features)
            column = f"{name}_score"
            predictions[column] = model.predict(validation[features], num_iteration=model.best_iteration_)
            score_columns[name] = column
            model.booster_.save_model(str(folder / f"{name}.txt"), num_iteration=model.best_iteration_)
            pd.DataFrame({
                "feature": features, "gain": model.booster_.feature_importance("gain"),
            }).sort_values("gain", ascending=False).to_csv(folder / f"{name}_feature_gain.csv", index=False)
            manifest["models"][name] = dict(
                features=features, best_iteration=model.best_iteration_, trials=trials,
                num_leaves=model.get_params()["num_leaves"],
                min_child_samples=model.get_params()["min_child_samples"],
            )
        per_query, summary = evaluate_validation(predictions, score_columns)
        predictions.to_parquet(folder / "validation_scores.parquet", index=False)
        per_query.to_csv(folder / "validation_by_query.csv", index=False)
        summary.to_csv(folder / "validation_metrics.csv", index=False)
        (folder / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)
    print(summary[summary.segment.eq("all")].to_string(index=False), flush=True)
    print(f"Saved rankers and validation scores: {output.resolve()}", flush=True)
    return output
