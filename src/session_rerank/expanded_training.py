"""Train the full study from disk-backed features and save every tuning trial."""

import gc
import json
from itertools import product
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from .adaptive_policies import ndcg, relevance_metrics
from .ranking_settings import top_k, reranking_settings
from .expanded_data import FEATURES, PURCHASE_FEATURES, read_candidates
from .study_files import write_json, save_table, stage_complete, finish_stage

GRID = ((15, 100), (15, 300), (31, 100), (31, 300))


DEFAULT_TRAINING = dict(
    learning_rate=.05, num_boost_round=500, early_stopping_rounds=30,
    log_evaluation_period=25, num_leaves=[15, 31], min_data_in_leaf=[100, 300],
    max_depth=-1, max_bin=255, lambda_l1=0., lambda_l2=0.,
    feature_fraction=1., bagging_fraction=1., bagging_freq=0,
    lambdarank_truncation_level=13,
)


def training_settings(config):
    """Validate human-editable hyperparameters before preparing any large data.

    num_leaves and min_data_in_leaf are lists: their Cartesian product is the
    tuning grid. Scalar settings apply to every trial. To compare a different
    grid or learning rate, copy the config and give it a different output folder.
    """
    reranking_settings(config)
    supplied = config.get("training", {})
    unknown = set(supplied) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    settings = DEFAULT_TRAINING | supplied
    for name, minimum in (("num_leaves", 2), ("min_data_in_leaf", 1)):
        values = settings[name]
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < minimum for value in values
        ):
            raise ValueError(f"training.{name} must be a nonempty list of integers >= {minimum}")
    for name, minimum in (("num_boost_round", 1), ("early_stopping_rounds", 1),
                          ("log_evaluation_period", 0), ("max_bin", 2),
                          ("bagging_freq", 0), ("lambdarank_truncation_level", 1)):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"training.{name} must be an integer >= {minimum}")
    depth = settings["max_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or (depth != -1 and depth < 1):
        raise ValueError("training.max_depth must be -1 (unlimited) or a positive integer")
    for name in ("learning_rate", "feature_fraction", "bagging_fraction"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"training.{name} must be in (0,1]")
    for name in ("lambda_l1", "lambda_l2"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} must be finite and nonnegative")
    if isinstance(config["threads"], bool) or not isinstance(config["threads"], int) or config["threads"] < 1:
        raise ValueError("threads must be a positive integer")
    return settings


def determine_primary(config, audit):
    """Use a declared support rule rather than choosing a metric after seeing scores."""
    declared = config.get("primary_metric", "auto")
    if declared in ("click_ndcg", "gain_ndcg"):
        return declared
    if declared != "auto":
        raise ValueError("primary_metric must be auto, click_ndcg, or gain_ndcg")
    train = audit["purchases"]["train"]["purchase_queries"]
    validation = audit["purchases"]["validation"]["purchase_queries"]
    supported = train >= config["purchase_primary_min_train_queries"] and validation >= config["purchase_primary_min_validation_queries"]
    return "gain_ndcg" if supported else "click_ndcg"


def validation_metrics(scores, prediction, label_gains, k=10):
    """Compute both objectives on identical query groups and report their denominators."""
    frame = scores.copy()
    frame["score"] = prediction
    rows = []
    for _, group in frame.groupby("query_id", sort=False):
        order = np.argsort(-group.score.to_numpy(), kind="stable")[:k]
        rows.append(relevance_metrics(order, group.clicked, group.purchased, label_gains, k))
    measured = pd.DataFrame(rows)
    return {name: float(measured[name].mean()) if measured[name].notna().any() else None
            for name in measured} | {name + "_queries": int(measured[name].count()) for name in measured}


def build_lightgbm_data(folder, split, objective, feature_indices, reference=None, *, config=None):
    """Construct a binned LightGBM Dataset, releasing its float matrix afterwards.

    All-zero query groups are excluded. Boolean row selection is a temporary
    bounded full-matrix allocation on this machine, while the permanent feature
    cache remains memory-mapped and the LightGBM representation is histogram bins.
    """
    metadata = pd.read_parquet(folder / f"{split}_features" / "queries.parquet")
    valid_queries = metadata.clicks.gt(0) if objective == "click" else (metadata.clicks + metadata.purchases).gt(0)
    if not valid_queries.any():
        raise ValueError(f"No positive {objective} query groups in {split}; inspect the sample audit.")
    row_mask = np.repeat(valid_queries.to_numpy(), metadata.candidates.to_numpy())
    matrix = np.load(folder / f"{split}_features" / "X.npy", mmap_mode="r")
    label_file = "click_labels.npy" if objective == "click" else "grades.npy"
    labels = np.load(folder / f"{split}_features" / label_file, mmap_mode="r")
    selected = np.ascontiguousarray(matrix[np.ix_(np.flatnonzero(row_mask), feature_indices)], dtype=np.float32)
    settings = training_settings(config) if config is not None else DEFAULT_TRAINING
    dataset = lgb.Dataset(
        selected, label=labels[row_mask], group=metadata.loc[valid_queries, "candidates"].tolist(),
        feature_name=[(FEATURES + PURCHASE_FEATURES)[i] for i in feature_indices], categorical_feature=[1],
        reference=reference, free_raw_data=True,
        params=dict(feature_pre_filter=False,
                    data_random_seed=config["seed"] if config is not None else 4285,
                    num_threads=config["threads"] if config is not None else 4,
                    max_bin=settings["max_bin"]),
    )
    dataset.construct()
    del selected, matrix
    gc.collect()
    return dataset


def train_expanded_models(config, folder):
    """Fit click and purchase-aware rankers with matching feature variants and grids."""
    if stage_complete(folder, "training"):
        return
    settings = training_settings(config)
    k = top_k(config)
    grid = list(product(settings["num_leaves"], settings["min_data_in_leaf"])) if "training" in config else GRID
    stage_label = "[3/4]" if config.get("validation_only", False) else "[3/7]"
    train_info = pd.read_parquet(folder / "train_features" / "queries.parquet")
    print(f"{stage_label} Train fresh LightGBM rankers on {len(train_info):,} queries; {len(grid)} grid trials per nominal model", flush=True)
    audit = json.loads((folder / "data_audit.json").read_text())
    primary = determine_primary(config, audit)
    write_json(folder / "objective.json", dict(
        primary=primary, primary_rule=config.get("primary_metric", "auto"), label_gains=config["label_gains"],
        training=settings, top_k=k,
        support_rule=(dict(train=config["purchase_primary_min_train_queries"],
                           validation=config["purchase_primary_min_validation_queries"])
                      if config.get("primary_metric", "auto") == "auto" else None),
        purchase_support=audit["purchases"],
    ))
    validation = read_candidates(folder, "validation")
    validation_X = np.load(folder / "validation_features" / "X.npy", mmap_mode="r")
    records, outputs = [], ["objective.json"]
    variants = [
        ("click_product", "click", list(range(7))),
        ("click_session", "click", list(range(len(FEATURES)))),
        ("purchase_product", "purchase", list(range(7)) + [15]),
        ("purchase_session", "purchase", list(range(len(FEATURES) + len(PURCHASE_FEATURES)))),

    ]
    # Global purchase popularity and earlier purchase counts are explicit additions
    # in the purchase/session model. Click-only variants match the pilot features.
    variants += [(f"purchase_session_gain_{gain}", "purchase", list(range(18)))
                 for gain in config.get("purchase_gain_sensitivity", [2, 5])]
    for name, objective, feature_indices in variants:
        feature_count = len(feature_indices)
        model_path = folder / f"{name}.txt"
        trial_path = folder / f"{name}_trials.json"
        # A completed model can survive interruption of a later model fit.
        checkpoint = f"model_{name}"
        if stage_complete(folder, checkpoint):
            records.append(json.loads(trial_path.read_text())["result"])
            outputs += [model_path.name, trial_path.name, f"{name}_feature_gain.csv", f"{name}_trials.csv"]
            continue
        started = perf_counter()
        print(f"Build datasets for {name} ({feature_count} features)", flush=True)
        train = build_lightgbm_data(folder, "train", objective, feature_indices, config=config)
        valid = build_lightgbm_data(folder, "validation", objective, feature_indices, train, config=config)
        fit_gains = [0, 1] if objective == "click" else config["label_gains"]
        trial_grid = grid
        if "_gain_" in name:
            gain = int(name.rsplit("_", 1)[1])
            fit_gains = [0, 1, gain]
            nominal = json.loads((folder / "purchase_session_trials.json").read_text())["trials"]
            best_nominal = max(nominal, key=lambda trial: trial[primary])
            trial_grid = ((best_nominal["leaves"], best_nominal["min_rows"]),)
        best_score, best_model, trials = -np.inf, None, []
        for trial_number, (leaves, minimum) in enumerate(trial_grid, start=1):
            trial_started = perf_counter()
            print(f"Fit {name}, trial {trial_number}/{len(trial_grid)}: leaves={leaves}, min_rows={minimum}, learning_rate={settings['learning_rate']}", flush=True)
            def custom_metric(prediction, data):
                # Dataset validation contains positive groups for this model's
                # objective. Selection below still uses the common full query set.
                labels = data.get_label().astype(int)
                gains = np.asarray(fit_gains)[labels]
                offsets = np.r_[0, np.cumsum(data.get_group())]
                values = [ndcg(np.argsort(-prediction[a:b], kind="stable")[:k], gains[a:b], k)
                          for a,b in zip(offsets[:-1], offsets[1:])]
                return f"rank_ndcg_at_{k}", float(np.nanmean(values)), True
            model = lgb.train(dict(
                objective="lambdarank", metric="None", label_gain=fit_gains,
                learning_rate=settings["learning_rate"], num_leaves=leaves, min_data_in_leaf=minimum,
                max_depth=settings["max_depth"], max_bin=settings["max_bin"],
                lambda_l1=settings["lambda_l1"], lambda_l2=settings["lambda_l2"],
                feature_fraction=settings["feature_fraction"], bagging_fraction=settings["bagging_fraction"],
                bagging_freq=settings["bagging_freq"],
                lambdarank_truncation_level=settings["lambdarank_truncation_level"], feature_pre_filter=False,
                deterministic=True, force_col_wise=True, seed=config["seed"],
                num_threads=config["threads"], verbosity=-1,
            ), train, num_boost_round=settings["num_boost_round"], valid_sets=[valid], feval=custom_metric,
               callbacks=[lgb.log_evaluation(settings["log_evaluation_period"]),
                          lgb.early_stopping(settings["early_stopping_rounds"], verbose=True)])
            prediction = model.predict(validation_X[:, feature_indices], num_threads=config["threads"])
            measured = validation_metrics(validation, prediction, config["label_gains"], k)
            score = measured[primary]
            if score is None:
                raise ValueError("No eligible validation queries for the primary metric.")
            trials.append(dict(top_k=k, leaves=leaves, min_rows=minimum, iterations=model.best_iteration,
                               seconds=perf_counter()-trial_started, **measured))
            print(f"Completed {name} trial {trial_number}: {primary}={score:.5f}, "
                  f"best_iteration={model.best_iteration}, {perf_counter()-trial_started:.1f}s", flush=True)
            if score > best_score:
                best_score, best_model = score, model
        best_model.save_model(str(model_path), num_iteration=best_model.best_iteration)
        save_table(pd.DataFrame(dict(
            feature=[(FEATURES + PURCHASE_FEATURES)[i] for i in feature_indices],
            gain=best_model.feature_importance("gain"),
        )), folder / f"{name}_feature_gain.csv")
        result = dict(model=name, objective=objective, feature_count=feature_count, feature_indices=feature_indices, fit_gains=fit_gains,
                      primary_score=best_score, seconds=perf_counter()-started)
        write_json(trial_path, dict(result=result, training=settings, trials=trials))
        # A flat CSV makes every attempted hyperparameter setting easy to inspect
        # without reading JSON. The JSON also records the full shared config.
        scalars = {key: value for key, value in settings.items()
                   if key not in ("num_leaves", "min_data_in_leaf")}
        save_table(pd.DataFrame(trials).assign(**scalars), folder / f"{name}_trials.csv")
        finish_stage(folder, checkpoint, [model_path.name, trial_path.name, f"{name}_feature_gain.csv", f"{name}_trials.csv"])
        records.append(result)
        outputs += [model_path.name, trial_path.name, f"{name}_feature_gain.csv", f"{name}_trials.csv"]
        del train, valid, best_model
        gc.collect()
    summary = pd.DataFrame(records)
    save_table(summary, folder / "model_comparison.csv")
    # Choose a session-aware scorer on validation only. A later policy comparison
    # must not change its relevance model between methods.
    session_models = summary[summary.model.str.endswith("_session")]
    selected = session_models.sort_values(["primary_score","model"], ascending=[False,True]).iloc[0]
    write_json(folder / "selected_model.json", dict(
        model=selected.model, feature_count=int(selected.feature_count), feature_indices=selected.feature_indices, primary=primary,
    ))
    finish_stage(folder, "training", outputs + ["model_comparison.csv", "selected_model.json"])


def score_expanded_split(config, folder, split):
    """Apply the selected frozen feature schema to a saved later split."""
    selected = json.loads((folder / "selected_model.json").read_text())
    matrix = np.load(folder / f"{split}_features" / "X.npy", mmap_mode="r")
    model = lgb.Booster(model_file=str(folder / f"{selected['model']}.txt"))
    scores = read_candidates(folder, split)
    scores["relevance_score"] = model.predict(matrix[:, selected["feature_indices"]], num_threads=config["threads"])
    return scores
