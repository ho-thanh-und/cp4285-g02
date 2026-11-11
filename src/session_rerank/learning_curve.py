"""Manual, one-size-at-a-time training with a comparable validation learning curve.

No function in this module opens selection or confirmation outcomes. The same
command resumes its size after an interruption; it never starts the next size.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .expanded_experiment import SOURCES, open_expanded_run
from .expanded_training import training_settings
from .ranking_settings import top_k, reranking_settings
from .expanded_data import read_candidates
from .samples import file_hash
from .study_files import write_json, save_table, stage_complete, finish_stage, verify_files


DEFAULT_CONFIG = Path("config/learning_curve.json")


def query_target(value):
    """Accept an explicit positive query target or 'all'; never silently clamp it."""
    if value is None or value == "all":
        return None
    if isinstance(value, bool):
        raise ValueError("--queries must be a positive integer or all")
    text = str(value)
    if not text.isdecimal() or int(text) < 1:
        raise ValueError("--queries must be a positive integer or all")
    return int(text)


def open_learning_curve_run(config_path=DEFAULT_CONFIG, queries=None, curve_root=None):
    """Open one size under a shared, immutable comparison configuration.

    A copied JSON config plus a new output_directory defines a new experiment.
    Reusing an old directory with changed settings is rejected so tuning cannot
    overwrite earlier scores or quietly mix incompatible learning curves.
    Call this while holding study_lock on the chosen output directory.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = query_target(config.get("training_target_queries", 2000) if queries is None else queries)
    config["validation_only"] = True
    config["primary_metric"] = config.get("primary_metric", "click_ndcg")
    # Keep the comparison endpoint constant even if purchase support grows with
    # sample size. Both click and graded-purchase measurements are still saved.
    if config["primary_metric"] not in ("click_ndcg", "gain_ndcg"):
        raise ValueError("A learning curve needs a fixed primary_metric: click_ndcg or gain_ndcg")
    training_settings(config)
    config["training_target_queries"] = target
    root = Path(curve_root or config["output_directory"])
    config["output_directory"] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    scientific_config = {key: value for key, value in config.items()
                         if key not in ("training_target_queries", "output_directory")}
    identity = dict(
        config=scientific_config,
        pilot_manifest_sha256=file_hash(Path(config["pilot_directory"]) / "manifest.json"),
        source_sha256={name: file_hash(Path(__file__).parent / name) for name in SOURCES},
    )
    manifest = root / "learning_curve.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError(
                "Settings, data, or code differ from this learning curve. Keep the old results; "
                "set a new output_directory in your copied config or use --curve-root NEW_FOLDER."
            )
    else:
        write_json(manifest, dict(identity=identity, created_at_utc=datetime.now(timezone.utc).isoformat()))
    name = "queries_all" if target is None else f"queries_{target}"
    folder = root / name
    if folder.exists():
        recorded, folder = open_expanded_run(resume=folder)
        if recorded != config:
            raise ValueError("This size has different saved settings; use a new output directory.")
        print(f"Resume size: {folder.resolve()}", flush=True)
    else:
        config, folder = open_expanded_run(config=config, run_name=name)
    return config, folder, root


def save_learning_result(config, folder):
    """Save best-trial scores for every model on the identical validation queries.

    Each size retrains its own models from scratch. We compare the same model
    family across sizes, rather than comparing only whichever family won a size.
    Actual query counts can slightly exceed the requested target because complete
    sessions are kept. Positive training groups are reported separately.
    """
    if stage_complete(folder, "learning"):
        return pd.read_csv(folder / "learning_metrics.csv")
    if not stage_complete(folder, "training"):
        raise ValueError("Complete model training before comparing sample sizes.")
    train = pd.read_parquet(folder / "train_features" / "queries.parquet")
    validation = read_candidates(folder, "validation").sort_values(["query_id", "original_rank"])
    label_hash = hashlib.sha256()
    for row in validation[["query_id", "item_id", "clicked", "purchased"]].itertuples(index=False, name=None):
        label_hash.update(f"{row[0]}|{row[1]}|{int(row[2])}|{int(row[3])}\n".encode())
    primary = json.loads((folder / "objective.json").read_text())["primary"]
    settings = reranking_settings(config)
    if not stage_complete(folder, "manual_reranking"):
        raise ValueError("Complete the configured validation reranking review first.")
    reranked = (pd.read_csv(folder / "reranking_metrics.csv").set_index(["model", "policy_id"])
                if settings["enabled"] else None)
    rows = []
    for record in pd.read_csv(folder / "model_comparison.csv").to_dict("records"):
        details = json.loads((folder / f"{record['model']}_trials.json").read_text())
        best = max(details["trials"], key=lambda trial: trial[primary])
        positive = train.clicks.gt(0) if record["objective"] == "click" else (train.clicks + train.purchases).gt(0)
        policy_scores = {"relevance_diversity_ild": None}
        for metric in ("click_ndcg", "gain_ndcg", "recall_at_k", "purchase_recall", "diversity_ild"):
            policy_scores["reranked_"+metric] = (float(reranked.loc[(record["model"], "configured"), metric])
                                                   if reranked is not None else None)
        if reranked is not None:
            policy_scores["relevance_diversity_ild"] = float(reranked.loc[(record["model"], "relevance"), "diversity_ild"])
        rows.append(dict(
            requested_queries=config["training_target_queries"] or "all",
            training_queries=len(train), training_sessions=train.session_key.nunique(),
            positive_training_queries=int(positive.sum()), training_candidates=int(train.candidates.sum()),
            model=record["model"], objective=record["objective"], primary_metric=primary,
            training_seconds=record["seconds"], validation_labels_sha256=label_hash.hexdigest(),
            decay_function_type=settings["decay_function_type"], decay_function_rate=settings["decay_function_rate"],
            history_decay_type=settings["history_decay_type"], history_decay_rate_per_minute=settings["history_decay_rate_per_minute"],
            **best, **policy_scores,
        ))
    result = pd.DataFrame(rows)
    save_table(result, folder / "learning_metrics.csv")
    finish_stage(folder, "learning", ["learning_metrics.csv"])
    print(f"Saved validation results: {(folder / 'learning_metrics.csv').resolve()}", flush=True)
    return result


def compare_learning_sizes(curve_root):
    """Export a table and plot from completed sizes, without fitting any model.

    Validate nested training queries and identical validation labels before
    presenting a curve. Partially trained sizes are skipped until complete.
    These are development scores used for tuning, never final-test claims.
    """
    root = Path(curve_root)
    manifest = root / "learning_curve.json"
    if not manifest.exists():
        raise ValueError(f"No learning curve at {root}; train one size first.")
    identity = json.loads(manifest.read_text())["identity"]
    frames, samples = [], []
    for folder in sorted(root.glob("queries_*")):
        if not folder.is_dir() or not stage_complete(folder, "learning"):
            continue
        saved = json.loads((folder / "expanded_run.json").read_text())
        scientific_config = {key: value for key, value in saved["config"].items()
                             if key not in ("training_target_queries", "output_directory")}
        if scientific_config != identity["config"] or saved["source_sha256"] != identity["source_sha256"]:
            raise ValueError(f"Incompatible experiment settings at {folder}")
        # Verify the metadata actually used to establish sample membership too.
        feature_receipt = json.loads((folder / "features_complete.json").read_text())["files"]
        # Windows writes backslashes in receipt keys; Linux writes slashes.
        # Normalize the lookup, preserving the original stored fingerprints.
        feature_receipt = {name.replace("\\", "/"): digest for name, digest in feature_receipt.items()}
        verify_files(folder, {"train_features/queries.parquet": feature_receipt["train_features/queries.parquet"]})
        metadata = pd.read_parquet(folder / "train_features" / "queries.parquet")
        samples.append((len(metadata), set(metadata.query_id)))
        frames.append(pd.read_csv(folder / "learning_metrics.csv"))
    if not frames:
        raise ValueError("No completed sizes yet. Train a size before comparing.")
    samples.sort(key=lambda pair: pair[0])
    for (_, smaller), (_, larger) in zip(samples[:-1], samples[1:]):
        if not smaller.issubset(larger):
            raise ValueError("Training query samples are not nested; do not compare them as this learning curve.")
    combined = pd.concat(frames, ignore_index=True).sort_values(["training_queries", "model"])
    if combined.validation_labels_sha256.nunique() != 1 or combined.primary_metric.nunique() != 1:
        raise ValueError("Validation labels or comparison objectives differ between sizes.")
    save_table(combined, root / "comparison.csv")
    # Plot the main model families; purchase-gain sensitivity remains in the CSV.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = combined[~combined.model.str.contains("_gain_")]
    k = int(combined.top_k.iloc[0])
    fig, grid = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    axes = grid.ravel()
    for model, group in shown.groupby("model", sort=False):
        for ax, metric, reranked_metric, title in zip(axes,
                ("click_ndcg", "gain_ndcg", "relevance_diversity_ild", "training_seconds"),
                ("reranked_click_ndcg", "reranked_gain_ndcg", "reranked_diversity_ild", None),
                (f"Click NDCG@{k}", f"Purchase-aware NDCG@{k}", f"Configured diversity ILD@{k}", "Training time (seconds)")):
            line, = ax.plot(group.training_queries, group[metric], marker="o", label=model)
            if reranked_metric is not None and group[reranked_metric].notna().any():
                ax.plot(group.training_queries, group[reranked_metric], marker="x", linestyle="--", color=line.get_color())
            ax.set(xlabel="Actual training queries", ylabel=title, title=title)
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(sorted(shown.training_queries.unique()),
                          [f"{n:,}" for n in sorted(shown.training_queries.unique())])
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Manual learning curve — solid: relevance-only; dashed: configured reranking")
    fig.tight_layout()
    temporary = root / "learning_curve.png.tmp"
    fig.savefig(temporary, format="png", dpi=160)
    plt.close(fig)
    temporary.replace(root / "learning_curve.png")
    print(combined[["training_queries", "top_k", "model", "click_ndcg", "reranked_click_ndcg", "gain_ndcg", "reranked_gain_ndcg",
                    "relevance_diversity_ild", "reranked_diversity_ild", "training_seconds"]]
          .to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    print(f"Comparison: {(root / 'comparison.csv').resolve()}", flush=True)
    print(f"Chart: {(root / 'learning_curve.png').resolve()}", flush=True)
    return root / "comparison.csv"
