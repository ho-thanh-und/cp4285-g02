"""Run the anonymous-session study, from sample preparation to the final report."""

import argparse
import os
import sys
from pathlib import Path

# Find the package in src/ even when the project itself is not installed
# (e.g. after `python -m pip install -r requirements.txt` on the cluster).
_SOURCE = str(Path(__file__).resolve().parent / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

# Ctrl+click this function to read the anonymous sample preparation code.
from session_rerank.samples import prepare_samples
from session_rerank.train_rankers import train_rankers
# Ctrl+click define_product_diversity to follow the five-step diversity review.
from session_rerank.diversity import define_product_diversity
from session_rerank.anonymous_study import (
    open_study, tune_diversification, compare_on_selection, freeze_and_evaluate,
)
from session_rerank.study_report import save_study_report
from session_rerank.study_files import study_lock

# This JSON file sets the CSV folder, output folder, split dates, and sample sizes.
DEFAULT_CONFIG = Path("config/anonymous_samples.json")


def main(config_path=DEFAULT_CONFIG, *, train=False, ranker_run=None, run=False, study_run=None):
    """Read the high-level stages here. Ctrl+click each call to follow its implementation."""
    # 1. Prepare the anonymous samples and their query-time histories.
    dataset_folder = prepare_samples(config_path)

    # 2. Learn relevance scores with and without current-session features.
    if train:
        return train_rankers(dataset_folder)

    # The run action reuses a trained ranker. A resume action reads its recorded
    # paths and validates completed stages before continuing after an interruption.
    if run:
        folder = open_study(dataset_folder, ranker_run, study_run)
        with study_lock(folder):
            # 3. Tune the quota, fixed weight, and session gate using validation.
            tune_diversification(folder)

            # 4. Choose among the four shortlisted settings on reserved selection queries.
            compare_on_selection(folder)

            # 5. Save frozen choices before opening reserved confirmation outcomes.
            freeze_and_evaluate(folder)

            # 6. Export figures, session confidence intervals, examples, and the report.
            return save_study_report(folder)

    # 3a. Define product diversity and measure saved validation recommendation lists.
    # Read the chosen training run, take its top 10 products per query,
    # calculate category/token distances, average the pairs, and save the review.
    # Run this path with main.py diversity --ranker-run <saved training folder>.
    if ranker_run is not None:
        return define_product_diversity(dataset_folder, ranker_run)
    return dataset_folder


def expanded_main(config_path=Path("config/expanded_study.json"), resume=None):
    """Run the expanded study. Read these calls in order, then Ctrl+click for detail."""
    from session_rerank.expanded_data import (
        prepare_expanded, build_feature_cache, prepare_confirmation,
    )
    from session_rerank.expanded_training import train_expanded_models
    from session_rerank.expanded_experiment import (
        open_expanded_run, tune_expanded, select_expanded, freeze_expanded,
        evaluate_expanded_confirmation,
    )
    from session_rerank.expanded_report import write_expanded_report

    config, folder = open_expanded_run(config_path, resume)
    if config.get("validation_only", False):
        raise ValueError("This is a manual training-only run. Resume it with train --queries; use the expanded config for a final study.")
    with study_lock(folder):
        # 1. Keep every eligible training session and reserve a fresh final test.
        prepare_expanded(config, folder)
        # 2. Materialize query-time features in bounded batches on disk.
        build_feature_cache(config, folder)
        # 3. Train click and purchase-aware relevance models using validation.
        train_expanded_models(config, folder)
        # 4. Tune linear/exponential adaptation and separate history-decay ablations.
        tune_expanded(config, folder)
        # 5. Select from the shortlist, then lock all model and policy choices.
        select_expanded(config, folder)
        freeze_expanded(folder)
        # 6. Only now open fresh confirmation outcomes and build their features.
        prepare_confirmation(config, folder)
        build_feature_cache(config, folder, confirmation=True)
        evaluate_expanded_confirmation(config, folder)
        # 7. Save the report, confidence intervals, and per-session behaviour graphs.
        return write_expanded_report(config, folder)


def manual_training_main(config_path=Path("config/learning_curve.json"), queries=None, curve_root=None):
    """Train exactly one requested size, print progress, then stop for human review.

    Ctrl+click these calls in order. There is deliberately no loop over sizes
    and no call to selection or confirmation: you decide when to start the next
    size or change hyperparameters in a separate named experiment directory.
    """
    if queries is None:
        raise ValueError("Supply the sample size explicitly: --queries 2000, 10000, 50000, or all")
    import json
    from session_rerank.learning_curve import open_learning_curve_run, save_learning_result, compare_learning_sizes
    from session_rerank.expanded_data import prepare_expanded, build_feature_cache
    from session_rerank.expanded_training import train_expanded_models
    from session_rerank.configured_reranking import evaluate_manual_reranking

    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = Path(curve_root or settings["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    with study_lock(root):
        # 1. Save settings and select a deterministic, nested sample size.
        config, folder, root = open_learning_curve_run(config_path, queries, root)
        # 2. Read only this size's complete training sessions and fixed validation.
        prepare_expanded(config, folder)
        build_feature_cache(config, folder)
        # 3. Fit fresh models with the JSON grid, printing iterations and scores.
        train_expanded_models(config, folder)
        # 4. Compare the configured decay policy with relevance-only on validation.
        evaluate_manual_reranking(config, folder)
        # 5. Save both scores and redraw completed sizes, then stop for your review.
        save_learning_result(config, folder)
        compare_learning_sizes(root)
    print("This size is finished. Review the scores, then start the next size manually.", flush=True)
    return folder


def prepare_data_main(config_path=Path("config/learning_curve.json"), queries=None, curve_root=None):
    """Prepare one size's data without fitting any model (steps 1-2 of manual training).

    The folder holds the samples, SQLite events and feature cache that the
    analysis actions (event-ablation, ranker-phase2, two-tower, ...) read. A later
    `train` with the same --queries and --curve-root resumes from these stages.
    """
    if queries is None:
        raise ValueError("Supply the sample size explicitly: --queries 50000 or all")
    import json
    from session_rerank.learning_curve import open_learning_curve_run
    from session_rerank.expanded_data import prepare_expanded, build_feature_cache

    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = Path(curve_root or settings["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    with study_lock(root):
        config, folder, root = open_learning_curve_run(config_path, queries, root)
        prepare_expanded(config, folder)
        build_feature_cache(config, folder)
    print(f"Data prepared without training: {folder}", flush=True)
    return folder


def cli(argv=None):
    """Read optional preparation settings and start the new pipeline.

    `uv run main.py train --queries 2000` trains one learning-curve size.
    `uv run main.py train --queries 2000 --training-config config/learning_curve.json` selects settings.
    `uv run main.py compare` prints completed sizes and redraws the curve.
    `uv run main.py alpha-sweep --trained-run results/learning_curve/queries_10000`
    compares session-adaptive alpha with a fixed-alpha curve on validation.
    `uv run main.py two-tower --trained-run results/learning_curve/queries_10000`
    trains the rank-weighted two-tower filter and measures recall at each K1.
    `uv run main.py filter-pipeline --trained-run results/learning_curve/queries_10000
    --two-tower-run results/two_tower/RUN` filters K1 products, scores them with
    LightGBM, reranks to a top 10 and reports NDCG@10 and ILD@10.
    `uv run main.py event-ablation --trained-run results/learning_curve/queries_10000`
    retrains the session rankers with and without product views, and on all vs
    multi-click training sessions, then compares them on the same validation.
    `uv run main.py ranker-phase2 --trained-run results/learning_curve/queries_10000`
    compares category-page ranker variants (window popularity, position-bias
    correction, loss, learning rate) with the popularity sort.
    `uv run main.py prepare-data --queries 50000 --curve-root results/learning_curve_v2`
    prepares one size's samples and features without fitting any model.
    `uv run main.py item-knn --trained-run results/learning_curve_v2/queries_50000`
    ranks category pages with the item-kNN co-click baseline and compares it
    with the popularity sort and any Phase 2 runs listed in config/item_knn.json.
    `uv run main.py heldout` freezes the category-page choices, retrains the frozen
    rankers and evaluates every method on the reserved selection queries.
    `uv run main.py ymal` evaluates item-page "You may also like" lists (item-kNN,
    relevance only, fixed alpha, 6 + 4) on one allowed week, clicks only.
    `uv run main.py home-page` trains the session two-tower and evaluates home page
    recommendations over the whole catalog on one allowed week.
    `uv run main.py error-analysis` breaks down where the frozen models win and lose
    on the selection results (config/error_analysis.json).
    `uv run main.py confirmation` runs the final test once: every page's frozen
    choice on the confirmation week (config/confirmation.json).
    `uv run main.py exposure` measures exposure and popularity bias of the frozen
    lists on all three pages (config/exposure.json), for the ethics section.
    `uv run main.py sequence-models` trains SASRec and GRU4Rec and compares them
    with the home-page methods on the same cases (config/sequence_models.json).
    `uv run main.py export-sequences` and `score-external --export DIR --label NAME`
    connect a recommender from another environment (Transformers4Rec, see
    scripts/t4rec/T4REC_GUIDE.md) to the same home-page cases and metrics.
    `uv run main.py pilot-train` retains the original click-only pilot workflow.
    `uv run main.py diversity --ranker-run results/anonymous_rankers/RUN` reviews ILD.
    `uv run main.py run --ranker-run results/anonymous_rankers/RUN` completes the study.
    `uv run main.py resume --study-run results/anonymous_study/RUN` continues a saved study.
    `uv run main.py expand` runs the full-data purchase and decay experiment.
    `uv run main.py expand-resume --study-run results/expanded_study/RUN` resumes it.
    --config selects the pilot sample protocol; --expanded-config selects the
    expanded study settings. --help performs no preparation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["prepare", "train", "pilot-train", "diversity", "run", "resume", "expand", "expand-resume", "compare", "alpha-sweep", "two-tower", "filter-pipeline", "event-ablation", "ranker-phase2", "prepare-data", "item-knn", "heldout", "ymal", "home-page", "error-analysis", "exposure", "confirmation", "sequence-models", "export-sequences", "score-external"], default="prepare",
                        help="Prepare, train, review diversity, run remaining stages, or resume a study")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Sample sizes, date periods, and history limit")
    parser.add_argument("--ranker-run", type=Path,
                        help="Saved ranker folder to review with the diversity action")
    parser.add_argument("--study-run", type=Path, help="Existing study folder for the resume action")
    parser.add_argument("--expanded-config", type=Path, default=Path("config/expanded_study.json"),
                        help="Full-data, purchase and decay experiment settings")
    parser.add_argument("--queries", help="Required with train: sample size such as 2000, 10000, 50000, or all")
    parser.add_argument("--training-config", type=Path,
                        help="Manual training JSON; default config/learning_curve.json with --queries")
    parser.add_argument("--curve-root", type=Path,
                        help="Output directory for this learning curve or tuning experiment")
    parser.add_argument("--trained-run", type=Path,
                        help="Learning-curve size folder whose models alpha-sweep reuses")
    parser.add_argument("--sweep-config", type=Path, default=Path("config/alpha_sweep.json"),
                        help="Fixed alphas, adaptive betas and bootstrap settings for alpha-sweep")
    parser.add_argument("--two-tower-config", type=Path, default=Path("config/two_tower.json"),
                        help="Tower sizes, rank weights, negatives and K1 values for two-tower")
    parser.add_argument("--two-tower-run", type=Path,
                        help="Saved two-tower folder that filter-pipeline uses as its K1 filter")
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/filter_pipeline.json"),
                        help="K1 values, LightGBM models and rerank policies for filter-pipeline")
    parser.add_argument("--ablation-config", type=Path, default=Path("config/event_ablation.json"),
                        help="Event rules, training sets and models for event-ablation")
    parser.add_argument("--phase2-config", type=Path, default=Path("config/ranker_phase2.json"),
                        help="Ranker variants, models and data choices for ranker-phase2")
    parser.add_argument("--knn-config", type=Path, default=Path("config/item_knn.json"),
                        help="Co-click basket size, shrinkage and runs to compare for item-knn")
    parser.add_argument("--heldout-config", type=Path, default=Path("config/heldout.json"),
                        help="Frozen runs and the held-out split for heldout (selection only for now)")
    parser.add_argument("--ymal-config", type=Path, default=Path("config/ymal.json"),
                        help="Week, pool sizes and policies for the item page 'You may also like' evaluation")
    parser.add_argument("--home-config", type=Path, default=Path("config/home_page.json"),
                        help="Session two-tower, week and policies for the home page evaluation")
    parser.add_argument("--errors-config", type=Path, default=Path("config/error_analysis.json"),
                        help="Selection results to analyse for error-analysis")
    parser.add_argument("--confirmation-config", type=Path, default=Path("config/confirmation.json"),
                        help="Page configs for the final confirmation test")
    parser.add_argument("--exposure-config", type=Path, default=Path("config/exposure.json"),
                        help="Selection runs whose lists exposure measures")
    parser.add_argument("--sequence-config", type=Path, default=Path("config/sequence_models.json"),
                        help="SASRec / GRU4Rec settings and the home-page run to compare with")
    parser.add_argument("--t4rec-config", type=Path, default=Path("config/t4rec.json"),
                        help="Export, model and comparison settings for export-sequences / score-external")
    parser.add_argument("--export", type=Path, help="Folder from export-sequences, for score-external")
    parser.add_argument("--label", help="Method name for the scored predictions, e.g. t4rec_xlnet")
    args = parser.parse_args(argv)
    external = args.export is not None or args.label is not None
    if (args.action == "score-external" and (args.export is None or args.label is None)) or \
            (args.action != "score-external" and external):
        parser.error("score-external needs --export and --label, and they are used only there")
    if args.action == "export-sequences":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.external_models import export_sequences
        export_sequences(args.t4rec_config)
        return
    if args.action == "score-external":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.external_models import score_predictions
        score_predictions(args.export, args.label, args.t4rec_config)
        return
    if args.action == "sequence-models":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.sequence_models import run_sequence_models
        run_sequence_models(args.sequence_config)
        return
    run_actions = ("alpha-sweep", "two-tower", "filter-pipeline", "event-ablation", "ranker-phase2", "item-knn")
    if (args.action in run_actions) != (args.trained_run is not None):
        parser.error(f"--trained-run is required with {', '.join(run_actions)}, and used only there")
    if args.action == "error-analysis":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.error_analysis import run_error_analysis
        run_error_analysis(args.errors_config)
        return
    if args.action == "confirmation":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.confirmation import run_confirmation
        run_confirmation(args.confirmation_config)
        return
    if args.action == "exposure":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.exposure import run_exposure
        run_exposure(args.exposure_config)
        return
    if args.action == "home-page":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.home_page import run_home_page
        run_home_page(args.home_config)
        return
    if args.action == "ymal":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.ymal import run_ymal
        run_ymal(args.ymal_config)
        return
    if args.action == "heldout":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.heldout import run_heldout
        run_heldout(args.heldout_config)
        return
    if args.action == "item-knn":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.item_knn import run_item_knn
        run_item_knn(args.trained_run, args.knn_config)
        return
    if args.action == "ranker-phase2":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.ranker_phase2 import run_phase2
        run_phase2(args.trained_run, args.phase2_config)
        return
    if args.action == "event-ablation":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.event_ablation import run_event_ablation
        run_event_ablation(args.trained_run, args.ablation_config)
        return
    if (args.action == "filter-pipeline") != (args.two_tower_run is not None):
        parser.error("--two-tower-run is required with filter-pipeline, and used only there")
    if args.action == "filter-pipeline":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.filter_pipeline import run_filter_pipeline
        run_filter_pipeline(args.trained_run, args.two_tower_run, args.pipeline_config)
        return
    if args.action == "alpha-sweep":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.alpha_sweep import run_alpha_sweep
        run_alpha_sweep(args.trained_run, args.sweep_config)
        return
    if args.action == "two-tower":
        os.chdir(Path(__file__).resolve().parent)
        from session_rerank.two_tower import train_two_tower
        train_two_tower(args.trained_run, args.two_tower_config)
        return
    if args.queries is not None and args.action not in ("train", "prepare-data"):
        parser.error("--queries is used with train or prepare-data")
    if (args.training_config is not None or args.curve_root is not None) and args.action not in ("train", "compare", "prepare-data"):
        parser.error("--training-config and --curve-root are used with train, prepare-data or compare")
    if args.action == "prepare-data":
        if args.queries is None:
            parser.error("prepare-data requires --queries (for example: prepare-data --queries 50000)")
        from session_rerank.learning_curve import query_target
        try:
            query_target(args.queries)
        except ValueError as error:
            parser.error(str(error))
        os.chdir(Path(__file__).resolve().parent)
        prepare_data_main(args.training_config or Path("config/learning_curve.json"), args.queries, args.curve_root)
        return
    manual = args.action == "train"
    if manual and args.queries is None:
        parser.error("train requires --queries (for example: train --queries 2000)")
    if manual or args.action == "compare":
        if args.ranker_run is not None or args.study_run is not None:
            parser.error("Manual training resumes by repeating its command; do not pass --ranker-run or --study-run")
        from session_rerank.learning_curve import query_target, compare_learning_sizes
        if args.queries is not None:
            try:
                query_target(args.queries)
            except ValueError as error:
                parser.error(str(error))
        os.chdir(Path(__file__).resolve().parent)
        config_path = args.training_config or Path("config/learning_curve.json")
        if manual:
            manual_training_main(config_path, args.queries, args.curve_root)
        else:
            import json
            root = args.curve_root or Path(json.loads(config_path.read_text(encoding="utf-8"))["output_directory"])
            if not root.is_dir():
                parser.error(f"No learning curve at {root}; train one size first")
            with study_lock(root):
                compare_learning_sizes(root)
        return
    if args.action in ("expand", "expand-resume"):
        if args.ranker_run is not None:
            parser.error("Expansion trains its own models")
        if (args.action == "expand-resume") != (args.study_run is not None):
            parser.error("--study-run is required with expand-resume")
        os.chdir(Path(__file__).resolve().parent)
        expanded_main(args.expanded_config, args.study_run)
        return
    if args.action in ("diversity", "run") and args.ranker_run is None:
        parser.error("--ranker-run is required for diversity and run")
    if args.action not in ("diversity", "run") and args.ranker_run is not None:
        parser.error("--ranker-run is used with diversity or run")
    if (args.action == "resume") != (args.study_run is not None):
        parser.error("--study-run is required with resume")
    # Resolve paths such as data/raw from the project folder, wherever the script was launched.
    os.chdir(Path(__file__).resolve().parent)
    if args.action == "resume":
        # Read only the saved protocol and paths. Resume uses the original study inputs.
        import json
        identity = json.loads((args.study_run / "study.json").read_text())["identity"]
        main(args.config, ranker_run=Path(identity["ranker_run"]), run=True, study_run=args.study_run)
    else:
        main(args.config, train=args.action == "pilot-train", ranker_run=args.ranker_run, run=args.action == "run")


# Start the CLI when this file is run as a script. Importing main leaves it idle.
if __name__ == "__main__":
    cli()
