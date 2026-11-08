"""Run a recommender from another library on the home-page cases, in its own environment.

Libraries such as Transformers4Rec need package versions that cannot share this
project's environment. They plug in through files:

1. export-sequences (this environment) writes, to a new folder,
   train_sessions.parquet  training sessions as lists of vocabulary ids,
   cases.parquet           the exact home-page evaluation cases (checked against
                           the saved home-page run): clicks so far as vocabulary
                           ids and product rows, and the later clicks,
   vocabulary.parquet      vocabulary id -> catalog row and product id (0 = padding),
   export.json             where everything came from.
2. The external script (its own environment) trains on train_sessions.parquet
   and writes predictions.parquet: one row per case with a ranked list of
   vocabulary ids (`item_ids`), most likely first.
3. score-external (this environment) turns each list into the top 10 products
   (dropping products already clicked; popularity when a case has no prediction),
   scores it with the same metrics as every home-page method, and compares it
   on identical cases with the saved home-page methods and any SASRec/GRU4Rec run.

The training window, data rules and cases are those of the chosen
sequence-model config (clicks only, eligible sessions, same validation cases).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .diversity import ProductDiversity
from .event_ablation import session_bootstrap
from .home_page import click_sequences, evaluation_cases
from .sequence_models import read_sequence_settings, vocabulary
from .study_files import save_table, write_json
from .ymal import DIVERSITY, RELEVANCE, list_metrics

HOME_METHODS = ["popularity", "item_knn", "two_tower", "fixed_0.8_item_knn"]


def read_external_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    sequence, home = read_sequence_settings(settings["sequence_config"])
    return settings, sequence, home


def export_sequences(settings_path=Path("config/t4rec.json"), output_root=None):
    settings, sequence, home = read_external_settings(settings_path)
    products = pd.read_parquet(Path(home["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    window = sequence["training_window"]
    print(f"[export] Training sessions {window['start']} to before {window['end_exclusive']}", flush=True)
    sessions = click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], item_index)
    vocab = vocabulary(sessions)
    limit = sequence["max_length"]
    train = []
    for key, date, items in sessions:
        known = [int(vocab[i]) for i in items if i in vocab.index]
        if len(known) >= 2:
            train.append(dict(session_key=key, eventdate=date, item_ids=known[-(limit + 1):]))
    print(f"[export] {home['window']['name']} week cases", flush=True)
    week = click_sequences(home["raw_directory"], home["window"]["start"], home["window"]["end_exclusive"], item_index)
    cases, total = evaluation_cases(week, home["max_cases"], home["seed"])
    saved = pd.read_parquet(Path(sequence["home_run"]) / "per_case.parquet")
    reference = saved[saved.method.eq("popularity")].sort_values("case")
    if len(reference) != len(cases) or not np.array_equal(reference.clicked_so_far.to_numpy(),
                                                          [len(c[1]) for c in cases]):
        raise ValueError("Exported cases do not match the saved home-page run")
    case_rows = [dict(case=number, session_key=key, prefix_rows=list(prefix), later_rows=sorted(later),
                      prefix_ids=[int(vocab[r]) for r in prefix if r in vocab.index][-limit:])
                 for number, (key, prefix, later) in enumerate(cases)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{home['window']['name']}"
    folder.mkdir(parents=True)
    save_table(pd.DataFrame(train), folder / "train_sessions.parquet")
    save_table(pd.DataFrame(case_rows), folder / "cases.parquet")
    save_table(pd.DataFrame(dict(vocab_id=vocab.to_numpy(), product_row=vocab.index.to_numpy(),
                                 item_id=products.item_id.to_numpy()[vocab.index.to_numpy()])),
               folder / "vocabulary.parquet")
    write_json(folder / "export.json", dict(
        exported_at_utc=datetime.now(timezone.utc).isoformat(), config=str(settings_path),
        sequence_config=settings["sequence_config"], home_run=sequence["home_run"], training_window=window,
        evaluation_week=home["window"], max_length=limit, vocabulary=int(len(vocab)),
        training_sessions=len(train), training_positions=int(sum(len(r["item_ids"]) - 1 for r in train)),
        evaluable_cases=total, cases=len(case_rows), model=settings["model"]))
    # The cluster job reads this to pass the folder to the external script and to scoring.
    (folder.parent / "latest_export.txt").write_text(folder.as_posix() + "\n", encoding="utf-8")
    print(f"[export] {len(train):,} sessions, {len(case_rows):,} cases, {len(vocab):,} products -> {folder}",
          flush=True)
    return folder


def score_predictions(export_folder, label, settings_path=Path("config/t4rec.json")):
    """Score an external predictions.parquet and compare it with the saved methods."""
    settings, sequence, home = read_external_settings(settings_path)
    folder = Path(export_folder)
    exported = json.loads((folder / "export.json").read_text())
    if exported["home_run"] != sequence["home_run"] or exported["evaluation_week"] != home["window"]:
        raise ValueError("The export does not belong to this config's home-page run")
    cases = pd.read_parquet(folder / "cases.parquet")
    vocab = pd.read_parquet(folder / "vocabulary.parquet")
    predictions = pd.read_parquet(folder / "predictions.parquet").set_index("case")
    to_row = dict(zip(vocab.vocab_id, vocab.product_row))
    products = pd.read_parquet(Path(home["trained_run"]) / "products.parquet")
    catalog_ids = products.item_id.to_numpy()
    categories = products.category_id.fillna(-1).to_numpy(np.int64)
    diversity = ProductDiversity(products)
    train = pd.read_parquet(folder / "train_sessions.parquet")
    counts = pd.Series(np.concatenate(train.item_ids.to_numpy())).value_counts()
    fallback = [to_row[i] for i in counts.index]
    k, slots = home["top_k"], home["relevance_slots"]
    rows = []
    for case in cases.itertuples(index=False):
        prefix = list(case.prefix_rows)
        exclude = set(prefix)
        ranked = predictions.item_ids.get(case.case)
        used = ranked is not None and len(ranked) > 0
        source = [to_row[int(i)] for i in ranked if int(i) in to_row] if used else fallback
        top = np.array([r for r in source if r not in exclude][:k])
        if len(top) < k:  # pad from popularity if a short list remains
            top = np.array(list(top) + [r for r in fallback if r not in exclude and r not in set(top)][:k - len(top)])
        rows.append(dict(session_key=case.session_key, clicked_so_far=len(prefix), targets=len(case.later_rows),
                         method=label, case=int(case.case), used_prediction=used,
                         **list_metrics(top, set(case.later_rows), prefix[-1], catalog_ids, categories, diversity,
                                        slots)))
    ours = pd.DataFrame(rows)
    saved = pd.read_parquet(Path(sequence["home_run"]) / "per_case.parquet")
    frames = [saved[saved.method.isin(HOME_METHODS)], ours]
    for run in settings.get("compare_sequence_runs", []):
        other = pd.read_parquet(Path(run) / "per_case.parquet")
        frames.append(other.assign(method=other.method + "_" + Path(run).name.split("_")[0]))
    combined = pd.concat(frames, ignore_index=True)
    combined["clicks_so_far"] = pd.cut(combined.clicked_so_far, [0, 1, 2, 5, np.inf], labels=["1", "2", "3-5", "6+"])
    summary = pd.concat([
        combined.groupby("method", sort=False)[RELEVANCE + DIVERSITY].mean().reset_index().assign(segment="all cases"),
        combined.groupby(["clicks_so_far", "method"], observed=True, sort=False)[RELEVANCE + ["combined_ild"]]
        .mean().reset_index().rename(columns={"clicks_so_far": "segment"})], ignore_index=True)
    rng = np.random.default_rng(settings["seed"])
    differences = []
    base_case = ours.set_index("case")
    for reference in combined.method.unique():
        if reference == label:
            continue
        base = combined[combined.method.eq(reference)].set_index("case").reindex(base_case.index)
        for metric in RELEVANCE + ["combined_ild"]:
            diff = base_case[metric] - base[metric]
            low, high = session_bootstrap(diff, base_case.session_key, settings["bootstrap_replicates"], rng)
            differences.append(dict(method=label, reference=reference, metric=metric,
                                    value=float(base_case[metric].mean()), reference_value=float(base[metric].mean()),
                                    difference=float(diff.mean()), ci_low=low, ci_high=high))
    output = folder / f"scored_{label}"
    output.mkdir(exist_ok=True)
    save_table(ours, output / "per_case.parquet")
    save_table(summary, output / "summary.csv")
    save_table(pd.DataFrame(differences), output / "differences.csv")
    write_json(output / "scoring.json", dict(label=label, cases=len(ours),
                                             cases_with_prediction=int(ours.used_prediction.sum())))
    print(f"[score] {label}: NDCG@10 {ours.ndcg.mean():.4f}, recall@10 {ours.recall.mean():.4f} -> {output}",
          flush=True)
    return output
