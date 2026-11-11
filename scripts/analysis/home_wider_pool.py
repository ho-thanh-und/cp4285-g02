"""Home page: SASRec + 6 + 4 with a wider pool for the four discovery places
(validation week only).

The frozen policy takes places 7-10 as the four most mutually different products
among SASRec's places 7-20 (pool 20). Idea (user, 2026-11-09): draw them from
SASRec's top 100 instead. Pools tested: 20 (frozen), 50, 100 and 200. Places 1-6
are SASRec's best six in every variant; the distance and greedy rule are the
frozen 6 + 4's (`diverse_tail_order`, combined category + name-word distance).

Reported, as declared before the run (2026-11-09):
  * the 6 + 4 rule: NDCG@10 at most 5% below plain SASRec, then most ILD;
  * the split view (user, 2026-11-09: places 7-10 are for discovery): NDCG@6 of
    places 1-6 (identical by construction), and for places 7-10 the share of lists
    adding a category the six do not show, and how often places 7-10 hold a later
    click when the shopper later clicks a category the six do not show.
Several home variants are being tried on the validation week; any that is adopted
must be confirmed on the selection week. The final test is not used.

    uv run python scripts/analysis/home_wider_pool.py [--cases N]
    # selection-week check of pool 100 against the frozen pool 20 (6p):
    uv run python scripts/analysis/home_wider_pool.py \
        --run results/sequence_models/20261108T182139Z_selection --pools 20 100
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import lightgbm  # noqa: F401  (import before pandas/torch: avoids an OpenMP clash on Windows)
import numpy as np
import pandas as pd
import torch

from session_rerank.adaptive_policies import diverse_tail_order
from session_rerank.diversity import ProductDiversity
from session_rerank.event_ablation import session_bootstrap
from session_rerank.home_page import click_sequences, evaluation_cases
from session_rerank.sequence_models import build_model, item_weights, ranked
from session_rerank.study_files import save_table, write_json
from session_rerank.ymal import list_metrics

RUN = ROOT / "results/sequence_models/20261108T175008Z_validation"
POOLS, MAX_LOSS = (20, 50, 100, 200), 0.05
METRICS = ["ndcg", "ndcg_at_6", "hit", "combined_ild", "distinct_categories", "tail_new_category",
           "tail_hit", "tail_hit_new_category_cases"]


def ndcg_at(top, later, n):
    hits = np.array([r in later for r in top[:n]], dtype=float)
    discounts = 1 / np.log2(np.arange(2, n + 2))
    return float(hits @ discounts / discounts[:min(n, len(later))].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN, help="Saved SASRec run whose week and cases are used")
    parser.add_argument("--pools", type=int, nargs="+", default=list(POOLS))
    parser.add_argument("--cases", type=int, help="Score only the first N cases (a quick trial)")
    args = parser.parse_args()
    run, pools, max_cases = ROOT / args.run, tuple(args.pools), args.cases
    if 20 not in pools:
        raise ValueError("pool 20 (the frozen policy) is the reference and must be included")
    saved = json.loads((run / "sequence_settings.json").read_text())
    home, model_settings = saved["home"], saved["settings"]
    k, slots = home["top_k"], home["relevance_slots"]
    products = pd.read_parquet(Path(home["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    catalog_ids = products.item_id.to_numpy()
    categories = products.category_id.fillna(-1).to_numpy(np.int64)
    diversity = ProductDiversity(products)
    vocab_rows = np.load(run / "vocabulary_rows.npy")
    vocab = pd.Series(np.arange(1, len(vocab_rows) + 1), index=vocab_rows)
    ids_to_rows = pd.Series(vocab.index.to_numpy(), index=vocab.to_numpy())
    window = model_settings["training_window"]
    clicks = np.zeros(len(products))
    for _, _, items in click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], item_index):
        np.add.at(clicks, items, 1)
    fallback_rows = np.argsort(-clicks, kind="stable")
    model = build_model("sasrec", len(vocab), model_settings)
    model.load_state_dict(torch.load(run / "sasrec.pt", map_location="cpu", weights_only=True))
    model.eval()
    weights = item_weights(model)

    week = home["window"]
    cases, _ = evaluation_cases(click_sequences(home["raw_directory"], week["start"], week["end_exclusive"],
                                                item_index), home["max_cases"], home["seed"])
    cases = cases[:max_cases] if max_cases else cases
    print(f"[wider pool] {week['name']} week, {len(cases):,} cases", flush=True)
    rows = []
    for number, (key, prefix, later) in enumerate(cases):
        members, scores, personal = ranked(model, prefix, vocab, ids_to_rows, fallback_rows, model_settings,
                                           max(pools), weights)
        reward = np.nan_to_num(diversity.distances(catalog_ids[members].tolist())["combined"], nan=0.0)
        lists = {"sasrec": members[:k]}
        for pool in pools:
            order = diverse_tail_order(scores[:pool], reward[:pool, :pool], slots, k)
            lists[f"six_four_pool{pool}"] = members[order]
        six_categories = {int(categories[r]) for r in members[:slots]}
        new_category_later = {int(categories[r]) for r in later} - six_categories - {-1}
        for method, top in lists.items():
            row = dict(session_key=key, case=number, method=method, used_session=personal,
                       **list_metrics(top, later, prefix[-1], catalog_ids, categories, diversity, slots))
            row["ndcg_at_6"] = ndcg_at(top, later, slots)
            row["tail_new_category"] = float(any(int(categories[r]) not in six_categories for r in top[slots:]))
            row["tail_hit"] = float(any(r in later for r in top[slots:]))
            row["tail_hit_new_category_cases"] = row["tail_hit"] if new_category_later else np.nan
            rows.append(row)
        if (number + 1) % 2000 == 0:
            print(f"[wider pool] {number + 1:,}/{len(cases):,}", flush=True)
    ours = pd.DataFrame(rows)
    if not max_cases:  # the frozen pool must reproduce the saved run
        reference = pd.read_parquet(run / "per_case.parquet")
        for mine, theirs in (("sasrec", "sasrec"), ("six_four_pool20", "six_four_sasrec")):
            a = ours[ours.method.eq(mine)].sort_values("case").ndcg.to_numpy()
            b = reference[reference.method.eq(theirs)].sort_values("case").ndcg.to_numpy()
            if not np.allclose(a, b):
                raise ValueError(f"{mine} does not reproduce the saved validation run")
        print("[wider pool] plain SASRec and the frozen 6 + 4 reproduce the saved run exactly", flush=True)

    summary = ours.groupby("method", sort=False)[METRICS].mean().reset_index()
    summary["ndcg_change_vs_sasrec"] = summary.ndcg / summary.set_index("method").ndcg["sasrec"] - 1
    summary["within_5_percent"] = summary.ndcg_change_vs_sasrec >= -MAX_LOSS
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    if max_cases:
        return
    rng = np.random.default_rng(4285)
    differences = []
    base = ours[ours.method.eq("six_four_pool20")].set_index("case")
    for method in ours.method.unique():
        if method == "six_four_pool20":
            continue
        other = ours[ours.method.eq(method)].set_index("case").reindex(base.index)
        for metric in METRICS:
            diff = other[metric] - base[metric]
            keep = diff.notna()
            low, high = session_bootstrap(diff[keep], base.session_key[keep], 2000, rng)
            differences.append(dict(method=method, reference="six_four_pool20", metric=metric,
                                    value=float(other[metric][keep].mean()),
                                    reference_value=float(base[metric][keep].mean()),
                                    difference=float(diff[keep].mean()), ci_low=low, ci_high=high,
                                    cases=int(keep.sum())))
    differences = pd.DataFrame(differences)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "results/home_wider_pool" / f"{stamp}_{week['name']}"
    folder.mkdir(parents=True)
    save_table(ours, folder / "per_case.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(differences, folder / "differences.csv")
    write_json(folder / "settings.json", dict(run=str(run.relative_to(ROOT)), pools=list(pools),
                                              max_ndcg_loss=MAX_LOSS, week=week, cases=len(cases)))
    print(differences[differences.metric.isin(["ndcg", "distinct_categories", "tail_new_category",
                                               "tail_hit_new_category_cases"])].to_string(index=False))
    print(f"[wider pool] Saved to {folder}")


if __name__ == "__main__":
    main()
