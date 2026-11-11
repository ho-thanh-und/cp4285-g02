"""Home page: fill the four discovery places from other categories (validation week only).

The frozen home policy (SASRec + 6 + 4) keeps SASRec's best six and adds the four
most mutually different products from SASRec's places 7-20. For mixed visits those
20 products are almost always one category, so the four rarely add a new one
(EXPERIMENT_LOG.md 6p).

Variant tested here, `six_other_categories_sasrec`:
  * places 1-6: SASRec's best six (as in 6 + 4);
  * places 7-10: walking down SASRec's ranking (top `POOL` products), the best
    product of each category that is not among the first six's categories, one
    product per category, in SASRec order. Products with no category are skipped.
    If fewer than four such categories exist in the pool, the next best remaining
    products fill the list.

Declared before the run (2026-11-09), the same rule that chose 6 + 4: the variant is
acceptable if it loses at most 5% NDCG@10 against plain SASRec on the validation
week; among acceptable policies the one with the most combined ILD is preferred.
The final test (May 16 - June 1) is not used.

Everything else is the frozen validation run's: the same 20,000 cases, the same
saved SASRec model, vocabulary, fallback and metrics (`list_metrics`). The frozen
6 + 4 and plain SASRec are recomputed and must match the saved run exactly.

    uv run python scripts/analysis/home_other_categories.py
"""
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

from session_rerank.diversity import ProductDiversity
from session_rerank.event_ablation import session_bootstrap
from session_rerank.home_page import click_sequences, evaluation_cases
from session_rerank.sequence_models import build_model, diverse_lists, item_weights, ranked
from session_rerank.study_files import save_table, write_json
from session_rerank.ymal import list_metrics

RUN = ROOT / "results/sequence_models/20261108T175008Z_validation"  # frozen validation run (6j)
POOL = 1000          # SASRec products searched for other categories
MAX_LOSS = 0.05      # declared rule: at most 5% NDCG loss against plain SASRec
METRICS = ["ndcg", "recall", "hit", "mrr", "combined_ild", "distinct_categories",
           "tail_new_category", "tail_hit", "tail_hit_new_category_cases"]


def other_categories(members, categories, k=10, keep=6):
    head = list(members[:keep])
    seen = {int(categories[r]) for r in head}
    tail = []
    for r in members[keep:]:
        c = int(categories[r])
        if c >= 0 and c not in seen:
            tail.append(r)
            seen.add(c)
            if len(tail) == k - keep:
                break
    if len(tail) < k - keep:  # not enough categories in the pool: next best remaining products
        chosen = set(head) | set(tail)
        tail += [r for r in members[keep:] if r not in chosen][:k - keep - len(tail)]
    return np.array(head + tail)


def main():
    saved = json.loads((RUN / "sequence_settings.json").read_text())
    home, model_settings = saved["home"], saved["settings"]
    k, slots = home["top_k"], home["relevance_slots"]
    products = pd.read_parquet(Path(home["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    catalog_ids = products.item_id.to_numpy()
    categories = products.category_id.fillna(-1).to_numpy(np.int64)
    diversity = ProductDiversity(products)

    vocab_rows = np.load(RUN / "vocabulary_rows.npy")
    vocab = pd.Series(np.arange(1, len(vocab_rows) + 1), index=vocab_rows)
    ids_to_rows = pd.Series(vocab.index.to_numpy(), index=vocab.to_numpy())
    window = model_settings["training_window"]
    history = click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], item_index)
    clicks = np.zeros(len(products))
    for _, _, items in history:
        np.add.at(clicks, items, 1)
    fallback_rows = np.argsort(-clicks, kind="stable")
    model = build_model("sasrec", len(vocab), model_settings)
    model.load_state_dict(torch.load(RUN / "sasrec.pt", map_location="cpu", weights_only=True))
    model.eval()
    weights = item_weights(model)

    week = home["window"]
    cases, _ = evaluation_cases(click_sequences(home["raw_directory"], week["start"], week["end_exclusive"],
                                                item_index), home["max_cases"], home["seed"])
    print(f"[other-categories] {week['name']} week, {len(cases):,} cases", flush=True)
    rows, shown = [], []
    for number, (key, prefix, later) in enumerate(cases):
        members, scores, personal = ranked(model, prefix, vocab, ids_to_rows, fallback_rows, model_settings,
                                           POOL, weights)
        pool = max(k, home["pool_size"])
        lists = {"sasrec": members[:k],
                 "six_four_sasrec": diverse_lists(members[:pool], scores[:pool], diversity, catalog_ids,
                                                  home)["six_four"],
                 "six_other_categories_sasrec": other_categories(members, categories, k, slots)}
        new_category_later = {int(categories[r]) for r in later} - {int(categories[r]) for r in members[:slots]}
        for method, top in lists.items():
            head_cats = {int(categories[r]) for r in top[:slots]}
            tail = top[slots:]
            row = dict(session_key=key, case=number, method=method, clicked_so_far=len(prefix),
                       used_session=personal,
                       **list_metrics(top, later, prefix[-1], catalog_ids, categories, diversity, slots))
            row["tail_new_category"] = float(any(int(categories[r]) not in head_cats for r in tail))
            row["tail_hit"] = float(any(r in later for r in tail))
            # Only cases where the shopper later clicks a category the first six do not show.
            row["tail_hit_new_category_cases"] = row["tail_hit"] if new_category_later - {-1} else np.nan
            rows.append(row)
            shown.append((number, method, catalog_ids[top]))
        if (number + 1) % 2000 == 0:
            print(f"[other-categories] {number + 1:,}/{len(cases):,}", flush=True)
    ours = pd.DataFrame(rows)

    # The recomputed frozen lists must reproduce the saved validation run.
    reference = pd.read_parquet(RUN / "per_case.parquet")
    for method in ("sasrec", "six_four_sasrec"):
        a = ours[ours.method.eq(method)].sort_values("case").ndcg.to_numpy()
        b = reference[reference.method.eq(method)].sort_values("case").ndcg.to_numpy()
        if not np.allclose(a, b):
            raise ValueError(f"{method} does not reproduce the saved validation run")
    print("[other-categories] plain SASRec and 6 + 4 reproduce the saved run exactly", flush=True)

    rng = np.random.default_rng(4285)
    differences = []
    for base_method in ("sasrec", "six_four_sasrec"):
        base = ours[ours.method.eq(base_method)].set_index("case")
        for method in ours.method.unique():
            if method == base_method:
                continue
            other = ours[ours.method.eq(method)].set_index("case").reindex(base.index)
            for metric in METRICS:
                diff = other[metric] - base[metric]
                keep = diff.notna()
                low, high = session_bootstrap(diff[keep], base.session_key[keep], 2000, rng)
                differences.append(dict(method=method, reference=base_method, metric=metric,
                                        value=float(other[metric][keep].mean()),
                                        reference_value=float(base[metric][keep].mean()),
                                        difference=float(diff[keep].mean()), ci_low=low, ci_high=high,
                                        cases=int(keep.sum())))
    differences = pd.DataFrame(differences)
    summary = ours.groupby("method", sort=False)[METRICS].mean().reset_index()
    plain = summary.set_index("method").ndcg["sasrec"]
    summary["ndcg_change_vs_sasrec"] = summary.ndcg / plain - 1
    summary["within_5_percent"] = summary.ndcg_change_vs_sasrec >= -MAX_LOSS

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "results/home_other_categories" / f"{stamp}_{week['name']}"
    folder.mkdir(parents=True)
    save_table(ours, folder / "per_case.parquet")
    save_table(pd.DataFrame([dict(case=c, method=m, position=p + 1, item_id=int(i))
                             for c, m, items in shown for p, i in enumerate(items)]), folder / "lists.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(differences, folder / "differences.csv")
    write_json(folder / "settings.json", dict(run=str(RUN.relative_to(ROOT)), pool=POOL, max_ndcg_loss=MAX_LOSS,
                                              week=week, cases=len(cases)))
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(differences[differences.metric.isin(["ndcg", "hit", "combined_ild", "distinct_categories",
                                               "tail_hit_new_category_cases"])].to_string(index=False))
    print(f"[other-categories] Saved to {folder}")


if __name__ == "__main__":
    main()
