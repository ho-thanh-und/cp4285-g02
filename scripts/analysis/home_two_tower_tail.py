"""Home page: SASRec's best six, then four discovery places from the session two-tower
(validation week only).

Idea (user, 2026-11-09): the four discovery places should come from other places but
still be relevant. The session two-tower describes products by name words, category,
price and popularity, so its ranking can reach related categories; it acts as the
relevance filter for the discovery places, and the most diverse products are then
taken from what it keeps.

Variants (places 1-6 are always SASRec's best six, as in the frozen 6 + 4; products
already clicked and the six are excluded from places 7-10):
  * `six_sasrec_four_two_tower`: the four most mutually different products (same
    distance and greedy rule as 6 + 4) among the two-tower's top 20.
  * `six_sasrec_four_two_tower_categories`: walking down the two-tower's top 100,
    its best product of each category not among the six's categories, one per
    category; if fewer than four, the two-tower's next best products fill in.

Declared before the run (2026-11-09), the 6 + 4 rule: acceptable if NDCG@10 is at
most 5% below plain SASRec on the validation week; among acceptable policies, most
combined ILD. Several variants are now being tried on the validation week, so a
variant that passes must be confirmed on the selection week before it is used; the
final test (May 16 - June 1) is not used.

Models: the frozen SASRec of the validation run (6j, trained Feb-Apr) and the
session two-tower saved by the home-page validation run (6g, trained April).
Cases and metrics are the same 20,000 validation cases and `list_metrics`.

    uv run python scripts/analysis/home_two_tower_tail.py
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

from session_rerank.adaptive_policies import diverse_tail_order
from session_rerank.event_ablation import session_bootstrap
from session_rerank.home_page import (HomeData, build_session_two_tower, click_sequences, evaluation_cases,
                                      item_vectors, read_home_settings, session_vector, top_excluding)
from session_rerank.item_knn import read_knn_settings
from session_rerank.sequence_models import build_model, diverse_lists, item_weights, ranked
from session_rerank.study_files import save_table, write_json
from session_rerank.ymal import list_metrics

SASREC_RUN = ROOT / "results/sequence_models/20261108T175008Z_validation"
TOWER_RUN = ROOT / "results/home_page/20261108T095146Z_validation"
TOWER_POOL, CATEGORY_POOL, MAX_LOSS = 20, 100, 0.05
METRICS = ["ndcg", "recall", "hit", "mrr", "combined_ild", "distinct_categories",
           "tail_new_category", "tail_hit", "tail_hit_new_category_cases"]


def main():
    saved = json.loads((SASREC_RUN / "sequence_settings.json").read_text())
    home, model_settings = saved["home"], saved["settings"]
    settings = read_home_settings(ROOT / "config/home_page.json")
    k, slots = home["top_k"], home["relevance_slots"]
    data = HomeData(settings, read_knn_settings(settings["knn_config"]))
    catalog_ids, categories = data.catalog_ids, data.categories

    # Session two-tower of the home-page validation run.
    tower_settings = json.loads((TOWER_RUN / "home_settings.json").read_text())["settings"]
    tower = build_session_two_tower(data.names, data.category_ids, data.prices, data.popularity_inputs,
                                    tower_settings)
    tower.load_state_dict(torch.load(TOWER_RUN / "session_two_tower.pt", weights_only=True))
    tower.eval()
    vectors = item_vectors(tower, data)

    # Frozen SASRec of the validation run, with its vocabulary and fallback.
    vocab_rows = np.load(SASREC_RUN / "vocabulary_rows.npy")
    vocab = pd.Series(np.arange(1, len(vocab_rows) + 1), index=vocab_rows)
    ids_to_rows = pd.Series(vocab.index.to_numpy(), index=vocab.to_numpy())
    window = model_settings["training_window"]
    clicks = np.zeros(len(catalog_ids))
    for _, _, items in click_sequences(home["raw_directory"], window["start"], window["end_exclusive"],
                                       data.item_index):
        np.add.at(clicks, items, 1)
    fallback_rows = np.argsort(-clicks, kind="stable")
    sasrec = build_model("sasrec", len(vocab), model_settings)
    sasrec.load_state_dict(torch.load(SASREC_RUN / "sasrec.pt", map_location="cpu", weights_only=True))
    sasrec.eval()
    weights = item_weights(sasrec)

    week = home["window"]
    cases, _ = evaluation_cases(click_sequences(home["raw_directory"], week["start"], week["end_exclusive"],
                                                data.item_index), home["max_cases"], home["seed"])
    print(f"[two-tower tail] {week['name']} week, {len(cases):,} cases", flush=True)
    rows, shown = [], []
    for number, (key, prefix, later) in enumerate(cases):
        members, scores, personal = ranked(sasrec, prefix, vocab, ids_to_rows, fallback_rows, model_settings,
                                           max(k, home["pool_size"]), weights)
        six = list(members[:slots])
        six_categories = {int(categories[r]) for r in six}
        tower_scores = vectors @ session_vector(tower, vectors, prefix, tower_settings["history_items"])
        candidates = top_excluding(tower_scores, set(prefix) | set(six), CATEGORY_POOL, scores=True)
        # Variant 1: four most mutually different among the two-tower's top 20.
        pool = candidates[:TOWER_POOL]
        reward = np.nan_to_num(data.diversity.distances(catalog_ids[pool].tolist())["combined"], nan=0.0)
        four = pool[diverse_tail_order(tower_scores[pool], reward, 0, k - slots)]
        # Variant 2: the two-tower's best product of each category the six do not show.
        tail, seen = [], set(six_categories)
        for r in candidates:
            c = int(categories[r])
            if c >= 0 and c not in seen:
                tail.append(r)
                seen.add(c)
                if len(tail) == k - slots:
                    break
        tail += [r for r in candidates if r not in tail][:k - slots - len(tail)]
        lists = {"sasrec": members[:k],
                 "six_four_sasrec": diverse_lists(members, scores, data.diversity, catalog_ids, home)["six_four"],
                 "six_sasrec_four_two_tower": np.array(six + list(four)),
                 "six_sasrec_four_two_tower_categories": np.array(six + tail)}
        new_category_later = {int(categories[r]) for r in later} - six_categories - {-1}
        for method, top in lists.items():
            head_categories = {int(categories[r]) for r in top[:slots]}
            row = dict(session_key=key, case=number, method=method, clicked_so_far=len(prefix),
                       used_session=personal,
                       **list_metrics(top, later, prefix[-1], catalog_ids, categories, data.diversity, slots))
            row["tail_new_category"] = float(any(int(categories[r]) not in head_categories for r in top[slots:]))
            row["tail_hit"] = float(any(r in later for r in top[slots:]))
            row["tail_hit_new_category_cases"] = row["tail_hit"] if new_category_later else np.nan
            rows.append(row)
            shown.append((number, method, catalog_ids[top]))
        if (number + 1) % 2000 == 0:
            print(f"[two-tower tail] {number + 1:,}/{len(cases):,}", flush=True)
    ours = pd.DataFrame(rows)

    reference = pd.read_parquet(SASREC_RUN / "per_case.parquet")
    for method in ("sasrec", "six_four_sasrec"):
        a = ours[ours.method.eq(method)].sort_values("case").ndcg.to_numpy()
        b = reference[reference.method.eq(method)].sort_values("case").ndcg.to_numpy()
        if not np.allclose(a, b):
            raise ValueError(f"{method} does not reproduce the saved validation run")
    print("[two-tower tail] plain SASRec and 6 + 4 reproduce the saved run exactly", flush=True)

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
    summary["ndcg_change_vs_sasrec"] = summary.ndcg / summary.set_index("method").ndcg["sasrec"] - 1
    summary["within_5_percent"] = summary.ndcg_change_vs_sasrec >= -MAX_LOSS

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "results/home_two_tower_tail" / f"{stamp}_{week['name']}"
    folder.mkdir(parents=True)
    save_table(ours, folder / "per_case.parquet")
    save_table(pd.DataFrame([dict(case=c, method=m, position=p + 1, item_id=int(i))
                             for c, m, items in shown for p, i in enumerate(items)]), folder / "lists.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(differences, folder / "differences.csv")
    write_json(folder / "settings.json", dict(sasrec_run=str(SASREC_RUN.relative_to(ROOT)),
                                              tower_run=str(TOWER_RUN.relative_to(ROOT)), tower_pool=TOWER_POOL,
                                              category_pool=CATEGORY_POOL, max_ndcg_loss=MAX_LOSS, week=week,
                                              cases=len(cases)))
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print(differences[differences.metric.isin(["ndcg", "hit", "distinct_categories",
                                               "tail_hit_new_category_cases"])].to_string(index=False))
    print(f"[two-tower tail] Saved to {folder}")


if __name__ == "__main__":
    main()
