"""Home page: order a visit's clicked products by first click (as trained and tested)
or by last click (re-clicks move to the end)? Validation or selection week only.

The evaluation sequences keep each product at its first click (home_page.click_sequences
uses the earliest timeframe), and the website copies that (serving.distinct_in_order).
On the website a shopper who returns to an earlier interest by re-clicking products
therefore does not move the home page (EXPERIMENT_LOG.md 6r).

This script rebuilds the cases from the raw click stream with re-clicks kept:
  * a case sits before each product's first click, exactly like evaluation_cases
    (same case list, same 20,000-case sample, same targets: products first clicked later);
  * input A: the products clicked so far in order of first click (reproduces the saved run);
  * input B: the same products in order of last click.
Policies: plain SASRec, 6 + 4 over the top 20 (tested) and over the top 100 (website).

Declared rule (6r): the website switches to B if, for 6 + 4 over the top 100 on the cases
where A and B differ, the 95% interval of the NDCG@10 difference B - A does not lie
entirely below zero, on the validation week and on the selection week.

    uv run python scripts/analysis/home_repeat_order.py                       # validation
    uv run python scripts/analysis/home_repeat_order.py --run results/sequence_models/20261108T182139Z_selection
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
from session_rerank.home_page import click_sequences
from session_rerank.sequence_models import build_model, item_weights, ranked
from session_rerank.study_files import save_table, write_json
from session_rerank.ymal import list_metrics

RUN = ROOT / "results/sequence_models/20261108T175008Z_validation"
POOLS = (20, 100)


def click_streams(raw_directory, start, end_exclusive, item_index):
    """Like click_sequences, but every click event is kept: (key, rows by time, repeats included)."""
    raw = Path(raw_directory)
    queries = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "sessionId", "eventdate", "is.test"])
    queries = queries[(queries.eventdate >= start) & (queries.eventdate < end_exclusive)]
    queries = queries[~queries.groupby(["sessionId", "eventdate"])["is.test"].transform("any")]
    clicks = pd.read_csv(raw / "train-clicks.csv", sep=";").merge(queries, on="queryId")
    clicks = clicks[clicks.itemId.isin(item_index.index)]
    clicks = clicks.sort_values(["sessionId", "eventdate", "timeframe", "itemId"], kind="stable")
    clicks["row"] = item_index.loc[clicks.itemId].to_numpy()
    return [(f"{sid}|{date}", group.row.tolist())
            for (sid, date), group in clicks.groupby(["sessionId", "eventdate"], sort=False)]


def first_order(events):
    seen, out = set(), []
    for r in events:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def last_order(events):
    return list(reversed(first_order(list(reversed(events)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    run = ROOT / args.run
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
    streams = click_streams(home["raw_directory"], week["start"], week["end_exclusive"], item_index)
    # The reference sequences, to check that the case list is evaluation_cases' list.
    reference = click_sequences(home["raw_directory"], week["start"], week["end_exclusive"], item_index)
    if [(k_, s) for k_, _, s in reference] != [(key, first_order(events)) for key, events in streams]:
        raise ValueError("The click streams do not reproduce click_sequences")
    cases = []
    for key, events in streams:
        distinct = first_order(events)
        first_at = {}
        for position, r in enumerate(events):
            first_at.setdefault(r, position)
        for t in range(1, len(distinct)):
            prefix_events = events[:first_at[distinct[t]]]
            cases.append((key, first_order(prefix_events), last_order(prefix_events), set(distinct[t:])))
    total = len(cases)
    if total > home["max_cases"]:
        keep = np.sort(np.random.default_rng(home["seed"]).choice(total, home["max_cases"], replace=False))
        cases = [cases[i] for i in keep]
    print(f"[repeat order] {week['name']} week, {len(cases):,} of {total:,} cases; "
          f"{np.mean([a != b for _, a, b, _ in cases]):.1%} have a different last-click order", flush=True)

    rows = []
    for number, (key, first, last, later) in enumerate(cases):
        differs = first != last
        start = len(rows)
        for order_name, prefix in (("first", first), ("last", last)):
            if order_name == "last" and not differs:  # same input, same lists: copy this case's rows
                rows.extend([dict(r, order="last") for r in rows[start:]])
                continue
            members, scores, personal = ranked(model, prefix, vocab, ids_to_rows, fallback_rows, model_settings,
                                               max(POOLS), weights)
            reward = np.nan_to_num(diversity.distances(catalog_ids[members].tolist())["combined"], nan=0.0)
            lists = {"sasrec": members[:k]}
            for pool in POOLS:
                lists[f"six_four_pool{pool}"] = members[diverse_tail_order(scores[:pool], reward[:pool, :pool], slots, k)]
            for method, top in lists.items():
                metrics = list_metrics(top, later, prefix[-1], catalog_ids, categories, diversity, slots)
                rows.append(dict(session_key=key, case=number, order=order_name, method=method, differs=differs,
                                 ndcg=metrics["ndcg"], hit=metrics["hit"], mrr=metrics["mrr"],
                                 recall=metrics["recall"]))
        if (number + 1) % 2000 == 0:
            print(f"[repeat order] {number + 1:,}/{len(cases):,}", flush=True)
    ours = pd.DataFrame(rows)

    saved_cases = pd.read_parquet(run / "per_case.parquet")
    for mine, theirs in (("sasrec", "sasrec"), ("six_four_pool20", "six_four_sasrec")):
        a = ours[ours.method.eq(mine) & ours.order.eq("first")].sort_values("case").ndcg.to_numpy()
        b = saved_cases[saved_cases.method.eq(theirs)].sort_values("case").ndcg.to_numpy()
        if not np.allclose(a, b):
            raise ValueError(f"first-click order does not reproduce the saved {theirs}")
    print("[repeat order] first-click order reproduces the saved run exactly", flush=True)

    rng = np.random.default_rng(4285)
    results = []
    for method in ("sasrec", "six_four_pool20", "six_four_pool100"):
        frame = ours[ours.method.eq(method)]
        first = frame[frame.order.eq("first")].set_index("case")
        last = frame[frame.order.eq("last")].set_index("case").reindex(first.index)
        for subset, mask in (("all cases", np.ones(len(first), bool)), ("order differs", first.differs.to_numpy())):
            for metric in ("ndcg", "hit", "mrr"):
                diff = (last[metric] - first[metric])[mask]
                low, high = session_bootstrap(diff, first.session_key[mask], 2000, rng)
                results.append(dict(method=method, subset=subset, cases=int(mask.sum()), metric=metric,
                                    first_order=float(first[metric][mask].mean()),
                                    last_order=float(last[metric][mask].mean()),
                                    difference=float(diff.mean()), ci_low=low, ci_high=high))
    results = pd.DataFrame(results)
    rule = results[(results.method == "six_four_pool100") & (results.subset == "order differs")
                   & (results.metric == "ndcg")].iloc[0]
    passed = not rule.ci_high < 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "results/home_repeat_order" / f"{stamp}_{week['name']}"
    folder.mkdir(parents=True)
    save_table(ours, folder / "per_case.parquet")
    save_table(results, folder / "differences.csv")
    write_json(folder / "settings.json", dict(run=str(run.relative_to(ROOT)), pools=list(POOLS), week=week,
                                              cases=len(cases), total_cases=total,
                                              rule_passed=bool(passed)))
    pd.set_option("display.width", 220)
    print(results.to_string(index=False))
    print(f"[repeat order] rule (pool 100, order differs, NDCG@10 interval not entirely below zero): "
          f"{'PASSED' if passed else 'FAILED'}")
    print(f"[repeat order] Saved to {folder}")


if __name__ == "__main__":
    main()
