"""Full raw-data audit and deterministic development/held-out query samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


SPLITS = {
    "train": ("2016-04-01", "2016-04-30", 12000),
    "validation": ("2016-05-01", "2016-05-15", 6000),
    "test": ("2016-05-16", "2016-06-02", 10000),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(args.raw / "train-queries.csv", sep=";", dtype={"searchstring.tokens": "str", "items": "str"})
    clicks = pd.read_csv(args.raw / "train-clicks.csv", sep=";")
    products = pd.read_csv(args.raw / "products.csv", sep=";")
    categories = pd.read_csv(args.raw / "product-categories.csv", sep=";")
    views = pd.read_csv(args.raw / "train-item-views.csv", sep=";")
    purchases = pd.read_csv(args.raw / "train-purchases.csv", sep=";")
    assert q.queryId.is_unique and products.itemId.is_unique
    assert not categories.itemId.duplicated().any(), "Multi-category products require a set representation"
    linked = clicks.merge(q[["queryId", "sessionId", "eventdate", "timeframe", "is.test"]], on="queryId", how="left", suffixes=("", "_query"), validate="many_to_one")
    assert linked.sessionId.notna().all()
    assert (linked.timeframe > linked.timeframe_query).all()
    linked.to_parquet(output / "clicks.parquet", index=False)
    q.drop(columns=["items", "searchstring.tokens"]).to_parquet(output / "queries_metadata.parquet", index=False)
    cat = categories.set_index("itemId").categoryId.to_dict()
    product_set = set(products.itemId)
    tokens_available = set(products.loc[products['product.name.tokens'].notna(), 'itemId'])
    candidate_counts, category_counts = [], []
    exposures = Counter()
    missing_products = missing_tokens = duplicate_lists = 0
    print("Auditing all candidate lists and building pre-April exposure counts...", flush=True)
    for i, row in enumerate(q.itertuples(index=False)):
        # Named tuples rename dotted column names; address stable ordinary columns.
        items = [int(x) for x in row.items.split(",")]
        candidate_counts.append(len(items))
        category_counts.append(len({cat.get(x, -1) for x in items}))
        duplicate_lists += len(set(items)) != len(items)
        missing_products += sum(x not in product_set for x in items)
        missing_tokens += sum(x not in tokens_available for x in items)
        if row.eventdate < "2016-03-31":
            exposures.update(items)
        if (i + 1) % 200000 == 0:
            print(f"  {i+1:,} / {len(q):,}", flush=True)
    q["candidate_count"] = candidate_counts
    q["candidate_category_count"] = category_counts
    pd.DataFrame(exposures.items(), columns=["itemId", "exposures"]).to_parquet(output / "exposures.parquet", index=False)
    split_report = {}
    for offset, (name, (start, end, size)) in enumerate(SPLITS.items()):
        eligible = q[(q.eventdate >= start) & (q.eventdate < end) & ~q['is.test']]
        # Uniform query sampling without replacement across the full date interval.
        selected = eligible.sample(n=min(size, len(eligible)), random_state=4285 + offset).sort_values(["eventdate", "sessionId", "timeframe", "queryId"])
        selected.to_parquet(output / f"{name}.parquet", index=False)
        identifiers = ",".join(map(str, sorted(selected.queryId)))
        split_report[name] = dict(start=start, end_exclusive=end, eligible_queries=len(eligible), sampled_queries=len(selected), query_ids_sha256=hashlib.sha256(identifiers.encode()).hexdigest(), daily_sample_counts=selected.eventdate.value_counts().sort_index().to_dict())
    audit = {
        "queries": len(q), "provided_sessions": q.sessionId.nunique(),
        "daily_sessions": len(q[["sessionId", "eventdate"]].drop_duplicates()),
        "session_ids_spanning_dates": int(q.groupby('sessionId').eventdate.nunique().gt(1).sum()),
        "anonymous_query_fraction": float(q.userId.isna().mean()),
        "hidden_label_queries": int(q['is.test'].sum()),
        "observed_label_queries": int((~q['is.test']).sum()),
        "queries_with_clicks": clicks.queryId.nunique(), "click_rows": len(clicks),
        "duplicate_click_rows": int(clicks.duplicated().sum()),
        "clicks_not_strictly_after_source_query": int((linked.timeframe <= linked.timeframe_query).sum()),
        "click_delay_quantiles": (linked.timeframe - linked.timeframe_query).quantile([0,.5,.9,.99,1]).to_dict(),
        "view_rows": len(views), "purchase_rows": len(purchases), "products": len(products),
        "categories": categories.categoryId.nunique(),
        "candidate_rows": int(np.sum(candidate_counts)),
        "candidate_count_quantiles": {str(x):float(np.quantile(candidate_counts,x)) for x in [0,.25,.5,.75,.9,1]},
        "single_category_candidate_lists": int(np.sum(np.array(category_counts)==1)),
        "single_category_candidate_fraction": float(np.mean(np.array(category_counts)==1)),
        "candidate_lists_with_duplicate_items": int(duplicate_lists),
        "candidate_rows_without_product": int(missing_products),
        "candidate_rows_without_name_tokens": int(missing_tokens),
        "query_timeframe_range": [int(q.timeframe.min()),int(q.timeframe.max())],
        "view_timeframe_range": [int(views.timeframe.min()),int(views.timeframe.max())],
        "daily_query_counts": q.eventdate.value_counts().sort_index().to_dict(),
        "splits": split_report,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/data_audit.json").write_text(json.dumps(audit, indent=2, default=int) + "\n")
    print(json.dumps({k:v for k,v in audit.items() if k not in ['daily_query_counts','splits']}, indent=2, default=int), flush=True)


if __name__ == "__main__":
    main()
