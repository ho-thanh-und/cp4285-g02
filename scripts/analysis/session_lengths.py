"""Per-session activity in the raw Diginetica files (session = sessionId + eventdate).

Counts queries, clicks, views, orders and purchased items per session for three
scopes: all sessions, sessions with a query, and the project's April training
pool. Prints distribution tables and saves session_counts.parquet.
Results are recorded in EXPERIMENT_LOG.md (section "Dataset facts").

    uv run python scripts/analysis/session_lengths.py data/raw <output folder>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
key = ["sessionId", "eventdate"]
q = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "sessionId", "eventdate", "is.test"])
c = pd.read_csv(raw / "train-clicks.csv", sep=";", usecols=["queryId"]).merge(q[["queryId"] + key], on="queryId")
v = pd.read_csv(raw / "train-item-views.csv", sep=";", usecols=key)
p = pd.read_csv(raw / "train-purchases.csv", sep=";", usecols=key + ["ordernumber"])

sessions = pd.concat([q[key], v[key], p[key]]).drop_duplicates().set_index(key)
counts = pd.DataFrame(index=sessions.index)
counts["queries"] = q.groupby(key).size()
counts["clicks"] = c.groupby(key).size()
counts["views"] = v.groupby(key).size()
counts["purchased_items"] = p.groupby(key).size()
counts["orders"] = p.groupby(key).ordernumber.nunique()
counts["has_hidden_query"] = q.groupby(key)["is.test"].any()
counts = counts.fillna({"queries": 0, "clicks": 0, "views": 0, "purchased_items": 0, "orders": 0,
                        "has_hidden_query": False})
counts = counts.reset_index()
counts.to_parquet(out / "session_counts.parquet", index=False)

scopes = {
    "A. all sessions (any activity)": counts,
    "B. sessions with >=1 query": counts[counts.queries > 0],
    "C. project training pool (Apr 1-29, no hidden query)": counts[
        (counts.queries > 0) & ~counts.has_hidden_query.astype(bool)
        & (counts.eventdate >= "2016-04-01") & (counts.eventdate < "2016-04-30")],
}
columns = ["queries", "clicks", "views", "orders", "purchased_items"]
for name, frame in scopes.items():
    print(f"\n== {name}: {len(frame):,} sessions")
    stats = pd.DataFrame({
        "total": frame[columns].sum(),
        "mean": frame[columns].mean(),
        "median": frame[columns].median(),
        "p75": frame[columns].quantile(.75),
        "p90": frame[columns].quantile(.9),
        "p99": frame[columns].quantile(.99),
        "max": frame[columns].max(),
        "share_with_any": (frame[columns] > 0).mean(),
    })
    print(stats.round(3).to_string())
    bins = [-1, 0, 1, 2, 3, 4, 5, 10, np.inf]
    labels = ["0", "1", "2", "3", "4", "5", "6-10", "11+"]
    shares = pd.DataFrame({col: pd.cut(frame[col], bins, labels=labels).value_counts(normalize=True)
                           .reindex(labels) for col in columns})
    print("share of sessions by count:")
    print((shares * 100).round(1).astype(str).add("%").to_string())
