"""Does a click on a product have a view of the same product in the same session?

Matches clicks and views on session, date and product, and reports how often a
view follows a click, how soon, and how many views have no click behind them.
Results are recorded in EXPERIMENT_LOG.md (section "Dataset facts").

    uv run python scripts/analysis/click_view.py data/raw
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw = Path(sys.argv[1])
q = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "sessionId", "eventdate", "timeframe", "is.test"])
q = q.rename(columns={"timeframe": "q_time"})
c = pd.read_csv(raw / "train-clicks.csv", sep=";").rename(columns={"timeframe": "c_time"}).merge(q, on="queryId")
v = pd.read_csv(raw / "train-item-views.csv", sep=";", usecols=["sessionId", "itemId", "timeframe", "eventdate"])
v = v.rename(columns={"timeframe": "v_time"})
key = ["sessionId", "eventdate", "itemId"]
print(f"clicks {len(c):,}  views {len(v):,}")

# Sessions that have any view at all vs click sessions.
click_sessions = c[["sessionId", "eventdate"]].drop_duplicates()
view_sessions = v[["sessionId", "eventdate"]].drop_duplicates()
both = click_sessions.merge(view_sessions)
print(f"click sessions {len(click_sessions):,}; with >=1 view {len(both):,} ({len(both)/len(click_sessions):.1%})")

# Same product viewed in the same session at any time.
c["row"] = np.arange(len(c))
m = c.merge(v, on=key, how="left")
any_view = m.groupby("row").v_time.apply(lambda s: s.notna().any())
print(f"\nclicks whose product is viewed in the same session at ANY time: {any_view.mean():.1%}")

# Timing relative to the click (same units; timeframe is ms from session start).
m["gap_s"] = (m.v_time - m.c_time) / 1000
after = m[m.gap_s >= 0].groupby("row").gap_s.min()
before = m[m.gap_s < 0].groupby("row").gap_s.max()
print(f"  with a view at/after the click: {len(after)/len(c):.1%}")
print(f"  with a view only before the click: {len(before.index.difference(after.index))/len(c):.1%}")
print("  seconds from click to first view after it (quantiles 10/50/90/99):",
      after.quantile([.1, .5, .9, .99]).round(1).tolist())
print(f"  view within 60 s after the click: {(after <= 60).sum()/len(c):.1%}")

# By visible sessions of the project: no hidden query, April window.
vis = c[(~c["is.test"]) & (c.eventdate >= "2016-04-01") & (c.eventdate < "2016-04-30")]
print(f"\nproject training pool clicks {len(vis):,}: product viewed in same session at any time "
      f"{any_view[vis.row].mean():.1%}, view at/after click {vis.row.isin(after.index).mean():.1%}")

# The other direction: views that have a click on the same product in the session.
cc = c[key + ["c_time"]]
w = v.reset_index().merge(cc, on=key, how="left")
w["gap_s"] = (w.v_time - w.c_time) / 1000
v_any = w.groupby("index").c_time.apply(lambda s: s.notna().any())
v_after = w[(w.gap_s >= 0) & (w.gap_s <= 600)].groupby("index").size()
print(f"\nviews with a click on the same product in the same session (any time): {v_any.mean():.1%}")
print(f"views with such a click in the 10 minutes before: {len(v_after)/len(v):.1%}")
