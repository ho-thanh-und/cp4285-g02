"""Prepare the full anonymous study in a disk-backed database and feature batches.

The database stores one candidate-list string per query. Expanding these lists
only when building a feature batch avoids holding millions of candidate rows
in a pandas table. Confirmation has its own database, opened after the freeze.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import psutil

import numpy as np
import pandas as pd

from .core import tokens, jaccard_distance
from .samples import select_sessions, file_hash
from .study_files import write_json, verify_files, stage_complete, finish_stage, save_table
from .ranker_features import PRODUCT_FEATURES, SESSION_FEATURES
from .ranking_settings import reranking_settings

FEATURES = PRODUCT_FEATURES + SESSION_FEATURES
PURCHASE_FEATURES = ["prior_product_purchases", "session_item_purchases", "session_purchases"]


def session_key(session_id, date):
    """Use the same anonymous daily-session key as the pilot preprocessing."""
    return "s_" + hashlib.sha256(f"{date}|{session_id}".encode()).hexdigest()[:16]


def reserve_expanded(config):
    """Keep all training sessions, fixed development sets, and new confirmation IDs."""
    pilot = Path(config["pilot_directory"])
    old = json.loads((pilot / "manifest.json").read_text())
    verify_files(pilot, old["output_sha256"])
    raw = Path(old["protocol"]["raw_directory"])
    verify_files(raw, old["source_sha256"])
    protocol = json.loads(json.dumps(old["protocol"]))
    protocol["splits"]["train"]["target_queries"] = 10**9
    protocol["splits"]["confirmation"]["target_queries"] = 10**9
    sessions, meta = select_sessions(raw, protocol)
    if config.get("validation_only", False):
        # A manual learning-curve run opens only training/validation outcomes.
        # select_sessions returns training sessions in a reproducible shuffled
        # order. Taking a prefix makes 2k a subset of 10k, then 50k, then all.
        train = sessions[sessions.split.eq("train")]
        target = config.get("training_target_queries")
        available = int(train.query_count.sum())
        if target is not None:
            if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
                raise ValueError("Training query target must be a positive integer or all.")
            if target > available:
                raise ValueError(f"Requested {target:,} queries but only {available:,} are eligible; use --queries all.")
            count = int(np.searchsorted(train.query_count.cumsum(), target, side="left")) + 1
            train = train.iloc[:count]
        sessions = pd.concat([train, sessions[sessions.split.eq("validation")]], ignore_index=True)
        meta = meta[meta.session_key.isin(sessions.session_key)]
        old_queries = pd.read_parquet(pilot / "queries.parquet")
        if set(meta.loc[meta.split.eq("validation"), "queryId"]) != set(old_queries.loc[old_queries.split.eq("validation"), "query_id"]):
            raise ValueError("Validation query IDs changed.")
        return raw, sessions, meta, old
    previous = set(pd.read_parquet(pilot / "confirmation_ids.parquet").session_key)
    pool = sessions[sessions.split.eq("confirmation") & ~sessions.session_key.isin(previous)]
    pool = pool.iloc[np.random.default_rng(config["confirmation_seed"]).permutation(len(pool))]
    cutoff = min(len(pool), int(np.searchsorted(
        pool.query_count.cumsum(), config["confirmation_target_queries"], side="left",
    )) + 1)
    fresh = pool.iloc[:cutoff]
    if fresh.query_count.sum() < config["confirmation_target_queries"]:
        raise ValueError("Not enough unused eligible sessions for the new confirmation sample.")
    sessions = pd.concat([sessions[~sessions.split.eq("confirmation")], fresh], ignore_index=True)
    meta = meta[meta.session_key.isin(sessions.session_key)]
    # Development queries must be identical to the pilot, including full sessions.
    old_queries = pd.read_parquet(pilot / "queries.parquet")
    for split in ("validation", "selection"):
        if set(meta.loc[meta.split.eq(split), "queryId"]) != set(old_queries.loc[old_queries.split.eq(split), "query_id"]):
            raise ValueError("Development query IDs changed.")
    return raw, sessions, meta, old


def create_database(raw, meta, path):
    """Load selected query rows and session events without loading personal user IDs."""
    temporary = path.with_suffix(".pending.sqlite")
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    try:
        db.executescript("""
        CREATE TABLE queries(query_id INTEGER PRIMARY KEY, session_key TEXT, split TEXT,
            eventdate TEXT, timeframe INTEGER, query_tokens TEXT, category INTEGER, items TEXT);
        CREATE TABLE events(event_id INTEGER PRIMARY KEY, session_key TEXT, eventdate TEXT,
            kind TEXT, timeframe INTEGER, query_id INTEGER, item_id INTEGER, order_number TEXT);
        CREATE TABLE attribution(event_id INTEGER PRIMARY KEY, query_id INTEGER,
            item_id INTEGER, status TEXT, eligible_queries INTEGER);
        """)
        identity = meta.set_index("queryId")[["session_key", "split"]].to_dict("index")
        fields = ["queryId", "sessionId", "eventdate", "timeframe", "searchstring.tokens", "categoryId", "items"]
        for chunk in pd.read_csv(raw / "train-queries.csv", sep=";", usecols=fields,
                                 dtype={"items": "string", "searchstring.tokens": "string"},
                                 chunksize=10000):
            selected = chunk[chunk.queryId.isin(identity)]
            # CSV column order may differ from the requested usecols order.
            # Named fields for dotted columns are unstable, so set an explicit order first.
            records = []
            for row in selected[fields].itertuples(index=False, name=None):
                qid, sid, date, time, text, category, items = row
                records.append((int(qid), identity[qid]["session_key"], identity[qid]["split"],
                                date, int(time), "" if pd.isna(text) else str(text),
                                None if pd.isna(category) else int(category), str(items)))
            db.executemany("INSERT INTO queries VALUES(?,?,?,?,?,?,?,?)", records)
        if db.execute("SELECT count(*) FROM queries").fetchone()[0] != len(meta):
            raise ValueError("Some selected query rows were missing.")
        db.execute("""INSERT INTO events(session_key,eventdate,kind,timeframe,query_id)
                      SELECT session_key,eventdate,'query',timeframe,query_id FROM queries""")
        mapping = meta.drop_duplicates(["sessionId", "eventdate"]).set_index(
            ["sessionId", "eventdate"]
        ).session_key.to_dict()
        # Source clicks have query IDs. Views/purchases are attached by daily session.
        for chunk in pd.read_csv(raw / "train-clicks.csv", sep=";", chunksize=50000):
            chunk = chunk[chunk.queryId.isin(identity)]
            values = [(identity[r.queryId]["session_key"], int(r.timeframe), int(r.queryId), int(r.itemId))
                      for r in chunk.itertuples(index=False)]
            db.executemany("""INSERT INTO events(session_key,kind,timeframe,query_id,item_id,eventdate)
                VALUES(?,'click',?,?,?,(SELECT eventdate FROM queries WHERE query_id=?))""",
                [(key, time, qid, item, qid) for key, time, qid, item in values])
        for filename, kind in (("train-item-views.csv", "view"), ("train-purchases.csv", "purchase")):
            fields = ["sessionId", "eventdate", "itemId", "timeframe"]
            if kind == "purchase":
                fields.append("ordernumber")
            for chunk in pd.read_csv(raw / filename, sep=";", usecols=fields,
                                     dtype={"ordernumber": "string"} if kind == "purchase" else None,
                                     chunksize=50000):
                records = []
                for row in chunk[fields].itertuples(index=False, name=None):
                    sid, date, item, time, *order = row
                    key = mapping.get((sid, date))
                    if key is not None:
                        records.append((key, date, kind, int(time), int(item), str(order[0]) if order else None))
                db.executemany("""INSERT INTO events(session_key,eventdate,kind,timeframe,item_id,order_number)
                                  VALUES(?,?,?,?,?,?)""", records)
        db.executescript("""
        CREATE INDEX query_session ON queries(session_key,timeframe);
        -- Cover split/session lookups without visiting large candidate-list rows.
        -- This matters on Windows drives, where many tiny random reads are slow.
        CREATE INDEX query_split_session ON queries(split,session_key);
        CREATE INDEX event_session ON events(session_key,timeframe);
        CREATE INDEX event_query ON events(query_id,kind);
        CREATE INDEX event_day ON events(eventdate,kind,item_id);
        """)
        attribute_purchases(db)
        db.commit()
    finally:
        db.close()
    temporary.replace(path)


def attribute_purchases(db):
    """Last-touch attribution using time and candidate membership within a session.

    A unique latest preceding query receives the purchase. Equal latest query
    times are ambiguous and receive no purchase label. Every event keeps an
    audit status, including purchases without a prior matching candidate.
    """
    purchases = db.execute(
        "SELECT event_id,session_key,timeframe,item_id FROM events WHERE kind='purchase'"
    ).fetchall()
    records = []
    for event_id, session, time, item in purchases:
        earlier = db.execute("""SELECT query_id,timeframe,items FROM queries
                                WHERE session_key=? AND timeframe<? ORDER BY timeframe DESC""",
                             (session, time)).fetchall()
        eligible = [(qid, qt) for qid, qt, items in earlier if item in parse_items(items)]
        latest = [qid for qid, qt in eligible if qt == eligible[0][1]] if eligible else []
        status = "matched" if len(latest) == 1 else "ambiguous_time" if latest else "no_prior_candidate"
        records.append((event_id, latest[0] if status == "matched" else None, item, status, len(eligible)))
    db.executemany("INSERT INTO attribution VALUES(?,?,?,?,?)", records)


def parse_items(value):
    """Preserve original candidate order while converting comma-separated IDs."""
    items = [int(item) for item in value.split(",") if item]
    if not items or len(set(items)) != len(items):
        raise ValueError("A query has empty or duplicate candidates.")
    return items


def purchase_audit(db):
    """Count purchase support, unmatched cases, and sensitivity to repeated searches."""
    audit = {}
    splits = [r[0] for r in db.execute("SELECT DISTINCT split FROM queries")]
    for split in splits:
        sessions = {r[0] for r in db.execute("SELECT DISTINCT session_key FROM queries WHERE split=?", (split,))}
        rows = db.execute("""SELECT a.status,a.query_id,a.item_id,a.eligible_queries,e.session_key
                             FROM attribution a JOIN events e USING(event_id)""").fetchall()
        rows = [row for row in rows if row[4] in sessions]
        audit[split] = dict(
            purchase_events=len(rows),
            status=dict(Counter(row[0] for row in rows)),
            purchase_queries=len({r[1] for r in rows if r[1] is not None}),
            purchase_sessions=len({r[4] for r in rows}),
            purchased_products=len({r[2] for r in rows}),
            multiple_eligible_query_events=sum(r[3] > 1 for r in rows),
        )
    return audit


def prepare_expanded(config, folder):
    """Publish the development database and confirmation reservation as one stage."""
    if stage_complete(folder, "data"):
        return
    if config.get("validation_only", False):
        target = config.get("training_target_queries")
        print(f"[1/4] Prepare {'all' if target is None else f'{target:,}'} training queries and fixed validation", flush=True)
    else:
        print("[1/7] Reserve fresh confirmation IDs and prepare all training sessions", flush=True)
    raw, sessions, meta, old = reserve_expanded(config)
    reserved = meta[meta.split.eq("confirmation")]
    columns = ["queryId", "session_key", "eventdate", "timeframe"]
    save_table(reserved[columns], folder / "confirmation_ids.parquet")
    save_table(sessions.drop(columns="sessionId"), folder / "sessions.parquet")
    development = meta[~meta.split.eq("confirmation")]
    create_database(raw, development, folder / "development.sqlite")
    products = pd.read_parquet(Path(config["pilot_directory"]) / "products.parquet")
    save_table(products, folder / "products.parquet")
    with sqlite3.connect(folder / "development.sqlite") as db:
        audit = purchase_audit(db)
    write_json(folder / "data_audit.json", dict(
        splits={name: dict(queries=int(group.query_count.sum()), sessions=len(group))
                for name, group in sessions.groupby("split")},
        purchases=audit, source_sha256=old["source_sha256"], raw_directory=str(raw),
        excluded_previous_confirmation_sessions=int(pd.read_parquet(
            Path(config["pilot_directory"]) / "confirmation_ids.parquet").session_key.nunique()),
        rule="Unique latest preceding query in the same daily session containing the purchased candidate",
        confirmation_outcomes_materialized=False,
        validation_only=config.get("validation_only", False),
    ))
    finish_stage(folder, "data", [
        "confirmation_ids.parquet", "sessions.parquet", "products.parquet",
        "development.sqlite", "data_audit.json",
    ])


def prepare_confirmation(config, folder):
    """Read reserved confirmation outcomes only after choices have been frozen."""
    frozen = folder / "frozen_choices.json"
    if not frozen.exists():
        raise ValueError("Freeze choices before opening confirmation.")
    if stage_complete(folder, "confirmation_data"):
        return
    opened_path = folder / "confirmation_opened.json"
    if opened_path.exists():
        if json.loads(opened_path.read_text())["frozen_sha256"] != file_hash(frozen):
            raise ValueError("Frozen choices changed after confirmation was opened.")
    else:
        write_json(opened_path, dict(
            frozen_sha256=file_hash(frozen), opened_at_utc=datetime.now(timezone.utc).isoformat(),
        ))
    reserved = pd.read_parquet(folder / "confirmation_ids.parquet")
    raw = Path(json.loads((folder / "data_audit.json").read_text())["raw_directory"])
    audit = json.loads((folder / "data_audit.json").read_text())
    verify_files(raw, audit["source_sha256"])
    fields = ["queryId", "sessionId", "eventdate", "timeframe"]
    found = []
    for chunk in pd.read_csv(raw / "train-queries.csv", sep=";", usecols=fields, chunksize=25000):
        found.append(chunk[chunk.queryId.isin(reserved.queryId)])
    meta = pd.concat(found)
    meta["session_key"] = [session_key(sid, date) for sid, date in zip(meta.sessionId, meta.eventdate)]
    meta["split"] = "confirmation"
    pd.testing.assert_frame_equal(
        meta[reserved.columns].sort_values("queryId").reset_index(drop=True),
        reserved.sort_values("queryId").reset_index(drop=True),
    )
    create_database(raw, meta, folder / "confirmation.sqlite")
    with sqlite3.connect(folder / "confirmation.sqlite") as db:
        write_json(folder / "confirmation_purchase_audit.json", purchase_audit(db))
    finish_stage(folder, "confirmation_data", [
        "confirmation_opened.json", "confirmation.sqlite", "confirmation_purchase_audit.json",
    ])


def query_features(query, events, catalog, popularity, history_limit):
    """Construct one complete query group with strictly earlier session features."""
    qid, session, split, date, time, text, query_category, item_string = query
    items = parse_items(item_string)
    n = len(items)
    rows = np.zeros((n, len(FEATURES) + len(PURCHASE_FEATURES)), dtype=np.float32)
    product = [catalog.get(item, (np.nan, None, frozenset())) for item in items]
    rows[:, 0] = [p[0] for p in product]
    rows[:, 1] = [p[1] if p[1] is not None else np.nan for p in product]
    rows[:, 2] = [int(p[1] == query_category) if p[1] is not None and query_category is not None else 0 for p in product]
    query_tokens = tokens(text)
    rows[:, 3] = int(not query_tokens)
    rows[:, 4] = [1 - jaccard_distance(query_tokens, p[2]) if query_tokens and p[2] else np.nan for p in product]
    for index, kind in ((5, "click"), (6, "view"), (15, "purchase")):
        rows[:, index] = [popularity[kind][item] for item in items]
    # Events at the query timestamp and the query's own labels cannot enter its prefix.
    earlier = [event for event in events if event[0] < time and event[3] != qid][-history_limit:]
    click = Counter(e[2] for e in earlier if e[1] == "click")
    view = Counter(e[2] for e in earlier if e[1] == "view")
    bought = Counter(e[2] for e in earlier if e[1] == "purchase")
    interactions = [e for e in earlier if e[1] in ("click", "view")]
    latest = {e[2]: e[0] for e in interactions}
    history_categories = {catalog[item][1] for item in latest if item in catalog and catalog[item][1] is not None}
    history_tokens = {catalog[item][2] for item in latest if item in catalog and catalog[item][2]}
    rows[:, 7] = sum(click.values())
    rows[:, 8] = sum(view.values())
    rows[:, 9] = int(not interactions)
    rows[:, 10] = [click[item] for item in items]
    rows[:, 11] = [view[item] for item in items]
    rows[:, 12] = [time - latest[item] if item in latest else np.nan for item in items]
    rows[:, 13] = [int(p[1] in history_categories) for p in product]
    if history_tokens:
        rows[:, 14] = [max(1 - jaccard_distance(p[2], past) for past in history_tokens) if p[2] else 0 for p in product]
    rows[:, 16] = [bought[item] for item in items]
    rows[:, 17] = sum(bought.values())
    return items, rows, earlier, max(0, len([e for e in events if e[0] < time and e[3] != qid]) - history_limit)


def build_feature_cache(config, folder, confirmation=False):
    """Stream query groups into a float32 memory map plus compressed audit batches."""
    stage = "confirmation_features" if confirmation else "features"
    if stage_complete(folder, stage):
        return
    dbpath = folder / ("confirmation.sqlite" if confirmation else "development.sqlite")
    stage_label = "[2/4]" if config.get("validation_only", False) else "[2/7]"
    print(f"{stage_label} Build {'confirmation' if confirmation else 'development'} features in batches", flush=True)
    started = perf_counter()
    peak_rss = psutil.Process().memory_info().rss
    products = pd.read_parquet(folder / "products.parquet")
    catalog = {int(r.item_id): (r.price_log2, None if pd.isna(r.category_id) else int(r.category_id), tokens(r.name_tokens))
               for r in products.itertuples(index=False)}
    # Popularity always comes from development training sessions, even for confirmation.
    with sqlite3.connect(folder / "development.sqlite") as source:
        daily = source.execute("""SELECT eventdate,kind,item_id,count(*) FROM events
            WHERE kind IN ('click','view','purchase') AND session_key IN
            (SELECT session_key FROM queries WHERE split='train')
            GROUP BY eventdate,kind,item_id ORDER BY eventdate""").fetchall()
        training_dates = [r[0] for r in source.execute("SELECT DISTINCT eventdate FROM queries WHERE split='train'")]
    daily_counts = defaultdict(list)
    for date, kind, item, count in daily:
        daily_counts[date].append((kind, item, count))
    snapshots = {}
    counters = {kind: Counter() for kind in ("click", "view", "purchase")}
    for date in sorted(set(daily_counts) | set(training_dates)):
        snapshots[date] = {kind: counter.copy() for kind, counter in counters.items()}
        for kind, item, count in daily_counts[date]:
            counters[kind][item] += count
    db = sqlite3.connect(dbpath)
    # Sorting one split is bounded by its query rows. Keep SQLite's temporary
    # sort in memory and give event lookups a modest page cache rather than
    # issuing repeated tiny reads across the Windows filesystem boundary.
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA cache_size=-65536")
    outputs, info = [], {}
    history_limit = (reranking_settings(config)["history_max_events"] if "reranking" in config else
                     json.loads((Path(config["pilot_directory"]) / "manifest.json").read_text())["protocol"]["history_max_events"])
    try:
        splits = (("confirmation",) if confirmation else
                  ("train", "validation") if config.get("validation_only", False) else
                  ("train", "validation", "selection"))
        for split in splits:
            query_count, total = db.execute("""SELECT count(*),sum(length(items)-length(replace(items,',',''))+1)
                                  FROM queries WHERE split=?""", (split,)).fetchone()
            print(f"{split}: prepare {query_count:,} queries and {total:,} candidate rows", flush=True)
            split_dir = folder / f"{split}_features"
            split_dir.mkdir(exist_ok=True)
            matrix_path = split_dir / "X.npy"
            matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype="float32",
                                               shape=(int(total), len(FEATURES) + len(PURCHASE_FEATURES)))
            click_path, grade_path = split_dir / "click_labels.npy", split_dir / "grades.npy"
            click_labels = np.lib.format.open_memmap(click_path, mode="w+", dtype="int8", shape=(int(total),))
            grades = np.lib.format.open_memmap(grade_path, mode="w+", dtype="int8", shape=(int(total),))
            cursor = db.execute("""SELECT * FROM queries WHERE split=?
                                   ORDER BY eventdate,session_key,timeframe,query_id""", (split,))
            metadata, offset, batch_number = [], 0, 0
            while True:
                queries = cursor.fetchmany(config["query_batch_size"])
                if not queries:
                    break
                sessions = list({q[1] for q in queries})
                marks = ",".join("?" for _ in sessions)
                events = defaultdict(list)
                for row in db.execute(f"""SELECT session_key,timeframe,kind,item_id,query_id
                    FROM events WHERE session_key IN ({marks}) ORDER BY timeframe,event_id""", sessions):
                    events[row[0]].append(row[1:])
                invalid_clicks = dict(db.execute(f"""SELECT e.query_id,count(*) FROM events e
                    JOIN queries q ON q.query_id=e.query_id WHERE q.session_key IN ({marks})
                    AND e.kind='click' AND e.timeframe<=q.timeframe GROUP BY e.query_id""", sessions))
                clicks = defaultdict(Counter)
                # Only strictly later clicks become outcome labels.
                for qid, item, count in db.execute(f"""SELECT e.query_id,e.item_id,count(*)
                    FROM events e JOIN queries q ON q.query_id=e.query_id
                    WHERE q.session_key IN ({marks}) AND e.kind='click' AND e.timeframe>q.timeframe
                    GROUP BY e.query_id,e.item_id""", sessions):
                    clicks[qid][item] = count
                purchases = defaultdict(Counter)
                for qid, item, count in db.execute(f"""SELECT a.query_id,a.item_id,count(*)
                    FROM attribution a JOIN queries q ON a.query_id=q.query_id
                    WHERE q.session_key IN ({marks}) AND a.status='matched' GROUP BY a.query_id,a.item_id""", sessions):
                    purchases[qid][item] = count
                candidate_frames, history_rows = [], []
                for query in queries:
                    qid, session, _, date, time, *_ = query
                    popularity = snapshots.get(date, counters) if split == "train" else counters
                    items, features, history, truncated = query_features(
                        query, events[session], catalog, popularity, history_limit,
                    )
                    matrix[offset:offset + len(items)] = features
                    clicked = np.array([clicks[qid][item] > 0 for item in items])
                    purchased = np.array([purchases[qid][item] > 0 for item in items])
                    click_labels[offset:offset + len(items)] = clicked
                    grades[offset:offset + len(items)] = np.where(purchased, 2, clicked.astype(int))
                    candidate_frames.append(pd.DataFrame(dict(
                        query_id=qid, session_key=session, split=split, item_id=items,
                        original_rank=np.arange(1, len(items) + 1), clicked=clicked,
                        purchased=purchased, grade=np.where(purchased, 2, clicked.astype(int)),
                    )))
                    metadata.append(dict(
                        query_id=qid, session_key=session, eventdate=date, query_timeframe_ms=time,
                        offset=offset, candidates=len(items), clicks=int(clicked.sum()),
                        missing_catalog_items=sum(item not in catalog for item in items),
                        invalid_click_events=invalid_clicks.get(qid, 0),
                        unmatched_click_events=sum(count for item,count in clicks[qid].items() if item not in set(items)),
                        purchases=int(purchased.sum()), history_truncated_events=truncated,
                    ))
                    history_rows.extend(dict(
                        query_id=qid, item_id=e[2], event_type=e[1], age_ms=time-e[0],
                    ) for e in history if e[1] in ("click", "view", "purchase") and e[2] is not None)
                    offset += len(items)
                batch = split_dir / f"candidates_{batch_number:05d}.parquet"
                save_table(pd.concat(candidate_frames, ignore_index=True), batch)
                history_path = split_dir / f"history_{batch_number:05d}.parquet"
                save_table(pd.DataFrame(history_rows, columns=["query_id","item_id","event_type","age_ms"]), history_path)
                outputs.extend([str(batch.relative_to(folder)), str(history_path.relative_to(folder))])
                batch_number += 1
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                if batch_number == 1 or batch_number % 10 == 0 or len(metadata) == query_count:
                    print(f"{split}: {len(metadata):,} queries, {offset:,} candidates", flush=True)
            if offset != total:
                raise ValueError("Candidate counts changed during feature construction.")
            matrix.flush()
            click_labels.flush()
            grades.flush()
            del matrix, click_labels, grades
            query_path = split_dir / "queries.parquet"
            save_table(pd.DataFrame(metadata), query_path)
            outputs.extend(str(path.relative_to(folder)) for path in (matrix_path, query_path, click_path, grade_path))
            info[split] = dict(queries=len(metadata), candidates=offset, batches=batch_number,
                               history_truncated_queries=sum(r["history_truncated_events"] > 0 for r in metadata),
                               missing_catalog_items=sum(r["missing_catalog_items"] for r in metadata),
                               invalid_click_events=sum(r["invalid_click_events"] for r in metadata),
                               unmatched_click_events=sum(r["unmatched_click_events"] for r in metadata),
                               zero_click_queries=sum(r["clicks"]==0 for r in metadata),
                               zero_gain_queries=sum(r["clicks"]+r["purchases"]==0 for r in metadata))
    finally:
        db.close()
    write_json(folder / f"{stage}_info.json", dict(splits=info, features=FEATURES + PURCHASE_FEATURES, history_max_events=history_limit,
        seconds=perf_counter()-started, peak_observed_rss_bytes=peak_rss))
    finish_stage(folder, stage, outputs + [f"{stage}_info.json"])


def read_candidates(folder, split):
    """Evaluation splits are small enough to combine their saved candidate batches."""
    return pd.concat([pd.read_parquet(p) for p in sorted((folder / f"{split}_features").glob("candidates_*.parquet"))],
                     ignore_index=True)


def read_histories(folder, split):
    """Read earlier interactions, preserving ages for time-decay comparisons."""
    return pd.concat([pd.read_parquet(p) for p in sorted((folder / f"{split}_features").glob("history_*.parquet"))],
                     ignore_index=True)
