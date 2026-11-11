"""Prepare small anonymous-session tables without training or opening confirmation labels."""

import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone


RAW_FILES = [
    "products.csv", "product-categories.csv", "train-queries.csv",
    "train-clicks.csv", "train-item-views.csv", "train-purchases.csv",
]


def file_hash(path):
    """Return the SHA-256 fingerprint used to identify the exact source bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_protocol(path):
    """Read sample sizes/date periods and reject overlapping split definitions."""
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    previous_end = None
    expected = ["train", "validation", "selection", "confirmation"]
    if list(protocol["splits"]) != expected:
        raise ValueError(f"Define splits in chronological order: {expected}")
    # The end date belongs outside the split. The next split can start on that date.
    for name, split in protocol["splits"].items():
        start, end = split["start"], split["end_exclusive"]
        if start >= end or (previous_end is not None and start < previous_end):
            raise ValueError(f"Overlapping or invalid date interval: {name}")
        if split["target_queries"] <= 0:
            raise ValueError("Sample targets must be positive")
        previous_end = end
    if protocol["history_max_events"] <= 0:
        raise ValueError("history_max_events must be positive")
    return protocol


def select_sessions(raw, protocol):
    """Choose complete daily sessions using query metadata only, never click outcomes.

    Shuffle eligible sessions with a fixed seed. Include whole sessions until
    the query target is reached, so actual query counts may exceed the target.
    Skip a whole daily session if any query in it has hidden labels.
    Returns selected session metadata and all its query metadata.
    """
    import numpy as np
    import pandas as pd

    # Personal IDs, candidate lists, and post-query duration are not read here.
    meta = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=[
        "queryId", "sessionId", "eventdate", "timeframe", "is.test",
    ])
    if not meta.queryId.is_unique or meta.isna().any().any():
        raise ValueError("Query IDs must be unique and sampling metadata complete")
    # A session is one source session ID on one date. This keeps its queries together.
    groups = meta.groupby(["sessionId", "eventdate"], sort=True).agg(
        query_count=("queryId", "size"), has_hidden=("is.test", "any"),
        first_query_timeframe_ms=("timeframe", "min"),
        last_query_timeframe_ms=("timeframe", "max"),
    ).reset_index()
    chosen = []
    for index, (name, spec) in enumerate(protocol["splits"].items()):
        pool = groups.loc[
            (groups.eventdate >= spec["start"])
            & (groups.eventdate < spec["end_exclusive"])
            & ~groups.has_hidden
        ]
        if pool.empty:
            raise ValueError(f"No eligible anonymous daily sessions for {name}")
        shuffled = pool.iloc[np.random.default_rng(protocol["seed"] + index).permutation(len(pool))]
        # Find the first point where the running query count reaches the target.
        # Keep that entire session even if it puts the sample over the target.
        count = min(len(shuffled), int(np.searchsorted(
            shuffled.query_count.cumsum(), spec["target_queries"], side="left"
        )) + 1)
        selected = shuffled.iloc[:count].copy()
        selected["split"] = name
        chosen.append(selected)
    selected = pd.concat(chosen, ignore_index=True)
    # Give each daily session a key for joining tables without exporting the source ID.
    selected["session_key"] = [
        "s_" + hashlib.sha256(f"{day}|{sid}".encode()).hexdigest()[:16]
        for sid, day in zip(selected.sessionId, selected.eventdate)
    ]
    if not selected.session_key.is_unique:
        raise ValueError("Session-key collision or overlapping selected sessions")
    selected_meta = meta.merge(
        selected[["sessionId", "eventdate", "session_key", "split"]],
        on=["sessionId", "eventdate"], validate="many_to_one",
    )
    return selected, selected_meta


def load_visible_queries(raw, selected_meta):
    """Read full rows only for training/validation/selection, preserving candidate order.

    Confirmation query IDs were reserved from metadata. Their candidate lists
    and click outcomes are deliberately not materialized by this stage.
    """
    import pandas as pd

    visible = selected_meta.loc[selected_meta.split != "confirmation"]
    return load_query_rows(raw, visible)


def load_query_rows(raw, metadata):
    """Read candidate lists for explicitly supplied IDs while retaining their split.

    Confirmation callers must check the saved frozen choices before calling this.
    Reading in chunks avoids loading every raw candidate list into memory.
    """
    import pandas as pd

    wanted = set(metadata.queryId)
    frames = []
    fields = ["queryId", "sessionId", "eventdate", "timeframe",
              "searchstring.tokens", "categoryId", "items"]
    # Read in batches because the candidate lists make this CSV large.
    for chunk in pd.read_csv(raw / "train-queries.csv", sep=";", usecols=fields,
                             dtype={"items": "string", "searchstring.tokens": "string"},
                             chunksize=25000):
        frames.append(chunk.loc[chunk.queryId.isin(wanted)])
    queries = pd.concat(frames, ignore_index=True).merge(
        metadata[["queryId", "session_key", "split"]], on="queryId", validate="one_to_one",
    )
    if len(queries) != len(wanted):
        raise ValueError("Selected queries were lost while reading the raw candidate lists")
    return queries.sort_values(["eventdate", "session_key", "timeframe", "queryId"])


def build_candidates_and_clicks(raw, source_queries):
    """Expand candidates and match each click to its original one-based result position.

    Missing/duplicate candidate matches receive an explicit status and no
    guessed rank. Nonclicks mean no recorded click, not verified dislike.
    Returns candidate labels and the original click-event table.
    """
    import numpy as np
    import pandas as pd

    candidates = source_queries[["queryId", "split", "items"]].copy()
    # Expand a list like "42,17,93" into three rows with ranks 1, 2, and 3.
    candidates["itemId"] = candidates["items"].map(
        lambda value: [int(item) for item in str(value).split(",") if item]
    )
    candidates = candidates.explode("itemId").drop(columns="items")
    if candidates.itemId.isna().any():
        raise ValueError("Selected query has no candidates")
    candidates["itemId"] = candidates.itemId.astype("int64")
    candidates["original_rank"] = candidates.groupby("queryId", sort=False).cumcount() + 1

    click_source = pd.read_csv(raw / "train-clicks.csv", sep=";")
    clicks = click_source.loc[click_source.queryId.isin(source_queries.queryId)].merge(
        source_queries[["queryId", "session_key", "split", "eventdate", "timeframe"]]
        .rename(columns={"timeframe": "query_timeframe_ms"}),
        on="queryId", validate="many_to_one",
    ).rename(columns={"timeframe": "click_timeframe_ms"})
    if (clicks.click_timeframe_ms <= clicks.query_timeframe_ms).any():
        raise ValueError("A target click is not strictly after its source query")

    # A click has a known rank only when its item occurs once in that query's list.
    positions = candidates.groupby(["queryId", "itemId"]).original_rank.agg(
        candidate_occurrences="size", original_rank="first",
    ).reset_index()
    clicks = clicks.merge(positions, on=["queryId", "itemId"], how="left", validate="many_to_one")
    clicks["rank_match_status"] = np.select(
        [clicks.candidate_occurrences.isna(), clicks.candidate_occurrences.gt(1)],
        ["missing_candidate", "ambiguous_duplicate"], default="matched",
    )
    clicks.loc[clicks.rank_match_status != "matched", "original_rank"] = pd.NA
    clicks["original_rank"] = clicks.original_rank.astype("Int64")
    # Equal click timestamps share an order number. We do not invent their order.
    clicks["click_time_order"] = clicks.groupby("queryId").click_timeframe_ms.rank(method="dense").astype("int64")
    clicks["click_delay_ms"] = clicks.click_timeframe_ms - clicks.query_timeframe_ms
    # Join click counts back to the shown items. Zero means no click was recorded.
    counts = clicks.groupby(["queryId", "itemId"]).size().rename("click_count").reset_index()
    candidates = candidates.merge(counts, on=["queryId", "itemId"], how="left", validate="many_to_one")
    candidates["click_count"] = candidates.click_count.fillna(0).astype("int64")
    candidates["clicked"] = candidates.click_count.gt(0)
    candidates = candidates.rename(columns={"queryId": "query_id", "itemId": "item_id"})
    clicks = clicks.rename(columns={"queryId": "query_id", "itemId": "item_id"})
    return (
        candidates[["query_id", "split", "item_id", "original_rank", "clicked", "click_count"]],
        clicks[["query_id", "session_key", "split", "eventdate", "item_id", "click_timeframe_ms",
                "original_rank", "rank_match_status", "click_time_order", "click_delay_ms"]],
    )


def build_events(raw, source_queries, clicks, selected_sessions, *, include_confirmation=False):
    """Combine current-session queries, clicks, views, and purchases into one event log.

    Personal user IDs are excluded at CSV reading. The log contains complete
    selected visits, including later events; only build_histories() makes
    prediction inputs. include_confirmation is reserved for the frozen evaluation.
    """
    import pandas as pd

    # Use the session ID and date to attach views and purchases to the right visit.
    allowed = selected_sessions if include_confirmation else selected_sessions.loc[
        selected_sessions.split != "confirmation"
    ]
    mapping = allowed[["sessionId", "eventdate", "session_key", "split"]]
    frames = []
    queries = source_queries[["queryId", "session_key", "split", "eventdate", "timeframe"]].rename(
        columns={"queryId": "query_id", "timeframe": "timeframe_ms"})
    queries["event_type"] = "query"
    frames.append(queries)
    click_events = clicks[["query_id", "session_key", "split", "eventdate", "item_id", "click_timeframe_ms"]].rename(
        columns={"click_timeframe_ms": "timeframe_ms"})
    click_events["event_type"] = "click"
    frames.append(click_events)
    for filename, event_type in [("train-item-views.csv", "view"), ("train-purchases.csv", "purchase")]:
        fields = ["sessionId", "eventdate", "itemId", "timeframe"]
        if event_type == "purchase":
            fields.append("ordernumber")
        # Read order numbers as text so long identifiers keep every digit.
        frame = pd.read_csv(raw / filename, sep=";", usecols=fields,
                            dtype={"ordernumber": "string"} if event_type == "purchase" else None).merge(
            mapping, on=["sessionId", "eventdate"], validate="many_to_one",
        ).rename(columns={"itemId": "item_id", "timeframe": "timeframe_ms", "ordernumber": "order_number"})
        frame["event_type"] = event_type
        frames.append(frame.drop(columns="sessionId"))
    events = pd.concat(frames, ignore_index=True)
    for field in ["query_id", "item_id"]:
        events[field] = events[field].astype("Int64")
    if "order_number" not in events:
        events["order_number"] = pd.Series(pd.NA, index=events.index, dtype="string")
    else:
        events["order_number"] = events.order_number.astype("string")
    events = events.sort_values(["session_key", "timeframe_ms", "event_type"], kind="stable").reset_index(drop=True)
    events["event_id"] = [f"e_{i:08d}" for i in range(len(events))]
    # Events recorded at the same millisecond share a time group.
    events["time_group"] = events.groupby("session_key").timeframe_ms.rank(method="dense").astype("int64")
    return events[["event_id", "session_key", "split", "eventdate", "event_type", "timeframe_ms",
                   "query_id", "item_id", "order_number", "time_group"]]


def build_histories(queries, events, max_events):
    """Return allowed query-time prefixes, limited to the most recent max_events.

    Use only this daily session and timestamps strictly below the query time.
    Exclude current-query events. The complete session remains in events;
    available/used counts expose any prefix truncation for review.
    """
    import pandas as pd

    groups = {key: frame for key, frame in events.groupby("session_key", sort=False)}
    history_frames = []
    available, used = [], []
    columns = ["query_id", "session_key", "split", "history_event_id", "event_type", "item_id",
               "source_query_id", "event_timeframe_ms", "age_ms", "history_position"]
    for query in queries.itertuples(index=False):
        session = groups[query.session_key]
        # Cut at the first event at or after the query time. Even ties are excluded.
        stop = session.timeframe_ms.searchsorted(query.query_timeframe_ms, side="left")
        earlier = session.iloc[:stop]
        # Views and purchases can have no query ID, so keep those earlier events too.
        earlier = earlier.loc[earlier.query_id.ne(query.query_id).fillna(True)]
        available.append(len(earlier))
        # Keep the most recent events and record both counts to show what was trimmed.
        prefix = earlier.tail(max_events).copy()
        used.append(len(prefix))
        prefix = prefix.rename(columns={"query_id": "source_query_id", "event_id": "history_event_id",
                                        "timeframe_ms": "event_timeframe_ms"})
        prefix["query_id"] = query.query_id
        prefix["age_ms"] = query.query_timeframe_ms - prefix.event_timeframe_ms
        prefix["history_position"] = range(1, len(prefix) + 1)
        history_frames.append(prefix[columns])
    histories = pd.concat(history_frames, ignore_index=True)
    queries = queries.copy()
    queries["history_available_count"] = available
    queries["history_used_count"] = used
    return queries, histories


def prepare_samples(config_path=Path("config/anonymous_samples.json")):
    """Build the eight reviewable tables for the anonymous-session pilot.

    This function prepares data only. It does not train a model, score
    confirmation queries, or replace the earlier benchmark's processed data.
    Completed outputs are reused only for the same protocol/source files.
    """
    import pandas as pd

    protocol = read_protocol(config_path)
    raw, output = Path(protocol["raw_directory"]), Path(protocol["output_directory"])
    for name in RAW_FILES:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    print("Prepare anonymous-session samples: verify source files", flush=True)
    # Fingerprints let us check whether the inputs and settings match a previous run.
    hashes = {name: file_hash(raw / name) for name in RAW_FILES}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    # Reuse the saved tables only if the inputs and table contents still match.
    if output.exists():
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Output folder exists without a complete manifest; choose a fresh output folder")
        saved = json.loads(manifest_path.read_text())
        if saved["protocol_sha256"] != protocol_hash or saved["source_sha256"] != hashes:
            raise RuntimeError("Existing samples use different settings/source data; choose a fresh output folder")
        for name, expected in saved["output_sha256"].items():
            if file_hash(output / name) != expected:
                raise RuntimeError(f"Prepared table changed: {name}")
        print(f"Reuse verified samples: {output.resolve()}", flush=True)
        return output

    print("Select whole sessions without opening confirmation outcomes", flush=True)
    # 1. Sample complete sessions across the four date splits.
    selected, meta = select_sessions(raw, protocol)
    # 2. Read the visible queries and turn each shown item into a labelled row.
    source_queries = load_visible_queries(raw, meta)
    candidates, clicks = build_candidates_and_clicks(raw, source_queries)
    # 3. Build the full event log, then cut it at each query to make its history.
    events = build_events(raw, source_queries, clicks, selected)
    queries = source_queries.rename(columns={
        "queryId": "query_id", "timeframe": "query_timeframe_ms",
        "searchstring.tokens": "query_tokens", "categoryId": "query_category_id",
    })[["query_id", "session_key", "split", "eventdate", "query_timeframe_ms", "query_tokens", "query_category_id"]]
    queries["query_tokens"] = queries.query_tokens.fillna("")
    queries["candidate_count"] = queries.query_id.map(candidates.groupby("query_id").size()).astype("int64")
    queries, histories = build_histories(queries, events, protocol["history_max_events"])
    # 4. Create the session summary, reserved query list, and product catalog.
    sessions = selected.rename(columns={"eventdate": "date"})[[
        "session_key", "split", "date", "query_count", "first_query_timeframe_ms", "last_query_timeframe_ms",
    ]].sort_values(["date", "session_key"]).reset_index(drop=True)
    # Save just enough metadata to find the confirmation queries for final evaluation.
    reserved = meta.loc[meta.split == "confirmation"].rename(columns={
        "queryId": "query_id", "timeframe": "query_timeframe_ms",
    })[["query_id", "session_key", "eventdate", "query_timeframe_ms"]].sort_values(["eventdate", "session_key", "query_timeframe_ms", "query_id"])
    # Keep every catalog item, including items with no category recorded.
    products = pd.read_csv(raw / "products.csv", sep=";", dtype={"product.name.tokens": "string"}).merge(
        pd.read_csv(raw / "product-categories.csv", sep=";"), on="itemId", how="left", validate="one_to_one",
    ).rename(columns={"itemId": "item_id", "categoryId": "category_id", "pricelog2": "price_log2", "product.name.tokens": "name_tokens"})
    products["category_id"] = products.category_id.astype("Int64")
    tables = dict(products=products, sessions=sessions, queries=queries, candidates=candidates,
                  clicks=clicks, events=events, histories=histories, confirmation_ids=reserved)

    # Validate that exported inputs really obey the anonymous-session contract.
    if any("user" in column.lower() for table in tables.values() for column in table.columns):
        raise AssertionError("A personal-user field was exported")
    visible_ids = set(queries.query_id)
    if visible_ids & set(reserved.query_id) or set(events.session_key) & set(sessions.loc[sessions.split == "confirmation", "session_key"]):
        raise AssertionError("Confirmation records leaked into visible data")
    if not histories.empty:
        checked = histories.merge(queries[["query_id", "session_key", "query_timeframe_ms"]], on="query_id", suffixes=("", "_query"), validate="many_to_one")
        if not checked.session_key.eq(checked.session_key_query).all() or not checked.age_ms.gt(0).all():
            raise AssertionError("A history contains another session or a current/future event")
        if checked.source_query_id.eq(checked.query_id).fillna(False).any():
            raise AssertionError("A query's labels entered its own history")
    actual_counts = meta.groupby("session_key").size()
    if not actual_counts.eq(sessions.set_index("session_key").query_count.reindex(actual_counts.index)).all():
        raise AssertionError("Selected sessions are incomplete")

    # 5. Record sample sizes and data issues that a reviewer can inspect.
    audit = dict(
        all_sessions_complete=True, personal_identifiers_exported=False,
        confirmation_outcomes_materialized=False,
        session_definition="Supplied sessionId plus eventdate; reset state at each daily-session boundary",
        hidden_label_rule="Exclude the entire daily session if it contains any hidden-label query",
        selection_rule="Seeded uniform session shuffle; keep complete sessions until the query target is reached",
        label_meaning="Clicked means an observed query-associated click; nonclick is not verified dislike",
        event_time_meaning="Relative source timeframe in milliseconds, not wall-clock time of day",
        tied_events="Equal timestamps share time_group; tie ordering is not known",
        history_max_events=protocol["history_max_events"],
        queries_with_truncated_history=int(queries.history_available_count.gt(queries.history_used_count).sum()),
        zero_click_queries=int((~candidates.groupby("query_id").clicked.any()).sum()),
        duplicate_candidate_rows=int(candidates.duplicated(["query_id", "item_id"]).sum()),
        click_rank_status=clicks.rank_match_status.value_counts().to_dict(),
        candidates_without_product=int((~candidates.item_id.isin(products.item_id)).sum()),
        queries_with_empty_tokens=int(queries.query_tokens.eq("").sum()),
        split_counts={name: dict(
            sessions=int((sessions.split == name).sum()), queries=int(sessions.loc[sessions.split == name, "query_count"].sum()),
            target_queries=spec["target_queries"], start=spec["start"], end_exclusive=spec["end_exclusive"],
        ) for name, spec in protocol["splits"].items()},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # 6. Write the complete set in a temporary folder before moving it into place.
    # A failed write leaves the final output path free for a retry.
    with tempfile.TemporaryDirectory(prefix=output.name + ".pending-", dir=output.parent) as temporary:
        folder = Path(temporary)
        for name, frame in tables.items():
            frame.to_parquet(folder / f"{name}.parquet", index=False)
        manifest = dict(
            format_version=1, status="complete", generated_at_utc=datetime.now(timezone.utc).isoformat(),
            protocol=protocol, protocol_sha256=protocol_hash, source_sha256=hashes,
            tables={name: dict(rows=len(frame), columns=list(frame.columns)) for name, frame in tables.items()},
            output_sha256={f"{name}.parquet": file_hash(folder / f"{name}.parquet") for name in tables},
        )
        (folder / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)
    print(f"Prepared {len(tables)} tables: {output.resolve()}", flush=True)
    return output
