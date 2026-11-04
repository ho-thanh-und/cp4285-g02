"""Build an offline review of every prepared table and column, plus a SQLite copy."""

import argparse
from html import escape
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from session_rerank.samples import file_hash


TABLES = {
    "products": ("Product catalogue", "Static catalogue metadata. It includes products beyond the sampled candidate lists.", "item_id"),
    "sessions": ("Selected anonymous sessions", "Complete daily sessions, including confirmation reservations. This is not a personal user profile.", "session_key"),
    "queries": ("Visible query samples", "Training, validation, and selection queries. Confirmation queries are not materialized here.", "query_id"),
    "candidates": ("Original query result lists", "One row per returned candidate. Original order is preserved. No recorded click is not verified dislike.", "query_id + original_rank"),
    "clicks": ("Observed click events", "Original click timestamps and recoverable ranks. Missing ranks are explicit, rather than guessed.", "Event rows; no unique natural key assumed"),
    "events": ("Complete selected session events", "Queries, clicks, views, and purchases in visible sessions. This log includes later events; use histories for prediction inputs.", "event_id"),
    "histories": ("Allowed query-time histories", "Only earlier events from the same daily session. The most recent configured event window is retained.", "query_id + history_event_id"),
    "confirmation_ids": ("Reserved confirmation IDs", "Metadata only. No candidate lists or confirmation click labels were exported.", "query_id"),
}

DEFINITIONS = {
    "item_id": ("identifier", "Anonymous product identifier. Join to products.item_id. Its numeric size has no semantic meaning."),
    "price_log2": ("numeric", "Source log2 price field. This is not a currency amount."),
    "name_tokens": ("tokens", "Comma-separated anonymous product-name token IDs. The chart measures token count, not semantic meaning."),
    "category_id": ("category", "Product category identifier. Categories are labels, not distances."),
    "session_key": ("identifier", "Anonymous key derived from source sessionId and date. It must not link personal history across visits."),
    "split": ("category", "Chronological role: train, validation, selection, or confirmation."),
    "date": ("date", "Source event date in YYYY-MM-DD format."),
    "query_count": ("numeric", "All visible queries in this selected daily session. Session sampling keeps them together."),
    "first_query_timeframe_ms": ("time", "First query's relative source clock in milliseconds. It is not a wall-clock time of day."),
    "last_query_timeframe_ms": ("time", "Last query's relative source clock in milliseconds."),
    "query_id": ("identifier", "Source search-query identifier. Join to queries.query_id, or confirmation_ids for reserved metadata."),
    "eventdate": ("date", "Source event date. Together with the original session identifier, it defines the daily visit."),
    "query_timeframe_ms": ("time", "Prediction time on the source relative clock. History events must occur strictly before this value."),
    "query_tokens": ("tokens", "Anonymous search token IDs. An empty string means no query tokens were supplied; it is separate from NULL."),
    "query_category_id": ("category", "Source query category. Zero is the source's unspecified category value."),
    "candidate_count": ("numeric", "Number of products in the original candidate list. The sampler does not add products."),
    "history_available_count": ("numeric", "Eligible earlier events in this same daily session, before applying the history window."),
    "history_used_count": ("numeric", "Earlier events retained in histories after the configured history window."),
    "original_rank": ("numeric", "One-based position in the original result list. In clicks, NULL means the exact position could not be matched."),
    "clicked": ("category", "True means an observed query-associated click. False means no recorded click, not confirmed irrelevance."),
    "click_count": ("numeric", "Number of raw click events for this query/product. This is an outcome label, not a query-time feature."),
    "click_timeframe_ms": ("time", "Click timestamp on the source relative clock, in milliseconds."),
    "rank_match_status": ("category", "matched, missing_candidate, or ambiguous_duplicate. Only a unique candidate match receives a click rank."),
    "click_time_order": ("numeric", "Dense chronological order within one query's clicks. Equal timestamps receive the same order."),
    "click_delay_ms": ("time", "Click time minus its source query time, in milliseconds. It must be positive."),
    "event_id": ("identifier", "Unique prepared event-row identifier. Join histories.history_event_id to this field."),
    "event_type": ("category", "query, click, view, or purchase. Each type has a different feedback meaning."),
    "timeframe_ms": ("time", "Relative event timestamp in milliseconds. It is not a time of day."),
    "order_number": ("identifier", "Purchase order identifier, present only on purchase events. NULL on other event types is expected."),
    "time_group": ("numeric", "Dense timestamp group within a session. Equal-time events share a group; their true internal order is unknown."),
    "history_event_id": ("identifier", "Reference to the actual past event in events.event_id."),
    "source_query_id": ("identifier", "Query linked to the past event. Views and purchases have no invented query attribution, so NULL is expected."),
    "event_timeframe_ms": ("time", "Past event's source-clock timestamp. It must be below the current query timestamp."),
    "age_ms": ("time", "Current query time minus past event time. Positive age verifies strict temporal ordering."),
    "history_position": ("numeric", "Position in the stored oldest-to-newest prefix. Equal-time events are not assumed to have a true internal ordering."),
}


def number(value):
    """Format chart values compactly without changing their underlying counts."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2g}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    if abs(value) >= 10_000:
        return f"{value / 1000:.2g}k"
    return f"{value:,.3g}" if value % 1 else f"{value:,.0f}"


def scalar(value):
    """Convert pandas/NumPy scalars into values safe for JSON and HTML samples."""
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def histogram(values, bins=12, logarithmic=False):
    """Return full-data histogram counts and actual bin boundaries, including zeros."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(counts=[], edges=[], logarithmic=logarithmic)
    transformed = np.log1p(values) if logarithmic else values
    counts, edges = np.histogram(transformed, bins=bins if np.ptp(transformed) else 1)
    if logarithmic:
        edges = np.expm1(edges)
    return dict(counts=counts.tolist(), edges=edges.tolist(), logarithmic=logarithmic)


def column_profile(series, kind):
    """Calculate every statistic from the complete column, never a row sample."""
    nonnull = series.dropna()
    result = dict(rows=len(series), nulls=int(series.isna().sum()), distinct=int(nonnull.nunique()),
                  examples=[scalar(v) for v in nonnull.drop_duplicates().head(3)])
    if kind == "identifier":
        # IDs are labels. Plot repetition, not the meaningless numeric ID magnitude.
        occurrences = nonnull.value_counts()
        distribution = occurrences.value_counts().sort_index()
        if len(distribution) <= 12:
            result.update(chart="bars", labels=[str(v) for v in distribution.index], values=distribution.tolist(),
                          x_label="Distinct IDs", y_label="Occurrences per ID")
        else:
            result.update(chart="hist", **histogram(occurrences.to_numpy(), bins=12), x_label="Occurrences per ID", y_label="Distinct IDs")
        result["plot_note"] = "Frequency of repetition; ID numbers are not measurements."
    elif kind == "category":
        counts = nonnull.astype(str).value_counts()
        top = counts.head(8)
        labels, values = top.index.tolist(), top.tolist()
        if len(counts) > 8:
            labels.append("Other values")
            values.append(int(counts.iloc[8:].sum()))
        result.update(chart="bars", labels=labels, values=values, x_label="Rows", y_label="Value")
        result["plot_note"] = "Top eight values plus the remainder. NULL is counted separately."
    elif kind == "date":
        counts = nonnull.astype(str).value_counts().sort_index()
        result.update(chart="dates", labels=counts.index.tolist(), values=counts.tolist(), x_label="Source date", y_label="Rows")
    else:
        if kind == "tokens":
            result["empty_strings"] = int(nonnull.astype(str).eq("").sum())
            numeric = nonnull.astype(str).map(lambda v: len([t for t in v.split(",") if t]))
            result["plot_note"] = "Number of anonymous tokens per row; empty strings have zero tokens."
            x_label = "Token count"
        else:
            numeric = pd.to_numeric(nonnull, errors="coerce")
            x_label = "Value (ms)" if kind == "time" else "Value"
        finite = numeric[np.isfinite(numeric.astype(float))]
        if len(finite):
            result.update(min=float(finite.min()), median=float(finite.median()), max=float(finite.max()),
                          nonfinite=int(len(numeric) - len(finite)))
        log = bool(kind == "time" and len(finite) > 0 and finite.min() >= 0)
        result.update(chart="hist", **histogram(finite, logarithmic=log), x_label=x_label, y_label="Rows")
        if log:
            result["plot_note"] = "Log-spaced millisecond bins, including zero; every finite value is counted."
    return result


def chart_svg(profile, title):
    """Draw an offline, directly labeled SVG distribution with accessible data text."""
    width = 360
    counts = profile.get("values", profile.get("counts", []))
    if not counts:
        return '<p class="muted">No non-NULL values to plot.</p>'
    maximum = max(counts) or 1
    parts = []
    if profile["chart"] == "bars":
        height = max(156, 72 + len(counts) * 25)
        left, right, top = 134, 45, 26
        parts.append(f'<text x="0" y="15">{escape(profile["y_label"])}</text>')
        plot_width = width - left - right
        for i, (label, value) in enumerate(zip(profile["labels"], counts)):
            y = top + i * 25
            label = str(label)
            shown = label if len(label) <= 18 else label[:16] + "…"
            parts.append(f'<text x="{left-8}" y="{y+13}" text-anchor="end"><title>{escape(label)}</title>{escape(shown)}</text>')
            parts.append(f'<rect x="{left}" y="{y}" width="{value/maximum*plot_width:.2f}" height="18" class="mark"><title>{escape(label)}: {value:,}</title></rect>')
            parts.append(f'<text x="{left+value/maximum*plot_width+5:.2f}" y="{y+13}">{number(value)}</text>')
        bottom = top + len(counts) * 25 + 3
        parts.append(f'<text x="{left}" y="{bottom+23}">0</text><text x="{width-right}" y="{bottom+23}" text-anchor="end">{number(maximum)}</text>')
        parts.append(f'<text x="{left+plot_width/2}" y="{height-2}" text-anchor="middle">{escape(profile["x_label"])}</text>')
    else:
        height = 188
        left, right, top, bottom = 54, 12, 18, 132
        plot_width, plot_height = width-left-right, bottom-top
        bar_width = plot_width / len(counts)
        for i, value in enumerate(counts):
            h = value / maximum * plot_height
            label = profile["labels"][i] if profile["chart"] == "dates" else f'{profile["edges"][i]:.5g} to {profile["edges"][i+1]:.5g}'
            parts.append(f'<rect x="{left+i*bar_width:.2f}" y="{bottom-h:.2f}" width="{max(.3,bar_width-1):.2f}" height="{h:.2f}" class="mark"><title>{escape(label)}: {value:,}</title></rect>')
        for fraction in [0, .5, 1]:
            y = bottom - fraction * plot_height
            parts.append(f'<text x="{left-7}" y="{y+5:.2f}" text-anchor="end">{number(maximum*fraction)}</text>')
        indices = sorted(set([0, len(counts)//2, len(counts)-1]))
        for i in indices:
            x = left + (i + .5)*bar_width
            anchor = "start" if i == 0 else "end" if i == len(counts)-1 else "middle"
            if profile["chart"] == "dates":
                label = profile["labels"][i][5:]
            else:
                value = profile["edges"][i] if i != len(counts)-1 else profile["edges"][-1]
                label = number(value)
            parts.append(f'<text x="{x:.2f}" y="{bottom+23}" text-anchor="{anchor}">{escape(label)}</text>')
        parts.append(f'<text x="{left+plot_width/2}" y="181" text-anchor="middle">{escape(profile["x_label"])}</text>')
        parts.append(f'<text x="4" y="12">{escape(profile["y_label"])}</text>')
    description = "; ".join(f"{label}: {count}" for label, count in zip(profile.get("labels", []), counts))
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}"><title>{escape(title)}</title><desc>{escape(description or "Full-column histogram. Hover a bar for its exact bin count.")}</desc>'+"".join(parts)+"</svg>"


def example_table(frame):
    """Show actual rows, keeping long strings accessible through native details."""
    rows = []
    # Include one non-first example to avoid representing only a sorted file's start.
    indices = [*range(min(4, len(frame))), len(frame)//2] if len(frame) else []
    # Include informative records instead of a first-rows-only preview.
    # The complete data remains available in SQLite and the CSV pack.
    masks = []
    if "rank_match_status" in frame:
        masks.append(frame.rank_match_status.ne("matched"))
    if "clicked" in frame:
        masks.append(frame.clicked)
    if "event_type" in frame:
        masks.append(frame.event_type.eq("purchase"))
    if "query_tokens" in frame:
        masks.append(frame.query_tokens.fillna("").ne(""))
    if "history_used_count" in frame:
        masks.append(frame.history_used_count.gt(0))
    for mask in masks:
        matches = np.flatnonzero(mask.fillna(False).to_numpy())
        if len(matches):
            indices.append(int(matches[0]))
    indices = sorted(set(indices))
    for row in frame.iloc[indices].itertuples(index=False, name=None):
        cells = []
        for value in row:
            value = scalar(value)
            text = "NULL" if value is None else str(value)
            if len(text) > 50:
                rendered = f'<details><summary>{escape(text[:42])}…</summary><code>{escape(text)}</code></details>'
            else:
                rendered = escape(text)
            cells.append(f'<td>{rendered}</td>')
        rows.append('<tr>'+"".join(cells)+'</tr>')
    return '<div class="table-scroll"><table><thead><tr>'+"".join(f'<th>{escape(c)}</th>' for c in frame.columns)+'</tr></thead><tbody>'+"".join(rows)+'</tbody></table></div>'


def relationship_diagram(tables):
    """Show the eight tables and the main parent/child joins without inventing keys."""
    positions = dict(products=(0,0),sessions=(2,0),candidates=(0,1),queries=(1,1),
                     confirmation_ids=(2,1),clicks=(0,2),events=(2,2),histories=(1,3))
    nodes = []
    for name, (x,y) in positions.items():
        px, py = 20+x*270, 22+y*150
        nodes.append(f'<g><rect x="{px}" y="{py}" width="225" height="92" rx="5" class="node"/><text x="{px+12}" y="{py+27}" class="node-name">{name}</text><text x="{px+12}" y="{py+53}">{len(tables[name]):,} rows · {len(tables[name].columns)} columns</text><text x="{px+12}" y="{py+76}">{escape(TABLES[name][2][:29])}</text></g>')
    edges = [
        (132,114,132,172,"item_id"),(620,114,425,172,"session_key"),
        (672,114,672,172,"session_key"),(290,215,245,215,"query_id"),
        (340,264,175,322,"query_id"),(505,264,650,322,"query_id"),
        (672,414,485,472,"event_id"),(425,264,425,472,"query_id"),
    ]
    paths = []
    for x1,y1,x2,y2,label in edges:
        paths.append(f'<path d="M{x1} {y1} L{x2} {y2}" marker-end="url(#arrow)"/>')
        if y1 != y2:
            paths.append(f'<text x="{(x1+x2)/2+4}" y="{(y1+y2)/2-5}" class="edge-label">{label}</text>')
    return '<div class="schema-scroll"><svg class="schema" viewBox="0 0 810 586" role="img" aria-label="Eight prepared tables and their main joins"><title>Prepared table relationships</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" class="arrow"/></marker></defs>'+"".join(paths+nodes)+'</svg></div>'


def write_database(tables, destination):
    """Export every row to a browsable SQLite file without changing the Parquet data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(temporary) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, index=False, if_exists="fail", chunksize=10000)
        keys = dict(products="item_id",sessions="session_key",queries="query_id",
                    candidates="query_id, original_rank",events="event_id",
                    histories="query_id, history_event_id",confirmation_ids="query_id")
        for name, key in keys.items():
            connection.execute(f'CREATE UNIQUE INDEX "{name}_key" ON "{name}" ({key})')
        for name, frame in tables.items():
            if "query_id" in frame and name not in ["queries", "confirmation_ids"]:
                connection.execute(f'CREATE INDEX "{name}_query" ON "{name}" (query_id)')
        for name, frame in tables.items():
            actual = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            if actual != len(frame):
                raise AssertionError(f"SQLite row count differs: {name}")
        connection.commit()
    temporary.replace(destination)


def make_review(source, output, database):
    """Profile all actual output columns, save review HTML/JSON, and export SQLite."""
    manifest = json.loads((source / "manifest.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        if file_hash(source / name) != digest:
            raise RuntimeError(f"Table changed since preparation: {name}")
    tables = {name: pd.read_parquet(source / f"{name}.parquet") for name in TABLES}
    profiles = {}
    sections = []
    for name, frame in tables.items():
        schema = pq.read_schema(source / f"{name}.parquet")
        fields = []
        schema_rows = []
        profiles[name] = {}
        for column in frame.columns:
            kind, purpose = DEFINITIONS[column]
            profile = column_profile(frame[column], kind)
            profiles[name][column] = profile
            null_fraction = 100 * profile["nulls"] / max(1, len(frame))
            stats = f'<span>NULL <strong>{profile["nulls"]:,}</strong> ({null_fraction:.2f}%)</span><span>Distinct <strong>{profile["distinct"]:,}</strong></span>'
            if "empty_strings" in profile:
                stats += f'<span>Empty strings <strong>{profile["empty_strings"]:,}</strong></span>'
            if "min" in profile:
                stats += f'<span>Min / median / max <strong>{number(profile["min"])} / {number(profile["median"])} / {number(profile["max"])}</strong></span>'
            examples = ", ".join(str(v)[:65] for v in profile["examples"])
            svg = chart_svg(profile, f"{name}.{column} distribution")
            fields.append(f'<article class="field" id="{name}-{column}"><h3>{escape(column)}</h3><p class="type">{escape(str(schema.field(column).type))} · {kind}</p><p>{escape(purpose)}</p><div class="field-stats">{stats}</div>{svg}<p class="caption">{escape(profile.get("plot_note", "Full non-NULL column values."))}</p><p class="examples">Example values: {escape(examples)}</p></article>')
            schema_rows.append(f'<tr><td><a href="#{name}-{column}">{escape(column)}</a></td><td>{escape(str(schema.field(column).type))}</td><td>{profile["nulls"]:,}</td><td>{escape(purpose)}</td></tr>')
        title, purpose, key = TABLES[name]
        sections.append(f'<section id="table-{name}"><h2>{name}</h2><p class="lead">{escape(title)} · {len(frame):,} rows · {len(frame.columns)} columns</p><p>{escape(purpose)}</p><p><strong>Key:</strong> {escape(key)}</p><details><summary>Column dictionary and types</summary><div class="table-scroll"><table><thead><tr><th>Column</th><th>Parquet type</th><th>NULL rows</th><th>Meaning</th></tr></thead><tbody>'+"".join(schema_rows)+'</tbody></table></div></details><h3 class="subheading">Actual sample rows</h3>'+example_table(frame)+'<h3 class="subheading">Every column</h3><div class="fields">'+"".join(fields)+'</div><a class="back" href="#contents">Back to contents</a></section>')

    # These are independent quality facts, not a manufactured composite score.
    issues = [
        f'{audit["click_rank_status"].get("missing_candidate",0):,} click events refer to products outside their recorded candidate list. Their original_rank is NULL.',
        f'{audit["zero_click_queries"]:,} visible queries have no clicked candidate. They remain in the sample for your review.',
        f'{audit["queries_with_empty_tokens"]:,} visible queries have no search tokens. Empty strings are distinct from missing NULL values.',
        f'{audit["queries_with_truncated_history"]:,} query histories reached the configured {audit["history_max_events"]}-event window.',
        'NULL item_id on query events, NULL query attribution on views/purchases, and NULL order_number on non-purchase events are expected.',
        'The confirmation table reserves IDs only. Outcome graphs, labels, and confirmation events have not been exported.',
    ]
    split_rows = []
    for name, spec in audit["split_counts"].items():
        split_rows.append(f'<tr><td>{name}</td><td>{spec["sessions"]:,}</td><td>{spec["queries"]:,}</td><td>{spec["target_queries"]:,}</td><td>{spec["start"]} to before {spec["end_exclusive"]}</td></tr>')
    contents = '<ul>'+"".join(f'<li><a href="#table-{name}">{name}</a> — {len(frame.columns)} columns</li>' for name, frame in tables.items())+'</ul>'
    page = HTML_HEAD + '<main><header><p class="eyebrow">CP4285 · Step 1 · Real prepared data</p><h1>Anonymous session samples</h1><p class="lead">'+f'{len(tables)} tables · {sum(len(f.columns) for f in tables.values())} column profiles · full-column statistics'+'</p><p>Inspect the table relationships, actual rows, and distribution of every column. All measurements use the complete prepared tables. Example rows are labeled samples.</p><p><strong>Source:</strong> '+escape(str(source.resolve()))+'</p><p><a href="'+escape(database.name)+'">Full SQLite database</a> · <a href="'+escape(output.with_suffix(".profiles.json").name)+'">Exact profile values</a></p></header><nav id="contents" aria-label="Contents"><h2>Tables</h2>'+contents+'</nav><section><h2>Before you approve the data</h2><ul>'+"".join('<li>'+escape(issue)+'</li>' for issue in issues)+'</ul><p><strong>Passed:</strong> whole selected sessions retained; no personal user fields; strictly earlier same-session histories; original candidate order retained.</p><p><strong>Scope:</strong> model training and model metrics have not run. Histories use the declared daily-session definition, not a verified 30-minute inactivity definition.</p></section><section><h2>Chronological samples</h2><p>Query targets are soft. Complete-session sampling can overshoot the target. The unused gaps are 30 April, 8 May, and 15 May.</p><div class="table-scroll"><table><thead><tr><th>Split</th><th>Sessions</th><th>Queries</th><th>Query target</th><th>Date interval</th></tr></thead><tbody>'+"".join(split_rows)+'</tbody></table></div></section><section><h2>Table relationships</h2>'+relationship_diagram(tables)+'<p class="caption">Main joins are shown. All stored columns appear in the profiles below. NULL references are expected where the source does not identify a product or query.</p></section>'+"".join(sections)+'<footer>Created '+escape(manifest["generated_at_utc"])+'. Source and prepared-file checksums are recorded in manifest.json. This page works offline and uses no external scripts or fonts.</footer></main></body></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    output.with_suffix(".profiles.json").write_text(json.dumps(profiles, indent=2, allow_nan=False, default=scalar) + "\n", encoding="utf-8")
    write_database(tables, database)
    print(f'Review: {output.resolve()}', flush=True)
    print(f'SQLite: {database.resolve()}', flush=True)
    print(f'{len(tables)} tables; {sum(len(f.columns) for f in tables.values())} column profiles', flush=True)


HTML_HEAD = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CP4285 Anonymous Sample Review</title><style>
:root{color-scheme:light dark;--bg:#f6f8f8;--paper:#ffffff;--fg:#183037;--muted:#52666e;--accent:#137c73;--line:#d6e1e2;--soft:#eef4f4}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:30px 32px 75px;background:var(--paper)}h1{font-size:2.4rem;line-height:1.15;margin:.3em 0}h2{font-size:1.5rem;line-height:1.25;margin:0 0 16px}h3{font-size:1.02rem;margin:0;overflow-wrap:anywhere}p{margin:9px 0;max-width:92ch}a{color:var(--accent);text-underline-offset:.18em}li{margin:8px 0}header{padding-bottom:25px}.eyebrow,.type,.caption,.examples{color:var(--muted)}.eyebrow{font-size:.85rem;letter-spacing:.06em}.lead{font-size:1.1rem}.type{font-size:.85rem;margin:3px 0 9px}.caption,.examples{font-size:.84rem;overflow-wrap:anywhere}section,nav{padding:28px 0;border-top:1px solid var(--line)}nav ul{display:flex;flex-wrap:wrap;gap:8px 30px;padding-left:20px}nav li{width:235px}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:35px 35px;margin-top:18px}.field{min-width:0;padding-top:18px;border-top:1px solid var(--line)}.field p{font-size:.92rem}.field-stats{display:flex;flex-wrap:wrap;gap:5px 16px;font-size:.84rem;margin:12px 0}.field svg{width:100%;max-width:450px;display:block;overflow:visible}.field svg text{font:15px system-ui,sans-serif;fill:var(--fg)}.mark{fill:var(--accent)}.table-scroll,.schema-scroll{overflow-x:auto;margin:14px 0}table{width:100%;border-collapse:collapse;font-size:.86rem}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);overflow-wrap:anywhere;max-width:250px}th{background:var(--soft);white-space:nowrap}td{font-variant-numeric:tabular-nums}details{margin:12px 0}summary{color:var(--accent);cursor:pointer}code{font-size:.9em;overflow-wrap:anywhere}td details{margin:0}td details code{white-space:normal}.subheading{margin:23px 0 8px;font-weight:600}.back{display:inline-block;margin-top:20px}.schema{width:100%;min-width:700px;max-width:950px}.schema text{fill:var(--fg);font:13px system-ui,sans-serif}.schema .node-name{font-size:17px;font-weight:600}.schema .node{fill:var(--soft);stroke:var(--line)}.schema path{fill:none;stroke:var(--muted);stroke-width:1.2}.schema .arrow{fill:var(--muted);stroke:none}.schema .edge-label{font-size:12px;fill:var(--muted)}footer{margin-top:36px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:.85rem}@media(max-width:720px){main{padding:22px 16px}h1{font-size:1.9rem}.fields{grid-template-columns:1fr;gap:18px}.field svg{max-width:450px}nav li{width:auto}nav ul{gap:6px 24px}}@media(prefers-color-scheme:dark){:root{--bg:#111b1e;--paper:#17252a;--fg:#e1eeee;--muted:#afc4c9;--accent:#70cdbf;--line:#3a5057;--soft:#20353c}}@media print{body,main{background:white;color:black}main{max-width:none;padding:0}.field,table{break-inside:avoid}nav{display:none}details{display:block}details>*{display:block!important}}
</style></head><body>'''


def main():
    """Choose prepared input tables and the destination HTML/SQLite review files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/anonymous_samples"))
    parser.add_argument("--output", type=Path, default=Path("results/anonymous-samples-review.html"))
    parser.add_argument("--database", type=Path, default=Path("results/anonymous-samples.sqlite"))
    args = parser.parse_args()
    make_review(args.source, args.output, args.database)


if __name__ == "__main__":
    main()
