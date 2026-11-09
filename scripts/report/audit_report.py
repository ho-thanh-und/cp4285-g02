"""Audit REPORT.md: recompute every checked number from the result files and raw data.

    uv run python scripts/report/audit_report.py

Each check computes a value from its source, formats it the way the report
writes it, and looks for that exact text in REPORT.md. Prints one line per
check and exits with an error if any check fails. Numbers the report quotes from
one-off analyses in EXPERIMENT_LOG.md (§2) are not recomputed here; the report
names them as dataset facts.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
R = ROOT / "results"
RUNS = dict(
    knn_validation=R / "item_knn/20261108T081521Z_queries_50000",
    category_selection=R / "heldout/20261108T082748Z_selection",
    category_confirmation=R / "heldout/20261108T193305Z_confirmation",
    item_validation=R / "ymal/20261108T090850Z_validation",
    item_selection=R / "ymal/20261108T092942Z_selection",
    item_confirmation=R / "ymal/20261108T193505Z_confirmation",
    home_validation=R / "sequence_models/20261108T175008Z_validation",
    home_selection=R / "sequence_models/20261108T182139Z_selection",
    home_confirmation=R / "sequence_models/20261108T200822Z_confirmation",
    home_cases_validation=R / "home_page/20261108T095146Z_validation",
    home_cases_selection=R / "home_page/20261108T102455Z_selection",
    home_cases_confirmation=R / "home_page/20261108T194937Z_confirmation",
    confirmation=R / "confirmation/20261108T193024Z_confirmation",
    errors=R / "error_analysis/20261108T105517Z",
    exposure=R / "exposure/20261108T184420Z",
    grid=R / "sequence_grid/grid_summary.csv",
    alpha=R / "alpha_sweep/20261108T044340Z_queries_10000/gap_vs_fixed_curve.csv",
    grid_best=R / "sequence_grid/sasrec_dim64_blocks2_dropout0.5_epochs30_lr0.001/20261108T153317Z_validation",
    data=R / "learning_curve_v2/queries_50000",
)
TEXT = " ".join((ROOT / "REPORT.md").read_text(encoding="utf-8").replace("−", "-").split())
results = []


def check(name, snippet):
    found = " ".join(snippet.replace("−", "-").split()) in TEXT
    results.append((found, name, snippet))


def d3(value):
    return f"{value:+.3f}"


def diff(row):
    return f"{row.difference:+.3f} ({row.ci_low:+.3f}, {row.ci_high:+.3f})"


def summary(run, segment, column="ndcg"):
    table = pd.read_csv(run / "summary.csv")
    return table[table.segment.eq(segment)].set_index("method")[column]


def difference(run, method, reference, metric="ndcg", segment=None):
    table = pd.read_csv(run / "differences.csv")
    mask = table.method.eq(method) & table.reference.eq(reference) & table.metric.eq(metric)
    if segment:
        mask &= table.segment.eq(segment)
    rows = table[mask]
    assert len(rows) == 1, (run, method, reference, metric, len(rows))
    return rows.iloc[0]


# ------------------------------------------------------------------ data ---
raw = ROOT / "data/raw"
queries = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["sessionId", "userId", "eventdate"])
check("queries", f"| Queries (category browses and text searches) | {len(queries):,} |")
sessions = queries[["sessionId", "eventdate"]].drop_duplicates()
check("daily sessions", f"| Daily sessions with at least one query (session ID + date) | {len(sessions):,} |")
check("no user ID", f"| Query rows without a user ID | {queries.userId.isna().mean():.0%} |")
purchases = pd.read_csv(raw / "train-purchases.csv", sep=";", usecols=["sessionId", "eventdate"]).drop_duplicates()
bought = purchases.merge(sessions, on=["sessionId", "eventdate"]).shape[0] / len(sessions)
check("sessions that buy", f"Purchases are rare** (about {bought:.0%} of sessions buy)")
products = pd.read_parquet(RUNS["data"] / "products.parquet")
check("catalog", f"| Catalog products (anonymised IDs, category, price band, name tokens) | {len(products):,} |")
check("categories", f"| Product categories | {products.category_id.nunique():,} |")
features = json.loads((RUNS["data"] / "features_info.json").read_text())["splits"]
check("training queries", f"{features['train']['queries']:,} sampled April queries "
                          f"({features['train']['candidates'] / 1e6:.1f} million")
best = json.loads((RUNS["grid_best"] / "sequence_settings.json").read_text())["training"]["sasrec"]
check("SASRec sessions", f"of which {best['training_sessions']:,} have two or more known clicks")
check("SASRec examples", f"give {best['training_positions']:,} next-click training examples")

# ---------------------------------------------------------- split sizes ---
for week, item_run, home_run in (("validation", "item_validation", "home_cases_validation"),
                                 ("selection", "item_selection", "home_cases_selection"),
                                 ("confirmation", "item_confirmation", "home_cases_confirmation")):
    pages = json.loads((RUNS[item_run] / "ymal_settings.json").read_text())["pages"]
    cases = json.loads((RUNS[home_run] / "home_settings.json").read_text())
    check(f"item pages {week}", f"{pages['evaluated_pages']:,} of {pages['evaluable_pages']:,}")
    check(f"home cases {week}", f"{cases['evaluated_cases']:,} of {cases['evaluable_cases']:,}")
category_queries = {week: int(summary(RUNS[run], "category_pages", "queries")["popularity_sort"])
                    for week, run in (("selection", "category_selection"), ("confirmation", "category_confirmation"))}
check("category queries selection", f"1,000 ({category_queries['selection']:,})")
check("category queries confirmation", f"1,000 ({category_queries['confirmation']:,})")
knn_validation = pd.read_csv(RUNS["knn_validation"] / "summary.csv")
validation_queries = int(knn_validation[knn_validation.segment.eq("category_pages")].queries.iloc[0])
check("category queries validation", f"503 ({validation_queries} category pages)")

# --------------------------------------------- table 5.3: three weeks ---
val = knn_validation[knn_validation.segment.eq("category_pages")].set_index("method").click_ndcg
sel = summary(RUNS["category_selection"], "category_pages", "click_ndcg")
conf = summary(RUNS["category_confirmation"], "category_pages", "click_ndcg")
lgbm = "lightgbm_purchase_session"
check("category LightGBM row", f"| Category pages | LightGBM `purchase_session` | "
                               f"{val['lightgbm_xendcg_purchase']:.3f} | {sel[lgbm]:.3f} | **{conf[lgbm]:.3f}** |")
check("category popularity row", f"| | Popularity sort | {val['popularity_sort']:.3f} | "
                                 f"{sel['popularity_sort']:.3f} | {conf['popularity_sort']:.3f} |")
knn_diff = pd.read_csv(RUNS["knn_validation"] / "differences.csv")
knn_diff = knn_diff[knn_diff.method.eq("lightgbm_xendcg_purchase") & knn_diff.reference.eq("popularity_sort")
                    & knn_diff.metric.eq("click_ndcg") & knn_diff.segment.eq("category_pages")].iloc[0]
check("category differences row", f"| | *Difference* | *{diff(knn_diff)}* | "
      f"*{diff(difference(RUNS['category_selection'], lgbm, 'popularity_sort', 'click_ndcg', 'category_pages'))}* | "
      f"***{diff(difference(RUNS['category_confirmation'], lgbm, 'popularity_sort', 'click_ndcg', 'category_pages'))}***"
      " |")
item = {week: summary(RUNS[f"item_{week}"], "all_pages") for week in ("validation", "selection", "confirmation")}
check("item frozen row", f"| Item page | Item-kNN + fixed 0.8 | {item['validation']['fixed_0.8_20']:.3f} | "
                         f"{item['selection']['fixed_0.8_20']:.3f} | **{item['confirmation']['fixed_0.8_20']:.3f}** |")
check("item baseline row", f"| | Category popularity | {item['validation']['category_popularity']:.3f} | "
      f"{item['selection']['category_popularity']:.3f} | {item['confirmation']['category_popularity']:.3f} |")
check("item differences row", "| | *Difference* | " + " | ".join(
    ("***" if week == "confirmation" else "*") + diff(difference(RUNS[f"item_{week}"], "fixed_0.8_20",
                                                                 "category_popularity"))
    + ("***" if week == "confirmation" else "*") for week in ("validation", "selection", "confirmation")) + " |")
home = {week: summary(RUNS[f"home_{week}"], "all cases") for week in ("validation", "selection", "confirmation")}
check("home frozen row", f"| Home page | SASRec + 6 + 4 | {home['validation']['six_four_sasrec']:.3f} | "
      f"{home['selection']['six_four_sasrec']:.3f} | **{home['confirmation']['six_four_sasrec']:.3f}** |")
check("home previous row", f"| | Item-kNN + fixed 0.8 (previous choice) | "
      f"{home['validation']['fixed_0.8_item_knn']:.3f} | {home['selection']['fixed_0.8_item_knn']:.3f} | "
      f"{home['confirmation']['fixed_0.8_item_knn']:.3f} |")
check("home differences row", "| | *Difference* | " + " | ".join(
    ("***" if week == "confirmation" else "*") + diff(difference(RUNS[f"home_{week}"], "six_four_sasrec",
                                                                 "fixed_0.8_item_knn"))
    + ("***" if week == "confirmation" else "*") for week in ("validation", "selection", "confirmation")) + " |")

# ------------------------------------------- table 5.4: final test ---
final = pd.read_csv(RUNS["confirmation"] / "final_table.csv")


def final_cell(method, reference, metric, bold=False):
    row = final[final.method.eq(method) & final.reference.eq(reference) & final.metric.eq(metric)].iloc[0]
    text = f"{row.difference:+.3f}"
    return (f"**{text}**" if bold else text) + f" ({row.ci_low:+.3f}, {row.ci_high:+.3f})"


check("final category vs popularity", "| Category pages | popularity sort | " + " | ".join([
    final_cell(lgbm, "popularity_sort", "click_ndcg", True), final_cell(lgbm, "popularity_sort", "recall_at_k"),
    final_cell(lgbm, "popularity_sort", "mrr")]) + " | – |")
check("final category vs item-kNN", "| Category pages | item-kNN | " + " | ".join([
    final_cell(lgbm, "item_knn", "click_ndcg"), final_cell(lgbm, "item_knn", "recall_at_k"),
    final_cell(lgbm, "item_knn", "mrr")]) + " | – |")
ild = final[final.metric.eq("combined_ild")].set_index(["method", "reference"]).difference
check("final item vs category popularity", "| Item page | category popularity | " + " | ".join([
    final_cell("fixed_0.8_20", "category_popularity", "ndcg"),
    final_cell("fixed_0.8_20", "category_popularity", "recall"),
    final_cell("fixed_0.8_20", "category_popularity", "mrr", True)])
    + f" | **{ild[('fixed_0.8_20', 'category_popularity')]:+.3f}** |")
for reference, label in (("fixed_0.8_item_knn", "item-kNN + fixed 0.8"), ("item_knn", "item-kNN"),
                         ("two_tower", "session two-tower")):
    bold = reference == "fixed_0.8_item_knn"
    check(f"final home vs {reference}", f"| Home page | {label} | " + " | ".join([
        final_cell("six_four_sasrec", reference, "ndcg", bold), final_cell("six_four_sasrec", reference, "recall"),
        final_cell("six_four_sasrec", reference, "mrr")]) + f" | {ild[('six_four_sasrec', reference)]:+.3f} |")
conf_item = summary(RUNS["item_confirmation"], "all_pages", "distinct_categories")
check("item categories in abstract", f"showing {conf_item['fixed_0.8_20']:.1f} categories instead of "
                                     f"{conf_item['category_popularity']:.1f}")
conf_home = summary(RUNS["home_confirmation"], "all cases", "distinct_categories")
check("home categories", f"({conf_home['six_four_sasrec']:.1f} vs {conf_home['fixed_0.8_item_knn']:.1f} categories")
relevance = difference(RUNS["item_confirmation"], "relevance_20", "category_popularity")
check("item relevance vs popularity", f"category popularity ({relevance.difference:+.3f}, {relevance.ci_low:+.3f} to "
                                      f"{relevance.ci_high:+.3f})")

# ------------------------------------------ table 5.2: variety step ---
hv = summary(RUNS["home_validation"], "all cases")
hild = summary(RUNS["home_validation"], "all cases", "combined_ild")
iv = summary(RUNS["item_validation"], "all_pages")
iild = summary(RUNS["item_validation"], "all_pages", "combined_ild")
for page, table, ild_table, base, method, label in (
        ("Item page", iv, iild, "relevance_20", "fixed_0.8_20", "fixed alpha 0.8, K1 = 20"),
        ("Item page", iv, iild, "relevance_20", "six_four_20", "6 + 4, K1 = 20"),
        ("Home page (SASRec)", hv, hild, "sasrec", "fixed_0.8_sasrec", "fixed alpha 0.8"),
        ("Home page (SASRec)", hv, hild, "sasrec", "six_four_sasrec", "6 + 4")):
    change = table[method] / table[base] - 1
    check(f"variety {page} {method}", f"| {page} | {label} | {change:+.1%} | {ild_table[method] - ild_table[base]:+.3f} |")

# ----------------------------------------------------- grid, Figure 1 ---
grid = pd.read_csv(RUNS["grid"]).sort_values("ndcg", ascending=False)
best_sasrec = grid[grid.model.eq("sasrec")].iloc[0]
best_gru = grid[grid.model.eq("gru4rec")].iloc[0]
knn = best_sasrec.ndcg - best_sasrec.ndcg_vs_item_knn
check("grid summary", f"best SASRec {best_sasrec.ndcg:.3f} vs item-kNN {knn:.3f}; GRU4Rec best {best_gru.ndcg:.3f}")
pairs = grid.assign(base=grid.config.str.replace("dropout0.5", "D").str.replace("dropout0.2", "D"))
pairs = pairs[pairs.model.eq("sasrec")].pivot_table(index="base", columns=grid.config.str.contains("dropout0.5"),
                                                    values="ndcg")
check("dropout 0.5 wins every pair", "Dropout 0.5 beats 0.2 in every pair" if (pairs[True] > pairs[False]).all()
      else "DROPOUT CLAIM FAILS")

# ------------------------------------------------- alpha sweep claim ---
gaps = pd.read_csv(RUNS["alpha"])
gaps = gaps[~gaps.family.eq("shuffled")]
check("adaptive gap range", f"(every gap between {gaps.ndcg_gap_vs_fixed_curve.min():+.3f} and "
                            f"{gaps.ndcg_gap_vs_fixed_curve.max():+.3f} NDCG")
check("no adaptive interval above zero", "no interval above zero" if not (gaps.gap_ci_low > 0).any()
      else "ADAPTIVE CLAIM FAILS")

# ------------------------------------------------- error analysis ---
errors = pd.read_csv(RUNS["errors"] / "category_pages_breakdown.csv")
for dimension, bucket, label in (("history", "no history", "No session history"),
                                 ("history", "with history", "With session history"),
                                 ("position_in_session", "3rd+", "3rd or later query of the session"),
                                 ("clicked_depth", "101+", "First click at the site's position 101+"),
                                 ("unseen_clicked_product", "a clicked product had no April clicks",
                                  "A clicked product had no April clicks")):
    row = errors[errors.dimension.eq(dimension) & errors.bucket.eq(bucket)].iloc[0]
    value = f"**{row.difference:+.3f}**" if row.difference > 0.05 else f"{row.difference:+.3f}"
    check(f"error {label}", f"| {label} | {row.cases} | {row.ndcg_model:.3f} | {row.ndcg_reference:.3f} | {value} |")
overview = json.loads((RUNS["errors"] / "overview.json").read_text())
check("both miss share", f"In {overview['category_pages']['both_zero']:.0%} of queries neither")
depth = errors[errors.dimension.eq("clicked_depth") & errors.bucket.eq("101+")].iloc[0]
check("deep first clicks", f"{depth.case_share:.0%} of first")
for page, file in (("item", "item_page_breakdown.csv"), ("home", "home_page_breakdown.csv")):
    table = pd.read_csv(RUNS["errors"] / file)
    table = table[table.ndcg_model.notna()]
    for bucket in ("a later click is in the 20-product pool", "no later click in the pool"):
        row = table[table.bucket.eq(bucket)].iloc[0]
        check(f"error {page} {bucket}", f"{row.case_share:.0%}{' of pages' if page == 'item' and 'is in' in bucket else ''}"
              f"{' of cases' if page == 'home' and 'is in' in bucket else ''}: {row.ndcg_model:.3f} / "
              f"{row.ndcg_reference:.3f}")
    row = table[table.bucket.eq("all other categories")].iloc[0]
    check(f"error {page} category switch", f"{row.case_share:.0%}: {row.ndcg_model:.3f} / {row.ndcg_reference:.3f}")
home_errors = pd.read_csv(RUNS["errors"] / "home_page_breakdown.csv")
row = home_errors[home_errors.bucket.eq("0 (no April clicks)")].iloc[0]
check("error home cold products", f"{row.case_share:.0%}: {row.ndcg_model:.3f} / {row.ndcg_reference:.3f}")

# ------------------------------------------------------ exposure ---
exposure = pd.read_csv(RUNS["exposure"] / "summary.csv").set_index(["page", "method"])
facts = json.loads((RUNS["exposure"] / "exposure.json").read_text())
check("head size", f"2% of products ({facts['head_products']:,})")
check("no-click share", f"and {facts['no_click_products'] / facts['catalog_products']:.0%}\nreceived none")
check("cold start products", f"shows the same {exposure.loc[('home', 'popularity'), 'distinct_products']} products")
check("item head share", f"concentration ({exposure.loc[('item', 'fixed_0.8_20'), 'head_share']:.0%} vs\n"
                         f"{exposure.loc[('item', 'shopper_clicks'), 'head_share']:.0%})")
check("category head share", f"puts {exposure.loc[('category', lgbm), 'head_share']:.0%} of slots on the popular 2% "
                             f"({exposure.loc[('category', lgbm), 'head_share_ratio']:.1f} times")
check("site order head", f"against {exposure.loc[('category', 'site_order'), 'head_share']:.0%} for the site's own order")
check("home head share", f"gives the popular 2% {exposure.loc[('home', 'six_four_sasrec'), 'head_share']:.0%} of slots, "
                         f"against\n{exposure.loc[('home', 'fixed_0.8_item_knn'), 'head_share']:.0%}")

# ------------------------------------------- abstract and conclusion ---
cat_conf = summary(RUNS["category_confirmation"], "category_pages", "click_ndcg")
cat_diff = difference(RUNS["category_confirmation"], lgbm, "popularity_sort", "click_ndcg", "category_pages")
check("abstract category", f"(click NDCG@10 {cat_conf[lgbm]:.3f} vs {cat_conf['popularity_sort']:.3f}; difference "
      f"{cat_diff.difference:+.3f}, 95% interval {cat_diff.ci_low:+.3f} to {cat_diff.ci_high:+.3f})")
home_conf = summary(RUNS["home_confirmation"], "all cases")
home_diff = difference(RUNS["home_confirmation"], "six_four_sasrec", "fixed_0.8_item_knn")
gain = home_conf["six_four_sasrec"] / home_conf["fixed_0.8_item_knn"] - 1
check("abstract home", f"by {gain:.0%} ({home_conf['six_four_sasrec']:.3f} vs {home_conf['fixed_0.8_item_knn']:.3f}; "
      f"{home_diff.difference:+.3f}, {home_diff.ci_low:+.3f} to {home_diff.ci_high:+.3f})")
item_diff = difference(RUNS["item_confirmation"], "fixed_0.8_20", "category_popularity")
check("abstract item", f"on NDCG ({item_diff.difference:+.3f}, {item_diff.ci_low:+.3f} to {item_diff.ci_high:+.3f})")
check("conclusion category gain", f"improves category pages by about {cat_conf[lgbm] / cat_conf['popularity_sort'] - 1:.0%}")
check("conclusion home gain", f"improves the home page by {gain:.0%}")
check("conclusion item categories", f"showing {conf_item['fixed_0.8_20'] / conf_item['category_popularity']:.1f} times")
sel_diff = difference(RUNS["category_selection"], lgbm, "popularity_sort", "click_ndcg", "category_pages")
check("category gain shrinks", f"smaller than on selection ({cat_diff.difference:+.3f} vs {sel_diff.difference:+.3f})")
knn_stats = json.loads((RUNS["knn_validation"] / "knn_settings.json").read_text())["stats"]
check("co-click pairs", f"This gives {knn_stats['similarity_pairs']:,} product pairs over "
      f"{knn_stats['products_with_clicks']:,} clicked products")
item_errors = pd.read_csv(RUNS["errors"] / "item_page_breakdown.csv")
pool_hit = item_errors[item_errors.dimension.eq("later_click_in_pool") & item_errors.hit_model.notna()]
pool_hit = pool_hit[pool_hit.bucket.str.startswith("a later")].hit_model.iloc[0]
check("pool hit rate value", f"the list finds it {pool_hit:.0%} of the time")
unseen = errors[errors.dimension.eq("unseen_clicked_product")].set_index("bucket").ndcg_model
check("new products", f"(NDCG {unseen['a clicked product had no April clicks']:.2f} vs "
      f"{unseen['all clicked products had April clicks']:.2f})")

check("SASRec relevance only", f"Only SASRec without the diversity step scores higher ({home_conf['sasrec']:.3f})")
check("SASRec no-click slots", f"({exposure.loc[('home', 'six_four_sasrec'), 'no_click_share']:.1%} of its slots go to products")
browse = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["searchstring.tokens"], dtype=str)
check("browse share", f"{browse['searchstring.tokens'].isna().mean():.0%} of all queries")

failed = [r for r in results if not r[0]]
for found, name, snippet in results:
    print(("ok    " if found else "FAIL  ") + name + ("" if found else f"\n      expected: {snippet!r}"))
print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
sys.exit(1 if failed else 0)
