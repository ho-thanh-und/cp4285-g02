"""Draw the report's figures from the saved result files (no numbers typed in).

    uv run python scripts/report/make_figures.py

Writes PNG files to figures/. Each figure reads the run folders named in
SOURCES, so it can be regenerated and audited. Colours follow the reference
palette of the data-visualisation guide: the frozen choice is blue, the main
alternative orange, other methods grey; at most three hues per figure, direct
labels, one axis per panel.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
SOURCES = dict(
    confirmation=ROOT / "results/confirmation/20261108T193024Z_confirmation",
    home_validation=ROOT / "results/sequence_models/20261108T175008Z_validation",
    item_validation=ROOT / "results/ymal/20261108T090850Z_validation",
    grid=ROOT / "results/sequence_grid/grid_summary.csv",
    exposure=ROOT / "results/exposure/20261108T184420Z/summary.csv",
)
BLUE, ORANGE, AQUA, GREY, LIGHT_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#d6d5d0"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LIGHT_GREY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / name, facecolor="white")
    plt.close(figure)
    print(f"wrote {OUT / name}")


def final_test():
    """Figure 3: NDCG@10 of every method on the confirmation week, frozen choice highlighted."""
    runs = json.loads((SOURCES["confirmation"] / "runs.json").read_text())
    category = pd.read_csv(Path(ROOT / runs["category"]) / "summary.csv")
    category = category[category.segment.eq("category_pages")].set_index("method").click_ndcg
    item = pd.read_csv(Path(ROOT / runs["item"]) / "summary.csv")
    item = item[item.segment.eq("all_pages")].set_index("method").ndcg
    home = pd.read_csv(Path(ROOT / runs["home_sequence"]) / "summary.csv")
    home = home[home.segment.eq("all cases")].set_index("method").ndcg
    panels = [
        ("Category pages (939 queries)", category, [
            ("site_order", "Site's original order"), ("popularity_sort", "Popularity sort"),
            ("item_knn", "Item-kNN (co-clicks)"), ("lightgbm_purchase_session", "LightGBM (frozen)")]),
        ("Item page, 'You may also like' (20,000 pages)", item, [
            ("category_popularity", "Category popularity"), ("relevance_20", "Item-kNN, relevance only"),
            ("six_four_20", "Item-kNN + 6 + 4"), ("fixed_0.8_20", "Item-kNN + fixed 0.8 (frozen)")]),
        ("Home page (20,000 cases)", home, [
            ("popularity", "Popularity"), ("two_tower", "Session two-tower"),
            ("fixed_0.8_item_knn", "Item-kNN + fixed 0.8 (before)"), ("item_knn", "Item-kNN"),
            ("sasrec", "SASRec, relevance only"), ("six_four_sasrec", "SASRec + 6 + 4 (frozen)")]),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), gridspec_kw=dict(height_ratios=[4, 4, 6], hspace=0.55))
    for axis, (title, values, rows) in zip(axes, panels):
        labels = [label for _, label in rows]
        numbers = [values[key] for key, _ in rows]
        colors = [BLUE if "(frozen)" in label else GREY for label in labels]
        bars = axis.barh(labels, numbers, color=colors, height=0.6)
        for bar, number in zip(bars, numbers):
            axis.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{number:.3f}",
                      va="center", fontsize=9, color=INK)
        axis.set_title(title)
        axis.set_xlim(0, max(numbers) * 1.18)
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("NDCG@10 on the confirmation week (May 16 – June 1)")
    save(figure, "fig3_final_test.png")


def trade_off():
    """Figure 2: relevance against list diversity on the validation week, home and item pages."""
    home = pd.read_csv(SOURCES["home_validation"] / "summary.csv")
    home = home[home.segment.eq("all cases")].set_index("method")
    item = pd.read_csv(SOURCES["item_validation"] / "summary.csv")
    item = item[item.segment.eq("all_pages")].set_index("method")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw=dict(wspace=0.32))
    families = [
        (axes[0], home, "SASRec", ["sasrec", "fixed_0.8_sasrec", "six_four_sasrec"], BLUE),
        (axes[0], home, "Item-kNN", ["item_knn", "fixed_0.8_item_knn"], ORANGE),
        (axes[1], item, "Item-kNN (K1 = 20)", ["relevance_20", "fixed_0.8_20", "six_four_20"], BLUE),
    ]
    names = {"sasrec": "relevance", "fixed_0.8_sasrec": "fixed 0.8", "six_four_sasrec": "6 + 4 (frozen)",
             "item_knn": "relevance", "fixed_0.8_item_knn": "fixed 0.8 (before)",
             "relevance_20": "relevance", "fixed_0.8_20": "fixed 0.8 (frozen)", "six_four_20": "6 + 4"}
    for axis, table, family, methods, color in families:
        x, y = table.loc[methods, "combined_ild"], table.loc[methods, "ndcg"]
        axis.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, label=family, zorder=3)
        for method in methods:
            below = method in ("fixed_0.8_item_knn",)  # keep this label off its line
            axis.annotate(names[method], (table.loc[method, "combined_ild"], table.loc[method, "ndcg"]),
                          textcoords="offset points", xytext=(8, -12) if below else (6, 5),
                          ha="left", fontsize=8.5, color=INK)
        # The declared rule: at most 5% below the family's relevance-only NDCG.
        axis.axhline(table.loc[methods[0], "ndcg"] * 0.95, color=color, linestyle=":", linewidth=1)
    for axis, table, points in ((axes[0], home, [("two_tower", "two-tower")]),
                                (axes[1], item, [("category_popularity", "category popularity")])):
        for method, label in points:
            axis.plot(table.loc[method, "combined_ild"], table.loc[method, "ndcg"], "o", color=GREY, markersize=7)
            axis.annotate(label, (table.loc[method, "combined_ild"], table.loc[method, "ndcg"]),
                          textcoords="offset points", xytext=(6, 5), fontsize=8.5, color=MUTED)
        axis.set_xlabel("Combined ILD@10 (list variety)")
        axis.set_ylabel("NDCG@10")
    axes[0].set_title("Home page (validation)")
    axes[1].set_title("Item page (validation)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_xlim(0.42, 0.78)
    axes[1].set_xlim(0.44, 0.72)
    figure.text(0.01, -0.04, "Dotted lines: 5% below each model's relevance-only NDCG (the declared selection rule).",
                fontsize=8.5, color=MUTED)
    save(figure, "fig2_trade_off.png")


def grid():
    """Figure 1: the 12-setting SASRec / GRU4Rec grid on the validation week."""
    table = pd.read_csv(SOURCES["grid"]).sort_values("ndcg")
    knn = table.ndcg.iloc[0] - table.ndcg_vs_item_knn.iloc[0]
    frozen = table.ndcg.iloc[0] - table["ndcg_vs_fixed_0.8_item_knn"].iloc[0]
    label = table.config.str.replace("_epochs30_lr0.001", "").str.replace("_dropout0.2_epochs60", "")
    label = label.str.replace("_", " ").str.replace("blocks", "blocks ").str.replace("dim", "dim ")
    label = label.str.replace("dropout", "dropout ").str.replace("lr", "lr ")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    # A dot plot, not bars: the axis is zoomed in, and bars would have to start at zero.
    positions = range(len(table))
    for model, color, name in (("sasrec", BLUE, "SASRec"), ("gru4rec", ORANGE, "GRU4Rec")):
        rows = [i for i, m in enumerate(table.model) if m == model]
        axis.plot(table.ndcg.iloc[rows], rows, "o", color=color, markersize=8, label=name, zorder=3)
    for y, value in zip(positions, table.ndcg):
        axis.text(value + 0.0007, y, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    axis.set_yticks(list(positions), label)
    axis.axvline(knn, color=GREY, linestyle="--", linewidth=1.2)
    axis.text(knn + 0.0003, -0.9, "item-kNN", color=MUTED, fontsize=8.5, va="bottom")
    axis.axvline(frozen, color=GREY, linestyle=":", linewidth=1.2)
    axis.text(frozen - 0.0003, -0.9, "item-kNN + fixed 0.8", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    axis.set_ylim(-1.2, len(table) - 0.4)
    axis.set_xlim(0.110, table.ndcg.max() + 0.006)
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    axis.set_xlabel("NDCG@10, home page, validation week (Feb–Apr training)")
    axis.set_title("SASRec and GRU4Rec settings (best first)")
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="y", length=0, labelsize=8.5)
    save(figure, "fig1_sequence_grid.png")


def exposure():
    """Figure 4: share of shown slots on the 2% most-clicked products, against shoppers' own clicks."""
    table = pd.read_csv(SOURCES["exposure"])
    rows = [
        ("home", "popularity", "Popularity (cold start)"), ("home", "fixed_0.8_item_knn", "Item-kNN + fixed 0.8"),
        ("home", "six_four_sasrec", "SASRec + 6 + 4 (frozen)"),
        ("item", "category_popularity", "Category popularity"), ("item", "fixed_0.8_20", "Item-kNN + fixed 0.8 (frozen)"),
        ("category", "site_order", "Site's original order"), ("category", "popularity_sort", "Popularity sort"),
        ("category", "lightgbm_purchase_session", "LightGBM (frozen)"),
    ]
    titles = dict(home="Home page", item="Item page", category="Category pages")
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), gridspec_kw=dict(height_ratios=[3, 2, 3], hspace=0.6))
    for axis, page in zip(axes, ("home", "item", "category")):
        page_rows = [(m, label) for p, m, label in rows if p == page]
        values = table[table.page.eq(page)].set_index("method")
        shares = [values.loc[m, "head_share"] for m, _ in page_rows]
        labels = [label for _, label in page_rows]
        colors = [BLUE if "(frozen)" in label else GREY for label in labels]
        bars = axis.barh(labels, shares, color=colors, height=0.6)
        for bar, share in zip(bars, shares):
            axis.text(share + 0.01, bar.get_y() + bar.get_height() / 2, f"{share:.0%}", va="center", fontsize=9,
                      color=INK)
        shopper = values.loc["shopper_clicks", "head_share"]
        axis.axvline(shopper, color=ORANGE, linewidth=1.6)
        axis.text(shopper, len(labels) - 0.45, f" shoppers' clicks {shopper:.0%}", color=ORANGE, fontsize=8.5,
                  va="bottom")
        axis.set_xlim(0, 1.12)
        axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axis.set_title(titles[page])
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Share of shown slots on the 2% most-clicked products (selection week)")
    save(figure, "fig4_exposure.png")


def main():
    missing = [name for name, path in SOURCES.items() if not path.exists()]
    if missing:
        sys.exit(f"Missing result files: {missing}")
    final_test()
    trade_off()
    grid()
    exposure()


if __name__ == "__main__":
    main()
