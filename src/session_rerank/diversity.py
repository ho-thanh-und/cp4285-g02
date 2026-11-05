"""Measure category and product-name variety in a recommendation list."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd

from .core import distance_matrix, tokens
from .samples import file_hash

# Starting choice for the combined measure. Give token and category distance equal weight.
TOKEN_WEIGHT = 0.5
MEASURE_DEFINITION = {
    "category_distance": "0 for the same category, 1 for different categories",
    "token_distance": "1 - shared name tokens / distinct name tokens in the pair",
    "combined_distance": "0.5 * token distance + 0.5 * category distance",
    "token_weight": TOKEN_WEIGHT,
    "ild": "Mean distance over every unordered pair in the first min(10, list length) items",
    "missing_metadata": "Unknown pair distances are omitted and their coverage is reported",
    "combined_missing_metadata": "Both token and category distances must be known",
    "short_lists": "Fewer than two items gives no pairs and an undefined ILD",
    "interpretation": "Distances range from 0 (similar) to 1 (different). Token overlap is a lexical proxy.",
}


class ProductDiversity:
    """Load metadata once, then calculate small distance matrices for requested items."""

    def __init__(self, products):
        """Keep the catalog keyed by unique product ID for later list comparisons."""
        if not products.item_id.is_unique:
            raise ValueError("Product metadata must have unique item IDs.")
        # Convert product names such as "12,45,78" into sets of anonymized word IDs.
        # Dictionaries let each recommendation look up its product metadata by ID.
        self.name_tokens = products.set_index("item_id").name_tokens.map(tokens).to_dict()
        self.categories = products.set_index("item_id").category_id.to_dict()

    def distances(self, item_ids):
        """Return symmetric token, category, and combined pair distances.

        Matrix row i and column j compare item_ids[i] with item_ids[j].
        An empty name or missing category makes that component unknown (NaN).
        This prevents missing metadata from receiving a diversity reward.
        """
        item_ids = list(item_ids)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Recommendation lists must contain unique product IDs.")
        unknown = set(item_ids) - self.name_tokens.keys()
        if unknown:
            raise ValueError(f"Products absent from catalog: {sorted(unknown)}")
        # A. Token distance compares product-name word sets with Jaccard distance.
        # Example: {1, 2} versus {2, 3} shares 1 token out of 3, so distance = 2/3.
        # Similar names get smaller distances. Names with no shared tokens get 1.
        # Reuse the existing token parser and Jaccard calculation.
        token = distance_matrix(item_ids, self.name_tokens).astype(float)
        # Both names must contain tokens for their comparison to be meaningful.
        # The two-dimensional mask marks every pair involving an empty name as unknown.
        token_known = np.array([bool(self.name_tokens[item]) for item in item_ids], dtype=bool)
        token[~(token_known[:, None] & token_known[None, :])] = np.nan

        # B. Category distance is 0 for matching IDs and 1 for different IDs.
        # Broadcast the category vector across rows and columns to compare every
        # pair at once. This keeps full candidate-list comparisons inexpensive.
        values = np.array([
            self.categories[item] if pd.notna(self.categories[item]) else None
            for item in item_ids
        ], dtype=object)
        category_known = np.array([value is not None for value in values], dtype=bool)
        category = (values[:, None] != values[None, :]).astype(float)
        category[~(category_known[:, None] & category_known[None, :])] = np.nan

        # C. Blend the two distances. For token distance 0.8 and category distance 0,
        # the combined distance is 0.5 * 0.8 + 0.5 * 0 = 0.4.
        # NaN propagates here, so the blend requires both metadata components.
        combined = TOKEN_WEIGHT * token + (1 - TOKEN_WEIGHT) * category
        # Self-distance is zero. ILD below only uses distinct item pairs.
        for matrix in (token, category, combined):
            np.fill_diagonal(matrix, 0)
        return dict(token=token, category=category, combined=combined)

    def measure_list(self, item_ids, k=10):
        """Return ILD and known-pair coverage for the top k items.

        Ten items produce 45 unordered pairs. For example, category distance
        for categories [A, A, B] is (0 + 1 + 1) / 3 = 2/3.
        Lists with fewer than two items have undefined diversity.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        # The caller supplies items in recommendation order. Keep only its first k.
        selected = list(item_ids)[:k]
        matrices = self.distances(selected)
        # Select the upper triangle above the diagonal. This includes (A, B) once
        # and excludes self-pairs (A, A). Ten products give 10 * 9 / 2 = 45 pairs.
        pairs = np.triu_indices(len(selected), k=1)
        total = len(pairs[0])
        metrics = dict(list_size=len(selected), pairs_total=total)
        for name, matrix in matrices.items():
            values = matrix[pairs]
            # Missing comparisons remain unknown. Keep their count visible through coverage.
            known = values[np.isfinite(values)]
            # ILD is the average pair distance. A higher value means a more varied list.
            # Zero or one item has no pairs, giving undefined ILD (NaN).
            metrics[f"{name}_ild"] = float(known.mean()) if len(known) else np.nan
            metrics[f"{name}_pairs_known"] = len(known)
            metrics[f"{name}_pair_coverage"] = len(known) / total if total else np.nan
        return metrics


def define_product_diversity(dataset_folder, ranker_run, output_root=Path("results/anonymous_diversity")):
    """Save the measure definition and review existing validation top-10 lists.

    Each saved ranker score determines one ordering. Equal scores retain the
    original order. Every validation query is included, even with zero clicks,
    because list diversity depends only on the recommended products.
    """
    # 1. Load the product catalog and saved ranker scores for the chosen dataset.
    # The fingerprints below check that the scores refer to these exact sample tables.
    dataset_folder, ranker_run = Path(dataset_folder), Path(ranker_run)
    manifest = json.loads((dataset_folder / "manifest.json").read_text())
    run = json.loads((ranker_run / "run.json").read_text())
    if run["sample_output_sha256"] != manifest["output_sha256"]:
        raise ValueError("Ranker scores and sample tables come from different datasets.")
    # Check the catalog and validation candidate lists before interpreting saved scores.
    for name in ("products", "candidates"):
        filename = f"{name}.parquet"
        if file_hash(dataset_folder / filename) != manifest["output_sha256"][filename]:
            raise ValueError(f"Prepared table changed: {filename}")
    catalog = ProductDiversity(pd.read_parquet(dataset_folder / "products.parquet"))
    scores = pd.read_parquet(ranker_run / "validation_scores.parquet")
    if scores.empty or not scores.split.eq("validation").all():
        raise ValueError("Expected saved validation candidates only.")
    candidates = pd.read_parquet(dataset_folder / "candidates.parquet", filters=[("split", "==", "validation")])
    keys = ["query_id", "item_id", "original_rank"]
    if scores.duplicated(["query_id", "item_id"]).any():
        raise ValueError("Saved validation candidates contain duplicate products.")
    # Require the same products and original positions for every validation query.
    pd.testing.assert_frame_equal(
        scores[keys].sort_values(keys).reset_index(drop=True),
        candidates[keys].sort_values(keys).reset_index(drop=True),
    )

    # 2. Build three top-10 lists per query using the original order and both models.
    # original_score is the negative original rank, so sorting high-to-low restores it.
    orderings = dict(
        original_order="original_score", query_product="query_product_score",
        query_product_session="query_product_session_score",
    )
    if not np.isfinite(scores[list(orderings.values())].to_numpy()).all():
        raise ValueError("Saved relevance scores must all be finite.")
    records = []
    for query_id, group in scores.sort_values(["query_id", "original_rank"]).groupby("query_id", sort=False):
        for model, column in orderings.items():
            # Stable sorting keeps the original candidate order when model scores tie.
            top = group.sort_values(column, ascending=False, kind="stable").item_id.head(10).tolist()
            # 3 and 4. Compare every product pair, then average its distances into ILD.
            # measure_list() above contains the category, token, and blended calculations.
            records.append(dict(
                query_id=int(query_id), model=model, top10_item_ids=json.dumps(top),
                **catalog.measure_list(top),
            ))
    by_query = pd.DataFrame(records)
    # Average query-level ILDs with equal weight per query for each model.
    # Pandas excludes undefined values from the mean. The measured-query counts
    # show how many lists contributed, including lists with no recorded clicks.
    summary = by_query.groupby("model", sort=False, as_index=False).agg(
        queries=("query_id", "size"),
        token_ild=("token_ild", "mean"),
        category_ild=("category_ild", "mean"),
        combined_ild=("combined_ild", "mean"),
        token_queries_measured=("token_ild", "count"),
        category_queries_measured=("category_ild", "count"),
        combined_queries_measured=("combined_ild", "count"),
        token_pair_coverage=("token_pair_coverage", "mean"),
        category_pair_coverage=("category_pair_coverage", "mean"),
        combined_pair_coverage=("combined_pair_coverage", "mean"),
    )
    # 5. Save the per-query review, model averages, and exact measure definition.
    # Each run gets a separate folder so earlier reviews remain available.
    # Publish the review only once every file has been written successfully.
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{stamp}-{uuid4().hex[:8]}"
    definition = dict(
        **MEASURE_DEFINITION, cutoff=10, ranker_run=str(ranker_run.resolve()),
        scores_sha256=file_hash(ranker_run / "validation_scores.parquet"),
        sample_output_sha256=manifest["output_sha256"],
        evaluation_population="All saved validation queries including zero-click queries",
        summary_aggregation="Equal weight per query, averaging defined ILDs only",
        original_rank_ties="Preserve original candidate order",
    )
    with TemporaryDirectory(dir=output_root) as temporary:
        folder = Path(temporary)
        # Per-query rows include the top-10 product IDs, ILDs, and metadata coverage.
        by_query.to_csv(folder / "validation_diversity_by_query.csv", index=False)
        summary.to_csv(folder / "validation_diversity_summary.csv", index=False)
        # Record the formulas, weights, and input fingerprints to reproduce the review.
        (folder / "measure.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)
    print(summary[["model", "queries", "token_ild", "category_ild", "combined_ild"]].to_string(index=False), flush=True)
    print(f"Saved diversity definition and review: {output.resolve()}", flush=True)
    return output
