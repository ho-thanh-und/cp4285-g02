"""Check the diversity definition with hand-calculated product lists."""

import json

import numpy as np
import pandas as pd
import pytest

from session_rerank.diversity import ProductDiversity, define_product_diversity
from session_rerank.samples import file_hash


@pytest.fixture
def products():
    return pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "name_tokens": ["1,2", "1,2", "3,4", ""],
        "category_id": pd.Series([10, 10, 20, pd.NA], dtype="Int64"),
    })


def test_distances_and_ild_match_hand_calculated_pairs(products):
    catalog = ProductDiversity(products)
    matrices = catalog.distances([1, 2, 3])
    for matrix in matrices.values():
        np.testing.assert_allclose(matrix, [[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    measured = catalog.measure_list([1, 2, 3])
    assert measured["pairs_total"] == 3
    assert measured["combined_ild"] == pytest.approx(2 / 3)
    assert measured["combined_pair_coverage"] == 1
    assert catalog.measure_list([3, 1, 2]) == measured


def test_combined_distance_gives_each_component_equal_weight(products):
    products.loc[1, "name_tokens"] = "2,3"
    catalog = ProductDiversity(products)
    # Token distance = 1 - 1/3, category distance = 0.
    assert catalog.measure_list([1, 2])["combined_ild"] == pytest.approx(1 / 3)


def test_missing_metadata_reports_coverage_without_rewarding_missing_values(products):
    catalog = ProductDiversity(products)
    measured = catalog.measure_list([1, 3, 4])
    assert measured["combined_pairs_known"] == 1
    assert measured["combined_pair_coverage"] == pytest.approx(1 / 3)
    assert measured["combined_ild"] == 1
    assert np.isnan(catalog.measure_list([1, 4])["combined_ild"])
    # A missing category still allows a token-only measurement.
    products.loc[3, "name_tokens"] = "1,2"
    measured = ProductDiversity(products).measure_list([1, 4])
    assert measured["token_ild"] == 0
    assert np.isnan(measured["category_ild"])
    assert np.isnan(measured["combined_ild"])


def test_short_lists_and_cutoff(products):
    catalog = ProductDiversity(products)
    for items in ([], [1]):
        measured = catalog.measure_list(items)
        assert measured["pairs_total"] == 0
        assert np.isnan(measured["combined_ild"])
    assert catalog.measure_list([1, 2, 3], k=2)["combined_ild"] == 0
    with pytest.raises(ValueError, match="positive"):
        catalog.measure_list([1], k=0)
    with pytest.raises(ValueError, match="unique"):
        catalog.measure_list([1, 1])
    with pytest.raises(ValueError, match="absent"):
        catalog.measure_list([999])


def test_review_uses_saved_rankings_and_includes_zero_click_queries(tmp_path, products):
    data = tmp_path / "data"
    run = tmp_path / "rankers"
    data.mkdir()
    run.mkdir()
    products.to_parquet(data / "products.parquet", index=False)
    candidates = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2, 2], "item_id": [1, 2, 3, 1, 2, 3],
        "original_rank": [1, 2, 3, 1, 2, 3], "split": ["validation"] * 6,
    })
    candidates.to_parquet(data / "candidates.parquet", index=False)
    hashes = {p.name: file_hash(p) for p in data.glob("*.parquet")}
    (data / "manifest.json").write_text(json.dumps(dict(output_sha256=hashes)))
    (run / "run.json").write_text(json.dumps(dict(sample_output_sha256=hashes)))
    scores = candidates.assign(
        clicked=False, original_score=-candidates.original_rank,
        query_product_score=1.0, query_product_session_score=[1, 3, 2, 1, 3, 2],
    )
    scores.to_parquet(run / "validation_scores.parquet", index=False)
    output = define_product_diversity(data, run, tmp_path / "output")
    reviewed = pd.read_csv(output / "validation_diversity_by_query.csv")
    assert len(reviewed) == 6
    assert reviewed.query("model == 'query_product'").top10_item_ids.eq("[1, 2, 3]").all()
    assert reviewed.query("model == 'query_product_session'").top10_item_ids.eq("[2, 3, 1]").all()
    assert set(pd.read_csv(output / "validation_diversity_summary.csv").queries) == {2}
    # Reject a stale run before generating misleading measurements.
    (run / "run.json").write_text(json.dumps(dict(sample_output_sha256={})))
    with pytest.raises(ValueError, match="different datasets"):
        define_product_diversity(data, run, tmp_path / "output")


def test_main_diversity_route_forwards_the_chosen_ranker_run(tmp_path, monkeypatch):
    import main

    calls = []
    monkeypatch.setattr(main, "prepare_samples", lambda config: tmp_path / "data")
    monkeypatch.setattr(main, "define_product_diversity", lambda data, run: calls.append((data, run)))
    main.cli(["diversity", "--ranker-run", str(tmp_path / "ranker")])
    assert calls == [(tmp_path / "data", tmp_path / "ranker")]
    with pytest.raises(SystemExit):
        main.cli(["diversity"])
