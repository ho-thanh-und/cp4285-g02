"""Check the export / external predictions / scoring bridge and the T4Rec script helpers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main
from session_rerank.external_models import export_sequences, score_predictions
from session_rerank.home_page import run_home_page
from test_home_page import SETTINGS as HOME_SETTINGS
from test_sequence_models import SEQUENCE

ROOT = Path(__file__).resolve().parents[1]


def t4rec_script():
    spec = importlib.util.spec_from_file_location("train_t4rec", ROOT / "scripts" / "t4rec" / "train_t4rec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Transformers4Rec itself is imported only inside functions.
    return module


def test_t4rec_helpers_pad_rank_and_measure_recall():
    script = t4rec_script()
    assert script.pad([[1, 2, 3], [4]], 2).tolist() == [[2, 3], [4, 0]]
    scores = np.array([9.0, 0.1, 0.9, 0.5, 0.7])
    # Padding id 0 and already-clicked id 2 are never recommended.
    assert script.ranked_ids(scores, [2], 3) == [4, 3, 1]
    cases = pd.DataFrame(dict(case=[0, 1], later_rows=[[10, 11], [12]]))
    vocabulary = pd.DataFrame(dict(vocab_id=[1, 2, 3], product_row=[10, 11, 12]))
    predictions = pd.DataFrame(dict(case=[0, 1], item_ids=[[1, 5], [4]]))
    assert script.recall_at_k(predictions, cases, vocabulary) == pytest.approx((0.5 + 0) / 2)


def synthetic(tmp_path):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    rng = np.random.default_rng(0)
    items = np.arange(1, 41)
    pd.DataFrame(dict(item_id=items, price_log2=items % 5, category_id=items % 4,
                      name_tokens=[f"{i % 7},{i % 5 + 10}" for i in items])).to_parquet(run / "products.parquet")
    queries, clicks, qid = [], [], 0
    for day, count in (("2016-04-03", 80), ("2016-04-20", 80), ("2016-05-02", 25)):
        for session in range(count):
            qid += 1
            queries.append((qid, f"{day}-{session}", day, False))
            for position, item in enumerate(rng.choice(items, 3, replace=False)):
                clicks.append((qid, 100 * position, int(item)))
    pd.DataFrame(queries, columns=["queryId", "sessionId", "eventdate", "is.test"]).to_csv(
        raw / "train-queries.csv", sep=";", index=False)
    pd.DataFrame(clicks, columns=["queryId", "timeframe", "itemId"]).to_csv(raw / "train-clicks.csv", sep=";", index=False)
    pd.DataFrame(dict(itemId=[1], eventdate=["2016-04-05"])).to_csv(raw / "train-purchases.csv", sep=";", index=False)
    knn = tmp_path / "knn.json"
    knn.write_text(json.dumps(dict(output_directory="x", raw_directory=str(raw), max_items_per_session=50,
                                   shrinkage=0, compare_with=[], bootstrap_replicates=10, seed=1)))
    home = tmp_path / "home.json"
    home.write_text(json.dumps(HOME_SETTINGS | dict(raw_directory=str(raw), trained_run=str(run),
                                                    knn_config=str(knn))))
    home_run = run_home_page(home, output_root=tmp_path / "home")
    sequence = tmp_path / "sequence.json"
    sequence.write_text(json.dumps(SEQUENCE | dict(home_config=str(home), home_run=str(home_run))))
    config = tmp_path / "t4rec.json"
    config.write_text(json.dumps(dict(output_directory=str(tmp_path / "t4rec"), sequence_config=str(sequence),
                                      compare_sequence_runs=[], model=dict(max_length=4, top_n=8, seed=1),
                                      bootstrap_replicates=10, seed=1)))
    return config, tmp_path


def test_export_then_score_external_predictions(tmp_path):
    config, root = synthetic(tmp_path)
    folder = export_sequences(config)
    assert (root / "t4rec" / "latest_export.txt").read_text().strip() == folder.as_posix()
    train = pd.read_parquet(folder / "train_sessions.parquet")
    cases = pd.read_parquet(folder / "cases.parquet")
    vocabulary = pd.read_parquet(folder / "vocabulary.parquet")
    assert train.item_ids.map(len).ge(2).all() and train.item_ids.map(len).le(5).all()
    assert len(cases) == 50 and cases.prefix_ids.map(len).le(4).all()
    assert vocabulary.vocab_id.min() == 1 and vocabulary.vocab_id.is_unique
    # A stand-in external model: every case gets the vocabulary in id order, except one empty case.
    ids = vocabulary.vocab_id.tolist()
    predictions = pd.DataFrame(dict(case=cases.case, item_ids=[ids if c else [] for c in cases.case]))
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    output = score_predictions(folder, "stand_in", config)
    per_case = pd.read_parquet(output / "per_case.parquet")
    assert len(per_case) == 50 and per_case.recall.between(0, 1).all()
    assert per_case.used_prediction.sum() == 49  # the empty case falls back to popularity
    differences = pd.read_csv(output / "differences.csv")
    assert set(differences.reference) == {"popularity", "item_knn", "two_tower", "fixed_0.8_item_knn"}
    assert "stand_in" in set(pd.read_csv(output / "summary.csv").method)


def test_cli_requires_export_and_label_together(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("session_rerank.external_models.score_predictions", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main.cli(["score-external", "--export", str(tmp_path)])
    with pytest.raises(SystemExit):
        main.cli(["prepare", "--label", "x"])
    main.cli(["score-external", "--export", str(tmp_path), "--label", "t4rec_xlnet"])
    assert calls and calls[0][:2] == (tmp_path, "t4rec_xlnet")
