"""Check SASRec and GRU4Rec on the home-page cases of a small synthetic catalog."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

import main
from session_rerank.home_page import run_home_page
from session_rerank.sequence_models import build_model, read_sequence_settings, run_sequence_models, training_batches, vocabulary
from test_home_page import SETTINGS as HOME_SETTINGS

SEQUENCE = dict(output_directory="unused", home_config="unused", home_run="unused", models=["sasrec", "gru4rec"],
                max_length=4, embedding_dim=8, sasrec_blocks=1, sasrec_heads=2, dropout=0.1, epochs=2,
                batch_size=16, learning_rate=0.01, negatives=10, popular_negative_share=0.5,
                popular_negative_power=0.75, epoch_check_cases=10, bootstrap_replicates=10, seed=1, threads=1)


def test_vocabulary_and_batches_keep_known_clicks_in_order():
    sequences = [("a", "d", [7, 3, 9]), ("b", "d", [3]), ("c", "d", [5, 3])]
    vocab = vocabulary(sequences)
    assert vocab.index.tolist() == [3, 5, 7, 9] and vocab.tolist() == [1, 2, 3, 4]
    inputs, targets = training_batches(sequences, vocab, max_length=2)
    # Session a: 7 -> 3 -> 9; session b has one click and is skipped.
    assert inputs.tolist() == [[3, 1], [2, 0]] and targets.tolist() == [[1, 4], [1, 0]]


@pytest.mark.parametrize("kind", ["sasrec", "gru4rec"])
def test_models_return_one_state_per_position(kind):
    model = build_model(kind, products=12, settings=SEQUENCE)
    hidden = model(torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]]))
    assert hidden.shape == (2, 4, SEQUENCE["embedding_dim"])
    if kind == "sasrec":
        # Causal attention: a later click cannot change an earlier position's state.
        model.eval()
        a = model(torch.tensor([[1, 2, 3, 0]]))
        b = model(torch.tensor([[1, 2, 9, 0]]))
        assert torch.allclose(a[0, :2], b[0, :2], atol=1e-6)


@pytest.mark.parametrize("window, ok", [
    (None, True), (dict(start="2016-02-01", end_exclusive="2016-04-30"), True),
    (dict(start="2016-04-01", end_exclusive="2016-05-02"), False),
    (dict(start="2015-12-01", end_exclusive="2016-04-30"), False)])
def test_training_window_must_end_before_the_evaluation_weeks(tmp_path, window, ok):
    home = tmp_path / "home.json"
    home.write_text(json.dumps(HOME_SETTINGS))
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(SEQUENCE | dict(home_config=str(home))
                               | ({"training_window": window} if window else {})))
    if ok:
        settings, _ = read_sequence_settings(path)
        assert settings["training_window"] == (window or HOME_SETTINGS["training_window"])
    else:
        with pytest.raises(ValueError, match="training_window"):
            read_sequence_settings(path)


def test_device_setting_is_checked_and_auto_falls_back_to_cpu(tmp_path):
    from session_rerank.sequence_models import choose_device
    home = tmp_path / "home.json"
    home.write_text(json.dumps(HOME_SETTINGS))
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(SEQUENCE | dict(home_config=str(home), device="tpu")))
    with pytest.raises(ValueError, match="device"):
        read_sequence_settings(path)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert choose_device({"device": "auto"}) == expected and choose_device({}) == expected
    assert choose_device({"device": "cpu"}) == "cpu"


def test_selection_needs_frozen_models(tmp_path):
    home = tmp_path / "home.json"
    home.write_text(json.dumps(HOME_SETTINGS | dict(window=dict(name="selection", start="2016-05-09",
                                                                end_exclusive="2016-05-15"),
                                                    frozen_run="x", frozen_policy="item_knn")))
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(SEQUENCE | dict(home_config=str(home))))
    with pytest.raises(ValueError, match="frozen_run"):
        read_sequence_settings(path)


def test_train_compare_and_reuse_on_selection(tmp_path):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    rng = np.random.default_rng(0)
    items = np.arange(1, 41)
    pd.DataFrame(dict(item_id=items, price_log2=items % 5, category_id=items % 4,
                      name_tokens=[f"{i % 7},{i % 5 + 10}" for i in items])).to_parquet(run / "products.parquet")
    queries, clicks, qid = [], [], 0
    for day, count in (("2016-04-03", 80), ("2016-04-20", 80), ("2016-05-02", 25), ("2016-05-10", 20)):
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
    common = dict(raw_directory=str(raw), trained_run=str(run), knn_config=str(knn))
    home_validation = tmp_path / "home.json"
    home_validation.write_text(json.dumps(HOME_SETTINGS | common))
    home_run = run_home_page(home_validation, output_root=tmp_path / "home")
    settings = tmp_path / "sequence.json"
    settings.write_text(json.dumps(SEQUENCE | dict(home_config=str(home_validation), home_run=str(home_run))))
    output = run_sequence_models(settings, output_root=tmp_path / "sequence")
    per_case = pd.read_parquet(output / "per_case.parquet")
    assert set(per_case.method) == {"sasrec", "gru4rec"} and per_case.groupby("method").size().iloc[0] == 50
    assert per_case.recall.between(0, 1).all()
    differences = pd.read_csv(output / "differences.csv")
    assert set(differences.reference) == {"item_knn", "fixed_0.8_item_knn", "two_tower"}
    assert (output / "sasrec.pt").exists() and (output / "gru4rec.pt").exists()
    # Selection reuses the saved models on the selection home-page cases.
    home_selection = tmp_path / "home_selection.json"
    home_selection.write_text(json.dumps(HOME_SETTINGS | common | dict(
        window=dict(name="selection", start="2016-05-09", end_exclusive="2016-05-15"),
        frozen_run=str(home_run), frozen_policy="fixed_0.8_item_knn")))
    selection_home_run = run_home_page(home_selection, output_root=tmp_path / "home")
    selection = tmp_path / "sequence_selection.json"
    selection.write_text(json.dumps(SEQUENCE | dict(home_config=str(home_selection),
                                                    home_run=str(selection_home_run), frozen_run=str(output))))
    held_out = run_sequence_models(selection, output_root=tmp_path / "sequence")
    assert (held_out / "frozen_choices.json").exists()
    assert pd.read_parquet(held_out / "per_case.parquet").groupby("method").size().iloc[0] == 40
