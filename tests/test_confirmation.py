"""Check the once-only confirmation rule and the final-test driver."""

import json

import pandas as pd
import pytest

import main
from session_rerank import confirmation
from session_rerank.windows import check_not_opened, check_window, completed_run


def finished(folder, marker="summary.csv"):
    folder.mkdir(parents=True)
    (folder / marker).write_text("x")
    return folder


def test_confirmation_counts_only_finished_runs(tmp_path):
    check_not_opened(tmp_path / "missing")
    (tmp_path / "20260101T000000Z_confirmation").mkdir()  # stopped before writing results
    assert completed_run(tmp_path) is None
    check_not_opened(tmp_path)
    finished(tmp_path / "20260102T000000Z_confirmation")
    with pytest.raises(ValueError, match="only once"):
        check_not_opened(tmp_path)
    # Selection runs are never blocked, and validation is not held out.
    finished(tmp_path / "20260103T000000Z_selection")
    check_not_opened(tmp_path, "selection")


@pytest.mark.parametrize("window, ok", [
    (dict(name="confirmation", start="2016-05-16", end_exclusive="2016-06-02"), True),
    (dict(name="confirmation", start="2016-05-15", end_exclusive="2016-06-02"), False),
    (dict(name="test", start="2016-05-16", end_exclusive="2016-06-02"), False)])
def test_confirmation_week_dates(window, ok):
    if ok:
        check_window(window)
    else:
        with pytest.raises(ValueError):
            check_window(window)


def test_cli_dispatches_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr("session_rerank.confirmation.run_confirmation", lambda *args: calls.append(args))
    main.cli(["confirmation"])
    assert len(calls) == 1


def test_real_configs_open_the_confirmation_week_with_frozen_choices():
    config = json.loads(open("config/confirmation.json").read())
    heldout, ymal, home, sequence = confirmation.read_page_settings(config)
    assert heldout["split"] == "confirmation" and ymal["frozen_policy"] == "fixed_0.8_20"
    assert sequence["frozen_policy"] == "six_four_sasrec" and home["window"]["name"] == "confirmation"


def fake_pages(tmp_path, monkeypatch, fail_item_once=False):
    """Page runners replaced by stand-ins that write the files the driver reads."""
    roots = {name: tmp_path / name for name in ("heldout", "ymal", "home", "sequence")}
    calls = []
    state = {"item_failed": False}

    def make(name, rows):
        def run(*args, **kwargs):
            calls.append(name)
            if name == "ymal" and fail_item_once and not state["item_failed"]:
                state["item_failed"] = True
                (roots[name] / "20260101T000000Z_confirmation").mkdir(parents=True)
                raise RuntimeError("crash")
            folder = finished(roots[name] / f"2026010{len(calls)}T000000Z_confirmation")
            pd.DataFrame(rows).to_csv(folder / "differences.csv", index=False)
            return folder
        return run

    row = dict(reference="base", metric="ndcg", value=0.2, reference_value=0.1, difference=0.1, ci_low=0.05,
               ci_high=0.15)
    monkeypatch.setattr(confirmation, "run_heldout", make("heldout", [
        row | dict(method="lightgbm_purchase_session", segment="category_pages"),
        row | dict(method="lightgbm_purchase_session", segment="all_queries"),
        row | dict(method="item_knn", segment="category_pages")]))
    monkeypatch.setattr(confirmation, "run_ymal", make("ymal", [row | dict(method="fixed_0.8_20"),
                                                                row | dict(method="relevance_20")]))
    monkeypatch.setattr(confirmation, "run_home_page", make("home", [row | dict(method="item_knn")]))
    monkeypatch.setattr(confirmation, "run_sequence_models", make("sequence", [row | dict(method="six_four_sasrec"),
                                                                               row | dict(method="sasrec")]))
    settings = dict(heldout=dict(output_directory=str(roots["heldout"]), split="confirmation",
                                 frozen_selection_run=str(tmp_path / "selection")),
                    ymal=dict(output_directory=str(roots["ymal"]), frozen_policy="fixed_0.8_20"),
                    home=dict(output_directory=str(roots["home"])),
                    sequence=dict(output_directory=str(roots["sequence"]), frozen_policy="six_four_sasrec",
                                  frozen_run="frozen"))
    monkeypatch.setattr(confirmation, "read_page_settings", lambda config: (
        settings["heldout"], settings["ymal"], settings["home"], settings["sequence"]))
    finished(tmp_path / "selection", "selection_choice.json")
    (tmp_path / "selection" / "selection_choice.json").write_text(json.dumps(dict(chosen="purchase_session")))
    sequence_config = tmp_path / "sequence_config.json"
    sequence_config.write_text(json.dumps(dict(models=["sasrec"])))
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(dict(output_directory=str(tmp_path / "confirmation"), heldout_config="h",
                                    ymal_config="y", home_config="hp", sequence_config=str(sequence_config))))
    return path, calls


def test_driver_freezes_first_then_builds_one_final_table(tmp_path, monkeypatch):
    path, calls = fake_pages(tmp_path, monkeypatch)
    folder = confirmation.run_confirmation(path)
    assert calls == ["heldout", "ymal", "home", "sequence"]
    frozen = json.loads((folder / "frozen_choices.json").read_text())
    assert frozen["pages"]["category pages"]["model"] == "lightgbm_purchase_session"
    assert frozen["pages"]["home page"]["policy"] == "six_four_sasrec" and (folder / "code.diff").exists()
    table = pd.read_csv(folder / "final_table.csv")
    # Only each page's frozen choice, and only the category-page segment.
    assert table.method.tolist() == ["lightgbm_purchase_session", "fixed_0.8_20", "six_four_sasrec"]
    # The sequence models were pointed at this confirmation's home-page cases.
    home_run = json.loads((folder / "runs.json").read_text())["home"]
    assert json.loads((folder / "sequence_config.json").read_text())["home_run"] == home_run
    with pytest.raises(ValueError, match="only once"):
        confirmation.run_confirmation(path)


def test_driver_resumes_after_a_crash_without_rescoring_finished_pages(tmp_path, monkeypatch):
    path, calls = fake_pages(tmp_path, monkeypatch, fail_item_once=True)
    with pytest.raises(RuntimeError):
        confirmation.run_confirmation(path)
    first = json.loads(next((tmp_path / "confirmation").glob("*_confirmation/frozen_choices.json")).read_text())
    folder = confirmation.run_confirmation(path)
    assert calls == ["heldout", "ymal", "ymal", "home", "sequence"]  # category pages scored once
    assert json.loads((folder / "frozen_choices.json").read_text()) == first
    assert list(folder.glob("code_resumed_*.diff")) and list(folder.glob("code_resumed_*.json"))
    assert (folder / "final_table.csv").exists()
