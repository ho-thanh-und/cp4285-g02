"""The final test: score every page's frozen choice once on the confirmation week.

Order:
1. Check every page config before anything is read: each must point at the
   confirmation week and carry its frozen choice.
2. Write frozen_choices.json (all three pages' choices, the code commit, and the
   uncommitted diff saved as code.diff) before any confirmation data is read.
3. Run the pages with their own code, which also refuses a second finished
   confirmation run:
     category pages   heldout.run_heldout     LightGBM model chosen on selection
     item page        ymal.run_ymal           item-kNN + fixed 0.8 (K1 = 20)
     home page        home_page.run_home_page the cases and comparison methods,
                      then sequence_models    SASRec + 6 + 4 on the same cases
4. Write final_table.csv: each frozen choice against its baselines, with the
   session-bootstrap intervals computed by each page's run.
A page that finished in an earlier attempt of this run is reused, not run again.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .heldout import git_state, read_heldout_settings, run_heldout
from .home_page import read_home_settings, run_home_page
from .sequence_models import read_sequence_settings, run_sequence_models
from .study_files import save_table, write_json
from .windows import check_not_opened, completed_run
from .ymal import read_ymal_settings, run_ymal

PAGES = ("category", "item", "home", "home_sequence")


def read_page_settings(config):
    """Every page config, checked to open the confirmation week with a frozen choice."""
    heldout = read_heldout_settings(config["heldout_config"])
    ymal = read_ymal_settings(config["ymal_config"])
    home = read_home_settings(config["home_config"])
    sequence, sequence_home = read_sequence_settings(config["sequence_config"])
    if heldout["split"] != "confirmation":
        raise ValueError("heldout_config must use the confirmation split")
    for name, window in (("ymal_config", ymal["window"]), ("home_config", home["window"]),
                         ("sequence_config", sequence_home["window"])):
        if window["name"] != "confirmation":
            raise ValueError(f"{name} must use the confirmation week")
    if Path(sequence["home_config"]).resolve() != Path(config["home_config"]).resolve():
        raise ValueError("sequence_config must use the same home-page config as home_config")
    if not sequence.get("frozen_policy"):
        raise ValueError("sequence_config needs frozen_policy: the home page's frozen choice")
    return heldout, ymal, home, sequence


def save_code(path):
    """The uncommitted changes (git diff HEAD) plus the full text of new, untracked files."""
    try:
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, check=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True,
                                   text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        diff, untracked = "", []
    for name in untracked:
        try:
            diff += f"\n=== untracked file: {name} ===\n" + Path(name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            diff += f"\n=== untracked file (not text): {name} ===\n"
    path.write_text(diff, encoding="utf-8")


def frozen_record(config, heldout, ymal, sequence, folder):
    chosen = json.loads((Path(heldout["frozen_selection_run"]) / "selection_choice.json").read_text())["chosen"]
    code = git_state(Path.cwd())
    save_code(folder / "code.diff")
    return dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), code=code, code_diff="code.diff",
        week=dict(name="confirmation", start="2016-05-16", end_exclusive="2016-06-02"),
        pages={
            "category pages": dict(model=f"lightgbm_{chosen}", chosen_on=heldout["frozen_selection_run"],
                                   config=config["heldout_config"]),
            "item page (You may also like)": dict(policy=ymal["frozen_policy"], config=config["ymal_config"]),
            "home page": dict(policy=sequence["frozen_policy"], model_run=sequence["frozen_run"],
                              config=config["sequence_config"]),
        },
        settings=config)


def page_run(name, output_directory, run):
    """A finished confirmation run of this page from an earlier attempt, or a new one."""
    earlier = completed_run(output_directory)
    if earlier:
        print(f"[confirmation] {name}: reusing the finished run {earlier}", flush=True)
        return earlier
    print(f"[confirmation] {name}: scoring the confirmation week", flush=True)
    return Path(run())


def final_table(runs, frozen):
    """Each page's frozen choice against its baselines (values and 95% session-bootstrap intervals)."""
    columns = ["method", "reference", "metric", "value", "reference_value", "difference", "ci_low", "ci_high"]
    category = pd.read_csv(runs["category"] / "differences.csv")
    category = category[category.segment.eq("category_pages")
                        & category.method.eq(frozen["category pages"]["model"])].assign(page="category pages")
    item = pd.read_csv(runs["item"] / "differences.csv")
    item = item[item.method.eq(frozen["item page (You may also like)"]["policy"])].assign(
        page="item page (You may also like)")
    home = pd.read_csv(runs["home_sequence"] / "differences.csv")
    home = home[home.method.eq(frozen["home page"]["policy"])].assign(page="home page")
    return pd.concat([frame[["page"] + columns] for frame in (category, item, home)], ignore_index=True)


def run_confirmation(settings_path=Path("config/confirmation.json"), output_root=None):
    config = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    heldout, ymal, home, sequence = read_page_settings(config)
    root = Path(output_root or config["output_directory"])
    check_not_opened(root, marker="final_table.csv")
    started = sorted(root.glob("*_confirmation")) if root.exists() else []
    if started:
        folder = started[-1]
        frozen = json.loads((folder / "frozen_choices.json").read_text())
        print(f"[confirmation] Resuming {folder} (choices frozen at {frozen['frozen_at_utc']})", flush=True)
        # The code may have been fixed after the stop: record it as it is now, too.
        resumed = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        save_code(folder / f"code_resumed_{resumed}.diff")
        write_json(folder / f"code_resumed_{resumed}.json", git_state(Path.cwd()))
    else:
        folder = root / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_confirmation"
        folder.mkdir(parents=True)
        frozen = frozen_record(config, heldout, ymal, sequence, folder)
        write_json(folder / "frozen_choices.json", frozen)
        print(f"[confirmation] Frozen choices written to {folder / 'frozen_choices.json'}", flush=True)
    runs = {"category": page_run("category pages", heldout["output_directory"],
                                 lambda: run_heldout(config["heldout_config"])),
            "item": page_run("item page", ymal["output_directory"], lambda: run_ymal(config["ymal_config"])),
            "home": page_run("home page cases", home["output_directory"],
                             lambda: run_home_page(config["home_config"]))}

    def sequence_run():
        # The sequence models score exactly the cases of this home-page run.
        settings = json.loads(Path(config["sequence_config"]).read_text(encoding="utf-8"))
        path = folder / "sequence_config.json"
        path.write_text(json.dumps(settings | dict(home_run=runs["home"].as_posix()), indent=2), encoding="utf-8")
        return run_sequence_models(path)

    runs["home_sequence"] = page_run("home page SASRec", sequence["output_directory"], sequence_run)
    write_json(folder / "runs.json", {name: run.as_posix() for name, run in runs.items()})
    table = final_table(runs, frozen["pages"])
    save_table(table, folder / "final_table.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(table.round(4).to_string(index=False))
    print(f"[confirmation] Saved to {folder}", flush=True)
    return folder
