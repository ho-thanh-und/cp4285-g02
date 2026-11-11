"""Evaluation weeks and the rule for opening the held-out ones.

  * validation   May 1–7: model and policy development
  * selection    May 9–14: choose between frozen candidates, opened with a frozen choice
  * confirmation May 16 – June 1: the final test, opened once with every page's
    choice frozen (user decision 2026-11-09: the whole period is used)
"""

from pathlib import Path

WINDOWS = {"validation": ("2016-05-01", "2016-05-08"), "selection": ("2016-05-09", "2016-05-15"),
           "confirmation": ("2016-05-16", "2016-06-02")}
HELD_OUT = ("selection", "confirmation")


def check_window(window):
    if WINDOWS.get(window["name"]) != (window["start"], window["end_exclusive"]):
        raise ValueError(f"window must be one of {WINDOWS}")


def completed_run(output_directory, name="confirmation", marker="summary.csv"):
    """The finished run of a week in this output folder (its marker file written), or None."""
    folder = Path(output_directory)
    finished = [run for run in sorted(folder.glob(f"*_{name}")) if (run / marker).exists()] if folder.exists() else []
    return finished[-1] if finished else None


def check_not_opened(output_directory, name="confirmation", marker="summary.csv"):
    """The confirmation week is scored once: refuse if this output folder already has a finished
    run of it. A run that stopped before writing its results (a crash) does not count."""
    earlier = completed_run(output_directory, name, marker)
    if name == "confirmation" and earlier:
        raise ValueError(f"The confirmation week was already scored here ({earlier}); it is opened only once")
