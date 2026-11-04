"""Step 2: fit the same reference, tree, GRU4Rec and SASRec candidates."""

from pathlib import Path

from .state import RESULTS, FROZEN_CHOICES, CANDIDATES, SELECTION_SCORES


def train_models():
    from .. import benchmark, optimize_ranker, recbole_model, neural_search, iteration3

    print("[2/6] Train multiple models", flush=True)
    if any(path.exists() for path in (FROZEN_CHOICES, CANDIDATES, SELECTION_SCORES)):
        print("Reuse the models recorded before selection; no further fitting.", flush=True)
        return

    if not Path("results/model.txt").exists():
        benchmark.develop_reference_model()

    columns = len(iteration3.FEATURE_NAMES) + len(iteration3.VIEW_NAMES)
    for name, leaves in (("views_60k", 31), ("views_60k_63", 63)):
        if not Path(f"results/optimization/{name}.txt").exists():
            optimize_ranker.train(name, columns, leaves=leaves, variant="views")

    if not Path("results/optimization/recbole_gru4rec.pth").exists():
        recbole_model.train()

    for name, leaves, no_user in (
        ("collab31", 31, False),
        ("collab63", 63, False),
        ("anonymous31", 31, True),
    ):
        if not (RESULTS / f"{name}.json").exists():
            iteration3.train_tree(name, leaves, no_user)

    for variant in ("sas_ce", "gru_ce"):
        if not (RESULTS / f"{variant}.json").exists():
            neural_search.train(variant)
