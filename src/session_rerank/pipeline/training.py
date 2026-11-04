"""Step 2: fit the same reference, tree, GRU4Rec and SASRec candidates."""

from pathlib import Path

from .state import RESULTS, FROZEN_CHOICES, CANDIDATES, SELECTION_SCORES


def train_models():
    """Fit every model that will enter the comparison in step 3.

    Uses the samples/features saved by prepare_dataset(). Saves tree models,
    GRU4Rec/SASRec checkpoints, validation predictions, and training summaries.
    Existing completed models are reused. Once candidates have been recorded
    for fresh selection, further fitting is disabled to keep evaluation fair.
    Returns nothing; model files are saved under results/.
    """
    from .. import benchmark, optimize_ranker, recbole_model, neural_search, iteration3

    print("[2/6] Train multiple models", flush=True)
    if any(path.exists() for path in (FROZEN_CHOICES, CANDIDATES, SELECTION_SCORES)):
        print("Reuse the models recorded before selection; no further fitting.", flush=True)
        return

    # The 12-feature reference gives us a simpler model to compare against.
    if not Path("results/model.txt").exists():
        benchmark.develop_reference_model()

    # These earlier trees use 40 base/temporal features + 8 view features.
    # More leaves let a tree express more complex rules.
    columns = len(iteration3.FEATURE_NAMES) + len(iteration3.VIEW_NAMES)
    for name, leaves in (("views_60k", 31), ("views_60k_63", 63)):
        if not Path(f"results/optimization/{name}.txt").exists():
            optimize_ranker.train(name, columns, leaves=leaves, variant="views")

    # Earlier GRU4Rec uses BPR: prefer a clicked item to a sampled negative.
    if not Path("results/optimization/recbole_gru4rec.pth").exists():
        recbole_model.train()

    # New trees also receive 15 collaborative/context features. The no_user
    # variant tests what happens without persistent-user profile features.
    for name, leaves, no_user in (
        ("collab31", 31, False),
        ("collab63", 63, False),
        ("anonymous31", 31, True),
    ):
        if not (RESULTS / f"{name}.json").exists():
            iteration3.train_tree(name, leaves, no_user)

    # CE (cross-entropy) models predict the target item over the vocabulary.
    # SASRec uses attention; GRU4Rec uses a recurrent sequence encoder.
    for variant in ("sas_ce", "gru_ce"):
        if not (RESULTS / f"{variant}.json").exists():
            neural_search.train(variant)
