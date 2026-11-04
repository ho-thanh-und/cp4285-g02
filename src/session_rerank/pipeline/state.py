"""Small result objects and the existing experiment's saved-file conventions."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

RESULTS = Path("results/iteration3")
FROZEN_CHOICES = RESULTS / "frozen_selection.json"
CANDIDATES = RESULTS / "candidate_manifest.json"
SELECTION_SCORES = RESULTS / "selection_summary.csv"
EVALUATION_CACHE = RESULTS / "confirmation_per_query.npz"
PROTOCOL = Path("config/iteration3_protocol.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_models(hashes):
    """Reject missing or changed models/configuration before using saved choices."""
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file():
            raise RuntimeError(f"Recorded model/configuration is missing: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Recorded model/configuration changed: {path}")


@dataclass
class ModelComparison:
    selected_name: str
    nominees: dict
    validation_scores: list
    selection_scores: dict
    model_hashes: dict
    selection_queries: int
    protocol_sha256: str

    @property
    def selected(self):
        return self.nominees[self.selected_name]

    def frozen_payload(self, policies):
        # Preserve the original frozen-selection format for existing runs.
        return dict(
            selected_name=self.selected_name,
            selected=self.selected,
            nominees=self.nominees,
            original_validation_scores=self.validation_scores,
            fresh_selection_scores=self.selection_scores,
            policies=policies,
            hashes=self.model_hashes,
            selection_queries=self.selection_queries,
            protocol_sha256=self.protocol_sha256,
        )

    @classmethod
    def from_frozen(cls, payload):
        return cls(
            selected_name=payload["selected_name"],
            nominees=payload["nominees"],
            validation_scores=payload["original_validation_scores"],
            selection_scores=payload["fresh_selection_scores"],
            model_hashes=payload["hashes"],
            selection_queries=payload["selection_queries"],
            protocol_sha256=payload["protocol_sha256"],
        )


@dataclass
class EvaluationResults:
    metrics: dict
    diversity_sweep: Any
    dataset: dict
    examples: list
    receipt: dict
