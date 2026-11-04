# Original-pipeline reference

`pipeline_reference.json` records the original `select()` and `confirm()`
functions from commit `3559fff`, executed before this refactor against the
three-query fixture defined in `make_toy_experiment()` in `tests/test_pipeline.py`.

Model predictions are deterministic stand-ins; normalization, tree/neural
fusion, re-ranking, policy search, metric calculations, and reporting use the
original project functions. The reference records the frozen choices, method
order, per-query metric arrays, diversification sweep, examples, and receipt.
It is a regression reference, not a full-data benchmark result.

Cross-platform tests compare recorded file hashes after normalizing path
separators. The receipt's frozen-file hash is checked against the current bytes
because platform-specific path separators change the JSON bytes.
