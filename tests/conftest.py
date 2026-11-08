"""Shared test setup.

Make src/ importable when the project is not installed (pip -r requirements.txt),
then load LightGBM before any test module imports pandas: on native Windows the
opposite order makes LightGBM training crash. See session_rerank/__init__.py.
"""

import sys
from pathlib import Path

_SOURCE = str(Path(__file__).resolve().parents[1] / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

import lightgbm  # noqa: E402,F401
