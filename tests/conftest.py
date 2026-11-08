"""Shared test setup.

Load LightGBM before any test module imports pandas: on native Windows the
opposite order makes LightGBM training crash. See session_rerank/__init__.py.
"""

import lightgbm  # noqa: F401
