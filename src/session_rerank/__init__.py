"""Diginetica session-aware re-ranking research benchmark."""

# On native Windows, LightGBM training crashes (access violation in
# LGBM_DatasetSetField) when pandas is imported before LightGBM. Loading
# LightGBM's native library first avoids the conflict. Importing this package
# triggers it, so scripts should import session_rerank before pandas.
import lightgbm  # noqa: F401
