"""Feature-subset selection: SHAP (ours) and the ablation baselines."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def feature_variances(meta: dict, X_sample: np.ndarray) -> np.ndarray:
    """Variance after the signed-log transform but before standardisation (standardised data has var 1).
    One-hot columns, which are not standardised, use their sample variance."""
    var = np.asarray(meta["scaler"]["var"], dtype=np.float64)
    if len(var) == X_sample.shape[1]:
        return var
    return np.concatenate([var, X_sample[:, len(var):].var(0)])


def select_features(method: str, k: int, n_features: int, *, shap_ranking: np.ndarray | None = None,
                    X_sample: np.ndarray | None = None, y_sample: np.ndarray | None = None,
                    meta: dict | None = None, seed: int = 0) -> np.ndarray:
    """Return sorted column indices of the chosen subset."""
    k = min(k, n_features)
    if method == "all":
        return np.arange(n_features)
    if method == "shap":
        chosen = np.asarray(shap_ranking)[:k]
    elif method == "random":
        chosen = np.random.default_rng(seed).choice(n_features, k, replace=False)
    elif method == "mi":
        mi = mutual_info_classif(X_sample, y_sample, random_state=seed)
        chosen = np.argsort(-mi)[:k]
    elif method == "variance":
        chosen = np.argsort(-feature_variances(meta, X_sample))[:k]
    else:
        raise ValueError(f"Unknown selection method {method!r}")
    return np.sort(chosen)
