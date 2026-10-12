"""Models, feature sets (ablation), group-aware CV and permutation importance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extraction.builder import feature_groups

# key -> (label, feature groups, timing columns that must be paid before predicting)
FEATURE_SETS: dict[str, tuple[str, list[str], list[str]]] = {
    "A": ("P1 only (baseline)", ["p1"], ["t_p1_ms"]),
    "B": ("P1 + P2 concat", ["p1", "p2"], ["t_p1_ms", "t_delta_ms"]),
    "C": ("P1 + Delta", ["p1", "delta"], ["t_p1_ms", "t_delta_ms"]),
    "D": ("P1 + Delta + Transform", ["p1", "delta", "transform"], ["t_p1_ms", "t_delta_ms"]),
    "E": ("D + static probe", ["p1", "delta", "transform", "probe_static"],
          ["t_p1_ms", "t_delta_ms", "t_probe_static_ms"]),
    "F": ("E + dynamic probe", ["p1", "delta", "transform", "probe_static", "probe_dyn"],
          ["t_p1_ms", "t_delta_ms", "t_probe_static_ms", "t_probe_dyn_ms"]),
}
MODEL_NAMES = ["Ridge", "RandomForest", "ExtraTrees", "HistGB"]


def columns_for(df: pd.DataFrame, key: str) -> list[str]:
    g = feature_groups(list(df.columns))
    cols: list[str] = []
    for grp in FEATURE_SETS[key][1]:
        cols += g[grp]
    return cols


def overhead_ms(df: pd.DataFrame, key: str) -> np.ndarray:
    return df[FEATURE_SETS[key][2]].sum(axis=1).to_numpy()


def make_model(name: str, seed: int = 0):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "HistGB":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed)
    raise ValueError(name)


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, p).statistic if np.std(p) > 0 else 0.0
    return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "spearman": float(rho)}


def cross_validate(df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, keys: list[str],
                   models: list[str], n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Group K-fold (all pairs of one base program stay in the same fold)."""
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for key in keys:
        X = df[columns_for(df, key)].to_numpy(dtype=float)
        for m in models:
            folds = []
            for tr, te in gkf.split(X, y, groups):
                mdl = make_model(m, seed).fit(X[tr], y[tr])
                folds.append(regression_metrics(y[te], mdl.predict(X[te])))
            f = pd.DataFrame(folds)
            row = {"set": key, "label": FEATURE_SETS[key][0], "model": m, "n_features": X.shape[1]}
            for c in f.columns:
                row[f"{c}_mean"], row[f"{c}_std"] = f[c].mean(), f[c].std()
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Fitted:
    key: str
    model_name: str
    columns: list[str]
    model: object


def fit_set(df: pd.DataFrame, y: np.ndarray, idx: np.ndarray, key: str, model_name: str,
            seed: int = 0) -> Fitted:
    cols = columns_for(df, key)
    mdl = make_model(model_name, seed).fit(df.iloc[idx][cols].to_numpy(float), y[idx])
    return Fitted(key, model_name, cols, mdl)


def predict(f: Fitted, df: pd.DataFrame) -> np.ndarray:
    return f.model.predict(df[f.columns].to_numpy(float))


def permutation_importance_individual(f: Fitted, df_te: pd.DataFrame, y_te: np.ndarray,
                                      n_repeats: int = 5, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    X = df_te[f.columns].to_numpy(float)
    base = mean_absolute_error(y_te, f.model.predict(X))
    out = {}
    for j, c in enumerate(f.columns):
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            inc.append(mean_absolute_error(y_te, f.model.predict(Xp)) - base)
        out[c] = float(np.mean(inc))
    return pd.Series(out).sort_values(ascending=False)


def permutation_importance_groups(f: Fitted, df_te: pd.DataFrame, y_te: np.ndarray,
                                  n_repeats: int = 10, seed: int = 0) -> pd.Series:
    """Permute whole feature groups jointly (rows shuffled together)."""
    rng = np.random.default_rng(seed)
    X = df_te[f.columns].to_numpy(float)
    base = mean_absolute_error(y_te, f.model.predict(X))
    g = feature_groups(f.columns)
    out = {}
    for grp, cols in g.items():
        if not cols:
            continue
        ix = [f.columns.index(c) for c in cols]
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, ix] = Xp[rng.permutation(len(Xp))][:, ix]
            inc.append(mean_absolute_error(y_te, f.model.predict(Xp)) - base)
        out[grp] = float(np.mean(inc))
    return pd.Series(out).sort_values(ascending=False)
