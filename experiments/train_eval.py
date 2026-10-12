"""Train, compare, ablate and explain the verification-difficulty predictor.

    python -m experiments.train_eval --data data/processed/dataset.csv

Outputs (experiments/results/): metrics.json, cv_results.csv, test_predictions.csv,
and the figures 01..07. Trained models go to data/processed/models/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.models import plots
from src.models.train import (FEATURE_SETS, MODEL_NAMES, cross_validate, fit_set, overhead_ms,
                              permutation_importance_groups, permutation_importance_individual,
                              predict, regression_metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.csv")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--ours", default="E", choices=list(FEATURE_SETS), help="feature set used as 'our' model")
    ap.add_argument("--model", default="auto", help="model name or 'auto' (best CV MAE on --ours)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    y = df["log_cost"].to_numpy()
    groups = df["base_id"].to_numpy()
    print(f"{len(df)} pairs, {df.base_id.nunique()} base programs, "
          f"{100 * df.is_timeout.mean():.1f}% timeouts")

    plots.label_distribution(df, out / "01_label_distribution.png")
    plots.cost_by_transform(df, out / "02_cost_by_transformation.png")

    # ---- 1. cross-validated comparison: every feature set x every model -------------------
    keys = list(FEATURE_SETS)
    cv = cross_validate(df, y, groups, keys, MODEL_NAMES, a.folds, a.seed)
    cv.to_csv(out / "cv_results.csv", index=False)
    plots.model_comparison(cv, out / "03_model_comparison.png")
    best_model = a.model
    if best_model == "auto":
        sub = cv[cv.set == a.ours].sort_values("mae_mean")
        best_model = str(sub.iloc[0]["model"])
    print(f"selected model: {best_model}")
    cv_best = cv[cv.model == best_model].set_index("set").loc[keys].reset_index()
    print(cv_best[["set", "label", "mae_mean", "r2_mean", "spearman_mean"]].round(3).to_string(index=False))

    # ---- 2. held-out test split (grouped by base program: no leakage) ----------------------
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=a.seed).split(df, y, groups))
    fitted = {k: fit_set(df, y, tr, k, best_model, a.seed) for k in keys}
    dte, yte = df.iloc[te].reset_index(drop=True), y[te]
    rows = []
    for k in keys:
        p = predict(fitted[k], dte)
        m = regression_metrics(yte, p)
        if 0 < dte["is_timeout"].sum() < len(dte):
            m["timeout_auc"] = float(roc_auc_score(dte["is_timeout"], p))
        m.update({"set": k, "label": FEATURE_SETS[k][0], "overhead_ms_mean": float(overhead_ms(dte, k).mean())})
        rows.append(m)
    test_metrics = pd.DataFrame(rows)
    print("\nHeld-out test:")
    print(test_metrics.round(3).to_string(index=False))
    plots.ablation(test_metrics, cv_best, out / "04_ablation.png")

    pb, po = predict(fitted["A"], dte), predict(fitted[a.ours], dte)
    plots.pred_vs_actual(yte, pb, po, out / "05_pred_vs_actual.png",
                         label_ours=f"{FEATURE_SETS[a.ours][0]}")

    # ---- 3. explanation -------------------------------------------------------------------
    g_imp = permutation_importance_groups(fitted[a.ours], dte, yte, seed=a.seed)
    i_imp = permutation_importance_individual(fitted[a.ours], dte, yte, seed=a.seed)
    plots.importance(g_imp, i_imp, out / "06_feature_importance.png")
    print("\nGroup importance (MAE increase):\n", g_imp.round(4).to_string())

    # ---- 4. predictions for the scheduling experiments -------------------------------------
    pred = dte[["pair_id", "base_id", "width", "transforms", "outcome", "verify_ms", "log_cost",
                "t_unroll_factor", "src1", "src2"]].copy()
    pred["pred_ours"], pred["pred_base"] = po, pb
    pred["ovh_ours"], pred["ovh_base"] = overhead_ms(dte, a.ours), overhead_ms(dte, "A")
    pred.to_csv(out / "test_predictions.csv", index=False)

    mdir = Path("data/processed/models")
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[a.ours], mdir / "model_ours.joblib")
    joblib.dump(fitted["A"], mdir / "model_baseline.joblib")

    metrics = {"n_pairs": int(len(df)), "n_base_programs": int(df.base_id.nunique()),
               "timeout_rate": float(df.is_timeout.mean()), "selected_model": best_model,
               "ours_set": a.ours, "cv": cv_best.to_dict("records"),
               "test": test_metrics.to_dict("records"),
               "group_importance": g_imp.to_dict(), "top_features": i_imp.head(15).to_dict()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}/metrics.json, cv_results.csv, test_predictions.csv and figures")


if __name__ == "__main__":
    main()
