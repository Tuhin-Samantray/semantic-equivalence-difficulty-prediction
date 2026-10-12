"""Phase 4c step 1: does the lean feature path reproduce the stored predictions?"""
import joblib
import numpy as np
import pandas as pd

from src.models.realrun import extract_for_model, score_batch

df = pd.read_csv("experiments/results_D/test_predictions.csv")
ours = joblib.load("data/processed/models/model_ours.joblib")
base = joblib.load("data/processed/models/model_baseline.joblib")
for f in (ours, base):
    f.model.n_jobs = 1
print("ours:", ours.key, ours.model_name, len(ours.columns), "features")
print("base:", base.key, base.model_name, len(base.columns), "features")
print("rows:", len(df), "| base programs:", df["base_id"].nunique())

sub = df.head(10).reset_index(drop=True)
for name, f, col in (("ours", ours, "pred_ours"), ("base", base, "pred_base")):
    scores, total_ms = score_batch(f, sub)
    diff = np.abs(scores - sub[col].to_numpy())
    print(f"{name}: max |diff| vs CSV = {diff.max():.2e} | {total_ms / len(sub):.1f} ms/pair "
          f"(CSV ovh column mean: {sub['ovh_' + ('ours' if name == 'ours' else 'base')].mean():.1f} ms)")
