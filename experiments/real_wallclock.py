"""Phase 4c: REAL wall-clock scheduling run (one worker, overhead included).

FIFO vs Pred-SJF (P1-only baseline, set A) vs Pred-SJF (ours, set D).
Batches are disjoint (split by base program) so they are independent.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import plots
from src.models.realrun import score_batch
from src.verification.equivalence import verify_pair

POLICIES = ["FIFO", "Pred-SJF (P1-only baseline)", "Pred-SJF (ours)"]
N_SPLITS = 5


def make_batches(df: pd.DataFrame, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(df["base_id"].unique()))
    rng.shuffle(ids)
    out = []
    for chunk in np.array_split(ids, N_SPLITS):
        sub = df[df["base_id"].isin(chunk)]
        out.append(sub.iloc[rng.permutation(len(sub))].reset_index(drop=True))
    return out


def run_policy(policy: str, batch: pd.DataFrame, models: dict) -> tuple[list[dict], float]:
    t_start = time.perf_counter()
    if policy == "FIFO":
        order, ovh_ms = np.arange(len(batch)), 0.0
    else:
        scores, ovh_ms = score_batch(models[policy], batch)
        order = np.argsort(scores, kind="stable")
    rows = []
    for pos, j in enumerate(order):
        r = batch.iloc[j]
        res = verify_pair(r["src1"], r["src2"], int(r["width"]), timeout_ms=5000)
        rows.append({"policy": policy, "pair_id": r["pair_id"], "position": pos,
                     "completion_s": time.perf_counter() - t_start, "outcome": res.outcome,
                     "verify_ms": res.total_ms, "stored_outcome": r["outcome"]})
    return rows, ovh_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results_real")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("experiments/results_D/test_predictions.csv")
    ours = joblib.load("data/processed/models/model_ours.joblib")
    base = joblib.load("data/processed/models/model_baseline.joblib")
    for f in (ours, base):
        f.model.n_jobs = 1
    models = {"Pred-SJF (ours)": ours, "Pred-SJF (P1-only baseline)": base}
    batches = make_batches(df, a.seed)[: a.batches]
    print("batch sizes:", [len(b) for b in batches], "| repeats:", a.repeats)

    warm = df.head(5).reset_index(drop=True)          # untimed warm-up
    for f in models.values():
        score_batch(f, warm)
    for _, r in warm.head(3).iterrows():
        verify_pair(r["src1"], r["src2"], int(r["width"]), timeout_ms=5000)

    rng = np.random.default_rng(a.seed + 1)
    raw, runs = [], []
    t_all = time.perf_counter()
    for rep in range(a.repeats):
        for bi, batch in enumerate(batches):
            for pol in rng.permutation(POLICIES):
                rows, ovh_ms = run_policy(str(pol), batch, models)
                d = np.array([r["completion_s"] for r in rows])
                for r in rows:
                    raw.append({"repeat": rep, "batch": bi, **r})
                runs.append({"repeat": rep, "batch": bi, "policy": str(pol), "overhead_s": ovh_ms / 1000,
                             "mean_completion_s": d.mean(), "makespan_s": d.max(),
                             "t50_s": np.quantile(d, 0.5), "t90_s": np.quantile(d, 0.9),
                             "n_timeouts": sum(r["outcome"] == "timeout" for r in rows)})
                print(f"rep {rep} batch {bi} {pol:<28} mean {d.mean():6.2f}s  makespan {d.max():6.2f}s  "
                      f"ovh {ovh_ms / 1000:5.2f}s  [{time.perf_counter() - t_all:6.0f}s elapsed]", flush=True)
                pd.DataFrame(raw).to_csv(out / "real_run_raw.csv", index=False)
    raw_df, runs_df = pd.DataFrame(raw), pd.DataFrame(runs)
    runs_df.to_csv(out / "real_run_runs.csv", index=False)

    # Summary: average repeats within each batch first, so the 5 batches are the independent units.
    per_batch = runs_df.groupby(["batch", "policy"]).mean(numeric_only=True).reset_index()
    rows = []
    for pol in POLICIES:
        g = per_batch[per_batch.policy == pol].set_index("batch")
        f = per_batch[per_batch.policy == "FIFO"].set_index("batch")
        sp, spm = f["mean_completion_s"] / g["mean_completion_s"], f["makespan_s"] / g["makespan_s"]
        rows.append({"policy": pol, "mean_completion_s": g["mean_completion_s"].mean(),
                     "makespan_s": g["makespan_s"].mean(), "t50_s": g["t50_s"].mean(),
                     "t90_s": g["t90_s"].mean(), "overhead_s": g["overhead_s"].mean(),
                     "speedup_mean": sp.mean(), "speedup_min": sp.min(), "speedup_max": sp.max(),
                     "speedup_makespan": spm.mean(), "batches_better": int((sp > 1).sum()),
                     "n_batches": len(sp)})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "real_run_summary.csv", index=False)
    print("\n", summ.round(3).to_string(index=False))
    print("\nrepeat-to-repeat spread of FIFO mean completion (s), per batch:")
    print(runs_df[runs_df.policy == "FIFO"].groupby("batch")["mean_completion_s"].agg(["min", "max"]).round(2))
    print("outcome differs from stored label in", f"{(raw_df.outcome != raw_df.stored_outcome).mean():.1%}", "of verifications")

    results = {pol: [np.concatenate([g["completion_s"].to_numpy()
                                     for _, g in raw_df[(raw_df.policy == pol) & (raw_df.repeat == r)].groupby("batch")])
                     for r in range(a.repeats)] for pol in POLICIES}
    print("figure:", plots.real_run_curves(results, out / "09_real_run.png"))


if __name__ == "__main__":
    main()
