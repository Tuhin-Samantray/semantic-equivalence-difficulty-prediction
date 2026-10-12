"""Batch-scheduling simulation driven by predicted verification cost.

A batch of pairs must all be verified. A *policy* decides the dispatch order.
Predicted-cost policies (shortest-predicted-job-first) first pay the feature
extraction overhead for the whole batch (conservatively: sequentially, before
any verification starts), then dispatch in predicted order.

Metrics
    mean_completion : mean time until each pair's result is available
                      (what an interactive/CI user experiences)
    makespan        : time until the whole batch is done
    t50 / t90       : time until 50% / 90% of the batch is verified
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

POLICIES = ["FIFO", "Pred-SJF (ours)", "Pred-SJF (P1-only baseline)", "Oracle-SJF", "Longest-first"]


def simulate(order: np.ndarray, cost: np.ndarray, n_workers: int, start: float = 0.0) -> np.ndarray:
    """List scheduling on ``n_workers`` identical workers; returns completion time per job."""
    free = [start] * n_workers
    heapq.heapify(free)
    done = np.zeros(len(cost))
    for j in order:
        t = heapq.heappop(free)
        done[j] = t + cost[j]
        heapq.heappush(free, done[j])
    return done


def _metrics(done: np.ndarray) -> dict[str, float]:
    return {"mean_completion": float(done.mean()), "makespan": float(done.max()),
            "t50": float(np.quantile(done, 0.5)), "t90": float(np.quantile(done, 0.9))}


def orders_for_batch(cost: np.ndarray, pred_ours: np.ndarray, pred_base: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "FIFO": np.arange(len(cost)),
        "Pred-SJF (ours)": np.argsort(pred_ours, kind="stable"),
        "Pred-SJF (P1-only baseline)": np.argsort(pred_base, kind="stable"),
        "Oracle-SJF": np.argsort(cost, kind="stable"),
        "Longest-first": np.argsort(-cost, kind="stable"),
    }


def run_replay(df: pd.DataFrame, batch_size: int = 60, n_batches: int = 300,
               workers: tuple[int, ...] = (1, 4), seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Replay recorded true costs under each policy over many random batches.

    ``df`` needs: verify_ms, pred_ours, pred_base, ovh_ours, ovh_base.
    Returns (per-batch metrics table, mean completion curves for plotting).
    """
    rng = np.random.default_rng(seed)
    rows = []
    curve_acc: dict[tuple[int, str], list[np.ndarray]] = {}
    batches = []
    for b in range(n_batches):
        idx = rng.choice(len(df), size=min(batch_size, len(df)), replace=False)
        batches.append(idx)
    ref_makespan = {}
    for k in workers:
        ref_makespan[k] = float(np.median([
            simulate(np.arange(len(i)), df["verify_ms"].to_numpy()[i], k).max() for i in batches]))
    grids = {k: np.linspace(0, ref_makespan[k] * 1.15, 200) for k in workers}

    for b, idx in enumerate(batches):
        sub = df.iloc[idx]
        cost = sub["verify_ms"].to_numpy()
        po, pb = sub["pred_ours"].to_numpy(), sub["pred_base"].to_numpy()
        ovh = {"Pred-SJF (ours)": sub["ovh_ours"].sum(), "Pred-SJF (P1-only baseline)": sub["ovh_base"].sum()}
        orders = orders_for_batch(cost, po, pb)
        for k in workers:
            for pol in POLICIES:
                done = simulate(orders[pol], cost, k, start=ovh.get(pol, 0.0))
                rows.append({"batch": b, "workers": k, "policy": pol, **_metrics(done)})
                curve_acc.setdefault((k, pol), []).append(
                    (done[None, :] <= grids[k][:, None]).mean(axis=1))
    table = pd.DataFrame(rows)
    curves = {"grids": grids, "mean_curves": {kp: np.mean(v, axis=0) for kp, v in curve_acc.items()}}
    return table, curves


def summarize(table: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per policy: mean metrics and speed-up over FIFO (mean ratio, 95% interval over batches)."""
    out = []
    for k, g in table.groupby("workers"):
        fifo = g[g.policy == "FIFO"].set_index("batch")
        for pol, gp in g.groupby("policy"):
            gp = gp.set_index("batch")
            sp = fifo["mean_completion"] / gp["mean_completion"]
            sp_make = fifo["makespan"] / gp["makespan"]
            out.append({"workers": k, "policy": pol,
                        "mean_completion_ms": gp["mean_completion"].mean(),
                        "makespan_ms": gp["makespan"].mean(),
                        "t50_ms": gp["t50"].mean(), "t90_ms": gp["t90"].mean(),
                        "speedup_mean_completion": sp.mean(),
                        "speedup_ci_lo": sp.quantile(0.025), "speedup_ci_hi": sp.quantile(0.975),
                        "speedup_makespan": sp_make.mean(),
                        "frac_batches_better_than_fifo": float((sp > 1.0).mean())})
    return pd.DataFrame(out)
