"""Replay scheduling experiment (fast, many batches).

    python -m experiments.scheduling_replay --batch-size 60 --n-batches 300

Uses the recorded true verification times of the held-out test pairs and the
model predictions from train_eval.py. Predictive policies are charged the
feature-extraction overhead of the whole batch up front.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models import plots
from src.models.scheduling import run_replay, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="experiments/results/test_predictions.csv")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--n-batches", type=int, default=300)
    ap.add_argument("--workers", default="1,4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.pred)
    workers = tuple(int(x) for x in a.workers.split(","))
    table, curves = run_replay(df, a.batch_size, a.n_batches, workers, a.seed)
    summ = summarize(table)
    out = Path(a.out)
    summ.to_csv(out / "scheduling_replay_summary.csv", index=False)
    plots.scheduling_bars(summ, out / "07_scheduling_replay_bars.png")
    plots.scheduling_curves(curves, summ, out / "08_scheduling_replay_curves.png")
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    (out / "scheduling_replay.json").write_text(json.dumps(summ.to_dict("records"), indent=2))
    print("\nwrote scheduling_replay_summary.csv, figures 07, 08")


if __name__ == "__main__":
    main()
