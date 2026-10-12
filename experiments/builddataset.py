"""Build the labelled dataset of transformed program pairs.

    python -m experiments.build_dataset --n-programs 400 --pairs 5 --workers 14

Outputs:
    data/processed/dataset.csv                  features + labels + sources
    data/verification_logs/verification_log.jsonl   raw verifier statistics
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.feature_extraction.builder import extract_all
from src.generation.programs import generate_program
from src.generation.transforms import apply_pipeline
from src.verification.equivalence import verify_pair


def process_program(seed: int, pairs: int, timeout_ms: int, probe_ms: int) -> list[dict]:
    prog = generate_program(seed)
    rng = random.Random(seed * 7919 + 13)
    rows, seen = [], {prog.src}
    attempts = 0
    while len(rows) < pairs and attempts < pairs * 6:
        attempts += 1
        try:
            src2, applied, info = apply_pipeline(prog.src, prog.width, rng, rng.randint(1, 3))
        except ValueError:
            continue
        if src2 in seen:
            continue
        seen.add(src2)
        feats, tim = extract_all(prog.src, src2, prog.width, applied, info, probe_ms)
        res = verify_pair(prog.src, src2, prog.width, timeout_ms)
        row = {"base_id": seed, "pair_id": f"{seed}_{len(rows)}", "width": prog.width,
               "transforms": "|".join(applied), "src1": prog.src, "src2": src2}
        row.update(feats)
        row.update(tim)
        row.update({"outcome": res.outcome, "build_ms": res.build_ms, "solve_ms": res.solve_ms,
                    "verify_ms": res.total_ms, "conflicts": res.conflicts,
                    "memory_mb": res.memory_mb})
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-programs", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=5, help="pairs per base program")
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--probe-ms", type=int, default=20)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default="data/processed/dataset.csv")
    a = ap.parse_args()

    t0 = time.time()
    seeds = range(a.seed_offset, a.seed_offset + a.n_programs)
    chunks = Parallel(n_jobs=a.workers, verbose=5)(
        delayed(process_program)(s, a.pairs, a.timeout_ms, a.probe_ms) for s in seeds)
    rows = [r for c in chunks for r in c]
    df = pd.DataFrame(rows)
    df["log_cost"] = df["verify_ms"].map(lambda x: math.log10(1.0 + x))
    df["is_timeout"] = (df["outcome"] == "timeout").astype(int)
    df["composite_cost"] = (0.7 * df["log_cost"]
                            + 0.2 * df["conflicts"].map(lambda x: math.log10(1 + x))
                            + 0.1 * df["memory_mb"].map(lambda x: math.log10(1 + x)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log = Path("data/verification_logs/verification_log.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("pair_id", "width", "transforms", "outcome",
                                                   "build_ms", "solve_ms", "verify_ms",
                                                   "conflicts", "memory_mb")}) + "\n")
    print(f"\n{len(df)} pairs from {df.base_id.nunique()} programs in {time.time() - t0:.0f}s")
    print(df["outcome"].value_counts().to_string())
    print(df["verify_ms"].describe(percentiles=[.1, .5, .9, .99]).round(1).to_string())
    print("NaNs in numeric columns:", int(df.select_dtypes("number").isna().sum().sum()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
