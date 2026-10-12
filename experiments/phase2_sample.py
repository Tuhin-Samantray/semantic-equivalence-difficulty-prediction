"""Phase 2 check: verify 20 sample pairs with Z3 and record the costs."""
from __future__ import annotations

import random

import pandas as pd

from src.generation.programs import generate_program
from src.generation.transforms import apply_pipeline
from src.verification.equivalence import verify_pair

rows = []
for seed in range(20):
    prog = generate_program(seed)
    rng = random.Random(seed)
    p2, names, info = apply_pipeline(prog.src, prog.width, rng, rng.randint(1, 3))
    r = verify_pair(prog.src, p2, prog.width, timeout_ms=5000)
    rows.append({"seed": seed, "width": prog.width, "n_args": prog.n_args,
                 "transforms": "+".join(names), **r.as_dict()})

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 40)
print(df.round(1).to_string(index=False))
print("outcomes:", df["outcome"].value_counts().to_dict())
s = df["solve_ms"]
print(f"solve_ms min/median/max: {s.min():.1f} / {s.median():.1f} / {s.max():.1f}")
df.to_csv("experiments/results/phase2_sample.csv", index=False)
