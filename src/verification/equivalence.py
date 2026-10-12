"""Z3-based semantic equivalence checking with cost instrumentation.

Two programs are equivalent iff the *miter* ``out1 != out2`` is UNSAT over all
bit-vector inputs. Loops have constant trip counts so symbolic execution fully
unrolls them: the check is complete, not bounded.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass, asdict

import z3

from src.verification.semantics import Z3Alg, run


@dataclass
class VerifyResult:
    outcome: str        # 'equivalent' | 'different' | 'timeout'
    build_ms: float     # parse + symbolic execution + miter construction
    solve_ms: float     # Z3 check() wall time
    total_ms: float     # build_ms + solve_ms  (the end-to-end verification cost)
    conflicts: float
    memory_mb: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Miter:
    ctx: z3.Context
    args: list
    out1: z3.ExprRef
    out2: z3.ExprRef
    formula: z3.BoolRef


def build_miter(src1: str, src2: str, width: int) -> Miter:
    ctx = z3.Context()
    alg = Z3Alg(width, ctx)
    t1, t2 = ast.parse(src1), ast.parse(src2)
    names = [a.arg for a in t1.body[0].args.args]
    args = [alg.arg(f"in{i}") for i in range(len(names))]
    o1, o2 = run(t1, args, alg), run(t2, args, alg)
    return Miter(ctx, args, o1, o2, o1 != o2)


def _stats(solver: z3.Solver) -> tuple[float, float]:
    d = {}
    st = solver.statistics()
    for k in st.keys():
        try:
            d[k] = st.get_key_value(k)
        except Exception:
            pass
    conflicts = float(d.get("conflicts", d.get("sat conflicts", 0.0)))
    mem = float(d.get("max memory", d.get("memory", 0.0)))
    return conflicts, mem


def solve_miter(m: Miter, timeout_ms: int) -> tuple[str, float, float, float]:
    s = z3.SolverFor("QF_BV", ctx=m.ctx)
    s.set("timeout", int(timeout_ms))
    s.add(m.formula)
    t0 = time.perf_counter()
    r = s.check()
    solve_ms = (time.perf_counter() - t0) * 1000
    conflicts, mem = _stats(s)
    outcome = "equivalent" if r == z3.unsat else "different" if r == z3.sat else "timeout"
    return outcome, solve_ms, conflicts, mem


def verify_pair(src1: str, src2: str, width: int, timeout_ms: int = 5000) -> VerifyResult:
    t0 = time.perf_counter()
    m = build_miter(src1, src2, width)
    build_ms = (time.perf_counter() - t0) * 1000
    outcome, solve_ms, conflicts, mem = solve_miter(m, timeout_ms)
    if outcome == "timeout":
        solve_ms = max(solve_ms, float(timeout_ms))  # censored at the cap
    return VerifyResult(outcome, build_ms, solve_ms, build_ms + solve_ms, conflicts, mem)
