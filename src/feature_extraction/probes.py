"""Cheap probe features, inspired by SATzilla-style 'probing'.

* static probe  : structural statistics of the Z3 miter formula (no solving).
* dynamic probe : a very short solver run (default 20 ms) whose outcome and
                  statistics are used as features.

Both are *pre-verification* signals: their cost is measured and charged to the
predictor in the scheduling experiments.
"""
from __future__ import annotations

import time

import z3

from src.verification.equivalence import Miter, solve_miter

_KINDS = {
    z3.Z3_OP_BMUL: "mul", z3.Z3_OP_BADD: "add", z3.Z3_OP_BSUB: "sub",
    z3.Z3_OP_BAND: "and", z3.Z3_OP_BOR: "or", z3.Z3_OP_BXOR: "xor",
    z3.Z3_OP_BSHL: "shl", z3.Z3_OP_BASHR: "shr", z3.Z3_OP_ITE: "ite",
}


def formula_stats(e: z3.ExprRef) -> dict[str, float]:
    """Unique-node DAG statistics (iterative, safe for very deep terms)."""
    depth: dict[int, int] = {}
    counts = {f"pf_{k}": 0.0 for k in _KINDS.values()}
    counts.update({"pf_mul_vv": 0.0, "pf_vars": 0.0, "pf_nodes": 0.0})
    stack = [(e, False)]
    while stack:
        n, done = stack.pop()
        nid = n.get_id()
        if done:
            kids = n.children()
            depth[nid] = 1 + max((depth[c.get_id()] for c in kids), default=0)
            continue
        if nid in depth:
            continue
        depth[nid] = -1  # in progress marker (DAG: no cycles)
        counts["pf_nodes"] += 1
        if z3.is_app(n):
            k = n.decl().kind()
            if k in _KINDS:
                counts[f"pf_{_KINDS[k]}"] += 1
                if k == z3.Z3_OP_BMUL and not any(z3.is_bv_value(c) for c in n.children()):
                    counts["pf_mul_vv"] += 1
            elif n.num_args() == 0 and k == z3.Z3_OP_UNINTERPRETED:
                counts["pf_vars"] += 1
        stack.append((n, True))
        for c in n.children():
            if c.get_id() not in depth:
                stack.append((c, False))
    counts["pf_depth"] = float(depth[e.get_id()])
    return counts


def static_probe(m: Miter) -> dict[str, float]:
    f = formula_stats(m.formula)
    simp = z3.simplify(m.formula)
    f["pf_trivially_equal"] = 1.0 if z3.is_false(simp) else 0.0
    f["pf_simp_nodes"] = formula_stats(simp)["pf_nodes"]
    f["pf_simp_ratio"] = f["pf_simp_nodes"] / max(f["pf_nodes"], 1.0)
    return f


def dynamic_probe(m: Miter, probe_ms: int = 20) -> dict[str, float]:
    t0 = time.perf_counter()
    outcome, _, conflicts, mem = solve_miter(m, probe_ms)
    elapsed = (time.perf_counter() - t0) * 1000
    return {"pd_solved": 0.0 if outcome == "timeout" else 1.0,
            "pd_ms": elapsed, "pd_conflicts": conflicts, "pd_mem_mb": mem}
