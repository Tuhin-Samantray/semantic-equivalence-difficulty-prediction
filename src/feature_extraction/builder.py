"""Assemble all feature groups for a program pair and time each group.

Groups (column prefixes):
    p1_  features of the original program            (single-program baseline)
    p2_  features of the transformed program
    d_   delta features: p2 - p1, ratios, AST edit proxies
    t_   transformation metadata (which passes were applied)
    pf_  static probe  (miter formula statistics, no solving)
    pd_  dynamic probe (20 ms solver run)
"""
from __future__ import annotations

import ast
import difflib
import time

from src.ast_extractor.features import node_histogram, program_features
from src.feature_extraction.probes import dynamic_probe, static_probe
from src.generation.transforms import TRANSFORM_NAMES
from src.verification.equivalence import build_miter

RATIO_KEYS = ["n_nodes", "dyn_mul_vv", "dyn_arith", "dyn_stmts", "n_mul_vv", "cfg_blocks",
              "expr_depth_max", "dyn_loop_iters"]


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def delta_features(t1: ast.Module, t2: ast.Module, f1: dict, f2: dict) -> dict[str, float]:
    d = {f"d_{k}": f2[k] - f1[k] for k in f1 if k not in ("width", "n_args")}
    for k in RATIO_KEYS:
        d[f"d_ratio_{k}"] = (f2[k] + 1.0) / (f1[k] + 1.0)
    h1, h2 = node_histogram(t1), node_histogram(t2)
    keys = set(h1) | set(h2)
    d["d_hist_l1"] = float(sum(abs(h1[k] - h2[k]) for k in keys))
    s1 = [type(n).__name__ for n in ast.walk(t1)]
    s2 = [type(n).__name__ for n in ast.walk(t2)]
    d["d_seq_sim"] = difflib.SequenceMatcher(None, s1, s2, autojunk=False).ratio()
    d["d_new_mul_vv"] = max(0.0, f2["dyn_mul_vv"] - f1["dyn_mul_vv"])
    return d


def transform_features(applied: list[str], info: dict) -> dict[str, float]:
    t = {f"t_{n}": float(applied.count(n)) for n in TRANSFORM_NAMES}
    t["t_n_applied"] = float(len(applied))
    t["t_unroll_factor"] = float(info.get("unroll_factor", 0))
    return t


def extract_all(src1: str, src2: str, width: int, applied: list[str], info: dict,
                probe_ms: int = 20, with_dynamic: bool = True
                ) -> tuple[dict[str, float], dict[str, float]]:
    """Return (features, timings_ms). Timings are per *cumulative* group."""
    tim: dict[str, float] = {}
    t0 = time.perf_counter()
    tr1 = ast.parse(src1)
    f1 = program_features(tr1, width)
    tim["t_p1_ms"] = _ms(t0)

    t0 = time.perf_counter()
    tr2 = ast.parse(src2)
    f2 = program_features(tr2, width)
    d = delta_features(tr1, tr2, f1, f2)
    t = transform_features(applied, info)
    tim["t_delta_ms"] = _ms(t0)  # p2 + delta + transform metadata

    t0 = time.perf_counter()
    miter = build_miter(src1, src2, width)
    pf = static_probe(miter)
    tim["t_probe_static_ms"] = _ms(t0)

    t0 = time.perf_counter()
    pd_ = dynamic_probe(miter, probe_ms) if with_dynamic else {}
    tim["t_probe_dyn_ms"] = _ms(t0)

    feats: dict[str, float] = {}
    feats.update({f"p1_{k}": v for k, v in f1.items()})
    feats.update({f"p2_{k}": v for k, v in f2.items()})
    feats.update(d)
    feats.update(t)
    feats.update(pf)
    feats.update(pd_)
    return feats, tim


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    g = {"p1": [], "p2": [], "delta": [], "transform": [], "probe_static": [], "probe_dyn": []}
    for c in columns:
        if c.startswith("p1_"): g["p1"].append(c)
        elif c.startswith("p2_"): g["p2"].append(c)
        elif c.startswith("d_"): g["delta"].append(c)
        elif c.startswith("t_") and not c.endswith("_ms"): g["transform"].append(c)
        elif c.startswith("pf_"): g["probe_static"].append(c)
        elif c.startswith("pd_"): g["probe_dyn"].append(c)
    return g
