"""Per-program structural features from the AST (+ CFG).

``dyn_*`` features weight each node by the product of enclosing loop trip
counts, i.e. they estimate the size of the fully unrolled symbolic program,
which is what the SMT solver actually sees.
"""
from __future__ import annotations

import ast
from collections import Counter

from src.cfg_builder.cfg import cfg_stats
from src.verification.semantics import const_int

_OPS = {ast.Add: "add", ast.Sub: "sub", ast.BitAnd: "and", ast.BitOr: "or",
        ast.BitXor: "xor", ast.LShift: "shl", ast.RShift: "shr"}


def _is_const(n: ast.expr) -> bool:
    return isinstance(n, ast.Constant) or (
        isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and _is_const(n.operand))


def _expr_depth(n: ast.AST) -> int:
    kids = [c for c in ast.iter_child_nodes(n) if isinstance(c, ast.expr)]
    return 1 + max((_expr_depth(c) for c in kids), default=0)


def node_histogram(tree: ast.AST) -> Counter:
    return Counter(type(n).__name__ for n in ast.walk(tree))


def program_features(tree: ast.Module, width: int) -> dict[str, float]:
    f: Counter = Counter()
    fn = tree.body[0]
    f["width"] = width
    f["n_args"] = len(fn.args.args)
    assigned: set[str] = set()
    max_expr_depth = 0

    def visit_expr(e: ast.expr, mult: int) -> None:
        nonlocal max_expr_depth
        max_expr_depth = max(max_expr_depth, _expr_depth(e))
        for n in ast.walk(e):
            if isinstance(n, ast.BinOp):
                if isinstance(n.op, ast.Mult):
                    kind = "mul_vc" if (_is_const(n.left) or _is_const(n.right)) else "mul_vv"
                else:
                    kind = _OPS.get(type(n.op), "other")
                f[f"n_{kind}"] += 1
                f[f"dyn_{kind}"] += mult
                f["dyn_arith"] += mult
            elif isinstance(n, ast.Constant):
                f["n_const"] += 1
            elif isinstance(n, ast.Name):
                f["n_name_refs"] += 1
            elif isinstance(n, ast.Compare):
                f["n_cmp"] += 1

    def visit_block(stmts: list[ast.stmt], mult: int, depth: int) -> None:
        f["loop_depth_max"] = max(f["loop_depth_max"], depth)
        for s in stmts:
            f["n_stmts"] += 1
            f["dyn_stmts"] += mult
            if isinstance(s, ast.Assign):
                f["n_assign"] += 1
                assigned.add(s.targets[0].id)
                visit_expr(s.value, mult)
            elif isinstance(s, ast.For):
                trip = const_int(s.iter.args[0])
                f["n_for"] += 1
                f["dyn_loop_iters"] += mult * trip
                f["max_trip"] = max(f["max_trip"], trip)
                visit_block(s.body, mult * trip, depth + 1)
            elif isinstance(s, ast.If):
                f["n_if"] += 1
                f["dyn_if"] += mult
                f["n_else"] += 1 if s.orelse else 0
                visit_expr(s.test, mult)
                visit_block(s.body, mult, depth)
                visit_block(s.orelse, mult, depth)
            elif isinstance(s, ast.Return):
                visit_expr(s.value, mult)

    visit_block(fn.body, 1, 0)
    f["n_nodes"] = sum(1 for _ in ast.walk(tree))
    f["n_locals"] = len(assigned)
    f["expr_depth_max"] = max_expr_depth
    out = {k: float(v) for k, v in f.items()}
    for k in ["n_mul_vv", "n_mul_vc", "dyn_mul_vv", "dyn_mul_vc", "n_add", "n_sub", "n_and",
              "n_or", "n_xor", "n_shl", "n_shr", "dyn_add", "dyn_sub", "dyn_and", "dyn_or",
              "dyn_xor", "dyn_shl", "dyn_shr", "n_for", "n_if", "n_else", "n_cmp", "n_const",
              "dyn_if", "dyn_loop_iters", "max_trip", "loop_depth_max", "dyn_arith",
              "dyn_stmts", "n_assign"]:
        out.setdefault(k, 0.0)
    out.update(cfg_stats(tree))
    return out
