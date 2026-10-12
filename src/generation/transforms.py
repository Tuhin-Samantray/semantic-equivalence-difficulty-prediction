"""Semantics-preserving, compiler-style AST transformations.

Every transformation takes ``(tree, rng, width)``, mutates ``tree`` in place
and returns an info dict if it changed something, or ``None`` if it was not
applicable. All of them preserve fixed-width two's complement semantics (this
is property-tested in ``tests/test_transforms.py``).
"""
from __future__ import annotations

import ast
import copy
import random
from typing import Callable, Optional

from src.verification.semantics import const_int

Info = Optional[dict]

COMMUTATIVE = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
NEGATE = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt,
          ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


# ---------------------------------------------------------------- helpers
def _all(tree: ast.AST, kind) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, kind)]


def _is_const(n: ast.expr) -> bool:
    return isinstance(n, ast.Constant) or (
        isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and _is_const(n.operand))


def _vars(tree: ast.AST) -> list[str]:
    fn = tree.body[0]
    names = [a.arg for a in fn.args.args]
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in names:
            names.append(n.id)
    return names


def _bodies(tree: ast.AST) -> list[list[ast.stmt]]:
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Module):
            continue
        for attr in ("body", "orelse"):
            b = getattr(n, attr, None)
            if isinstance(b, list) and b and isinstance(b[0], ast.stmt):
                out.append(b)
    return out


class _Subst(ast.NodeTransformer):
    def __init__(self, name: str, repl_factory: Callable[[], ast.expr]):
        self.name, self.repl = name, repl_factory

    def visit_Name(self, node: ast.Name):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return self.repl()
        return node


# ---------------------------------------------------------- transformations
def rename(tree, rng: random.Random, width: int) -> Info:
    """Consistent alpha-renaming of arguments and locals."""
    names = _vars(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.For):
            names.append(n.target.id)
    names = list(dict.fromkeys(names))
    mapping = {n: f"v{rng.randint(0, 999)}_{k}" for k, n in enumerate(names)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in mapping:
            n.id = mapping[n.id]
        elif isinstance(n, ast.arg) and n.arg in mapping:
            n.arg = mapping[n.arg]
    return {}


def commute(tree, rng, width) -> Info:
    cands = [n for n in _all(tree, ast.BinOp) if isinstance(n.op, COMMUTATIVE)]
    if not cands:
        return None
    for n in rng.sample(cands, k=max(1, len(cands) // 2)):
        n.left, n.right = n.right, n.left
    return {}


def dead_code(tree, rng, width) -> Info:
    """Insert unused assignments and always-false branches."""
    names = [a.arg for a in tree.body[0].args.args]  # args are always in scope
    bodies = _bodies(tree)
    for k in range(rng.randint(1, 3)):
        body = rng.choice(bodies)
        pos = rng.randint(0, len(body) - 1 if isinstance(body[-1], ast.Return) else len(body))
        v1, v2 = rng.choice(names), rng.choice(names)
        e = ast.BinOp(ast.Name(v1, ast.Load()), rng.choice([ast.Add(), ast.Mult(), ast.BitXor()]),
                      ast.Name(v2, ast.Load()))
        if rng.random() < 0.5:
            stmt: ast.stmt = ast.Assign([ast.Name(f"_dead{k}", ast.Store())], e)
        else:  # (v - v) != 0 is always false
            cond = ast.Compare(ast.BinOp(ast.Name(v1, ast.Load()), ast.Sub(), ast.Name(v1, ast.Load())),
                               [ast.NotEq()], [ast.Constant(0)])
            tgt = rng.choice(names)
            stmt = ast.If(cond, [ast.Assign([ast.Name(tgt, ast.Store())], e)], [])
        body.insert(pos, stmt)
    return {}


def const_fold(tree, rng, width) -> Info:
    mask = (1 << width) - 1
    changed = False

    class F(ast.NodeTransformer):
        def visit_BinOp(self, node):
            nonlocal changed
            self.generic_visit(node)
            l, r = node.left, node.right
            if isinstance(l, ast.Constant) and isinstance(r, ast.Constant):
                ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                       ast.Mult: lambda a, b: a * b, ast.BitAnd: lambda a, b: a & b,
                       ast.BitOr: lambda a, b: a | b, ast.BitXor: lambda a, b: a ^ b}
                f = ops.get(type(node.op))
                if f:
                    changed = True
                    return ast.Constant(f(l.value, r.value) & mask)
            if isinstance(node.op, (ast.Add, ast.Sub)) and isinstance(r, ast.Constant) and r.value == 0:
                changed = True
                return l
            if isinstance(node.op, ast.Mult) and isinstance(r, ast.Constant) and r.value == 1:
                changed = True
                return l
            return node

    F().visit(tree)
    return {} if changed else None


def unroll(tree, rng, width) -> Info:
    """Partial (factor k with remainder) or full loop unrolling."""
    loops = [n for n in _all(tree, ast.For) if const_int(n.iter.args[0]) >= 2]
    if not loops:
        return None
    loop = rng.choice(loops)
    n = const_int(loop.iter.args[0])
    k = rng.choice([2, 3, 4, n]) if n <= 6 else rng.choice([2, 3, 4])
    k = min(k, n)
    var = loop.target.id

    def copies(expr_factory_for_iter, count):
        out: list[ast.stmt] = []
        for t in range(count):
            for st in loop.body:
                st2 = copy.deepcopy(st)
                st2 = _Subst(var, expr_factory_for_iter(t)).visit(st2)
                out.append(st2)
        return out

    new: list[ast.stmt] = []
    if k == n:  # full unroll
        new = copies(lambda t: (lambda: ast.Constant(t)), n)
    else:
        main_trips, rem = divmod(n, k)
        body = copies(lambda t: (lambda: ast.BinOp(
            ast.BinOp(ast.Name(var, ast.Load()), ast.Mult(), ast.Constant(k)),
            ast.Add(), ast.Constant(t))), k)
        new.append(ast.For(ast.Name(var, ast.Store()),
                           ast.Call(ast.Name("range", ast.Load()), [ast.Constant(main_trips)], []),
                           body, []))
        new += copies(lambda t: (lambda: ast.Constant(main_trips * k + t)), rem)
    for body in _bodies(tree):
        for i, st in enumerate(body):
            if st is loop:
                body[i:i + 1] = new
                return {"unroll_factor": k, "full": k == n}
    return None


def strength_reduce(tree, rng, width) -> Info:
    cands = []
    for n in _all(tree, ast.BinOp):
        if isinstance(n.op, ast.Mult):
            for cst, other in ((n.right, n.left), (n.left, n.right)):
                if isinstance(cst, ast.Constant) and isinstance(cst.value, int) and cst.value >= 2:
                    c = cst.value
                    if c & (c - 1) == 0 or (c - 1) & (c - 2) == 0 and c > 2 or (c + 1) & c == 0:
                        cands.append((n, other, c))
                    break
    if not cands:
        return None
    n, other, c = rng.choice(cands)
    if c & (c - 1) == 0:
        repl: ast.expr = ast.BinOp(other, ast.LShift(), ast.Constant(c.bit_length() - 1))
    elif (c - 1) & (c - 2) == 0:  # 2^k + 1
        repl = ast.BinOp(ast.BinOp(copy.deepcopy(other), ast.LShift(), ast.Constant((c - 1).bit_length() - 1)),
                         ast.Add(), other)
    else:  # 2^k - 1
        repl = ast.BinOp(ast.BinOp(copy.deepcopy(other), ast.LShift(), ast.Constant((c + 1).bit_length() - 1)),
                         ast.Sub(), other)
    n.__class__, n.__dict__ = repl.__class__, repl.__dict__
    return {}


def distribute(tree, rng, width) -> Info:
    """a*(b+c) -> a*b + a*c  (adds variable multipliers: hard for bit-blasting)."""
    cands = []
    for n in _all(tree, ast.BinOp):
        if isinstance(n.op, ast.Mult):
            if isinstance(n.right, ast.BinOp) and isinstance(n.right.op, (ast.Add, ast.Sub)):
                cands.append((n, n.left, n.right))
            elif isinstance(n.left, ast.BinOp) and isinstance(n.left.op, (ast.Add, ast.Sub)):
                cands.append((n, n.right, n.left))
    if not cands:
        return None
    n, a, s = rng.choice(cands)
    new = ast.BinOp(ast.BinOp(copy.deepcopy(a), ast.Mult(), s.left), s.op,
                    ast.BinOp(copy.deepcopy(a), ast.Mult(), s.right))
    n.__class__, n.__dict__ = new.__class__, new.__dict__
    return {}


def reassociate(tree, rng, width) -> Info:
    """(a op b) op c -> a op (b op c) for + , * , ^ , & , |."""
    ok = (ast.Add, ast.Mult, ast.BitXor, ast.BitAnd, ast.BitOr)
    cands = [n for n in _all(tree, ast.BinOp)
             if isinstance(n.op, ok) and isinstance(n.left, ast.BinOp)
             and type(n.left.op) is type(n.op)]
    if not cands:
        return None
    n = rng.choice(cands)
    a, b, c = n.left.left, n.left.right, n.right
    new = ast.BinOp(a, n.op, ast.BinOp(b, n.op, c))
    n.__class__, n.__dict__ = new.__class__, new.__dict__
    return {}


def extract_temp(tree, rng, width) -> Info:
    """Hoist a sub-expression into a fresh temporary (compiler CSE/ANF-style)."""
    sites = []
    for body in _bodies(tree):
        for i, st in enumerate(body):
            if isinstance(st, ast.Assign):
                subs = [n for n in ast.walk(st.value) if isinstance(n, ast.BinOp)]
                if subs:
                    sites.append((body, i, st, subs))
    if not sites:
        return None
    body, i, st, subs = rng.choice(sites)
    target = rng.choice(subs)
    tmp = f"_t{rng.randint(0, 9999)}"
    saved = copy.deepcopy(target)
    new = ast.Name(tmp, ast.Load())
    target.__class__, target.__dict__ = new.__class__, new.__dict__
    body.insert(i, ast.Assign([ast.Name(tmp, ast.Store())], saved))
    return {}


def if_flip(tree, rng, width) -> Info:
    cands = [n for n in _all(tree, ast.If) if n.orelse and type(n.test.ops[0]) in NEGATE]
    if not cands:
        return None
    n = rng.choice(cands)
    n.test.ops = [NEGATE[type(n.test.ops[0])]()]
    n.body, n.orelse = n.orelse, n.body
    return {}


REGISTRY: dict[str, Callable] = {
    "rename": rename, "commute": commute, "dead_code": dead_code,
    "const_fold": const_fold, "unroll": unroll, "strength_reduce": strength_reduce,
    "distribute": distribute, "reassociate": reassociate,
    "extract_temp": extract_temp, "if_flip": if_flip,
}
WEIGHTS = {"rename": 1.0, "commute": 1.0, "dead_code": 1.0, "const_fold": 1.0,
           "unroll": 1.6, "strength_reduce": 1.0, "distribute": 1.6,
           "reassociate": 1.6, "extract_temp": 1.0, "if_flip": 0.7}
TRANSFORM_NAMES = list(REGISTRY)


def apply_pipeline(src: str, width: int, rng: random.Random, n_steps: int,
                   max_attempts: int = 30) -> tuple[str, list[str], dict]:
    """Apply ``n_steps`` applicable transformations; returns (src2, names, info).

    ``info`` carries per-transformation metadata (e.g. the unroll factor).
    The result is guaranteed to differ structurally from ``src``.
    """
    tree = ast.parse(src)
    applied: list[str] = []
    info: dict = {"unroll_factor": 0}
    names = list(REGISTRY)
    weights = [WEIGHTS[n] for n in names]
    for _ in range(max_attempts):
        if len(applied) >= n_steps:
            break
        name = rng.choices(names, weights)[0]
        trial = copy.deepcopy(tree)
        res = REGISTRY[name](trial, rng, width)
        if res is None:
            continue
        ast.fix_missing_locations(trial)
        tree = trial
        applied.append(name)
        if "unroll_factor" in res:
            info["unroll_factor"] = max(info["unroll_factor"], res["unroll_factor"])
    out = ast.unparse(tree)
    if not applied or ast.dump(ast.parse(out)) == ast.dump(ast.parse(src)):
        raise ValueError("pipeline produced no structural change")
    return out, applied, info
