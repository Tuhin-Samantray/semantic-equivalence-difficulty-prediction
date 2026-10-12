"""One interpreter, two algebras: concrete fixed-width ints and Z3 bit-vectors.

Sharing a single evaluator guarantees that the concrete oracle used in tests
and the symbolic semantics used for verification can never drift apart.

Semantics: two's complement of ``width`` bits, signed comparisons, arithmetic
right shift, and ``if`` executes both branches and merges with ``ite`` (only
variables that existed before the branch are merged; branch-local temporaries
are dropped).
"""
from __future__ import annotations

import ast
from typing import Any

import z3


class ConcreteAlg:
    def __init__(self, width: int):
        self.w = width
        self.mask = (1 << width) - 1
        self.half = 1 << (width - 1)

    def wrap(self, x: int) -> int:
        return ((x + self.half) & self.mask) - self.half

    def const(self, v: int): return self.wrap(v)
    def arg(self, name: str, v: int): return self.wrap(v)
    def add(self, a, b): return self.wrap(a + b)
    def sub(self, a, b): return self.wrap(a - b)
    def mul(self, a, b): return self.wrap(a * b)
    def and_(self, a, b): return self.wrap(a & b)
    def or_(self, a, b): return self.wrap(a | b)
    def xor(self, a, b): return self.wrap(a ^ b)
    def shl(self, a, k: int): return self.wrap(a << k) if k < self.w else 0
    def shr(self, a, k: int): return a >> min(k, self.w)
    def neg(self, a): return self.wrap(-a)
    def inv(self, a): return self.wrap(~a)

    def cmp(self, op, a, b):
        return {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b,
                ast.Eq: a == b, ast.NotEq: a != b}[op]

    def ite(self, c, a, b): return a if c else b


class Z3Alg:
    def __init__(self, width: int, ctx: z3.Context):
        self.w, self.ctx = width, ctx

    def const(self, v: int): return z3.BitVecVal(v, self.w, self.ctx)
    def arg(self, name: str, v=None): return z3.BitVec(name, self.w, self.ctx)
    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b
    def mul(self, a, b): return a * b
    def and_(self, a, b): return a & b
    def or_(self, a, b): return a | b
    def xor(self, a, b): return a ^ b
    def shl(self, a, k: int): return a << self.const(k)
    def shr(self, a, k: int): return a >> self.const(k)  # arithmetic shift
    def neg(self, a): return -a
    def inv(self, a): return ~a

    def cmp(self, op, a, b):  # signed comparisons
        return {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b,
                ast.Eq: a == b, ast.NotEq: a != b}[op]

    def ite(self, c, a, b): return z3.If(c, a, b)


class _Return(Exception):
    def __init__(self, v):
        self.v = v


def const_int(node: ast.expr) -> int:
    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -const_int(node.operand)
    raise ValueError("shift amount / range bound must be constant")


def _expr(n: ast.expr, env: dict, alg: Any):
    if isinstance(n, ast.Constant):
        return alg.const(int(n.value))
    if isinstance(n, ast.Name):
        return env[n.id]
    if isinstance(n, ast.UnaryOp):
        v = _expr(n.operand, env, alg)
        if isinstance(n.op, ast.USub):
            return alg.neg(v)
        if isinstance(n.op, ast.Invert):
            return alg.inv(v)
        raise ValueError(f"unsupported unary {n.op}")
    if isinstance(n, ast.BinOp):
        if isinstance(n.op, (ast.LShift, ast.RShift)):
            a, k = _expr(n.left, env, alg), const_int(n.right)
            return alg.shl(a, k) if isinstance(n.op, ast.LShift) else alg.shr(a, k)
        a, b = _expr(n.left, env, alg), _expr(n.right, env, alg)
        f = {ast.Add: alg.add, ast.Sub: alg.sub, ast.Mult: alg.mul,
             ast.BitAnd: alg.and_, ast.BitOr: alg.or_, ast.BitXor: alg.xor}.get(type(n.op))
        if f is None:
            raise ValueError(f"unsupported binop {n.op}")
        return f(a, b)
    if isinstance(n, ast.Compare):
        return alg.cmp(type(n.ops[0]), _expr(n.left, env, alg),
                       _expr(n.comparators[0], env, alg))
    raise ValueError(f"unsupported expression {ast.dump(n)}")


def _exec(stmts: list[ast.stmt], env: dict, alg: Any) -> None:
    for s in stmts:
        if isinstance(s, ast.Assign):
            env[s.targets[0].id] = _expr(s.value, env, alg)
        elif isinstance(s, ast.For):
            for it in range(const_int(s.iter.args[0])):
                env[s.target.id] = alg.const(it)
                _exec(s.body, env, alg)
        elif isinstance(s, ast.If):
            c = _expr(s.test, env, alg)
            e1, e2 = dict(env), dict(env)
            _exec(s.body, e1, alg)
            _exec(s.orelse, e2, alg)
            for k in env:
                env[k] = alg.ite(c, e1[k], e2[k])
        elif isinstance(s, ast.Return):
            raise _Return(_expr(s.value, env, alg))
        else:
            raise ValueError(f"unsupported statement {type(s).__name__}")


def run(tree: ast.Module, args: list, alg: Any):
    fn = tree.body[0]
    env = {a.arg: v for a, v in zip(fn.args.args, args)}
    try:
        _exec(fn.body, env, alg)
    except _Return as r:
        return r.v
    raise ValueError("function has no return")


def eval_concrete(src: str, width: int, args: list[int]) -> int:
    """Run ``src`` on concrete integer ``args`` under ``width``-bit semantics."""
    alg = ConcreteAlg(width)
    return run(ast.parse(src), [alg.wrap(a) for a in args], alg)
