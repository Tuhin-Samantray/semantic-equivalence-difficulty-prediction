"""Random base-program generator.

Programs are small integer functions ``f(a, b, ...)`` built from assignments,
constant-trip ``for i in range(N)`` loops, ``if/else`` and arithmetic/bitwise
expressions. All arithmetic is fixed-width two's complement (see
``src.verification.semantics``), so every program is total and loop-bounded,
which makes equivalence checking complete (not merely bounded).
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field

ARG_NAMES = ["a", "b", "c", "d"]
CMP_OPS = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]


@dataclass(frozen=True)
class Program:
    src: str
    width: int
    seed: int
    n_args: int


@dataclass
class GenConfig:
    n_args: tuple[int, int] = (2, 4)
    n_locals: tuple[int, int] = (2, 4)
    n_top_stmts: tuple[int, int] = (2, 5)
    max_loop_depth: int = 2
    loop_trip: tuple[int, int] = (2, 6)
    max_total_iters: int = 24
    expr_depth: int = 2
    widths: tuple[int, ...] = (8, 12, 16, 24, 32)


class _Gen:
    def __init__(self, rng: random.Random, cfg: GenConfig):
        self.rng = rng
        self.cfg = cfg
        # per-program "personality" => wide spread of difficulty across programs
        self.mul_p = rng.uniform(0.05, 0.55)
        self.loop_p = rng.uniform(0.2, 0.7)
        self.if_p = rng.uniform(0.1, 0.5)

    # ---- expressions -------------------------------------------------
    def leaf(self, names: list[str]) -> ast.expr:
        if self.rng.random() < 0.75 and names:
            return ast.Name(self.rng.choice(names), ast.Load())
        return ast.Constant(self.rng.choice([1, 2, 3, 3, 5, 7, 9, 11, 13, 100]))

    def expr(self, names: list[str], depth: int) -> ast.expr:
        if depth == 0 or self.rng.random() < 0.25:
            return self.leaf(names)
        r = self.rng.random()
        if r < self.mul_p:
            op: ast.operator = ast.Mult()
        elif r < self.mul_p + 0.12:
            k = ast.Constant(self.rng.randint(1, 3))
            return ast.BinOp(self.expr(names, depth - 1),
                             self.rng.choice([ast.LShift(), ast.RShift()]), k)
        else:
            op = self.rng.choice([ast.Add(), ast.Add(), ast.Sub(),
                                  ast.BitXor(), ast.BitAnd(), ast.BitOr()])
        return ast.BinOp(self.expr(names, depth - 1), op, self.expr(names, depth - 1))

    def cond(self, names: list[str]) -> ast.expr:
        return ast.Compare(self.expr(names, 1), [self.rng.choice(CMP_OPS)()],
                           [self.expr(names, 1)])

    # ---- statements --------------------------------------------------
    def assign_existing(self, names: list[str], targets: list[str]) -> ast.stmt:
        t = self.rng.choice(targets)
        e = self.expr(names, self.cfg.expr_depth)
        if self.rng.random() < 0.6:
            e = ast.BinOp(ast.Name(t, ast.Load()),
                          self.rng.choice([ast.Add(), ast.Add(), ast.BitXor(), ast.Sub()]), e)
        return ast.Assign([ast.Name(t, ast.Store())], e)

    def block(self, names: list[str], targets: list[str], n: int,
              loop_depth: int, iters_left: int, allow_if: bool = True) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for _ in range(n):
            r = self.rng.random()
            if (r < self.loop_p and loop_depth < self.cfg.max_loop_depth
                    and iters_left >= self.cfg.loop_trip[0]):
                trip = self.rng.randint(self.cfg.loop_trip[0],
                                        min(self.cfg.loop_trip[1], iters_left))
                var = f"i{loop_depth}"
                body = self.block(names + [var], targets, self.rng.randint(1, 2),
                                  loop_depth + 1, iters_left // trip, allow_if)
                out.append(ast.For(ast.Name(var, ast.Store()),
                                   ast.Call(ast.Name("range", ast.Load()),
                                            [ast.Constant(trip)], []), body, []))
            elif r < self.loop_p + self.if_p and allow_if:
                body = self.block(names, targets, self.rng.randint(1, 2),
                                  loop_depth, iters_left, False)
                orelse = (self.block(names, targets, self.rng.randint(1, 2),
                                     loop_depth, iters_left, False)
                          if self.rng.random() < 0.7 else [])
                out.append(ast.If(self.cond(names), body, orelse))
            else:
                out.append(self.assign_existing(names, targets))
        return out

    def program(self, n_args: int) -> ast.Module:
        args = ARG_NAMES[:n_args]
        n_loc = self.rng.randint(*self.cfg.n_locals)
        body: list[ast.stmt] = []
        names = list(args)
        locs: list[str] = []
        for k in range(n_loc):
            v = f"x{k}"
            body.append(ast.Assign([ast.Name(v, ast.Store())], self.expr(names, 2)))
            names.append(v)
            locs.append(v)
        body += self.block(names, locs, self.rng.randint(*self.cfg.n_top_stmts),
                           0, self.cfg.max_total_iters)
        ret = self.leaf(locs)
        for v in locs[1:]:
            ret = ast.BinOp(ret, self.rng.choice([ast.Add(), ast.BitXor(), ast.Sub()]),
                            ast.Name(v, ast.Load()))
        body.append(ast.Return(ret))
        fn = ast.FunctionDef("f", ast.arguments([], [ast.arg(a) for a in args], None,
                                                [], [], None, []), body, [], None)
        return ast.fix_missing_locations(ast.Module([fn], []))


def generate_program(seed: int, cfg: GenConfig | None = None) -> Program:
    """Deterministically generate one base program from ``seed``."""
    cfg = cfg or GenConfig()
    rng = random.Random(seed)
    n_args = rng.randint(*cfg.n_args)
    mod = _Gen(rng, cfg).program(n_args)
    return Program(ast.unparse(mod), rng.choice(cfg.widths), seed, n_args)
