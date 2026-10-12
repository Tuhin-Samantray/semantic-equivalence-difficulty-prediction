"""Tests for the Z3 equivalence checker (src/verification/equivalence.py)."""
from __future__ import annotations

import random

import pytest
import z3

from src.generation.programs import generate_program
from src.generation.transforms import apply_pipeline
from src.verification.equivalence import VerifyResult, build_miter, verify_pair
from src.verification.semantics import eval_concrete

MUL2 = "def f(a, b):\n    return a * 2 + b\n"
SHL1 = "def f(a, b):\n    return (a << 1) + b\n"
ADD = "def f(a, b):\n    return a + b\n"
SUB = "def f(a, b):\n    return a - b\n"
ID = "def f(a):\n    r = a\n    return r\n"
QUIRK = "def f(a):\n    r = a\n    if a == 5:\n        r = 0\n    return r\n"
FACTORED = "def f(a, b, c, d):\n    return (a + b) * (c + d)\n"
EXPANDED = "def f(a, b, c, d):\n    return a * c + a * d + b * c + b * d\n"


# Real timeout pair taken from the dataset (24-bit, multiplication-heavy, strength_reduce: b * 2 -> b << 1).
TIMEOUT_SRC1 = """def f(a, b, c, d):
    x0 = (11 + d) * (d * d)
    x1 = c * 3 * a
    x2 = d * (d * b)
    x3 = x0 * (c * 100)
    if a < 5 | 7:
        x1 = x1 + c * a * (b * 11)
        x0 = x0 + 9
    x0 = x0 + a * (c * x1)
    if x1 | x2 == x2:
        x2 = 1
        x0 = x0 - 13
    for i0 in range(6):
        for i1 in range(4):
            x2 = x2 + 3 * x0 * (b * x3)
            x3 = x3 - b
        if 11 * x2 == 13 * b:
            for i1 in range(3):
                x1 = (x2 & x1) + b * 2
                x2 = x2 + 13 * 13 * (c + x1)
        else:
            x2 = x3 * x1 * (b * 100)
    return (3 - x1 ^ x2) + x3
"""
TIMEOUT_SRC2 = TIMEOUT_SRC1.replace("(x2 & x1) + b * 2", "(x2 & x1) + (b << 1)")


def _counterexample(src1: str, src2: str, width: int) -> list[int]:
    """Ask Z3 for an input on which the two programs differ (signed values)."""
    m = build_miter(src1, src2, width)
    s = z3.Solver(ctx=m.ctx)
    s.add(m.formula)
    assert s.check() == z3.sat
    model = s.model()
    return [model.eval(a, model_completion=True).as_signed_long() for a in m.args]


def test_equivalent_pair_is_proved() -> None:
    res = verify_pair(MUL2, SHL1, width=8)
    assert res.outcome == "equivalent"


def test_different_pair_is_detected_with_valid_counterexample() -> None:
    res = verify_pair(ADD, SUB, width=8)
    assert res.outcome == "different"
    args = _counterexample(ADD, SUB, 8)
    assert eval_concrete(ADD, 8, args) != eval_concrete(SUB, 8, args)


def test_single_point_difference_is_found() -> None:
    assert verify_pair(ID, QUIRK, width=8).outcome == "different"
    args = _counterexample(ID, QUIRK, 8)
    assert args == [5]
    assert eval_concrete(ID, 8, [5]) == 5
    assert eval_concrete(QUIRK, 8, [5]) == 0


def test_miter_has_one_symbolic_input_per_argument() -> None:
    assert len(build_miter(ADD, SUB, 8).args) == 2
    assert len(build_miter(FACTORED, EXPANDED, 8).args) == 4


def test_timing_fields_are_consistent() -> None:
    res = verify_pair(MUL2, SHL1, width=8)
    assert res.build_ms > 0
    assert res.solve_ms >= 0
    assert res.total_ms == pytest.approx(res.build_ms + res.solve_ms)


def test_result_as_dict_has_all_fields() -> None:
    res = verify_pair(MUL2, SHL1, width=8)
    assert isinstance(res, VerifyResult)
    assert set(res.as_dict()) == {"outcome", "build_ms", "solve_ms", "total_ms", "conflicts", "memory_mb"}


def test_timeout_is_censored_at_the_cap() -> None:
    assert TIMEOUT_SRC1 != TIMEOUT_SRC2
    res = verify_pair(TIMEOUT_SRC1, TIMEOUT_SRC2, width=24, timeout_ms=200)
    assert res.outcome == "timeout"
    assert res.solve_ms >= 200.0
    assert res.total_ms == pytest.approx(res.build_ms + res.solve_ms)


def test_pipeline_pairs_are_never_reported_different() -> None:
    checked = 0
    for seed in range(10):
        prog = generate_program(seed)
        rng = random.Random(seed)
        try:
            p2, _names, _info = apply_pipeline(prog.src, prog.width, rng, rng.randint(1, 3))
        except ValueError:
            continue
        res = verify_pair(prog.src, p2, prog.width, timeout_ms=1000)
        assert res.outcome in ("equivalent", "timeout"), f"seed {seed}: {res.outcome}"
        checked += 1
    assert checked >= 3
