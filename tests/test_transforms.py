"""Property tests: every transformation must preserve program semantics."""
from __future__ import annotations

import ast
import random

import pytest

from src.generation.programs import generate_program
from src.generation.transforms import REGISTRY, TRANSFORM_NAMES, apply_pipeline
from src.verification.semantics import eval_concrete


def _inputs(width: int, n_args: int, rng: random.Random, n_random: int = 25) -> list[list[int]]:
    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    edge = [0, 1, -1, lo, hi]
    rows = [[rng.choice(edge) for _ in range(n_args)] for _ in range(5)]
    rows += [[rng.randint(lo, hi) for _ in range(n_args)] for _ in range(n_random)]
    return rows


def _assert_same(p1: str, p2: str, width: int, n_args: int, rng: random.Random) -> None:
    for args in _inputs(width, n_args, rng):
        r1 = eval_concrete(p1, width, args)
        r2 = eval_concrete(p2, width, args)
        assert r1 == r2, f"mismatch on {args}: {r1} != {r2}\n--- P1 ---\n{p1}\n--- P2 ---\n{p2}"


def test_generator_is_deterministic() -> None:
    assert generate_program(7).src == generate_program(7).src
    assert generate_program(7).src != generate_program(8).src


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_single_transform_preserves_semantics(name: str) -> None:
    applied = 0
    for seed in range(150):
        prog = generate_program(seed)
        rng = random.Random(seed)
        tree = ast.parse(prog.src)
        res = REGISTRY[name](tree, rng, prog.width)
        if res is None:
            continue
        ast.fix_missing_locations(tree)
        applied += 1
        _assert_same(prog.src, ast.unparse(tree), prog.width, prog.n_args, rng)
    assert applied > 0, f"{name} never applied on 150 programs"
    print(f"{name}: applied on {applied}/150 programs")


def test_pipeline_preserves_semantics() -> None:
    produced = 0
    for seed in range(200):
        prog = generate_program(seed)
        rng = random.Random(seed)
        try:
            p2, names, _info = apply_pipeline(prog.src, prog.width, rng, rng.randint(1, 3))
        except ValueError:
            continue
        produced += 1
        assert names
        _assert_same(prog.src, p2, prog.width, prog.n_args, rng)
    print(f"pipeline pairs produced: {produced}/200")
    assert produced >= 50
