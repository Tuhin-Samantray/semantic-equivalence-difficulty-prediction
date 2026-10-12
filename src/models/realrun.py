"""Helpers for the real wall-clock scheduling run (Phase 4c).

Lean feature extraction: only what each fitted model needs, nothing more,
so the measured overhead is what a deployed predictor would really pay.
"""
from __future__ import annotations

import ast
import time

import numpy as np
import pandas as pd

from src.ast_extractor.features import program_features
from src.feature_extraction.builder import delta_features, transform_features
from src.models.train import Fitted, predict


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def parse_transforms(row) -> tuple[list[str], dict]:
    s = row["transforms"]
    applied = [] if (not isinstance(s, str) or s == "") else s.split("|")
    return applied, {"unroll_factor": row["t_unroll_factor"]}


def features_p1(src1: str, width: int) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    f1 = program_features(ast.parse(src1), width)
    return {f"p1_{k}": v for k, v in f1.items()}, _ms(t0)


def features_p1_delta_transform(src1: str, src2: str, width: int, applied: list[str],
                                info: dict) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    tr1 = ast.parse(src1)
    f1 = program_features(tr1, width)
    tr2 = ast.parse(src2)
    f2 = program_features(tr2, width)
    feats = {f"p1_{k}": v for k, v in f1.items()}
    feats.update(delta_features(tr1, tr2, f1, f2))
    feats.update(transform_features(applied, info))
    return feats, _ms(t0)


def extract_for_model(fitted: Fitted, row) -> tuple[dict[str, float], float]:
    """Lean extraction for one pair; returns (features, extraction_ms)."""
    if fitted.key == "A":
        return features_p1(row["src1"], int(row["width"]))
    if fitted.key == "D":
        applied, info = parse_transforms(row)
        return features_p1_delta_transform(row["src1"], row["src2"], int(row["width"]), applied, info)
    raise ValueError(f"lean extractor not implemented for set {fitted.key}")


def score_batch(fitted: Fitted, batch: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Extract + predict for a whole batch, sequentially. Returns (scores, total_ms)."""
    t0 = time.perf_counter()
    feats = [extract_for_model(fitted, r)[0] for _, r in batch.iterrows()]
    scores = predict(fitted, pd.DataFrame(feats))
    return scores, _ms(t0)
