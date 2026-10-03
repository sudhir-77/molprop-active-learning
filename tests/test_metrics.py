"""Metric helpers and success-criterion logic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from molprop.metrics import (
    Z_90,
    build_coverage_criterion,
    build_hit_criterion,
    build_rmse_criterion,
    fit_std_scale,
    fraction_count,
    interval_coverage,
    mark_top_fraction,
    rmse,
)


def test_rmse_matches_hand_calculation() -> None:
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert rmse([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(math.sqrt(2.0 / 3.0))


def test_interval_coverage_is_closed_at_1_645_sd() -> None:
    assert interval_coverage([0.0, 2.0], [0.0, 0.0], [1.0, 1.0]) == 0.5
    assert interval_coverage([Z_90], [0.0], [1.0]) == 1.0
    assert interval_coverage([Z_90 + 0.001], [0.0], [1.0]) == 0.0


def test_fraction_count_and_top5_ties() -> None:
    assert fraction_count(4200, 0.20) == 840
    assert fraction_count(4200, 0.10) == 420
    assert fraction_count(4200, 0.05) == 210
    flags = mark_top_fraction(["b", "a", "c"], [1.0, 1.0, 0.0], fraction=0.5)
    # 50% of 3 rounds to 2. The value 1.0 is tied, so the smaller id "a" ranks ahead of "b".
    assert flags == [True, True, False]


def test_std_scale_corrects_underdispersion() -> None:
    rng = np.random.RandomState(0)
    y = rng.normal(size=5000)
    scale = fit_std_scale(y, np.zeros_like(y), np.full_like(y, 0.5))
    assert 1.7 < scale < 2.4
    coverage = interval_coverage(y, np.zeros_like(y), np.full_like(y, 0.5) * scale)
    assert 0.86 < coverage < 0.94


def test_rmse_criterion_requires_beating_the_forest() -> None:
    easy = build_rmse_criterion("ESOL test RMSE (scaffold split)", 0.70, 1.10, 0.90)
    assert easy["passed"] is True
    assert easy["op"] == "<="
    assert easy["threshold"] == 0.90
    stronger_forest = build_rmse_criterion("ESOL test RMSE (scaffold split)", 0.80, 0.75, 0.90)
    assert stronger_forest["passed"] is False
    assert stronger_forest["op"] == "<"
    assert stronger_forest["threshold"] == pytest.approx(0.75)


def test_coverage_criterion_fails_when_either_dataset_is_outside() -> None:
    good = build_coverage_criterion([0.90, 0.91])
    assert good["passed"] is True
    assert good["op"] == "between"
    bad = build_coverage_criterion([0.90, 0.99])
    assert bad["passed"] is False
    assert bad["value"] == pytest.approx(0.99)
    low = build_coverage_criterion([0.79, 0.90])
    assert low["passed"] is False
    assert low["value"] == pytest.approx(0.79)


def test_hit_criterion_threshold() -> None:
    assert build_hit_criterion(1.5)["passed"] is True
    assert build_hit_criterion(1.49)["passed"] is False
