"""Regression metrics, interval coverage, and success-criterion records."""

from __future__ import annotations

import math

import numpy as np


Z_90 = 1.645


def rmse(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch {truth.shape} vs {pred.shape}")
    if truth.size == 0:
        raise ValueError("rmse received no rows")
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def interval_coverage(y_true, y_pred, y_std, z: float = Z_90) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    std = np.asarray(y_std, dtype=np.float64)
    if truth.shape != pred.shape or truth.shape != std.shape:
        raise ValueError("coverage inputs must share a shape")
    if truth.size == 0:
        raise ValueError("coverage received no rows")
    inside = np.abs(truth - pred) <= (z * std)
    return float(np.mean(inside))


def fraction_count(n: int, fraction: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return max(1, int(math.floor(n * fraction + 0.5)))


def mark_top_fraction(ids, y, fraction: float = 0.05) -> list[bool]:
    """Mark the highest-valued molecules. Ties break toward the smaller id."""
    values = np.asarray(y, dtype=np.float64)
    labels = [str(item) for item in ids]
    if len(labels) != len(values):
        raise ValueError("ids and y must have the same length")
    k = fraction_count(len(labels), fraction)
    order = sorted(range(len(labels)), key=lambda index: (-float(values[index]), labels[index]))
    chosen = set(order[:k])
    return [index in chosen for index in range(len(labels))]


def fit_std_scale(y_true, y_pred, y_std, z: float = Z_90, target: float = 0.90) -> float:
    """Pick a scalar on the standard deviation so validation coverage is near 90%.

    The scale is fit only on the split passed in (the validation split). It does
    not look at test labels.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    std = np.maximum(np.asarray(y_std, dtype=np.float64), 1e-8)
    residual = np.abs(truth - pred)
    best_scale = 1.0
    best_error = float("inf")
    for scale in np.exp(np.linspace(np.log(0.05), np.log(20.0), 80)):
        coverage = float(np.mean(residual <= z * scale * std))
        error = abs(coverage - target)
        closer_to_one = abs(error - best_error) <= 1e-12 and abs(scale - 1.0) < abs(best_scale - 1.0)
        if error < best_error - 1e-12 or closer_to_one:
            best_scale = float(scale)
            best_error = error
    return best_scale


def build_rmse_criterion(name: str, gnn_rmse: float, rf_rmse: float, threshold: float) -> dict:
    """Pass only when the ensemble is under the absolute cap and strictly under the RF."""
    value = float(gnn_rmse)
    baseline = float(rf_rmse)
    passed = value <= threshold and value < baseline
    if baseline <= threshold:
        op = "<"
        cutoff = baseline
    else:
        op = "<="
        cutoff = float(threshold)
    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": float(cutoff),
        "threshold_high": None,
        "passed": bool(passed),
    }


def build_coverage_criterion(coverages: list[float]) -> dict:
    low = 0.80
    high = 0.97
    values = [float(item) for item in coverages]
    if not values:
        raise ValueError("coverage criterion needs at least one value")
    inside = [low <= item <= high for item in values]
    passed = all(inside)
    if passed:
        reported = float(np.mean(values))
    else:
        def distance(item: float) -> float:
            if item < low:
                return low - item
            if item > high:
                return item - high
            return 0.0

        reported = max(values, key=distance)
    return {
        "name": "90% prediction-interval coverage",
        "value": float(reported),
        "op": "between",
        "threshold": low,
        "threshold_high": high,
        "passed": bool(passed),
    }


def build_hit_criterion(ratio: float) -> dict:
    value = float(ratio)
    return {
        "name": "Top-5% hit recovery at 20% budget (Lipophilicity pool)",
        "value": value,
        "op": ">=",
        "threshold": 1.5,
        "threshold_high": None,
        "passed": bool(value >= 1.5),
    }
