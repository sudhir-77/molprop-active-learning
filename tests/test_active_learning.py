"""Acquisition functions and the label-gated oracle."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from molprop.active_learning import (
    Oracle,
    acquire,
    best_strategy_ratio,
    expected_improvement,
    recoveries_at_budget,
    run_pool_simulation,
    select_top_k,
    validate_trace,
)


def test_expected_improvement_matches_erf_formula() -> None:
    mu = np.asarray([1.0])
    sigma = np.asarray([1.0])
    best = 0.0
    z = 1.0
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    hand = (1.0 - best) * cdf + 1.0 * phi
    assert expected_improvement(mu, sigma, best)[0] == pytest.approx(hand)
    # A zero-variance prediction below the best has no improvement.
    assert expected_improvement(np.asarray([0.2]), np.asarray([0.0]), 1.0)[0] == 0.0


def test_acquire_known_rankings() -> None:
    ids = ["a", "b", "c"]
    mu = np.asarray([0.0, 1.0, 0.5])
    std = np.asarray([10.0, 0.0, 0.1])
    assert acquire("greedy", ids, mu, std, y_best=0.0, k=1, kappa=1.0) == ["b"]
    assert acquire("ucb", ids, mu, std, y_best=0.0, k=1, kappa=1.0) == ["a"]
    assert select_top_k(["b", "a"], [1.0, 1.0], k=1) == ["a"]


def test_oracle_refuses_unrevealed_labels() -> None:
    oracle = Oracle(["a", "b"], [1.0, 2.0])
    with pytest.raises(PermissionError, match="not been evaluated"):
        oracle.labels(["a"])
    oracle.reveal(["a"])
    assert oracle.labels(["a"]).tolist() == [1.0]
    with pytest.raises(PermissionError):
        oracle.labels(["a", "b"])
    with pytest.raises(KeyError):
        oracle.reveal(["missing"])


def test_simulation_never_passes_unlabeled_targets_and_greedy_finds_hits() -> None:
    n = 40
    ids = [f"m{i:02d}" for i in range(n)]
    smiles = ["C"] * n
    y = np.arange(n, dtype=np.float64)
    truth = {mol_id: float(value) for mol_id, value in zip(ids, y)}
    calls = {"n": 0}

    def fit_predict(labeled_ids, labeled_y, candidate_ids, state):
        calls["n"] += 1
        labeled_ids = [str(mol_id) for mol_id in labeled_ids]
        candidate_ids = [str(mol_id) for mol_id in candidate_ids]
        assert set(labeled_ids).isdisjoint(candidate_ids)
        for mol_id, value in zip(labeled_ids, labeled_y):
            assert value == pytest.approx(truth[mol_id])
        mu = np.asarray([truth[mol_id] for mol_id in candidate_ids], dtype=np.float64)
        std = np.full(len(candidate_ids), 0.05)
        return mu, std, {"round": calls["n"]}

    trace, pool = run_pool_simulation(
        ids,
        smiles,
        y,
        seeds=(0, 1, 2, 3, 4),
        init_fraction=0.10,
        budget_fraction=0.20,
        n_rounds=2,
        fit_predict=fit_predict,
    )
    validate_trace(trace, pool)
    recoveries = recoveries_at_budget(trace, pool, 0.20)
    assert recoveries["greedy"][0] == pytest.approx(1.0)
    best, ratio, means = best_strategy_ratio(recoveries)
    assert means["greedy"] == pytest.approx(1.0)
    assert means["greedy"] > means["random"]
    assert ratio >= 1.5
    assert best in {"greedy", "ucb", "ei"}
    # Random acquisition does not fit a model. One shared fit per seed, then one fit
    # per informed strategy on the second round.
    assert calls["n"] == 5 * (1 + 3)


def test_recovery_from_a_hand_built_trace() -> None:
    pool = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "smiles": ["C", "CC", "CCC", "CCCC"],
            "y_true": [0.0, 1.0, 2.0, 3.0],
            "is_top5": ["false", "false", "false", "true"],
        }
    )
    trace = pd.DataFrame(
        [
            {"strategy": "greedy", "seed": 0, "round": 0, "n_labeled": 1, "selected_ids": "a"},
            {"strategy": "greedy", "seed": 0, "round": 1, "n_labeled": 2, "selected_ids": "d"},
            {"strategy": "random", "seed": 0, "round": 0, "n_labeled": 1, "selected_ids": "b"},
            {"strategy": "random", "seed": 0, "round": 1, "n_labeled": 2, "selected_ids": "c"},
        ]
    )
    recoveries = recoveries_at_budget(trace, pool, 0.50)
    assert recoveries["greedy"][0] == pytest.approx(1.0)
    assert recoveries["random"][0] == pytest.approx(0.0)
