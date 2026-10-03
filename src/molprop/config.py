"""Run configuration. Full-run hyperparameters are fixed before looking at the test set."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    quick: bool
    seed: int
    ensemble_size: int
    ensemble_seeds: tuple[int, ...]
    hidden: int
    depth: int
    dropout: float
    fp_bits: int
    fp_radius: int
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    min_epochs: int
    patience: int
    rf_trees: int
    al_ensemble_size: int
    al_epochs: int
    al_warm_epochs: int
    al_seeds: tuple[int, ...]
    al_init_fraction: float
    al_budget_fraction: float
    al_rounds: int
    al_kappa: float
    curve_fractions: tuple[float, ...]
    max_molecules: dict[str, int] | None


def make_config(quick: bool) -> RunConfig:
    if quick:
        return RunConfig(
            quick=True,
            seed=42,
            ensemble_size=1,
            ensemble_seeds=(0,),
            hidden=32,
            depth=2,
            dropout=0.0,
            fp_bits=2048,
            fp_radius=2,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=32,
            max_epochs=1,
            min_epochs=1,
            patience=1,
            rf_trees=10,
            al_ensemble_size=1,
            al_epochs=1,
            al_warm_epochs=1,
            al_seeds=(0, 1),
            al_init_fraction=0.10,
            al_budget_fraction=0.20,
            al_rounds=1,
            al_kappa=1.0,
            curve_fractions=(1.0,),
            max_molecules={"esol": 80, "lipophilicity": 120},
        )
    return RunConfig(
        quick=False,
        seed=42,
        ensemble_size=5,
        ensemble_seeds=(0, 1, 2, 3, 4),
        hidden=128,
        depth=3,
        dropout=0.05,
        fp_bits=2048,
        fp_radius=2,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=64,
        max_epochs=50,
        min_epochs=12,
        patience=12,
        rf_trees=500,
        al_ensemble_size=2,
        al_epochs=12,
        al_warm_epochs=4,
        al_seeds=(0, 1, 2, 3, 4),
        al_init_fraction=0.10,
        al_budget_fraction=0.20,
        al_rounds=2,
        al_kappa=1.0,
        curve_fractions=(0.25, 0.5, 1.0),
        max_molecules=None,
    )
