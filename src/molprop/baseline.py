"""Random-forest baseline on Morgan fingerprints."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from molprop.featurize import MolGraph
from molprop.train import stack_fingerprints


def train_random_forest(
    graphs: list[MolGraph],
    y: np.ndarray,
    *,
    n_trees: int,
    seed: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_trees,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
        bootstrap=True,
    )
    model.fit(stack_fingerprints(graphs), np.asarray(y, dtype=np.float64))
    return model


def predict_random_forest(model: RandomForestRegressor, graphs: list[MolGraph]) -> np.ndarray:
    return np.asarray(model.predict(stack_fingerprints(graphs)), dtype=np.float64)
