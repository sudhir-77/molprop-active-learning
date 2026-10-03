"""Random-forest baselines on Morgan fingerprints, with or without RDKit descriptors."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from molprop.featurize import MolGraph
from molprop.train import seed_everything


def stack_model_inputs(graphs: list[MolGraph], include_descriptors: bool) -> np.ndarray:
    if not graphs:
        raise ValueError("no graphs to stack")
    fingerprints = np.stack([graph.fp for graph in graphs]).astype(np.float64)
    if not include_descriptors:
        return fingerprints
    descriptors = np.stack([graph.desc for graph in graphs]).astype(np.float64)
    return np.concatenate([fingerprints, descriptors], axis=1)


def train_random_forest(
    graphs: list[MolGraph],
    y: np.ndarray,
    *,
    n_trees: int,
    seed: int,
    include_descriptors: bool = False,
) -> RandomForestRegressor:
    # One job keeps tree bootstrap order fixed. n_jobs=-1 changes results between processes.
    seed_everything(seed)
    model = RandomForestRegressor(
        n_estimators=n_trees,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
        bootstrap=True,
    )
    features = stack_model_inputs(graphs, include_descriptors)
    model.fit(features, np.asarray(y, dtype=np.float64))
    model.include_descriptors_ = bool(include_descriptors)
    return model


def predict_random_forest(
    model: RandomForestRegressor,
    graphs: list[MolGraph],
    include_descriptors: bool | None = None,
) -> np.ndarray:
    if include_descriptors is None:
        include_descriptors = bool(getattr(model, "include_descriptors_", False))
    features = stack_model_inputs(graphs, include_descriptors)
    if features.shape[1] != int(model.n_features_in_):
        raise ValueError(
            f"random forest expected {int(model.n_features_in_)} features, got {features.shape[1]}"
        )
    return np.asarray(model.predict(features), dtype=np.float64)
