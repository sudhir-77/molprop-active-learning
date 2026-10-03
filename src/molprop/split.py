"""Bemis-Murcko scaffold split.

The assignment follows the DeepChem scaffold splitter: scaffolds are ordered from
largest to smallest, and each scaffold is placed entirely in train, validation, or
test. A fixed seed breaks ties between scaffolds of equal size. The test split is
therefore a set of scaffolds that do not appear in training or validation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def scaffold_split(
    scaffolds: list[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> list[str]:
    if len(scaffolds) == 0:
        raise ValueError("scaffold_split received no molecules")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, scaffold in enumerate(scaffolds):
        groups[scaffold].append(index)
    rng = np.random.RandomState(seed)
    tie_break = {scaffold: float(rng.rand()) for scaffold in groups}
    ordered = sorted(groups, key=lambda scaffold: (-len(groups[scaffold]), tie_break[scaffold]))
    n_mols = len(scaffolds)
    train_cutoff = fractions[0] * n_mols
    valid_cutoff = (fractions[0] + fractions[1]) * n_mols
    labels = [""] * n_mols
    n_train = 0
    n_valid = 0
    for scaffold in ordered:
        indices = groups[scaffold]
        if n_train + len(indices) <= train_cutoff:
            bucket = "train"
            n_train += len(indices)
        elif n_train + n_valid + len(indices) <= valid_cutoff:
            bucket = "val"
            n_valid += len(indices)
        else:
            bucket = "test"
        for index in indices:
            labels[index] = bucket
    if any(label == "" for label in labels):
        raise RuntimeError("scaffold split left a molecule unassigned")
    return labels


def assign_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["split"] = scaffold_split(labeled["scaffold"].astype(str).tolist(), seed=seed)
    return labeled


def assert_split_integrity(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("split frame is empty")
    if frame["id"].duplicated().any():
        raise ValueError("split ids are not unique")
    observed = set(frame["split"].astype(str))
    if observed != {"train", "val", "test"}:
        raise ValueError(f"expected train/val/test, found {sorted(observed)}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_scaffolds = set(frame.loc[frame["split"] == left, "scaffold"].astype(str))
        right_scaffolds = set(frame.loc[frame["split"] == right, "scaffold"].astype(str))
        overlap = left_scaffolds & right_scaffolds
        if overlap:
            raise ValueError(f"scaffold leakage between {left} and {right}: {len(overlap)} scaffolds")
    id_sets = {
        name: set(frame.loc[frame["split"] == name, "id"].astype(str))
        for name in ("train", "val", "test")
    }
    if id_sets["train"] & id_sets["val"] or id_sets["train"] & id_sets["test"] or id_sets["val"] & id_sets["test"]:
        raise ValueError("molecule ids overlap across splits")
