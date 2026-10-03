"""Featurizer and scaffold split."""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from molprop.featurize import featurize_mol, featurize_table, morgan_generator
from molprop.split import assert_split_integrity, assign_splits, scaffold_split


def _graph(smiles: str, n_bits: int = 128):
    mol = Chem.MolFromSmiles(smiles)
    generator = morgan_generator(n_bits, 2)
    return featurize_mol(mol, smiles, smiles, generator, n_bits)


def test_atom_and_bond_counts() -> None:
    ethanol = _graph("CCO")
    benzene = _graph("c1ccccc1")
    methane = _graph("C")
    assert ethanol.atom_x.shape[0] == 3
    assert ethanol.src.shape[0] == 4
    assert benzene.atom_x.shape[0] == 6
    assert benzene.src.shape[0] == 12
    assert methane.atom_x.shape[0] == 1
    assert methane.src.shape[0] == 0
    assert ethanol.atom_x.shape[1] == benzene.atom_x.shape[1] == methane.atom_x.shape[1]
    assert ethanol.bond_x.shape[1] == benzene.bond_x.shape[1] == methane.bond_x.shape[1]
    assert ethanol.fp.shape == (128,)


def test_shared_murcko_scaffold() -> None:
    benzene = _graph("c1ccccc1")
    phenol = _graph("Oc1ccccc1")
    pyridine = _graph("c1ccncc1")
    assert benzene.scaffold == phenol.scaffold
    assert benzene.scaffold != pyridine.scaffold


def test_invalid_smiles_are_dropped() -> None:
    table = pd.DataFrame({"id": ["ok", "bad"], "smiles": ["CCO", "not_a_molecule"], "y": [0.1, 0.2]})
    graphs, frame, dropped = featurize_table(table, fp_bits=64, fp_radius=2, cache_path=None)
    assert dropped == ["bad"]
    assert list(frame["id"]) == ["ok"]
    assert len(graphs) == 1


def test_scaffold_split_keeps_scaffolds_together_and_is_reproducible() -> None:
    smiles = [
        "c1ccccc1",
        "Oc1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "CCO",
        "CCN",
        "CCC",
        "c1ccc2ccccc2c1",
        "CC(=O)O",
        "CC(=O)N",
        "c1ccc(O)cc1",
        "Clc1ccccc1",
        "c1ccoc1",
        "c1ccsc1",
        "CC(C)O",
        "c1ccccc1N",
        "CCCl",
        "c1ccc(N)cc1",
        "C1CCCCC1",
    ]
    table = pd.DataFrame({"id": [f"m{i}" for i in range(len(smiles))], "smiles": smiles, "y": range(len(smiles))})
    _, frame, _ = featurize_table(table, fp_bits=64, fp_radius=2, cache_path=None)
    labeled = assign_splits(frame, seed=42)
    again = assign_splits(frame, seed=42)
    assert labeled["split"].tolist() == again["split"].tolist()
    assert_split_integrity(labeled)
    benzene_id = labeled.loc[labeled["smiles"] == "c1ccccc1", "scaffold"].iloc[0]
    shared = labeled.loc[labeled["scaffold"] == benzene_id, "split"].unique()
    assert len(shared) == 1
    counts = labeled["split"].value_counts()
    assert set(counts.index) == {"train", "val", "test"}
    assert counts["train"] > counts["val"]
    assert counts["train"] > counts["test"]


def test_scaffold_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        scaffold_split(["a", "b"], fractions=(0.5, 0.5, 0.5), seed=0)


def test_integrity_catches_scaffold_leakage() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "smiles": ["C", "CC", "CCC"],
            "scaffold": ["same", "same", "other"],
            "split": ["train", "test", "val"],
            "y": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_split_integrity(frame)
