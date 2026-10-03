"""Message passing, a tiny fit, and the fingerprint forest."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from rdkit import Chem

from molprop.baseline import predict_random_forest, train_random_forest
from molprop.config import make_config
from molprop.featurize import featurize_mol, morgan_generator
from molprop.metrics import rmse
from molprop.model import DMPNN, collate_graphs
from molprop.train import predict, train_ensemble


def _graphs(smiles_list: list[str], n_bits: int = 256):
    generator = morgan_generator(n_bits, 2)
    graphs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        graphs.append(featurize_mol(mol, smiles, smiles, generator, n_bits))
    return graphs


def test_predictions_do_not_depend_on_batch_order() -> None:
    graphs = _graphs(["C", "CCO", "c1ccccc1", "CC(=O)O"])
    model = DMPNN(graphs[0].atom_x.shape[1], graphs[0].bond_x.shape[1], hidden=16, depth=2, fp_dim=256, dropout=0.0)
    model.eval()
    device = torch.device("cpu")
    forward = model(collate_graphs(graphs, device))
    reverse = model(collate_graphs(list(reversed(graphs)), device))
    assert forward.shape == (4,)
    assert torch.allclose(forward, reverse.flip(0), atol=1e-5)
    methane = model(collate_graphs([graphs[0]], device))
    assert methane.shape == (1,)
    assert torch.isfinite(methane).all()


def test_gradients_reach_the_readout() -> None:
    graphs = _graphs(["CCO"])
    model = DMPNN(graphs[0].atom_x.shape[1], graphs[0].bond_x.shape[1], hidden=16, depth=2, fp_dim=256, dropout=0.0)
    model.train()
    pred = model(collate_graphs(graphs, torch.device("cpu")))
    pred.sum().backward()
    assert model.readout[-1].weight.grad is not None
    assert torch.isfinite(model.readout[-1].weight.grad).all()


def test_ensemble_fits_heavy_atom_count() -> None:
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "c1ccccc1", "c1ccncc1", "CC(=O)O"]
    graphs = _graphs(smiles, n_bits=128)
    y = np.asarray([Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in smiles], dtype=np.float64)
    cfg = replace(
        make_config(True),
        hidden=32,
        depth=2,
        dropout=0.0,
        max_epochs=30,
        min_epochs=30,
        patience=30,
        batch_size=8,
        fp_bits=128,
    )
    mu, std, _, _, _ = train_ensemble(
        graphs,
        y,
        None,
        None,
        graphs,
        cfg,
        seeds=(0, 1),
        device=torch.device("cpu"),
        log_prefix="toy",
    )
    assert std.shape == mu.shape
    assert np.isfinite(std).all()
    assert rmse(y, mu) < float(np.std(y)) * 0.5


def test_predict_empty_is_empty() -> None:
    graphs = _graphs(["CCO"])
    model = DMPNN(graphs[0].atom_x.shape[1], graphs[0].bond_x.shape[1], hidden=8, depth=1, fp_dim=256)
    assert predict(model, [], batch_size=4, device=torch.device("cpu")).shape == (0,)


def test_random_forest_runs_on_fingerprints() -> None:
    graphs = _graphs(["CCO", "c1ccccc1", "CCN", "CCC"], n_bits=64)
    y = np.asarray([0.1, 1.0, 0.2, 0.3])
    model = train_random_forest(graphs, y, n_trees=5, seed=0)
    pred = predict_random_forest(model, graphs)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
