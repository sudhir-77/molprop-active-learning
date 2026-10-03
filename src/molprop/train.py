"""Training and prediction for the deep ensemble."""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch import nn

from molprop.config import RunConfig
from molprop.featurize import MolGraph
from molprop.metrics import rmse
from molprop.model import DMPNN, collate_graphs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_fingerprints(graphs: list[MolGraph]) -> np.ndarray:
    if not graphs:
        raise ValueError("no graphs to stack")
    return np.stack([graph.fp for graph in graphs]).astype(np.float64)


def predict(model: nn.Module, graphs: list[MolGraph], batch_size: int, device: torch.device) -> np.ndarray:
    if not graphs:
        return np.zeros((0,), dtype=np.float64)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(graphs), batch_size):
            batch = collate_graphs(graphs[start : start + batch_size], device)
            pred = model(batch)
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_regressor(
    model: nn.Module,
    train_graphs: list[MolGraph],
    y_train: np.ndarray,
    val_graphs: list[MolGraph] | None,
    y_val: np.ndarray | None,
    *,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    y_scale: float = 1.0,
    log_prefix: str = "",
) -> nn.Module:
    """Minimize MSE. When a validation set is given, restore the best-val weights.

    `y_train` is already standardized. `y_scale` is only used to print original-unit RMSE.
    """
    if len(train_graphs) != len(y_train):
        raise ValueError("train graphs and labels differ in length")
    if len(train_graphs) == 0:
        raise ValueError("training set is empty")
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_train = np.asarray(y_train, dtype=np.float64)
    use_val = val_graphs is not None and y_val is not None and len(val_graphs) > 0
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        if use_val
        else None
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    rng = np.random.RandomState(seed)
    prefix = f"{log_prefix} " if log_prefix else ""
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_graphs))
        step = min(batch_size, len(train_graphs))
        total_sq = 0.0
        total_n = 0
        for start in range(0, len(order), step):
            batch_index = order[start : start + step]
            graphs = [train_graphs[int(i)] for i in batch_index]
            target = torch.tensor(y_train[batch_index], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(collate_graphs(graphs, device))
            loss = torch.nn.functional.mse_loss(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_sq += float(loss.detach()) * len(batch_index)
            total_n += len(batch_index)
        train_rmse = math.sqrt(total_sq / total_n) * y_scale
        val_rmse = None
        if use_val:
            val_pred = predict(model, val_graphs, batch_size, device)
            val_rmse = rmse(y_val, val_pred) * y_scale
            assert scheduler is not None
            scheduler.step(val_rmse)
            if val_rmse + 1e-6 < best_val:
                best_val = val_rmse
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        message = f"{prefix}epoch {epoch + 1}/{epochs} train_rmse {train_rmse:.4f}"
        if val_rmse is not None:
            message += f" val_rmse {val_rmse:.4f}"
        print(message, flush=True)
        if use_val and epoch + 1 >= min_epochs and stale >= patience:
            print(f"{prefix}early stop at epoch {epoch + 1}, best val_rmse {best_val:.4f}", flush=True)
            break
    if use_val:
        model.load_state_dict(best_state)
    return model


def _make_model(example: MolGraph, cfg: RunConfig) -> DMPNN:
    return DMPNN(
        atom_dim=int(example.atom_x.shape[1]),
        bond_dim=int(example.bond_x.shape[1]),
        hidden=cfg.hidden,
        depth=cfg.depth,
        fp_dim=int(example.fp.shape[0]),
        desc_dim=int(example.desc.shape[0]),
        dropout=cfg.dropout,
    )


def train_ensemble(
    train_graphs: list[MolGraph],
    y_train: np.ndarray,
    val_graphs: list[MolGraph] | None,
    y_val: np.ndarray | None,
    pred_graphs: list[MolGraph],
    cfg: RunConfig,
    seeds: tuple[int, ...] | list[int],
    device: torch.device,
    *,
    epochs: int | None = None,
    min_epochs: int | None = None,
    patience: int | None = None,
    warm_states: list[dict] | None = None,
    fixed_mean: float | None = None,
    fixed_scale: float | None = None,
    log_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, list[dict], float, float]:
    """Train one model per seed and return mean, std, and CPU state dicts on the original scale.

    Target mean and scale are computed from `y_train` only unless a fixed scale is passed
    in so later active-learning rounds stay on the same target units as the warm start.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    if fixed_mean is None or fixed_scale is None:
        y_mean = float(np.mean(y_train))
        y_scale = float(np.std(y_train))
        if y_scale < 1e-8:
            y_scale = 1.0
    else:
        y_mean = float(fixed_mean)
        y_scale = float(fixed_scale)
        if y_scale < 1e-8:
            raise ValueError("y_scale must be positive")
    y_train_s = (y_train - y_mean) / y_scale
    y_val_s = None if y_val is None else (np.asarray(y_val, dtype=np.float64) - y_mean) / y_scale
    epoch_budget = cfg.max_epochs if epochs is None else epochs
    min_budget = cfg.min_epochs if min_epochs is None else min_epochs
    patience_budget = cfg.patience if patience is None else patience
    member_preds = []
    states = []
    for index, seed in enumerate(seeds):
        seed_everything(int(seed))
        model = _make_model(train_graphs[0], cfg).to(device)
        if warm_states is not None:
            model.load_state_dict({key: value.to(device) for key, value in warm_states[index].items()})
        train_regressor(
            model,
            train_graphs,
            y_train_s,
            val_graphs,
            y_val_s,
            epochs=epoch_budget,
            min_epochs=min(min_budget, epoch_budget),
            patience=patience_budget,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            seed=int(seed),
            device=device,
            y_scale=y_scale,
            log_prefix=f"{log_prefix} member {index}".strip(),
        )
        raw = predict(model, pred_graphs, cfg.batch_size, device)
        member_preds.append(raw * y_scale + y_mean)
        states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    stacked = np.stack(member_preds, axis=0)
    mean = stacked.mean(axis=0)
    if stacked.shape[0] == 1:
        std = np.zeros(stacked.shape[1], dtype=np.float64)
    else:
        std = stacked.std(axis=0, ddof=1)
    return mean, std, states, y_mean, y_scale


def ensemble_predict_from_states(
    states: list[dict],
    example_graph: MolGraph,
    graphs: list[MolGraph],
    y_mean: float,
    y_scale: float,
    cfg: RunConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    for state in states:
        model = _make_model(example_graph, cfg).to(device)
        model.load_state_dict({key: value.to(device) for key, value in state.items()})
        raw = predict(model, graphs, cfg.batch_size, device)
        preds.append(raw * y_scale + y_mean)
    stacked = np.stack(preds, axis=0)
    mean = stacked.mean(axis=0)
    if stacked.shape[0] == 1:
        std = np.zeros(stacked.shape[1], dtype=np.float64)
    else:
        std = stacked.std(axis=0, ddof=1)
    return mean, std
