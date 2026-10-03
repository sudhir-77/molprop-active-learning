"""End-to-end property prediction and active-learning pipeline."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molprop.active_learning import (
    best_strategy_ratio,
    recoveries_at_budget,
    recovery_points,
    run_pool_simulation,
    validate_trace,
)
from molprop.baseline import predict_random_forest, train_random_forest
from molprop.config import RunConfig, make_config
from molprop.data import DATASETS, ensure_dataset, load_table
from molprop.featurize import FEATURE_VERSION, MolGraph, featurize_table
from molprop.metrics import (
    build_coverage_criterion,
    build_hit_criterion,
    build_rmse_criterion,
    fit_std_scale,
    fraction_count,
    interval_coverage,
    rmse,
)
from molprop.plots import plot_recovery, plot_rmse_curve
from molprop.report import (
    collect_hardware,
    prediction_metrics,
    update_readme,
    write_metrics,
    write_prediction_csv,
    write_resume_notes,
    write_split_csv,
)
from molprop.split import assert_split_integrity, assign_splits
from molprop.train import ensemble_predict_from_states, seed_everything, train_ensemble


@dataclass
class Prepared:
    name: str
    frame: pd.DataFrame
    graphs: list[MolGraph]
    dropped: list[str]

    def part(self, split: str) -> tuple[list[MolGraph], np.ndarray, list[str], list[str]]:
        mask = self.frame["split"].to_numpy() == split
        index = np.flatnonzero(mask)
        graphs = [self.graphs[int(i)] for i in index]
        y = self.frame["y"].to_numpy(dtype=np.float64)[index]
        ids = self.frame["id"].astype(str).to_numpy()[index].tolist()
        smiles = self.frame["smiles"].astype(str).to_numpy()[index].tolist()
        return graphs, y, ids, smiles


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name != "cpu":
        raise RuntimeError(f"unknown device {name}")
    return torch.device("cpu")


def take_subset(
    frame: pd.DataFrame,
    graphs: list[MolGraph],
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, list[MolGraph]]:
    if n >= len(frame):
        return frame.reset_index(drop=True), graphs
    rng = np.random.RandomState(seed)
    chosen = np.sort(rng.choice(len(frame), size=n, replace=False))
    subset = frame.iloc[chosen].reset_index(drop=True)
    kept = [graphs[int(i)] for i in chosen]
    return subset, kept


def prepare_dataset(name: str, cfg: RunConfig, data_dir: Path) -> Prepared:
    spec = DATASETS[name]
    path = ensure_dataset(spec, data_dir)
    table = load_table(path, spec)
    cache = data_dir / "cache" / f"{spec.name}_fp{cfg.fp_bits}_r{cfg.fp_radius}_v{FEATURE_VERSION}.pkl"
    graphs, frame, dropped = featurize_table(table, cfg.fp_bits, cfg.fp_radius, cache)
    print(f"{name}: featurized {len(frame)} molecules, dropped {len(dropped)}", flush=True)
    if cfg.max_molecules is not None and name in cfg.max_molecules:
        frame, graphs = take_subset(frame, graphs, cfg.max_molecules[name], cfg.seed)
        print(f"{name}: quick subset uses {len(frame)} molecules", flush=True)
    frame = assign_splits(frame, cfg.seed)
    assert_split_integrity(frame)
    covered = sum(int((frame["split"] == split).sum()) for split in ("train", "val", "test"))
    if covered != len(frame):
        raise RuntimeError(f"{name} split does not cover every molecule")
    counts = frame["split"].value_counts().to_dict()
    print(f"{name} split train/val/test {counts.get('train')} {counts.get('val')} {counts.get('test')}", flush=True)
    return Prepared(name, frame.reset_index(drop=True), graphs, dropped)


def assert_prediction_matches_split(split_path: Path, prediction_path: Path) -> None:
    split = pd.read_csv(split_path, keep_default_na=False)
    prediction = pd.read_csv(prediction_path)
    test_ids = set(split.loc[split["split"] == "test", "id"].astype(str))
    pred_ids = set(prediction["id"].astype(str))
    if test_ids != pred_ids:
        raise ValueError(f"{prediction_path.name} ids do not match the test split")


def fit_supervised(ds: Prepared, cfg: RunConfig, results_dir: Path, device: torch.device) -> dict:
    """Train on the train split, stop on validation, and score the frozen test split."""
    train_graphs, y_train, _, _ = ds.part("train")
    val_graphs, y_val, _, _ = ds.part("val")
    test_graphs, y_test, test_ids, _ = ds.part("test")
    if set(test_ids) & set(ds.frame.loc[ds.frame["split"] == "train", "id"].astype(str)):
        raise RuntimeError("test ids leaked into the training ids")
    mu_test, std_test, states, y_mean, y_scale = train_ensemble(
        train_graphs,
        y_train,
        val_graphs,
        y_val,
        test_graphs,
        cfg,
        cfg.ensemble_seeds,
        device,
        log_prefix=f"{ds.name} gnn",
    )
    mu_val, std_val = ensemble_predict_from_states(
        states, train_graphs[0], val_graphs, y_mean, y_scale, cfg, device
    )
    scale = fit_std_scale(y_val, mu_val, std_val)
    std_test = std_test * scale
    std_val = std_val * scale
    print(
        f"{ds.name} gnn val_rmse {rmse(y_val, mu_val):.4f} "
        f"val_coverage {interval_coverage(y_val, mu_val, std_val):.4f} std_scale {scale:.3f}",
        flush=True,
    )
    forest = train_random_forest(train_graphs, y_train, n_trees=cfg.rf_trees, seed=cfg.seed)
    rf_test = predict_random_forest(forest, test_graphs)
    rf_val = predict_random_forest(forest, val_graphs)
    print(
        f"{ds.name} rf val_rmse {rmse(y_val, rf_val):.4f} heldout_rows {len(test_ids)}",
        flush=True,
    )
    split_path = results_dir / "splits" / f"{ds.name}_split.csv"
    gnn_path = results_dir / "predictions" / f"{ds.name}_gnn_ensemble_test.csv"
    rf_path = results_dir / "predictions" / f"{ds.name}_rf_baseline_test.csv"
    write_split_csv(split_path, ds.frame)
    reloaded = pd.read_csv(split_path, keep_default_na=False)
    assert_split_integrity(reloaded)
    write_prediction_csv(gnn_path, test_ids, y_test, mu_test, std_test)
    write_prediction_csv(rf_path, test_ids, y_test, rf_test, None)
    assert_prediction_matches_split(split_path, gnn_path)
    assert_prediction_matches_split(split_path, rf_path)
    gnn_metrics = prediction_metrics(gnn_path, with_coverage=True)
    rf_metrics = prediction_metrics(rf_path, with_coverage=False)
    gnn_metrics["val_rmse"] = rmse(y_val, mu_val)
    rf_metrics["val_rmse"] = rmse(y_val, rf_val)
    gnn_metrics["std_scale"] = float(scale)
    return {"gnn_ensemble": gnn_metrics, "rf_baseline": rf_metrics, "split_path": split_path}


def learning_curve_rows(ds: Prepared, cfg: RunConfig, device: torch.device) -> list[dict]:
    """Test RMSE at nested training sizes.

    Epoch choice and architecture stay fixed. These test numbers are only a plot;
    they are not used to pick the model that is written to the prediction files.
    """
    train_graphs, y_train, _, _ = ds.part("train")
    val_graphs, y_val, _, _ = ds.part("val")
    test_graphs, y_test, _, _ = ds.part("test")
    rng = np.random.RandomState(cfg.seed)
    order = rng.permutation(len(train_graphs))
    rows: list[dict] = []
    for fraction in cfg.curve_fractions:
        size = min(len(order), fraction_count(len(order), fraction))
        selected = order[:size]
        subset = [train_graphs[int(i)] for i in selected]
        subset_y = y_train[selected]
        mu, _, _, _, _ = train_ensemble(
            subset,
            subset_y,
            val_graphs,
            y_val,
            test_graphs,
            cfg,
            seeds=(0,),
            device=device,
            log_prefix=f"{ds.name} curve {size}",
        )
        rows.append(
            {
                "dataset": ds.name,
                "model": "gnn",
                "train_size": int(size),
                "test_rmse": rmse(y_test, mu),
            }
        )
        forest = train_random_forest(subset, subset_y, n_trees=cfg.rf_trees, seed=cfg.seed)
        rf_pred = predict_random_forest(forest, test_graphs)
        rows.append(
            {
                "dataset": ds.name,
                "model": "rf_baseline",
                "train_size": int(size),
                "test_rmse": rmse(y_test, rf_pred),
            }
        )
    return rows


def run_lipophilicity_active_learning(ds: Prepared, cfg: RunConfig, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen train and validation molecules only, so test labels are never read."""
    mask = ds.frame["split"].to_numpy() != "test"
    index = np.flatnonzero(mask)
    pool_ids = ds.frame["id"].astype(str).to_numpy()[index].tolist()
    pool_smiles = ds.frame["smiles"].astype(str).to_numpy()[index].tolist()
    pool_y = ds.frame["y"].to_numpy(dtype=np.float64)[index]
    test_ids = set(ds.frame.loc[ds.frame["split"] == "test", "id"].astype(str))
    if set(pool_ids) & test_ids:
        raise RuntimeError("active-learning pool includes test molecules")
    graph_by_id = {pool_ids[pos]: ds.graphs[int(raw_index)] for pos, raw_index in enumerate(index)}
    calls = {"n": 0}

    def fit_predict(labeled_ids, labeled_y, candidate_ids, state):
        labeled_ids = [str(mol_id) for mol_id in labeled_ids]
        candidate_ids = [str(mol_id) for mol_id in candidate_ids]
        if set(labeled_ids) & set(candidate_ids):
            raise RuntimeError("labeled and candidate sets overlap")
        if any(mol_id in test_ids for mol_id in labeled_ids + candidate_ids):
            raise RuntimeError("active learning requested a test-split molecule")
        y = np.asarray(labeled_y, dtype=np.float64)
        if len(y) != len(labeled_ids):
            raise RuntimeError("oracle label vector does not match labeled ids")
        calls["n"] += 1
        warm = None if state is None else state["members"]
        fixed_mean = None if state is None else state["y_mean"]
        fixed_scale = None if state is None else state["y_scale"]
        epochs = cfg.al_epochs if state is None else cfg.al_warm_epochs
        mu, std, members, y_mean, y_scale = train_ensemble(
            [graph_by_id[mol_id] for mol_id in labeled_ids],
            y,
            None,
            None,
            [graph_by_id[mol_id] for mol_id in candidate_ids],
            cfg,
            seeds=tuple(range(cfg.al_ensemble_size)),
            device=device,
            epochs=epochs,
            min_epochs=1,
            patience=epochs,
            warm_states=warm,
            fixed_mean=fixed_mean,
            fixed_scale=fixed_scale,
            log_prefix=f"al fit {calls['n']}",
        )
        return mu, std, {"members": members, "y_mean": y_mean, "y_scale": y_scale}

    print(
        f"active learning pool {len(pool_ids)} (test split excluded) "
        f"seeds {list(cfg.al_seeds)} rounds {cfg.al_rounds}",
        flush=True,
    )
    return run_pool_simulation(
        pool_ids,
        pool_smiles,
        pool_y,
        seeds=cfg.al_seeds,
        init_fraction=cfg.al_init_fraction,
        budget_fraction=cfg.al_budget_fraction,
        n_rounds=cfg.al_rounds,
        kappa=cfg.al_kappa,
        fit_predict=fit_predict,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the molecular property active-learning pipeline")
    parser.add_argument("--quick", action="store_true", help="Small subset and few epochs, under 3 minutes")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args(argv)


def run_pipeline(
    project_root: Path,
    *,
    quick: bool = False,
    device: str = "cpu",
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    started = time.perf_counter()
    cfg = make_config(quick)
    torch_device = resolve_device(device)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    seed_everything(cfg.seed)
    data_path = data_dir if data_dir is not None else project_root / "data"
    results_path = results_dir if results_dir is not None else project_root / "results"
    data_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    hardware = collect_hardware(torch_device)
    print(f"device {torch_device} python {hardware['python']} cpu {hardware['cpu']}", flush=True)

    prepared = {name: prepare_dataset(name, cfg, data_path) for name in ("esol", "lipophilicity")}
    supervised = {
        name: fit_supervised(dataset, cfg, results_path, torch_device) for name, dataset in prepared.items()
    }
    curve_rows: list[dict] = []
    for dataset in prepared.values():
        curve_rows.extend(learning_curve_rows(dataset, cfg, torch_device))
    curve_frame = pd.DataFrame(curve_rows)
    curve_path = results_path / "learning_curve.csv"
    curve_frame.to_csv(curve_path, index=False)
    curve_frame = pd.read_csv(curve_path)

    trace, pool = run_lipophilicity_active_learning(prepared["lipophilicity"], cfg, torch_device)
    al_dir = results_path / "active_learning"
    al_dir.mkdir(parents=True, exist_ok=True)
    trace_path = al_dir / "trace.csv"
    pool_path = al_dir / "pool.csv"
    trace.to_csv(trace_path, index=False)
    pool.to_csv(pool_path, index=False)
    trace = pd.read_csv(trace_path)
    pool = pd.read_csv(pool_path)
    validate_trace(trace, pool)
    split = pd.read_csv(results_path / "splits" / "lipophilicity_split.csv", keep_default_na=False)
    test_ids = set(split.loc[split["split"] == "test", "id"].astype(str))
    if set(pool["id"].astype(str)) & test_ids:
        raise RuntimeError("saved active-learning pool includes test ids")

    points = recovery_points(trace, pool)
    plot_rmse_curve(curve_frame, results_path / "plots" / "test_rmse_vs_train_size.png")
    plot_recovery(points, results_path / "plots" / "hit_recovery_vs_budget.png", cfg.al_budget_fraction)

    recoveries = recoveries_at_budget(trace, pool, cfg.al_budget_fraction)
    best_name, ratio, means = best_strategy_ratio(recoveries)
    metrics_body = {}
    split_counts = {}
    for name, block in supervised.items():
        split_frame = pd.read_csv(block["split_path"], keep_default_na=False)
        split_counts[name] = {
            "train": int((split_frame["split"] == "train").sum()),
            "val": int((split_frame["split"] == "val").sum()),
            "test": int((split_frame["split"] == "test").sum()),
        }
        metrics_body[name] = {
            "gnn_ensemble": block["gnn_ensemble"],
            "rf_baseline": block["rf_baseline"],
            "dropped_smiles": len(prepared[name].dropped),
        }
    metrics_body["active_learning"] = {
        "budget_fraction": cfg.al_budget_fraction,
        "pool_size": int(len(pool)),
        "pool_excludes_test_split": True,
        "hit_recovery_mean": means,
        "hit_recovery_by_seed": recoveries,
        "best_strategy": best_name,
        "best_over_random": ratio,
        "kappa": cfg.al_kappa,
    }
    metrics_body["learning_curve"] = curve_frame.to_dict(orient="records")
    criteria = [
        build_rmse_criterion(
            "ESOL test RMSE (scaffold split)",
            metrics_body["esol"]["gnn_ensemble"]["test_rmse"],
            metrics_body["esol"]["rf_baseline"]["test_rmse"],
            0.90,
        ),
        build_rmse_criterion(
            "Lipophilicity test RMSE (scaffold split)",
            metrics_body["lipophilicity"]["gnn_ensemble"]["test_rmse"],
            metrics_body["lipophilicity"]["rf_baseline"]["test_rmse"],
            0.85,
        ),
        build_coverage_criterion(
            [
                metrics_body["esol"]["gnn_ensemble"]["coverage_90"],
                metrics_body["lipophilicity"]["gnn_ensemble"]["coverage_90"],
            ]
        ),
        build_hit_criterion(ratio),
    ]
    payload = {
        "project": "molprop-active-learning",
        "success_criteria": criteria,
        "metrics": metrics_body,
        "baseline": {
            "esol_rf_test_rmse": metrics_body["esol"]["rf_baseline"]["test_rmse"],
            "lipophilicity_rf_test_rmse": metrics_body["lipophilicity"]["rf_baseline"]["test_rmse"],
            "active_learning_random_hit_recovery": means["random"],
        },
        "splits": split_counts,
        "seeds": sorted(set(cfg.ensemble_seeds) | set(cfg.al_seeds)),
        "hardware": hardware,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    written = write_metrics(results_path / "metrics.json", payload)
    default_results = (project_root / "results").resolve()
    if (not cfg.quick) and results_path.resolve() == default_results:
        update_readme(project_root / "README.md", written)
        write_resume_notes(written, project_root / "RESUME_NOTES.md")
    for item in written["success_criteria"]:
        print(f"criterion {item['name']}: {item['value']:.4f} passed={item['passed']}", flush=True)
    print(f"runtime_seconds {written['runtime_seconds']}", flush=True)
    return written


def run_from_argv(argv: list[str] | None, project_root: Path) -> dict:
    args = parse_args(argv)
    return run_pipeline(
        project_root,
        quick=args.quick,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
