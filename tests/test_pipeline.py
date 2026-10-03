"""Reporting helpers and the quick end-to-end run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from molprop.active_learning import recovery_points
from molprop.pipeline import parse_args, run_from_argv
from molprop.plots import plot_recovery, plot_rmse_curve
from molprop.report import (
    assert_plain_language,
    collect_hardware,
    render_results_markdown,
    validate_metrics_payload,
    write_resume_notes,
)
import torch


def _sample_metrics() -> dict:
    return {
        "project": "molprop-active-learning",
        "success_criteria": [
            {
                "name": "ESOL test RMSE (scaffold split)",
                "value": 0.7,
                "op": "<=",
                "threshold": 0.9,
                "threshold_high": None,
                "passed": True,
            }
        ],
        "metrics": {
            "esol": {
                "gnn_ensemble": {"test_rmse": 0.6123, "coverage_90": 0.9012},
                "gnn_no_desc": {"test_rmse": 0.6401},
                "rf_baseline": {"test_rmse": 0.9804},
                "rf_fp_desc": {"test_rmse": 0.7011},
            },
            "lipophilicity": {
                "gnn_ensemble": {"test_rmse": 0.7015, "coverage_90": 0.8891},
                "gnn_no_desc": {"test_rmse": 0.7444},
                "rf_baseline": {"test_rmse": 0.8552},
                "rf_fp_desc": {"test_rmse": 0.7333},
            },
            "active_learning": {
                "hit_recovery_mean": {"random": 0.2, "greedy": 0.48, "ucb": 0.4, "ei": 0.3},
                "ratio_vs_random": {"random": 1.0, "greedy": 2.4, "ucb": 2.0, "ei": 1.5},
                "best_strategy": "greedy",
                "best_over_random": 2.4,
            },
        },
        "baseline": {"esol_rf_test_rmse": 0.9804},
        "splits": {"esol": {"train": 8, "val": 1, "test": 1}},
        "seeds": [0, 1, 2, 3, 4],
        "hardware": {"cpu": "cpu", "ram_gb": 8, "gpu": None, "python": "3.12.10", "os": "test"},
        "runtime_seconds": 12.5,
    }


def test_metrics_schema_and_resume_language(tmp_path: Path) -> None:
    metrics = _sample_metrics()
    validate_metrics_payload(metrics)
    notes = tmp_path / "RESUME_NOTES.md"
    write_resume_notes(metrics, notes)
    text = notes.read_text(encoding="utf-8")
    assert "0.612" in text
    assert "0.701" in text
    assert "0.980" in text
    assert "greedy" in text
    assert "2.40x" in text
    assert "0.901" in text
    assert "0.889" in text
    assert_plain_language(text)
    rendered = render_results_markdown(metrics)
    assert "0.6123" in rendered
    assert "0.7011" in rendered
    assert "| greedy | 0.4800 | 2.4000 |" in rendered
    hardware = collect_hardware(torch.device("cpu"))
    assert hardware["torch_threads"] >= 1
    assert hardware["python"]
    assert hardware["ram_gb"] > 0
    assert hardware["gpu"] is None


def test_parse_args_and_plots(tmp_path: Path) -> None:
    args = parse_args(["--quick", "--device", "cpu"])
    assert args.quick is True
    curve = pd.DataFrame(
        [
            {"dataset": "esol", "model": "gnn", "train_size": 10, "test_rmse": 1.0},
            {"dataset": "esol", "model": "rf_baseline", "train_size": 10, "test_rmse": 1.2},
            {"dataset": "esol", "model": "gnn", "train_size": 20, "test_rmse": 0.8},
            {"dataset": "esol", "model": "rf_baseline", "train_size": 20, "test_rmse": 1.0},
        ]
    )
    plot_path = tmp_path / "rmse.png"
    plot_rmse_curve(curve, plot_path)
    assert plot_path.stat().st_size > 0
    pool = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "smiles": ["C"] * 4,
            "y_true": [0, 1, 2, 3],
            "is_top5": ["false", "false", "false", "true"],
        }
    )
    trace = pd.DataFrame(
        [
            {"strategy": "random", "seed": 0, "round": 0, "n_labeled": 1, "selected_ids": "a"},
            {"strategy": "random", "seed": 0, "round": 1, "n_labeled": 2, "selected_ids": "b"},
            {"strategy": "greedy", "seed": 0, "round": 0, "n_labeled": 1, "selected_ids": "a"},
            {"strategy": "greedy", "seed": 0, "round": 1, "n_labeled": 2, "selected_ids": "d"},
        ]
    )
    points = recovery_points(trace, pool)
    recovery_path = tmp_path / "recovery.png"
    plot_recovery(points, recovery_path, 0.5)
    assert recovery_path.stat().st_size > 0


def test_cuda_request_without_gpu() -> None:
    from molprop.pipeline import resolve_device

    if not torch.cuda.is_available():
        try:
            resolve_device("cuda")
        except RuntimeError as exc:
            assert "CUDA" in str(exc)
        else:
            raise AssertionError("expected a CUDA error")
    assert resolve_device("cpu").type == "cpu"


def test_quick_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = run_from_argv(
        ["--quick", "--data-dir", str(root / "data"), "--results-dir", str(tmp_path / "results")],
        root,
    )
    results = tmp_path / "results"
    assert metrics["project"] == "molprop-active-learning"
    assert (results / "metrics.json").is_file()
    for name in ("esol", "lipophilicity"):
        split_path = results / "splits" / f"{name}_split.csv"
        gnn_path = results / "predictions" / f"{name}_gnn_ensemble_test.csv"
        rf_path = results / "predictions" / f"{name}_rf_baseline_test.csv"
        rf_desc_path = results / "predictions" / f"{name}_rf_fp_desc_test.csv"
        ablate_path = results / "predictions" / f"{name}_gnn_no_desc_test.csv"
        assert split_path.is_file()
        assert gnn_path.is_file()
        assert rf_path.is_file()
        assert rf_desc_path.is_file()
        assert ablate_path.is_file()
        split = pd.read_csv(split_path, keep_default_na=False)
        from molprop.split import assert_split_integrity

        assert_split_integrity(split)
        prediction = pd.read_csv(gnn_path)
        test_ids = set(split.loc[split["split"] == "test", "id"].astype(str))
        assert set(prediction["id"].astype(str)) == test_ids
        assert prediction["y_std"].notna().all()
        rf = pd.read_csv(rf_path)
        assert rf["y_std"].isna().all()
        rf_desc = pd.read_csv(rf_desc_path)
        assert set(rf_desc["id"].astype(str)) == test_ids
        assert rf_desc["y_std"].isna().all()
    assert metrics["hardware"]["torch_threads"] == 4
    pool = pd.read_csv(results / "active_learning" / "pool.csv")
    trace = pd.read_csv(results / "active_learning" / "trace.csv")
    lipo = pd.read_csv(results / "splits" / "lipophilicity_split.csv", keep_default_na=False)
    test_ids = set(lipo.loc[lipo["split"] == "test", "id"].astype(str))
    assert set(pool["id"].astype(str)).isdisjoint(test_ids)
    assert set(trace["strategy"]) == {"random", "greedy", "ucb", "ei"}
    assert (results / "plots" / "test_rmse_vs_train_size.png").is_file()
    assert (results / "plots" / "hit_recovery_vs_budget.png").is_file()


def test_quick_runs_match_except_runtime(tmp_path: Path) -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    first = run_from_argv(
        ["--quick", "--data-dir", str(data_dir), "--results-dir", str(tmp_path / "first")],
        root,
    )
    second = run_from_argv(
        ["--quick", "--data-dir", str(data_dir), "--results-dir", str(tmp_path / "second")],
        root,
    )
    assert first["hardware"]["torch_threads"] == 4
    assert second["hardware"]["torch_threads"] == 4
    left = json.loads((tmp_path / "first" / "metrics.json").read_text(encoding="utf-8"))
    right = json.loads((tmp_path / "second" / "metrics.json").read_text(encoding="utf-8"))
    left.pop("runtime_seconds")
    right.pop("runtime_seconds")
    assert left == right
