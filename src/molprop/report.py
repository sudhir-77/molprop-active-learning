"""Hardware, metrics JSON, README results, and resume notes."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molprop.metrics import interval_coverage, rmse


RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"
FORBIDDEN_WORDS = ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded")


def collect_hardware(device: torch.device) -> dict:
    ram_gb = 0.0
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            ram_gb = round(status.ullTotalPhys / (1024**3), 2)
        else:
            import os

            ram_gb = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3), 2)
    except Exception:
        ram_gb = 0.0
    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": ram_gb,
        "gpu": gpu,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def validate_metrics_payload(payload: dict) -> None:
    required = [
        "project",
        "success_criteria",
        "metrics",
        "baseline",
        "splits",
        "seeds",
        "hardware",
        "runtime_seconds",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"metrics.json missing {missing}")
    if not payload["success_criteria"]:
        raise ValueError("success_criteria is empty")
    for item in payload["success_criteria"]:
        for key in ("name", "value", "op", "threshold", "threshold_high", "passed"):
            if key not in item:
                raise ValueError(f"criterion missing {key}")
        if item["op"] not in {"<=", ">=", "<", ">", "between"}:
            raise ValueError(f"bad operator {item['op']}")
        if item["op"] == "between" and item["threshold_high"] is None:
            raise ValueError("between criterion needs threshold_high")
    for key in ("cpu", "ram_gb", "gpu", "python", "os"):
        if key not in payload["hardware"]:
            raise ValueError(f"hardware missing {key}")


def write_metrics(path: Path, payload: dict) -> dict:
    clean = to_builtin(payload)
    validate_metrics_payload(clean)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return clean


def write_split_csv(path: Path, frame: pd.DataFrame) -> None:
    columns = ["id", "smiles", "scaffold", "split"]
    output = frame[columns].copy()
    output["id"] = output["id"].astype(str)
    output["scaffold"] = output["scaffold"].fillna("").astype(str)
    output = output.sort_values("id").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, na_rep="")


def write_prediction_csv(path: Path, ids, y_true, y_pred, y_std) -> None:
    frame = pd.DataFrame(
        {
            "id": [str(item) for item in ids],
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )
    if y_std is None:
        frame["y_std"] = np.nan
    else:
        frame["y_std"] = np.asarray(y_std, dtype=np.float64)
    if len(frame) != len(set(frame["id"])):
        raise ValueError("prediction ids are not unique")
    frame = frame.sort_values("id").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def prediction_metrics(path: Path, with_coverage: bool) -> dict:
    frame = pd.read_csv(path)
    result = {"test_rmse": rmse(frame["y_true"], frame["y_pred"])}
    if with_coverage:
        if frame["y_std"].isna().any():
            raise ValueError(f"{path.name} is missing ensemble standard deviations")
        result["coverage_90"] = interval_coverage(frame["y_true"], frame["y_pred"], frame["y_std"])
    elif frame["y_std"].notna().any():
        raise ValueError(f"{path.name} should leave y_std empty for the random forest")
    return result


def assert_plain_language(text: str) -> None:
    lowered = text.lower()
    for word in FORBIDDEN_WORDS:
        if word in lowered:
            raise ValueError(f"resume notes contain '{word}'")
    if "\u2014" in text:
        raise ValueError("resume notes contain an em dash")


def render_results_markdown(metrics: dict) -> str:
    failed = [item["name"] for item in metrics["success_criteria"] if not item["passed"]]
    if failed:
        intro = "Some success criteria failed: " + "; ".join(failed) + "."
    else:
        intro = "All success criteria recorded in `results/metrics.json` passed."
    lines = [
        intro,
        "",
        "| Criterion | Value | Rule | Passed |",
        "|---|---:|---|---|",
    ]
    for item in metrics["success_criteria"]:
        if item["op"] == "between":
            rule = f"between {item['threshold']} and {item['threshold_high']}"
        else:
            rule = f"{item['op']} {item['threshold']}"
        lines.append(f"| {item['name']} | {item['value']:.4f} | {rule} | {item['passed']} |")
    lines.extend(
        [
            "",
            "| Dataset | GNN ensemble test RMSE | RF baseline test RMSE | 90% interval coverage |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in ("esol", "lipophilicity"):
        block = metrics["metrics"][dataset]
        lines.append(
            "| {name} | {gnn:.4f} | {rf:.4f} | {cov:.4f} |".format(
                name=dataset,
                gnn=block["gnn_ensemble"]["test_rmse"],
                rf=block["rf_baseline"]["test_rmse"],
                cov=block["gnn_ensemble"]["coverage_90"],
            )
        )
    active = metrics["metrics"]["active_learning"]
    recovery = ", ".join(f"{name} {value:.4f}" for name, value in active["hit_recovery_mean"].items())
    lines.extend(
        [
            "",
            (
                "Active learning at a 20% budget, mean over seeds: "
                f"{recovery}. Best strategy {active['best_strategy']} is "
                f"{active['best_over_random']:.4f} times random."
            ),
            "",
            f"Runtime {metrics['runtime_seconds']:.1f} seconds.",
            "",
            "Plots: `results/plots/test_rmse_vs_train_size.png`, `results/plots/hit_recovery_vs_budget.png`.",
        ]
    )
    return "\n".join(lines) + "\n"


def update_readme(readme_path: Path, metrics: dict) -> None:
    block = render_results_markdown(metrics)
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    replacement = f"{RESULTS_START}\n{block}{RESULTS_END}"
    if RESULTS_START in text and RESULTS_END in text:
        before, rest = text.split(RESULTS_START, 1)
        _, after = rest.split(RESULTS_END, 1)
        text = before + replacement + after
    else:
        text = text.rstrip() + "\n\n" + replacement + "\n"
    readme_path.write_text(text, encoding="utf-8")


def write_resume_notes(metrics: dict, path: Path) -> None:
    esol = metrics["metrics"]["esol"]
    lipo = metrics["metrics"]["lipophilicity"]
    active = metrics["metrics"]["active_learning"]
    esol_rmse = esol["gnn_ensemble"]["test_rmse"]
    esol_rf = esol["rf_baseline"]["test_rmse"]
    lipo_rmse = lipo["gnn_ensemble"]["test_rmse"]
    lipo_rf = lipo["rf_baseline"]["test_rmse"]
    esol_cov = esol["gnn_ensemble"]["coverage_90"]
    lipo_cov = lipo["gnn_ensemble"]["coverage_90"]
    ratio = active["best_over_random"]
    best = active["best_strategy"]
    text = f"""# Resume notes

## Project summary

This project trains a message-passing graph neural network in PyTorch to predict aqueous solubility and lipophilicity from molecular graphs, and estimates uncertainty with a five-model deep ensemble. A pool-based active-learning loop then chooses which molecules to evaluate next under random, greedy, upper-confidence-bound, and expected-improvement acquisition.

## GitHub description

Message-passing GNN ensemble for molecular solubility and lipophilicity prediction with pool-based active learning.

## Skills and tools

Python, PyTorch, RDKit, scikit-learn, pandas, matplotlib, pytest, graph neural networks, active learning, cheminformatics, uncertainty quantification

## Draft bullets

- Built a message-passing GNN ensemble in PyTorch for molecular property prediction, reaching {esol_rmse:.3f} RMSE on scaffold-split ESOL vs. {esol_rf:.3f} for a fingerprint random forest.
- Predicted lipophilicity on a Bemis-Murcko scaffold split with ensemble test RMSE {lipo_rmse:.3f} vs. {lipo_rf:.3f} for the fingerprint random forest.
- Ran an active-learning screening loop whose best strategy ({best}) recovered {ratio:.2f}x more top-5% molecules than random selection at a 20% evaluation budget.
- Quantified prediction uncertainty with a 5-model deep ensemble (ESOL 90% interval coverage {esol_cov:.3f}, lipophilicity {lipo_cov:.3f}) to choose which molecules to evaluate next.
"""
    assert_plain_language(text)
    path.write_text(text, encoding="utf-8")
