"""Plots written under results/plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_rmse_curve(frame: pd.DataFrame, path: Path) -> None:
    datasets = list(dict.fromkeys(frame["dataset"].astype(str).tolist()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 4.0), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        subset = frame[frame["dataset"] == dataset]
        for model, style in (("gnn", "-o"), ("rf_baseline", "--s")):
            part = subset[subset["model"] == model].sort_values("train_size")
            if part.empty:
                continue
            axis.plot(part["train_size"], part["test_rmse"], style, label=model)
        axis.set_title(dataset)
        axis.set_xlabel("Training molecules")
        axis.set_ylabel("Test RMSE")
        axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_recovery(points: pd.DataFrame, path: Path, budget_fraction: float) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for strategy in ("random", "greedy", "ucb", "ei"):
        subset = points[points["strategy"] == strategy]
        if subset.empty:
            continue
        grouped = subset.groupby("round", as_index=False).agg(
            fraction_labeled=("fraction_labeled", "mean"),
            hit_recovery=("hit_recovery", "mean"),
        )
        axis.plot(grouped["fraction_labeled"], grouped["hit_recovery"], marker="o", label=strategy)
    axis.axvline(budget_fraction, color="0.45", linestyle=":", linewidth=1)
    axis.set_xlabel("Fraction of the pool evaluated")
    axis.set_ylabel("Fraction of top-5% molecules found")
    axis.set_title("Top-5% hit recovery vs evaluation budget")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
