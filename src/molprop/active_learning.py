"""Pool-based active learning.

The oracle reveals a label only after that molecule has been selected. Acquisition
functions see model scores for unlabeled molecules and the labels of the evaluated
set, never the labels of molecules still in the pool.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from molprop.metrics import fraction_count, mark_top_fraction

STRATEGY_ORDER = ("random", "greedy", "ucb", "ei")


def norm_cdf(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    out = np.empty_like(values)
    scale = math.sqrt(2.0)
    for index, value in enumerate(values.ravel()):
        out.ravel()[index] = 0.5 * (1.0 + math.erf(float(value) / scale))
    return out


def norm_pdf(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * values * values) / math.sqrt(2.0 * math.pi)


def expected_improvement(mu, sigma, best: float) -> np.ndarray:
    """Expected improvement for a maximization objective."""
    mean = np.asarray(mu, dtype=np.float64)
    std = np.asarray(sigma, dtype=np.float64)
    improvement = np.empty_like(mean)
    tiny = std < 1e-12
    improvement[tiny] = np.maximum(mean[tiny] - best, 0.0)
    if np.any(~tiny):
        sig = std[~tiny]
        z = (mean[~tiny] - best) / sig
        improvement[~tiny] = (mean[~tiny] - best) * norm_cdf(z) + sig * norm_pdf(z)
    return improvement


def select_top_k(ids, scores, k: int) -> list[str]:
    labels = [str(item) for item in ids]
    values = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(values):
        raise ValueError("ids and scores must align")
    if k <= 0:
        raise ValueError("k must be positive")
    order = sorted(range(len(labels)), key=lambda index: (-float(values[index]), labels[index]))
    return [labels[index] for index in order[: min(k, len(labels))]]


def acquire(strategy: str, ids, mu, std, y_best: float, k: int, kappa: float) -> list[str]:
    if strategy == "greedy":
        scores = np.asarray(mu, dtype=np.float64)
    elif strategy == "ucb":
        scores = np.asarray(mu, dtype=np.float64) + kappa * np.asarray(std, dtype=np.float64)
    elif strategy == "ei":
        scores = expected_improvement(mu, std, y_best)
    else:
        raise ValueError(f"unknown acquisition strategy {strategy}")
    return select_top_k(ids, scores, k)


class Oracle:
    """Serves assay labels only for molecules that have already been evaluated."""

    def __init__(self, ids, y) -> None:
        self._y = {str(mol_id): float(value) for mol_id, value in zip(ids, y)}
        self.revealed: set[str] = set()

    def reveal(self, ids) -> None:
        unknown = [str(mol_id) for mol_id in ids if str(mol_id) not in self._y]
        if unknown:
            raise KeyError(f"unknown molecule ids: {unknown[:5]}")
        self.revealed.update(str(mol_id) for mol_id in ids)

    def labels(self, ids) -> np.ndarray:
        missing = [str(mol_id) for mol_id in ids if str(mol_id) not in self.revealed]
        if missing:
            raise PermissionError(
                f"Oracle refused labels for {len(missing)} molecules that have not been evaluated"
            )
        return np.asarray([self._y[str(mol_id)] for mol_id in ids], dtype=np.float64)


FitPredict = Callable[[list[str], np.ndarray, list[str], dict | None], tuple[np.ndarray, np.ndarray, dict]]


def batch_sizes_for(n_pool: int, init_fraction: float, budget_fraction: float, n_rounds: int) -> tuple[int, int, list[int]]:
    init_n = fraction_count(n_pool, init_fraction)
    budget_n = fraction_count(n_pool, budget_fraction)
    if budget_n <= init_n:
        raise ValueError("budget must leave molecules to acquire after the initial set")
    if budget_n > n_pool:
        raise ValueError("budget exceeds the pool")
    remaining = budget_n - init_n
    rounds = max(1, n_rounds)
    if remaining < rounds:
        rounds = remaining
    base, extra = divmod(remaining, rounds)
    sizes = [base + (1 if index < extra else 0) for index in range(rounds)]
    if any(size <= 0 for size in sizes):
        raise RuntimeError("batch sizes must be positive")
    return init_n, budget_n, sizes


def run_pool_simulation(
    pool_ids,
    smiles,
    y_true,
    *,
    seeds: list[int] | tuple[int, ...],
    strategies: list[str] | tuple[str, ...] = ("random", "greedy", "ucb", "ei"),
    init_fraction: float = 0.10,
    budget_fraction: float = 0.20,
    n_rounds: int = 2,
    kappa: float = 1.0,
    fit_predict: FitPredict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate screening. `fit_predict` receives labels only for the evaluated set."""
    ids = [str(mol_id) for mol_id in pool_ids]
    if any(";" in mol_id for mol_id in ids):
        raise ValueError("molecule ids cannot contain ';'")
    y = np.asarray(y_true, dtype=np.float64)
    if len(ids) != len(y) or len(ids) != len(list(smiles)):
        raise ValueError("pool ids, smiles, and labels must align")
    init_n, budget_n, sizes = batch_sizes_for(len(ids), init_fraction, budget_fraction, n_rounds)
    top_flags = mark_top_fraction(ids, y, 0.05)
    pool = pd.DataFrame(
        {
            "id": ids,
            "smiles": [str(item) for item in smiles],
            "y_true": y,
            "is_top5": ["true" if flag else "false" for flag in top_flags],
        }
    )
    rows: list[dict] = []
    id_to_pos = {mol_id: index for index, mol_id in enumerate(ids)}
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(ids))
        init_ids = [ids[int(index)] for index in perm[:init_n]]
        random_queue = [ids[int(index)] for index in perm[init_n:]]
        oracles = {strategy: Oracle(ids, y) for strategy in strategies}
        labeled = {strategy: list(init_ids) for strategy in strategies}
        labeled_set = {strategy: set(init_ids) for strategy in strategies}
        for strategy in strategies:
            oracles[strategy].reveal(init_ids)
            rows.append(
                {
                    "strategy": strategy,
                    "seed": int(seed),
                    "round": 0,
                    "n_labeled": len(init_ids),
                    "selected_ids": ";".join(init_ids),
                }
            )
        informed = [strategy for strategy in strategies if strategy != "random"]
        shared_state = None
        shared_mu = None
        shared_std = None
        unlabeled0 = [mol_id for mol_id in ids if mol_id not in labeled_set[informed[0]]] if informed else []
        if informed:
            init_y = oracles[informed[0]].labels(init_ids)
            shared_mu, shared_std, shared_state = fit_predict(list(init_ids), init_y, unlabeled0, None)
            if len(shared_mu) != len(unlabeled0) or len(shared_std) != len(unlabeled0):
                raise RuntimeError("fit_predict returned a score vector of the wrong length")
        strategy_state = {strategy: None if shared_state is None else _copy_state(shared_state) for strategy in informed}
        for round_index, batch_size in enumerate(sizes, start=1):
            for strategy in strategies:
                if strategy == "random":
                    picked = random_queue[:batch_size]
                    random_queue = random_queue[batch_size:]
                else:
                    unlabeled = [mol_id for mol_id in ids if mol_id not in labeled_set[strategy]]
                    if round_index == 1:
                        position = {mol_id: index for index, mol_id in enumerate(unlabeled0)}
                        order = [position[mol_id] for mol_id in unlabeled]
                        mu = np.asarray(shared_mu, dtype=np.float64)[order]
                        std = np.asarray(shared_std, dtype=np.float64)[order]
                        cand_ids = unlabeled
                    else:
                        labeled_y = oracles[strategy].labels(labeled[strategy])
                        mu, std, strategy_state[strategy] = fit_predict(
                            list(labeled[strategy]),
                            labeled_y,
                            unlabeled,
                            strategy_state[strategy],
                        )
                        cand_ids = unlabeled
                    y_best = float(np.max(oracles[strategy].labels(labeled[strategy])))
                    picked = acquire(strategy, cand_ids, mu, std, y_best, batch_size, kappa)
                if len(picked) != batch_size:
                    raise RuntimeError(f"{strategy} selected {len(picked)} molecules, expected {batch_size}")
                if any(mol_id in labeled_set[strategy] for mol_id in picked):
                    raise RuntimeError(f"{strategy} re-selected an evaluated molecule")
                oracles[strategy].reveal(picked)
                labeled[strategy].extend(picked)
                labeled_set[strategy].update(picked)
                rows.append(
                    {
                        "strategy": strategy,
                        "seed": int(seed),
                        "round": round_index,
                        "n_labeled": len(labeled[strategy]),
                        "selected_ids": ";".join(picked),
                    }
                )
        for strategy in strategies:
            if len(labeled[strategy]) != budget_n:
                raise RuntimeError(
                    f"{strategy} seed {seed} labeled {len(labeled[strategy])}, budget is {budget_n}"
                )
    trace = pd.DataFrame(rows).sort_values(["strategy", "seed", "round"]).reset_index(drop=True)
    validate_trace(trace, pool)
    # Touch the map so a typo in id alignment fails during development of callers.
    if any(mol_id not in id_to_pos for mol_id in pool["id"]):
        raise RuntimeError("pool ids drifted")
    return trace, pool


def _copy_state(state: dict) -> dict:
    import copy

    return copy.deepcopy(state)


def parse_selected_ids(value) -> list[str]:
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    if text == "" or text.lower() == "nan":
        return []
    return [item for item in text.split(";") if item]


def validate_trace(trace: pd.DataFrame, pool: pd.DataFrame) -> None:
    pool_ids = set(pool["id"].astype(str))
    required = {"strategy", "seed", "round", "n_labeled", "selected_ids"}
    if not required.issubset(trace.columns):
        raise ValueError(f"trace missing columns {required - set(trace.columns)}")
    for (_, _), group in trace.groupby(["strategy", "seed"], sort=False):
        ordered = group.sort_values("round")
        seen: list[str] = []
        expected_round = 0
        for row in ordered.itertuples(index=False):
            if int(row.round) != expected_round:
                raise ValueError("trace rounds are not contiguous from 0")
            expected_round += 1
            picked = parse_selected_ids(row.selected_ids)
            if len(picked) != len(set(picked)):
                raise ValueError("a round repeats a molecule")
            if any(mol_id not in pool_ids for mol_id in picked):
                raise ValueError("trace contains an id outside the pool")
            if any(mol_id in seen for mol_id in picked):
                raise ValueError("a molecule was selected in more than one round")
            seen.extend(picked)
            if len(seen) != int(row.n_labeled):
                raise ValueError("n_labeled does not match the cumulative selected ids")


def _top_id_set(pool: pd.DataFrame) -> set[str]:
    flags = pool["is_top5"]
    if flags.dtype == bool:
        mask = flags.to_numpy(dtype=bool)
    else:
        mask = flags.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    return set(pool["id"].astype(str).to_numpy()[mask])


def recovery_points(trace: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Cumulative hit recovery after every round, for plotting."""
    top_ids = _top_id_set(pool)
    if not top_ids:
        raise ValueError("pool has no top-5% molecules")
    n_pool = len(pool)
    rows: list[dict] = []
    for (strategy, seed), group in trace.groupby(["strategy", "seed"], sort=False):
        chosen: list[str] = []
        for row in group.sort_values("round").itertuples(index=False):
            chosen.extend(parse_selected_ids(row.selected_ids))
            rows.append(
                {
                    "strategy": str(strategy),
                    "seed": int(seed),
                    "round": int(row.round),
                    "n_labeled": int(row.n_labeled),
                    "fraction_labeled": int(row.n_labeled) / n_pool,
                    "hit_recovery": len(set(chosen) & top_ids) / len(top_ids),
                }
            )
    return pd.DataFrame(rows)


def recoveries_at_budget(
    trace: pd.DataFrame,
    pool: pd.DataFrame,
    budget_fraction: float = 0.20,
) -> dict[str, dict[int, float]]:
    """Fraction of is_top5 molecules evaluated once the labeled count reaches the budget."""
    top_ids = _top_id_set(pool)
    if not top_ids:
        raise ValueError("pool has no top-5% molecules")
    budget_n = fraction_count(len(pool), budget_fraction)
    recoveries: dict[str, dict[int, float]] = {}
    for (strategy, seed), group in trace.groupby(["strategy", "seed"], sort=False):
        chosen: list[str] = []
        reached = False
        for row in group.sort_values("round").itertuples(index=False):
            chosen.extend(parse_selected_ids(row.selected_ids))
            if int(row.n_labeled) >= budget_n:
                reached = True
                break
        if not reached:
            raise ValueError(f"{strategy} seed {seed} never reaches the budget")
        hit_count = len(set(chosen) & top_ids)
        recoveries.setdefault(str(strategy), {})[int(seed)] = hit_count / len(top_ids)
    ordered: dict[str, dict[int, float]] = {}
    for strategy in _strategy_order(recoveries):
        per_seed = recoveries[strategy]
        ordered[strategy] = {seed: per_seed[seed] for seed in sorted(per_seed)}
    return ordered


def _strategy_order(mapping: dict) -> list[str]:
    present = {str(name) for name in mapping}
    ordered = [name for name in STRATEGY_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def ratios_vs_random(means: dict[str, float]) -> dict[str, float]:
    if "random" not in means:
        raise ValueError("random acquisition is required")
    random_mean = float(means["random"])
    ratios: dict[str, float] = {}
    for name in _strategy_order(means):
        if name == "random":
            ratios[name] = 1.0
        elif random_mean <= 1e-12:
            ratios[name] = float("inf")
        else:
            ratios[name] = float(means[name] / random_mean)
    return ratios


def best_strategy_ratio(recoveries: dict[str, dict[int, float]]) -> tuple[str, float, dict[str, float]]:
    means = {}
    for strategy in _strategy_order(recoveries):
        per_seed = recoveries[strategy]
        values = [per_seed[seed] for seed in sorted(per_seed)]
        means[strategy] = float(np.mean(values))
    if "random" not in means:
        raise ValueError("random acquisition is required")
    informed = [name for name in ("greedy", "ucb", "ei") if name in means]
    if not informed:
        raise ValueError("at least one informed strategy is required")
    priority = {"greedy": 2, "ucb": 1, "ei": 0}
    best = max(informed, key=lambda name: (means[name], priority[name]))
    ratio = ratios_vs_random(means)[best]
    return best, float(ratio), means
