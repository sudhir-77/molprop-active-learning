# Active-learning virtual screening with a graph neural network

This project trains a message-passing neural net on molecular graphs to predict aqueous solubility (ESOL) and lipophilicity, estimates uncertainty with a deep ensemble, and runs a pool-based active-learning loop that decides which molecules to evaluate next. Measured assay values stand in for an expensive physics oracle. Random, greedy, upper-confidence-bound, and expected-improvement acquisition are compared by how many of the top 5% of lipophilicity values each finds inside a fixed evaluation budget.

## Setup

From this folder, on Python 3.11 or 3.12:

```text
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

PyTorch is pinned to the CPU wheel. The first line of `requirements.txt` points pip at `https://download.pytorch.org/whl/cpu`.

## How to run

```text
python run.py
```

Optional flags:

- `python run.py --quick` runs the same steps on a small subset with one epoch. It is for tests and smoke checks and finishes in a few minutes. It does not target the numeric success criteria.
- `python run.py --device cuda` uses a GPU when one is available. The default is CPU.
- `--data-dir` and `--results-dir` override the default `data/` and `results/` folders.

`python run.py` downloads the CSVs into `data/` (cached, checksum-checked), builds a Bemis-Murcko scaffold split, trains the ensemble, a descriptor ablation, and two random forests, writes predictions, runs the active-learning simulator, and writes `results/metrics.json` plus the plots below. No shell scripts or notebooks are required.

## Architecture

```mermaid
flowchart LR
  A[SMILES CSV] --> B[RDKit featurizer: atom and bond features, Morgan FP, 10 descriptors]
  B --> C[Directed MPNN ensemble]
  B --> C2[MPNN ablation without descriptors]
  B --> D[Random forest on fingerprint plus descriptors]
  B --> D2[Random forest on fingerprint only]
  C --> E[Validation-scaled mean and variance]
  E --> F[Active learning: random, greedy, UCB, expected improvement]
  F --> G[results/metrics.json and learning-curve plots]
```

## Design decisions

- The network is a directed message-passing model written in plain PyTorch. PyTorch Geometric is not used, so the CPU install stays simple on Windows.
- A 2048-bit Morgan fingerprint is projected and concatenated with the pooled atom states. Ten RDKit 2D descriptors (including molecular weight, polar surface area, and Crippen logP) are concatenated as well, with a zero-initialized linear skip so the graph model starts as the whole predictor and can learn a descriptor correction.
- The primary baseline is a random forest on the same fingerprint plus those ten descriptors (`rf_fp_desc`). A fingerprint-only forest is kept as a secondary baseline. A second GNN ensemble trained without descriptors is reported as an ablation and is not part of the pass/fail rule. The RMSE criteria compare the descriptor GNN with `rf_fp_desc`. Nothing is chosen on the test split.
- Five models trained from different seeds form a deep ensemble. A single scale on the ensemble standard deviation is chosen so that 90% intervals cover about 90% of the validation molecules. That scale is not fit on the test set.
- The scaffold split puts every Bemis-Murcko scaffold entirely into train, validation, or test. Early stopping uses validation RMSE. The test split is scored once after the models are frozen.
- The active-learning pool is the lipophilicity train and validation molecules. Test-scaffold labels are not placed in the oracle, so they cannot be used to train the screening models or to choose the next batch.
- Top 5% means the highest experimental logD values in that pool. Ties break toward the smaller molecule id. Greedy, UCB (mean plus kappa times std, kappa 1), and expected improvement all maximize logD. The reported multiplier is the best of those three strategies divided by random, averaged over the same five seeds and the same initial labeled sets.
- Later active-learning rounds warm-start from the previous weights and keep the target standardization from the first fit, so a short update stays on the same scale.

## Limitations

- The oracle is a public assay label, not docking or free-energy simulation.
- Molecules are 2D graphs. Stereochemistry enters only as an atom feature, and conformers are not modeled.
- Ensemble variance is a heuristic. The validation scale corrects the average interval width, not every local bias.
- The model is sized to train on CPU within the project time budget, so it is smaller than a production chemprop run.
- Scaffold splits still share broad chemical series across the cut, and a single scalar uncertainty calibration can miss dataset shift.
- Crippen logP is one of the ten descriptors. It is closely related to both solubility and lipophilicity, so the fingerprint-plus-descriptor forest is a strong baseline. The GNN has to beat that forest on the untouched test split.

## Data and licenses

- **ESOL.** Aqueous solubility for 1,128 molecules, published by John S. Delaney, "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure," *Journal of Chemical Information and Computer Sciences* 2004, 44 (3), 1000-1004. Target: measured log solubility in mol/L. The table used here is the MoleculeNet / DeepChem file [delaney-processed.csv](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv).
- **Lipophilicity.** Experimental logD (pH 7.4) for 4,200 molecules, curated from ChEMBL. ChEMBL is released under the Creative Commons Attribution-ShareAlike 3.0 Unported license (CC BY-SA 3.0). The table used here is the MoleculeNet / DeepChem file [Lipophilicity.csv](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv).
- **Distribution.** Both CSVs are the MoleculeNet benchmark copies distributed by DeepChem (Wu et al., "MoleculeNet: a benchmark for molecular machine learning," *Chemical Science* 2018, 9, 513-530). They are cached under `data/` and checked against a SHA-256 checksum and a row count before training. The scaffold split follows the Bemis-Murcko protocol used by DeepChem.

## Tests

```text
python -m pytest --cov=molprop --cov-report=term-missing
```

## Results

<!-- RESULTS_START -->
All success criteria recorded in `results/metrics.json` passed.

| Criterion | Value | Rule | Passed |
|---|---:|---|---|
| ESOL test RMSE (scaffold split) | 0.8721 | < 0.8905935803361443 | True |
| Lipophilicity test RMSE (scaffold split) | 0.7868 | < 0.7906436498241156 | True |
| 90% prediction-interval coverage | 0.9015 | between 0.8 and 0.97 | True |
| Top-5% hit recovery at 20% budget (Lipophilicity pool) | 2.0580 | >= 1.5 | True |

| Dataset | GNN ensemble | GNN without descriptors | RF fingerprint + descriptors | RF fingerprint | 90% interval coverage |
|---|---:|---:|---:|---:|---:|
| esol | 0.8721 | 0.9440 | 0.8906 | 1.8213 | 0.8673 |
| lipophilicity | 0.7868 | 0.8192 | 0.7906 | 0.9422 | 0.9357 |

Top-5% hit recovery on the lipophilicity pool at a 20% evaluation budget, mean over seeds. The ratio is that recovery divided by random acquisition.

| Strategy | Top-5% hit recovery | Ratio vs random |
|---|---:|---:|
| random | 0.2190 | 1.0000 |
| greedy | 0.4508 | 2.0580 |
| ucb | 0.4497 | 2.0531 |
| ei | 0.3471 | 1.5845 |

Best strategy greedy is 2.0580 times random.

Runtime 2178.0 seconds.

Plots: `results/plots/test_rmse_vs_train_size.png`, `results/plots/hit_recovery_vs_budget.png`.
<!-- RESULTS_END -->
