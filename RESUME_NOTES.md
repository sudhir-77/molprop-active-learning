# Resume notes

## Project summary

This project trains a message-passing graph neural network in PyTorch to predict aqueous solubility and lipophilicity from molecular graphs, and estimates uncertainty with a five-model deep ensemble. A pool-based active-learning loop then chooses which molecules to evaluate next under random, greedy, upper-confidence-bound, and expected-improvement acquisition.

## GitHub description

Message-passing GNN ensemble for molecular solubility and lipophilicity prediction with pool-based active learning.

## Skills and tools

Python, PyTorch, RDKit, scikit-learn, pandas, matplotlib, pytest, graph neural networks, active learning, cheminformatics, uncertainty quantification

## Draft bullets

- Built a message-passing GNN ensemble in PyTorch for molecular property prediction, reaching 0.872 RMSE on scaffold-split ESOL vs. 0.891 for a random forest on the same Morgan fingerprints and RDKit descriptors (1.821 for fingerprints alone).
- Predicted lipophilicity on a Bemis-Murcko scaffold split with ensemble test RMSE 0.787 vs. 0.791 for the fingerprint-plus-descriptor random forest.
- Ran an active-learning screening loop whose greedy acquisition recovered 2.06x more top-5% molecules than random selection at a 20% evaluation budget.
- Quantified prediction uncertainty with a 5-model deep ensemble (ESOL 90% interval coverage 0.867, lipophilicity 0.936) to choose which molecules to evaluate next.
