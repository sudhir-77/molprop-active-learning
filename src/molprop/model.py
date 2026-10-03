"""Directed message-passing network in plain PyTorch (no PyTorch Geometric)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from molprop.featurize import MolGraph


class DMPNN(nn.Module):
    """Edge-centered message passing with a Morgan fingerprint readout branch.

    Messages live on directed bonds. At each step the update for bond v->w sums
    incoming messages at v except the message that arrived on w->v, then adds that
    sum to the initial edge embedding. Atom states pool incoming edge messages and
    are combined with a projected fingerprint and a few RDKit 2D descriptors before the regression head.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden: int = 128,
        depth: int = 3,
        fp_dim: int = 2048,
        desc_dim: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.hidden = hidden
        self.depth = depth
        self.fp_dim = fp_dim
        self.desc_dim = desc_dim
        self.edge_init = nn.Linear(atom_dim + bond_dim, hidden)
        self.edge_update = nn.Linear(hidden, hidden)
        self.atom_out = nn.Linear(atom_dim + hidden, hidden)
        self.fp_proj = nn.Sequential(nn.Linear(fp_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.desc_skip = nn.Linear(desc_dim, 1)
        nn.init.zeros_(self.desc_skip.weight)
        nn.init.zeros_(self.desc_skip.bias)
        readout_in = hidden * 3 + desc_dim
        self.graph_norm = nn.LayerNorm(readout_in)
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        atom_x = batch["atom_x"]
        bond_x = batch["bond_x"]
        src = batch["src"]
        dst = batch["dst"]
        rev = batch["rev"]
        graph_index = batch["batch"]
        fp = batch["fp"]
        n_graphs = int(batch["n_graphs"])
        if fp.shape[1] != self.fp_dim:
            raise ValueError(f"expected fingerprint width {self.fp_dim}, got {fp.shape[1]}")
        messages = self._atom_messages(atom_x, bond_x, src, dst, rev)
        atom_h = torch.relu(self.atom_out(torch.cat([atom_x, messages], dim=-1)))
        sum_pool = atom_h.new_zeros(n_graphs, self.hidden)
        sum_pool.index_add_(0, graph_index, atom_h)
        counts = atom_h.new_zeros(n_graphs)
        counts.index_add_(
            0,
            graph_index,
            torch.ones(atom_h.size(0), device=atom_h.device, dtype=atom_h.dtype),
        )
        mean_pool = sum_pool / counts.clamp(min=1.0).unsqueeze(1)
        fp_h = self.fp_proj(fp)
        desc = batch["desc"]
        if desc.shape[1] != self.desc_dim:
            raise ValueError(f"expected descriptor width {self.desc_dim}, got {desc.shape[1]}")
        graph_h = self.graph_norm(torch.cat([sum_pool, mean_pool, fp_h, desc], dim=-1))
        return self.readout(graph_h).squeeze(-1) + self.desc_skip(desc).squeeze(-1)

    def _atom_messages(
        self,
        atom_x: torch.Tensor,
        bond_x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rev: torch.Tensor,
    ) -> torch.Tensor:
        if src.numel() == 0:
            return atom_x.new_zeros(atom_x.size(0), self.hidden)
        h0 = torch.relu(self.edge_init(torch.cat([atom_x[src], bond_x], dim=-1)))
        h = h0
        for _ in range(self.depth):
            incoming = h.new_zeros(atom_x.size(0), self.hidden)
            incoming.index_add_(0, dst, h)
            gathered = incoming[src] - h[rev]
            h = torch.relu(h0 + self.edge_update(gathered))
        atom_messages = h.new_zeros(atom_x.size(0), self.hidden)
        atom_messages.index_add_(0, dst, h)
        return atom_messages


def collate_graphs(graphs: list[MolGraph], device: torch.device) -> dict[str, torch.Tensor]:
    if not graphs:
        raise ValueError("cannot collate an empty batch")
    atom_blocks = []
    bond_blocks = []
    src_blocks = []
    dst_blocks = []
    rev_blocks = []
    batch_blocks = []
    fp_blocks = []
    desc_blocks = []
    atom_offset = 0
    edge_offset = 0
    for graph_id, graph in enumerate(graphs):
        n_atoms = int(graph.atom_x.shape[0])
        n_edges = int(graph.src.shape[0])
        atom_blocks.append(graph.atom_x)
        batch_blocks.append(np.full(n_atoms, graph_id, dtype=np.int64))
        fp_blocks.append(graph.fp)
        desc_blocks.append(graph.desc)
        if n_edges:
            bond_blocks.append(graph.bond_x)
            src_blocks.append(graph.src + atom_offset)
            dst_blocks.append(graph.dst + atom_offset)
            rev_blocks.append(graph.rev + edge_offset)
        atom_offset += n_atoms
        edge_offset += n_edges
    bond_dim = int(graphs[0].bond_x.shape[1])
    if bond_blocks:
        bond_x = np.concatenate(bond_blocks, axis=0)
        src = np.concatenate(src_blocks, axis=0)
        dst = np.concatenate(dst_blocks, axis=0)
        rev = np.concatenate(rev_blocks, axis=0)
    else:
        bond_x = np.zeros((0, bond_dim), dtype=np.float32)
        src = np.zeros((0,), dtype=np.int64)
        dst = np.zeros((0,), dtype=np.int64)
        rev = np.zeros((0,), dtype=np.int64)
    return {
        "atom_x": torch.tensor(np.concatenate(atom_blocks, axis=0), dtype=torch.float32, device=device),
        "bond_x": torch.tensor(bond_x, dtype=torch.float32, device=device),
        "src": torch.tensor(src, dtype=torch.long, device=device),
        "dst": torch.tensor(dst, dtype=torch.long, device=device),
        "rev": torch.tensor(rev, dtype=torch.long, device=device),
        "batch": torch.tensor(np.concatenate(batch_blocks, axis=0), dtype=torch.long, device=device),
        "fp": torch.tensor(np.stack(fp_blocks, axis=0), dtype=torch.float32, device=device),
        "desc": torch.tensor(np.stack(desc_blocks, axis=0), dtype=torch.float32, device=device),
        "n_graphs": len(graphs),
    }
