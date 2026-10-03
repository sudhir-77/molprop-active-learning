"""Directed message-passing network in plain PyTorch (no PyTorch Geometric)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from molprop.featurize import MolGraph


def segment_sum(values: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Sum rows into buckets without atomic adds.

    Rows are sorted by bucket, cumulatively summed, and written once per bucket.
    That matches `index_add` and stays valid under deterministic PyTorch mode.
    """
    squeezed = values.ndim == 1
    if squeezed:
        values = values.unsqueeze(1)
    n_rows, width = int(values.shape[0]), int(values.shape[1])
    if n_rows == 0 or n_segments == 0:
        out = values.new_zeros((n_segments, width))
        return out.squeeze(1) if squeezed else out
    index = index.to(dtype=torch.long)
    order = torch.argsort(index, stable=True)
    index_sorted = index.index_select(0, order)
    cumulative = torch.cumsum(values.index_select(0, order), dim=0)
    if n_rows == 1:
        end_pos = torch.zeros(1, dtype=torch.long, device=values.device)
    else:
        changed = index_sorted.narrow(0, 1, n_rows - 1) != index_sorted.narrow(0, 0, n_rows - 1)
        end_flags = torch.empty(n_rows, dtype=torch.bool, device=values.device)
        end_flags[:-1] = changed
        end_flags[-1] = True
        end_pos = torch.arange(n_rows, device=values.device)[end_flags]
    group_end = cumulative.index_select(0, end_pos)
    if group_end.shape[0] == 1:
        summed = group_end
    else:
        previous = torch.cat([group_end.new_zeros(1, width), group_end[:-1]], dim=0)
        summed = group_end - previous
    segment_ids = index_sorted.index_select(0, end_pos)
    out = values.new_zeros((n_segments, width)).index_copy(0, segment_ids, summed)
    return out.squeeze(1) if squeezed else out


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
        use_descriptors: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if use_descriptors and desc_dim < 1:
            raise ValueError("desc_dim must be positive when descriptors are used")
        self.hidden = hidden
        self.depth = depth
        self.fp_dim = fp_dim
        self.desc_dim = desc_dim if use_descriptors else 0
        self.use_descriptors = use_descriptors
        self.edge_init = nn.Linear(atom_dim + bond_dim, hidden)
        self.edge_update = nn.Linear(hidden, hidden)
        self.atom_out = nn.Linear(atom_dim + hidden, hidden)
        self.fp_proj = nn.Sequential(nn.Linear(fp_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        if use_descriptors:
            self.desc_skip = nn.Linear(desc_dim, 1)
            nn.init.zeros_(self.desc_skip.weight)
            nn.init.zeros_(self.desc_skip.bias)
        else:
            self.desc_skip = None
        readout_in = hidden * 3 + (desc_dim if use_descriptors else 0)
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
        sum_pool = segment_sum(atom_h, graph_index, n_graphs)
        counts = segment_sum(
            torch.ones(atom_h.size(0), device=atom_h.device, dtype=atom_h.dtype),
            graph_index,
            n_graphs,
        )
        mean_pool = sum_pool / counts.clamp(min=1.0).unsqueeze(1)
        fp_h = self.fp_proj(fp)
        parts = [sum_pool, mean_pool, fp_h]
        skip = atom_h.new_zeros(n_graphs)
        if self.use_descriptors:
            desc = batch["desc"]
            if desc.shape[1] != self.desc_dim:
                raise ValueError(f"expected descriptor width {self.desc_dim}, got {desc.shape[1]}")
            parts.append(desc)
            assert self.desc_skip is not None
            skip = self.desc_skip(desc).squeeze(-1)
        graph_h = self.graph_norm(torch.cat(parts, dim=-1))
        return self.readout(graph_h).squeeze(-1) + skip

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
        n_atoms = int(atom_x.size(0))
        for _ in range(self.depth):
            incoming = segment_sum(h, dst, n_atoms)
            gathered = incoming[src] - h[rev]
            h = torch.relu(h0 + self.edge_update(gathered))
        return segment_sum(h, dst, n_atoms)


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
