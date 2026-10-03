"""RDKit atom, bond, and Morgan fingerprint featurization."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

FEATURE_VERSION = 2

ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [0, 1, 2, 3]
HYDROGEN_COUNTS = [0, 1, 2, 3, 4]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


@dataclass
class MolGraph:
    mol_id: str
    smiles: str
    scaffold: str
    atom_x: np.ndarray
    bond_x: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    rev: np.ndarray
    fp: np.ndarray
    desc: np.ndarray


def rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """A small set of RDKit 2D descriptors, scaled so a linear layer can use them."""
    return np.asarray(
        [
            Descriptors.MolWt(mol) * 0.01,
            Descriptors.TPSA(mol) * 0.01,
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
            float(Crippen.MolLogP(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(mol.GetNumHeavyAtoms()),
        ],
        dtype=np.float32,
    )


def one_hot(value, choices: list) -> list[float]:
    vector = [0.0] * (len(choices) + 1)
    try:
        index = choices.index(value)
    except ValueError:
        index = len(choices)
    vector[index] = 1.0
    return vector


def atom_features(atom: Chem.Atom) -> list[float]:
    features: list[float] = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOMIC_NUMBERS))
    features.extend(one_hot(atom.GetTotalDegree(), DEGREES))
    features.extend(one_hot(atom.GetFormalCharge(), FORMAL_CHARGES))
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend(one_hot(int(atom.GetChiralTag()), CHIRAL_TAGS))
    features.extend(one_hot(atom.GetTotalNumHs(), HYDROGEN_COUNTS))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    features.append(float(atom.GetMass()) * 0.01)
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    features: list[float] = []
    features.extend(one_hot(bond.GetBondType(), BOND_TYPES))
    features.extend(one_hot(bond.GetStereo(), BOND_STEREOS))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    return features


def murcko_scaffold(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""
    return scaffold or ""


def morgan_generator(n_bits: int, radius: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_fingerprint(mol: Chem.Mol, generator, n_bits: int) -> np.ndarray:
    if hasattr(generator, "GetFingerprintAsNumPy"):
        fp = np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.float32)
        if fp.shape != (n_bits,):
            raise ValueError(f"unexpected fingerprint shape {fp.shape}")
        return fp
    bitvect = generator.GetFingerprint(mol)
    array = np.zeros((n_bits,), dtype=np.float32)
    from rdkit.Chem import DataStructs

    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def featurize_mol(mol: Chem.Mol, mol_id: str, smiles: str, generator, n_bits: int) -> MolGraph:
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    if not atoms:
        raise ValueError(f"{mol_id} has no atoms")
    bond_rows: list[list[float]] = []
    src: list[int] = []
    dst: list[int] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        src.extend((begin, end))
        dst.extend((end, begin))
        bond_rows.append(features)
        bond_rows.append(features)
    n_edges = len(src)
    rev = np.empty(n_edges, dtype=np.int64)
    for index in range(0, n_edges, 2):
        rev[index] = index + 1
        rev[index + 1] = index
    atom_x = np.asarray(atoms, dtype=np.float32)
    if bond_rows:
        bond_x = np.asarray(bond_rows, dtype=np.float32)
    else:
        bond_dim = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
        bond_x = np.zeros((0, bond_dim), dtype=np.float32)
    graph = MolGraph(
        mol_id=str(mol_id),
        smiles=smiles,
        scaffold=murcko_scaffold(mol),
        atom_x=atom_x,
        bond_x=bond_x,
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        rev=rev,
        fp=morgan_fingerprint(mol, generator, n_bits),
        desc=rdkit_descriptors(mol),
    )
    if n_edges:
        if not np.array_equal(graph.src[graph.rev], graph.dst):
            raise ValueError(f"reverse-edge map failed for {mol_id}")
        if not np.array_equal(graph.dst[graph.rev], graph.src):
            raise ValueError(f"reverse-edge map failed for {mol_id}")
    return graph


def featurize_table(
    table: pd.DataFrame,
    fp_bits: int,
    fp_radius: int,
    cache_path: Path | None = None,
) -> tuple[list[MolGraph], pd.DataFrame, list[str]]:
    """Featurize a table of id/smiles/y. Invalid SMILES are dropped."""
    ids = table["id"].astype(str).tolist()
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            payload.get("version") == FEATURE_VERSION
            and payload.get("fp_bits") == fp_bits
            and payload.get("fp_radius") == fp_radius
            and payload.get("source_ids") == ids
        ):
            return payload["graphs"], payload["frame"], payload["dropped"]

    generator = morgan_generator(fp_bits, fp_radius)
    graphs: list[MolGraph] = []
    kept_rows: list[dict] = []
    dropped: list[str] = []
    total = len(table)
    for row_index, row in enumerate(table.itertuples(index=False), start=1):
        if row_index == 1 or row_index % 1000 == 0 or row_index == total:
            print(f"featurizing {row_index}/{total}", flush=True)
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None:
            dropped.append(str(row.id))
            continue
        try:
            graph = featurize_mol(mol, str(row.id), str(row.smiles), generator, fp_bits)
        except Exception as exc:
            if not dropped:
                print(f"featurization failed for {row.id}: {exc}", flush=True)
            dropped.append(str(row.id))
            continue
        graphs.append(graph)
        kept_rows.append(
            {
                "id": graph.mol_id,
                "smiles": graph.smiles,
                "y": float(row.y),
                "scaffold": graph.scaffold,
            }
        )
    if not graphs:
        raise ValueError("featurization produced no molecules")
    atom_dim = graphs[0].atom_x.shape[1]
    bond_dim = graphs[0].bond_x.shape[1]
    fp_dim = graphs[0].fp.shape[0]
    desc_dim = graphs[0].desc.shape[0]
    for graph in graphs:
        widths_match = (
            graph.atom_x.shape[1] == atom_dim
            and graph.bond_x.shape[1] == bond_dim
            and graph.fp.shape[0] == fp_dim
            and graph.desc.shape[0] == desc_dim
        )
        if not widths_match:
            raise ValueError(f"inconsistent feature width for {graph.mol_id}")
    frame = pd.DataFrame(kept_rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "version": FEATURE_VERSION,
                    "fp_bits": fp_bits,
                    "fp_radius": fp_radius,
                    "source_ids": ids,
                    "graphs": graphs,
                    "frame": frame,
                    "dropped": dropped,
                },
                handle,
                protocol=4,
            )
    return graphs, frame, dropped
