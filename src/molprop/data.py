"""Dataset download, checksum verification, and table loading."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    sha256: str
    n_rows: int
    id_col: str
    smiles_col: str
    y_col: str


DATASETS: dict[str, DatasetSpec] = {
    "esol": DatasetSpec(
        name="esol",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        filename="delaney-processed.csv",
        sha256="8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
        n_rows=1128,
        id_col="Compound ID",
        smiles_col="smiles",
        y_col="measured log solubility in mols per litre",
    ),
    "lipophilicity": DatasetSpec(
        name="lipophilicity",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        filename="Lipophilicity.csv",
        sha256="aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
        n_rows=4200,
        id_col="CMPD_CHEMBLID",
        smiles_col="smiles",
        y_col="exp",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_data_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def verify_file(path: Path, n_rows: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != sha256:
        raise ValueError(f"checksum mismatch for {path.name}: {actual_hash} != {sha256}")
    actual_rows = count_data_rows(path)
    if actual_rows != n_rows:
        raise ValueError(f"row count mismatch for {path.name}: {actual_rows} != {n_rows}")


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
        partial.replace(dest)
    finally:
        if partial.exists():
            partial.unlink()


def ensure_dataset(spec: DatasetSpec, data_dir: Path) -> Path:
    """Download a dataset if needed, then verify checksum and row count."""
    dest = data_dir / spec.filename
    if dest.exists():
        try:
            verify_file(dest, spec.n_rows, spec.sha256)
            return dest
        except ValueError:
            dest.unlink()
    download_file(spec.url, dest)
    verify_file(dest, spec.n_rows, spec.sha256)
    return dest


def load_table(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in (spec.id_col, spec.smiles_col, spec.y_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"{spec.name} is missing columns {missing}")
    table = pd.DataFrame(
        {
            "id": frame[spec.id_col].astype(str),
            "smiles": frame[spec.smiles_col].astype(str),
            "y": pd.to_numeric(frame[spec.y_col], errors="raise").astype(float),
        }
    )
    if table["id"].duplicated().any():
        dupes = table["id"].duplicated(keep=False)
        suffix = table.groupby("id").cumcount()
        table.loc[dupes, "id"] = [
            f"{mol_id}#{int(n)}" for mol_id, n in zip(table.loc[dupes, "id"], suffix[dupes])
        ]
    if table["id"].duplicated().any():
        raise ValueError(f"{spec.name} still has duplicate ids after disambiguation")
    if len(table) != spec.n_rows:
        raise ValueError(f"{spec.name} loaded {len(table)} rows, expected {spec.n_rows}")
    return table.reset_index(drop=True)
