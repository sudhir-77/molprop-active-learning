"""Data download checks and table loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from molprop.data import DatasetSpec, count_data_rows, download_file, ensure_dataset, load_table, sha256_file, verify_file


def _write(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_verify_file_accepts_matching_hash_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    digest = _write(path, b"id,smiles,y\n1,CCO,0.2\n2,CC,0.3\n")
    verify_file(path, 2, digest)
    assert count_data_rows(path) == 2
    assert sha256_file(path) == digest


def test_verify_file_rejects_bad_hash_and_bad_row_count(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    _write(path, b"id,smiles,y\n1,CCO,0.2\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_file(path, 1, "0" * 64)
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="row count"):
        verify_file(path, 5, digest)


def test_ensure_dataset_downloads_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"id,smiles,y\na,CCO,1.0\n"
    digest = hashlib.sha256(content).hexdigest()
    spec = DatasetSpec(
        name="toy",
        url="https://example.invalid/toy.csv",
        filename="toy.csv",
        sha256=digest,
        n_rows=1,
        id_col="id",
        smiles_col="smiles",
        y_col="y",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 0):
            yield content

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

    def fake_get(url, timeout, stream):
        assert url == spec.url
        assert stream is True
        return FakeResponse()

    monkeypatch.setattr("molprop.data.requests.get", fake_get)
    path = ensure_dataset(spec, tmp_path)
    frame = load_table(path, spec)
    assert frame.loc[0, "smiles"] == "CCO"
    assert frame.loc[0, "y"] == 1.0
    # A second call must reuse the verified file and must not download again.
    monkeypatch.setattr("molprop.data.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("redownload")))
    assert ensure_dataset(spec, tmp_path) == path


def test_load_table_disambiguates_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dups.csv"
    path.write_text("id,smiles,y\nA,CCO,1\nA,CC,2\n", encoding="utf-8")
    spec = DatasetSpec("toy", "http://x", "dups.csv", "0" * 64, 2, "id", "smiles", "y")
    frame = load_table(path, spec)
    assert list(frame["id"]) == ["A#0", "A#1"]


def test_download_file_writes_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello"

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 0):
            yield payload

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

    monkeypatch.setattr("molprop.data.requests.get", lambda *args, **kwargs: FakeResponse())
    dest = tmp_path / "out.bin"
    download_file("https://example.invalid/out.bin", dest)
    assert dest.read_bytes() == payload
    assert not (tmp_path / "out.bin.partial").exists()
