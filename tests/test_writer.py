"""Tests for writer.py hardening."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from omnifold_publication.exceptions import PackageWriteError
from omnifold_publication.writer import write_package


def test_write_package_missing_input_raises(tmp_path):
    with pytest.raises(PackageWriteError, match="Input file not found"):
        write_package(
            input_path=tmp_path / "nonexistent.h5",
            output_dir=tmp_path / "out",
        )


def test_write_package_creates_checksum(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    with (out / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    assert "checksum_sha256" in metadata["publication"]
    assert len(metadata["publication"]["checksum_sha256"]) == 64


def test_write_package_custom_observables(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        observables=["pT_ll"],
        event_count=100,
    )
    df = pd.read_parquet(out / "events.parquet")
    assert "pT_ll" in df.columns
    assert "pT_l1" not in df.columns


def test_write_package_custom_observables_metadata(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        observables=["pT_ll"],
        event_count=100,
    )
    with (out / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    assert [observable["name"] for observable in metadata["observables"]] == ["pT_ll"]


def test_write_package_preserves_event_id_when_present(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    df = pd.read_parquet(out / "events.parquet")
    assert "event_id" in df.columns


def test_write_package_returns_output_path(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    assert out == tmp_path / "pkg"
    assert Path(out, "metadata.yaml").exists()
