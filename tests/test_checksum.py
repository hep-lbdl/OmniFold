"""Tests for checksum verification."""

import hashlib

import pandas as pd
import yaml

from omnifold_publication.validation import validate_package
from omnifold_publication.writer import write_package


def test_checksum_passes_on_valid_package(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    errors = validate_package(out)
    assert errors == []


def test_checksum_fails_on_tampered_parquet(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    events_path = out / "events.parquet"
    with events_path.open("ab") as stream:
        stream.write(b"tampered")
    errors = validate_package(out)
    assert any("Checksum mismatch" in error for error in errors)


def test_checksum_is_actual_sha256(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    with (out / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    actual = hashlib.sha256((out / "events.parquet").read_bytes()).hexdigest()
    assert metadata["publication"]["checksum_sha256"] == actual


def test_validation_without_checksum_still_passes(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    metadata_path = out / "metadata.yaml"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    metadata["publication"].pop("checksum_sha256")
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    assert validate_package(out) == []


def test_checksum_refresh_allows_other_validation_failures(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    events_path = out / "events.parquet"
    metadata_path = out / "metadata.yaml"

    df = pd.read_parquet(events_path).drop(columns=["weights_nominal"])
    df.to_parquet(events_path, index=False)

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    metadata["publication"]["checksum_sha256"] = hashlib.sha256(
        events_path.read_bytes()
    ).hexdigest()
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    errors = validate_package(out)
    assert any("weights_nominal" in error for error in errors)
