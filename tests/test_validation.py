from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from omnifold_publication import closure_test, load_package, validate_package, write_package


def test_validation_detects_event_count_mismatch(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    metadata_path = Path(package_dir) / "metadata.yaml"

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    metadata["publication"]["event_count"] += 1
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    errors = validate_package(package_dir)
    assert any("event_count mismatch" in error for error in errors)


def test_validation_detects_missing_events_file(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    events_path = Path(package_dir) / "events.parquet"
    events_path.unlink()

    errors = validate_package(package_dir)
    assert any("Missing events file" in error for error in errors)


def test_validation_requires_format_version(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    metadata_path = Path(package_dir) / "metadata.yaml"

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    metadata.pop("format_version")
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    errors = validate_package(package_dir)
    assert any("Missing metadata key: format_version" in error for error in errors)


def test_validation_rejects_unsupported_format_version(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    metadata_path = Path(package_dir) / "metadata.yaml"

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    metadata["format_version"] = "99.0"
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    errors = validate_package(package_dir)
    assert any("Unsupported format_version" in error for error in errors)

    with pytest.raises(ValueError, match="Unsupported format_version"):
        load_package(package_dir)


def test_validation_detects_missing_declared_systematic_column(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    events_path = Path(package_dir) / "events.parquet"
    df = pd.read_parquet(events_path).drop(columns=["weights_ensemble_0"])
    df.to_parquet(events_path, index=False)

    errors = validate_package(package_dir)
    assert any("weights_ensemble_0" in error for error in errors)


def test_validation_detects_duplicate_event_ids(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    events_path = Path(package_dir) / "events.parquet"
    df = pd.read_parquet(events_path)
    df.loc[1, "event_id"] = df.loc[0, "event_id"]
    df.to_parquet(events_path, index=False)

    errors = validate_package(package_dir)
    assert any("duplicate values" in error for error in errors)


def test_validation_detects_normalization_mismatch(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    metadata_path = Path(package_dir) / "metadata.yaml"

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    metadata["normalization"]["expected_nominal_sumw"] += 10.0
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    errors = validate_package(package_dir)
    assert any("nominal weight sum mismatch" in error for error in errors)


def test_closure_test_reports_small_difference_for_full_range(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )

    result = closure_test(package_dir, observable="pT_ll", bins=[0.0, 200.0])
    assert result["relative_difference"] == 0.0
