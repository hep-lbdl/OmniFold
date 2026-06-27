"""Tests for reader API methods and typed exceptions."""

import pytest
import yaml

from omnifold_publication.exceptions import PackageReadError, UnsupportedFormatVersion
from omnifold_publication.reader import get_weights, load_metadata, load_package
from omnifold_publication.writer import write_package


def test_list_weights(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    pkg = load_package(out)
    weights = pkg.list_weights()
    assert "nominal" in weights
    assert "base_mc_weight" in weights


def test_list_observables(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    pkg = load_package(out)
    observables = pkg.list_observables()
    assert observables == ["pT_ll", "pT_l1"]


def test_summary_keys(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    pkg = load_package(out)
    summary = pkg.summary()
    assert "format_version" in summary
    assert "event_count" in summary
    assert "observables" in summary
    assert "weights" in summary
    assert "checksum_sha256" in summary


def test_summary_includes_checksum(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    pkg = load_package(out)
    assert len(pkg.summary()["checksum_sha256"]) == 64


def test_unsupported_format_version_raises(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    metadata_path = out / "metadata.yaml"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    metadata["format_version"] = "99.0"
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream)
    with pytest.raises(UnsupportedFormatVersion):
        load_package(out)


def test_get_unknown_weight_raises_typed_exception(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    pkg = load_package(out)
    with pytest.raises(PackageReadError, match="Unknown metadata-declared weight"):
        pkg.get_weights("does_not_exist")


def test_function_get_weights_missing_column_raises_typed_exception(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    metadata = load_metadata(out)
    events = pkg_events_without_weights = load_package(out).load_events(columns=["pT_ll"])
    assert "weights_nominal" not in pkg_events_without_weights.columns
    with pytest.raises(PackageReadError, match="not present"):
        get_weights(events, metadata)
