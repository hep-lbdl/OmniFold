"""Tests for metadata schema validation."""

import pytest
import yaml

from omnifold_publication.schema import Metadata, validate_metadata
from omnifold_publication.writer import write_package


def test_schema_validates_spec_metadata():
    with open("spec/metadata.yaml", "r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    validate_metadata(metadata)


def test_schema_validates_generated_package_metadata(source_hdf, tmp_path):
    out = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "pkg",
        event_count=100,
    )
    with (out / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    validate_metadata(metadata)


def test_schema_rejects_missing_required_top_level_field():
    metadata = {
        "format_version": "0.1",
        "weights": {"nominal": "weights_nominal", "base_mc_weight": "weight_mc"},
        "normalization": {
            "luminosity": "unknown",
            "cross_section": "unknown",
            "event_weights_normalized": True,
        },
    }
    with pytest.raises(ValueError, match="observables"):
        validate_metadata(metadata)


def test_schema_accepts_minimal_metadata():
    metadata = {
        "format_version": "0.1",
        "observables": [{"name": "pT_ll", "description": "pT", "units": "GeV"}],
        "weights": {"nominal": "weights_nominal", "base_mc_weight": "weight_mc"},
        "normalization": {
            "luminosity": "unknown",
            "cross_section": "unknown",
            "event_weights_normalized": True,
        },
    }
    validate_metadata(metadata)


def test_schema_accepts_unknown_iteration_count():
    metadata = {
        "format_version": "0.1",
        "observables": [{"name": "pT_ll", "description": "pT", "units": "GeV"}],
        "weights": {"nominal": "weights_nominal", "base_mc_weight": "weight_mc"},
        "normalization": {
            "luminosity": "unknown",
            "cross_section": "unknown",
            "event_weights_normalized": True,
        },
        "training": {
            "algorithm": "OmniFold",
            "iterations": "unknown",
            "architecture": "unknown",
        },
    }
    model = Metadata(**metadata)
    assert model.training is not None
    assert model.training.iterations == "unknown"
