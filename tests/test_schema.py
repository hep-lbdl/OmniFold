"""Tests for the OmniFold metadata schema."""

import json

import pytest

from omnifold.schema import OmniFoldMetadata, ModelInfo, DatasetInfo


class TestModelInfo:
    def test_defaults(self):
        m = ModelInfo()
        assert m.optimizer == "Adam"
        assert m.epochs_step1 == 20
        assert m.batch_size_step1 == 10000

    def test_custom(self):
        m = ModelInfo(architecture="Dense(50)->Dense(1)", epochs_step1=50)
        assert m.architecture == "Dense(50)->Dense(1)"
        assert m.epochs_step1 == 50


class TestDatasetInfo:
    def test_defaults(self):
        d = DatasetInfo()
        assert d.name == ""
        assert d.n_events_sim == 0

    def test_custom(self):
        d = DatasetInfo(name="Z+jets", n_events_sim=100_000)
        assert d.name == "Z+jets"
        assert d.n_events_sim == 100_000


class TestOmniFoldMetadata:
    def test_defaults(self):
        m = OmniFoldMetadata(iterations=3)
        assert m.iterations == 3
        assert m.schema_version == "1.0"
        assert m.omnifold_version == "0.1.0"
        assert m.created_at  # auto-populated

    def test_dict_roundtrip(self):
        m = OmniFoldMetadata(
            iterations=4,
            model=ModelInfo(architecture="Dense(50)->Dense(1)"),
            dataset=DatasetInfo(name="test", n_events_sim=1000),
            description="unit test",
        )
        d = m.to_dict()
        loaded = OmniFoldMetadata.from_dict(d)

        assert loaded.iterations == 4
        assert loaded.model.architecture == "Dense(50)->Dense(1)"
        assert loaded.dataset.name == "test"
        assert loaded.description == "unit test"

    def test_json_roundtrip(self, tmp_path):
        m = OmniFoldMetadata(
            iterations=4,
            model=ModelInfo(architecture="Dense(50)->Dense(1)"),
            dataset=DatasetInfo(name="test", n_events_sim=1000),
        )
        path = str(tmp_path / "meta.json")
        m.to_json(path)
        loaded = OmniFoldMetadata.from_json(path)

        assert loaded.iterations == 4
        assert loaded.model.architecture == "Dense(50)->Dense(1)"
        assert loaded.dataset.name == "test"
        # created_at should be preserved, not regenerated
        assert loaded.created_at == m.created_at

    def test_json_is_valid_json(self):
        m = OmniFoldMetadata(iterations=2)
        s = m.to_json()
        parsed = json.loads(s)
        assert parsed["iterations"] == 2

    def test_from_dict_ignores_unknown_keys(self):
        """Forward compatibility: unknown keys should be silently dropped."""
        d = {
            "iterations": 3,
            "schema_version": "1.0",
            "omnifold_version": "0.1.0",
            "future_field": "some_value",
            "model": {"architecture": "test", "future_model_field": 42},
            "dataset": {},
        }
        m = OmniFoldMetadata.from_dict(d)
        assert m.iterations == 3
        assert m.model.architecture == "test"

    # -- Validation -------------------------------------------------------

    def test_validate_valid(self):
        m = OmniFoldMetadata(iterations=3)
        assert m.validate() is True

    def test_validate_bad_iterations(self):
        m = OmniFoldMetadata(iterations=0)
        with pytest.raises(ValueError, match="iterations must be positive"):
            m.validate()

    def test_validate_bad_format(self):
        m = OmniFoldMetadata(iterations=3, weight_format="csv")
        with pytest.raises(ValueError, match="unsupported weight_format"):
            m.validate()

    def test_validate_bad_normalization(self):
        m = OmniFoldMetadata(iterations=3, normalization="invalid")
        with pytest.raises(ValueError, match="unsupported normalization"):
            m.validate()
