"""Tests for the publish / load API."""

import os

import numpy as np
import pytest

from omnifold.publication import publish, load
from omnifold.schema import OmniFoldMetadata, ModelInfo, DatasetInfo
from omnifold.weights import OmniFoldWeights


# -- Helpers -----------------------------------------------------------------

def _make_weights(n_iter=3, n_events=100, seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(n_iter, 2, n_events)


def _make_metadata(n_iter=3):
    return OmniFoldMetadata(
        iterations=n_iter,
        model=ModelInfo(architecture="Dense(50,relu)->Dense(1,sigmoid)"),
        dataset=DatasetInfo(
            name="Gaussian test",
            n_events_sim=100,
            n_events_data=100,
        ),
        description="Unit test publication",
    )


# -- HDF5 publish / load ----------------------------------------------------

class TestPublishHDF5:
    def test_creates_expected_files(self, tmp_path):
        pytest.importorskip("h5py")
        w = _make_weights()
        meta = _make_metadata()

        pub_dir = str(tmp_path / "pub")
        publish(w, pub_dir, metadata=meta, weight_format="hdf5")

        assert os.path.exists(os.path.join(pub_dir, "metadata.json"))
        assert os.path.exists(os.path.join(pub_dir, "weights.h5"))

    def test_roundtrip(self, tmp_path):
        pytest.importorskip("h5py")
        w = _make_weights()
        meta = _make_metadata()

        pub_dir = str(tmp_path / "pub")
        publish(w, pub_dir, metadata=meta, weight_format="hdf5")

        result = load(pub_dir)
        np.testing.assert_array_almost_equal(result.weights, w)
        assert result.metadata.iterations == 3
        assert result.metadata.model.architecture == "Dense(50,relu)->Dense(1,sigmoid)"
        assert result.metadata.dataset.name == "Gaussian test"

    def test_load_direct_file(self, tmp_path):
        pytest.importorskip("h5py")
        w = _make_weights()
        meta = _make_metadata()
        ow = OmniFoldWeights(w, meta)

        path = str(tmp_path / "direct.h5")
        ow.to_hdf5(path)

        result = load(path)
        np.testing.assert_array_almost_equal(result.weights, w)


# -- Parquet publish / load --------------------------------------------------

class TestPublishParquet:
    def test_creates_expected_files(self, tmp_path):
        pytest.importorskip("pyarrow")
        w = _make_weights()
        meta = _make_metadata()

        pub_dir = str(tmp_path / "pub")
        publish(w, pub_dir, metadata=meta, weight_format="parquet")

        assert os.path.exists(os.path.join(pub_dir, "metadata.json"))
        assert os.path.exists(os.path.join(pub_dir, "weights.parquet"))

    def test_roundtrip(self, tmp_path):
        pytest.importorskip("pyarrow")
        w = _make_weights()
        meta = _make_metadata()

        pub_dir = str(tmp_path / "pub")
        publish(w, pub_dir, metadata=meta, weight_format="parquet")

        result = load(pub_dir)
        np.testing.assert_array_almost_equal(result.weights, w)
        assert result.metadata.iterations == 3

    def test_load_direct_file(self, tmp_path):
        pytest.importorskip("pyarrow")
        w = _make_weights()
        meta = _make_metadata()
        ow = OmniFoldWeights(w, meta)

        path = str(tmp_path / "direct.parquet")
        ow.to_parquet(path)

        result = load(path)
        np.testing.assert_array_almost_equal(result.weights, w)


# -- OmniFoldWeights input ---------------------------------------------------

class TestPublishWithContainer:
    def test_accepts_container(self, tmp_path):
        pytest.importorskip("h5py")
        w = _make_weights()
        meta = _make_metadata()
        ow = OmniFoldWeights(w, meta)

        pub_dir = str(tmp_path / "pub")
        publish(ow, pub_dir, weight_format="hdf5")

        result = load(pub_dir)
        np.testing.assert_array_almost_equal(result.weights, w)


# -- Auto-detect format ------------------------------------------------------

class TestAutoDetect:
    def test_auto_detect_hdf5_without_metadata(self, tmp_path):
        """If metadata.json is missing, auto-detect from file extension."""
        pytest.importorskip("h5py")
        w = _make_weights()
        meta = _make_metadata()
        ow = OmniFoldWeights(w, meta)

        pub_dir = str(tmp_path / "pub")
        os.makedirs(pub_dir)
        ow.to_hdf5(os.path.join(pub_dir, "weights.h5"))

        result = load(pub_dir)
        np.testing.assert_array_almost_equal(result.weights, w)

    def test_no_file_raises(self, tmp_path):
        pub_dir = str(tmp_path / "empty")
        os.makedirs(pub_dir)

        with pytest.raises(FileNotFoundError, match="No weight file"):
            load(pub_dir)


# -- Validation on publish ---------------------------------------------------

class TestPublishValidation:
    def test_bad_iterations_rejected(self, tmp_path):
        pytest.importorskip("h5py")
        w = _make_weights()
        meta = OmniFoldMetadata(iterations=0)  # invalid

        with pytest.raises(ValueError, match="iterations must be positive"):
            publish(w, str(tmp_path / "bad"), metadata=meta)

    def test_bad_format_rejected(self, tmp_path):
        w = _make_weights()
        meta = _make_metadata()

        with pytest.raises(ValueError, match="unsupported weight_format"):
            publish(w, str(tmp_path / "bad"), metadata=meta, weight_format="csv")
