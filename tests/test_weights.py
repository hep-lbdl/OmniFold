"""Tests for the OmniFoldWeights container."""

import numpy as np
import pytest

from omnifold.schema import OmniFoldMetadata
from omnifold.weights import OmniFoldWeights


# -- Helpers -----------------------------------------------------------------

def _make_weights(n_iter=3, n_events=100, seed=42):
    rng = np.random.RandomState(seed)
    w = rng.rand(n_iter, 2, n_events)
    meta = OmniFoldMetadata(iterations=n_iter)
    return w, meta


# -- Construction ------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)
        assert ow.n_iterations == 3
        assert ow.n_events == 100

    def test_infers_metadata(self):
        w, _ = _make_weights()
        ow = OmniFoldWeights(w)
        assert ow.metadata.iterations == 3

    def test_rejects_bad_shape_ndim(self):
        with pytest.raises(ValueError, match="shape"):
            OmniFoldWeights(np.ones((3, 100)))

    def test_rejects_bad_shape_axis1(self):
        with pytest.raises(ValueError, match="shape"):
            OmniFoldWeights(np.ones((3, 3, 100)))

    def test_casts_to_float64(self):
        w = np.ones((2, 2, 50), dtype=np.float32)
        ow = OmniFoldWeights(w)
        assert ow.weights.dtype == np.float64


# -- Accessors ---------------------------------------------------------------

class TestAccessors:
    def test_nominal(self):
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)
        np.testing.assert_array_equal(ow.nominal(), w[-1, 1, :])

    def test_get_weights_pull(self):
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)
        np.testing.assert_array_equal(
            ow.get_weights(0, "pull"), w[0, 0, :]
        )

    def test_get_weights_push(self):
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)
        np.testing.assert_array_equal(
            ow.get_weights(1, "push"), w[1, 1, :]
        )

    def test_get_weights_negative_index(self):
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)
        np.testing.assert_array_equal(
            ow.get_weights(-1, "push"), w[-1, 1, :]
        )

    def test_repr(self):
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)
        assert "iterations=3" in repr(ow)
        assert "n_events=100" in repr(ow)


# -- HDF5 round-trip ---------------------------------------------------------

class TestHDF5:
    def test_roundtrip(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)

        path = str(tmp_path / "weights.h5")
        ow.to_hdf5(path)
        loaded = OmniFoldWeights.from_hdf5(path)

        np.testing.assert_array_almost_equal(loaded.weights, w)
        assert loaded.metadata.iterations == 3
        assert loaded.n_events == 100

    def test_structured_groups(self, tmp_path):
        """Verify per-iteration groups exist for human inspection."""
        h5py = pytest.importorskip("h5py")
        w, meta = _make_weights(n_iter=2)
        ow = OmniFoldWeights(w, meta)

        path = str(tmp_path / "weights.h5")
        ow.to_hdf5(path)

        with h5py.File(path, "r") as f:
            assert "weights/iteration_0/pull" in f
            assert "weights/iteration_0/push" in f
            assert "weights/iteration_1/pull" in f
            assert "weights/iteration_1/push" in f


# -- Parquet round-trip ------------------------------------------------------

class TestParquet:
    def test_roundtrip(self, tmp_path):
        pytest.importorskip("pyarrow")
        w, meta = _make_weights()
        ow = OmniFoldWeights(w, meta)

        path = str(tmp_path / "weights.parquet")
        ow.to_parquet(path)
        loaded = OmniFoldWeights.from_parquet(path)

        np.testing.assert_array_almost_equal(loaded.weights, w)
        assert loaded.metadata.iterations == 3
        assert loaded.n_events == 100

    def test_column_naming(self, tmp_path):
        """Verify parquet columns follow naming convention."""
        pq = pytest.importorskip("pyarrow.parquet")
        w, meta = _make_weights(n_iter=2)
        ow = OmniFoldWeights(w, meta)

        path = str(tmp_path / "weights.parquet")
        ow.to_parquet(path)

        table = pq.read_table(path)
        names = table.column_names
        assert names == ["pull_iter0", "push_iter0", "pull_iter1", "push_iter1"]
