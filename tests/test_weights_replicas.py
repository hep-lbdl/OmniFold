"""Tests for replica and systematic support in OmniFoldWeights."""

import numpy as np
import pytest

from omnifold.schema import OmniFoldMetadata
from omnifold.weights import OmniFoldWeights


def _make_container(n_iter=3, n_events=100, seed=42):
    rng = np.random.RandomState(seed)
    w = rng.rand(n_iter, 2, n_events)
    meta = OmniFoldMetadata(iterations=n_iter)
    return OmniFoldWeights(w, meta)


class TestAddReplica:
    def test_add_and_retrieve(self):
        ow = _make_container()
        rep = np.ones(100) * 0.5
        ow.add_replica("ensemble_0", rep)
        np.testing.assert_array_equal(ow.replica("ensemble_0"), rep)

    def test_n_replicas(self):
        ow = _make_container()
        assert ow.n_replicas() == 0
        ow.add_replica("a", np.ones(100))
        ow.add_replica("b", np.ones(100))
        assert ow.n_replicas() == 2

    def test_replica_names(self):
        ow = _make_container()
        ow.add_replica("bootstrap_0", np.ones(100))
        ow.add_replica("bootstrap_1", np.ones(100))
        assert set(ow.replica_names()) == {"bootstrap_0", "bootstrap_1"}

    def test_wrong_shape_raises(self):
        ow = _make_container()
        with pytest.raises(ValueError, match="shape"):
            ow.add_replica("bad", np.ones(50))

    def test_missing_name_raises(self):
        ow = _make_container()
        with pytest.raises(KeyError, match="No replica"):
            ow.replica("nonexistent")

    def test_replicas_iterator(self):
        ow = _make_container()
        ow.add_replica("r0", np.ones(100) * 1.0)
        ow.add_replica("r1", np.ones(100) * 2.0)
        pairs = dict(ow.replicas())
        assert set(pairs.keys()) == {"r0", "r1"}


class TestAddSystematic:
    def test_add_and_retrieve(self):
        ow = _make_container()
        sys_w = np.ones(100) * 1.1
        ow.add_systematic("JET_JER_up", sys_w)
        np.testing.assert_array_equal(ow.systematic("JET_JER_up"), sys_w)

    def test_n_systematics(self):
        ow = _make_container()
        assert ow.n_systematics() == 0
        ow.add_systematic("JET_JER_up", np.ones(100))
        ow.add_systematic("JET_JER_down", np.ones(100))
        assert ow.n_systematics() == 2

    def test_systematic_names(self):
        ow = _make_container()
        ow.add_systematic("JET_JER_up", np.ones(100))
        assert "JET_JER_up" in ow.systematic_names()

    def test_wrong_shape_raises(self):
        ow = _make_container()
        with pytest.raises(ValueError, match="shape"):
            ow.add_systematic("bad", np.ones((100, 2)))

    def test_missing_name_raises(self):
        ow = _make_container()
        with pytest.raises(KeyError, match="No systematic"):
            ow.systematic("nonexistent")


class TestRepr:
    def test_repr_without_extras(self):
        ow = _make_container()
        r = repr(ow)
        assert "iterations=3" in r
        assert "n_replicas" not in r

    def test_repr_with_replicas_and_systematics(self):
        ow = _make_container()
        ow.add_replica("r0", np.ones(100))
        ow.add_systematic("JET_up", np.ones(100))
        r = repr(ow)
        assert "n_replicas=1" in r
        assert "n_systematics=1" in r


class TestHDF5WithReplicas:
    def test_roundtrip(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        ow = _make_container()
        rng = np.random.RandomState(0)
        ow.add_replica("ensemble_0", rng.rand(100))
        ow.add_replica("ensemble_1", rng.rand(100))
        ow.add_systematic("JET_JER_up", rng.rand(100))

        path = str(tmp_path / "weights.h5")
        ow.to_hdf5(path)
        loaded = OmniFoldWeights.from_hdf5(path)

        assert loaded.n_replicas() == 2
        assert loaded.n_systematics() == 1
        np.testing.assert_array_almost_equal(
            loaded.replica("ensemble_0"), ow.replica("ensemble_0")
        )
        np.testing.assert_array_almost_equal(
            loaded.systematic("JET_JER_up"), ow.systematic("JET_JER_up")
        )

    def test_no_replicas_group_absent(self, tmp_path):
        """HDF5 file without replicas should load cleanly."""
        pytest.importorskip("h5py")
        ow = _make_container()
        path = str(tmp_path / "weights.h5")
        ow.to_hdf5(path)
        loaded = OmniFoldWeights.from_hdf5(path)
        assert loaded.n_replicas() == 0
        assert loaded.n_systematics() == 0


class TestParquetWithReplicas:
    def test_roundtrip(self, tmp_path):
        pytest.importorskip("pyarrow")
        ow = _make_container()
        rng = np.random.RandomState(1)
        ow.add_replica("bootstrap_0", rng.rand(100))
        ow.add_systematic("JET_JES_up", rng.rand(100))
        ow.add_systematic("JET_JES_down", rng.rand(100))

        path = str(tmp_path / "weights.parquet")
        ow.to_parquet(path)
        loaded = OmniFoldWeights.from_parquet(path)

        assert loaded.n_replicas() == 1
        assert loaded.n_systematics() == 2
        np.testing.assert_array_almost_equal(
            loaded.replica("bootstrap_0"), ow.replica("bootstrap_0")
        )
        np.testing.assert_array_almost_equal(
            loaded.systematic("JET_JES_up"), ow.systematic("JET_JES_up")
        )

    def test_column_names_prefixed(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        ow = _make_container(n_iter=2)
        ow.add_replica("r0", np.ones(100))
        ow.add_systematic("sys_up", np.ones(100))

        path = str(tmp_path / "weights.parquet")
        ow.to_parquet(path)
        table = pq.read_table(path)
        names = table.column_names
        assert "replica__r0" in names
        assert "systematic__sys_up" in names
