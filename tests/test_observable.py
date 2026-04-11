"""Tests for Observable and reinterpretation API."""

import json

import numpy as np
import pytest

from omnifold.observable import Observable, uncertainty_band
from omnifold.schema import OmniFoldMetadata
from omnifold.weights import OmniFoldWeights


# -- Helpers -----------------------------------------------------------------

def _make_result(n_iter=3, n_events=200, seed=42):
    rng = np.random.RandomState(seed)
    w = rng.rand(n_iter, 2, n_events)
    meta = OmniFoldMetadata(iterations=n_iter)
    return OmniFoldWeights(w, meta)


def _make_events(n_events=200, seed=0):
    rng = np.random.RandomState(seed)
    return {"pt": rng.exponential(50, n_events), "eta": rng.uniform(-2.5, 2.5, n_events)}


# -- Construction / basics ---------------------------------------------------

class TestObservableConstruction:
    def test_minimal(self):
        obs = Observable(name="jet_pt")
        assert obs.name == "jet_pt"
        assert obs.units == ""
        assert obs.binning is None

    def test_full(self):
        obs = Observable(
            name="jet_pt",
            expression="leading-jet pT",
            compute_fn=lambda ev: ev["pt"],
            units="GeV",
            binning=[20, 50, 100, 200],
            phase_space={"pt_min": 20},
        )
        assert obs.expression == "leading-jet pT"
        assert obs.units == "GeV"
        assert obs.phase_space == {"pt_min": 20}

    def test_no_compute_fn_raises(self):
        obs = Observable(name="jet_pt")
        with pytest.raises(ValueError, match="no compute_fn"):
            obs.histogram({})


# -- Histogram ---------------------------------------------------------------

class TestHistogram:
    def test_basic_shape(self):
        obs = Observable(
            name="pt",
            compute_fn=lambda ev: ev["pt"],
            binning=[0, 25, 50, 100, 200],
        )
        events = _make_events()
        counts, errors, edges = obs.histogram(events)
        assert len(counts) == 4
        assert len(errors) == 4
        assert len(edges) == 5

    def test_unit_weights_equal_unweighted(self):
        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100, 200])
        events = _make_events()
        w = np.ones(len(events["pt"]))
        counts_w, _, _ = obs.histogram(events, weights=w)
        counts_none, _, _ = obs.histogram(events)
        np.testing.assert_array_equal(counts_w, counts_none)

    def test_zero_weights_give_zero_counts(self):
        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100])
        events = _make_events()
        w = np.zeros(len(events["pt"]))
        counts, errors, _ = obs.histogram(events, weights=w)
        np.testing.assert_array_equal(counts, 0)
        np.testing.assert_array_equal(errors, 0)

    def test_nan_filtered_from_values(self):
        obs = Observable(name="pt", compute_fn=lambda ev: ev, binning=[0, 1, 2, 3])
        values = np.array([0.5, np.nan, 1.5, 2.5])
        counts, _, _ = obs.histogram(values)
        assert counts.sum() == 3  # NaN filtered

    def test_nan_filtered_from_weights(self):
        obs = Observable(name="pt", compute_fn=lambda ev: ev, binning=[0, 1, 2])
        values = np.array([0.5, 0.5, 1.5])
        weights = np.array([1.0, np.nan, 1.0])
        counts, _, _ = obs.histogram(values, weights=weights)
        assert counts[0] == pytest.approx(1.0)
        assert counts[1] == pytest.approx(1.0)

    def test_error_is_sqrt_sum_w2(self):
        obs = Observable(name="pt", compute_fn=lambda ev: ev, binning=[0, 10])
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([2.0, 3.0, 4.0])
        counts, errors, _ = obs.histogram(values, weights=weights)
        expected_error = np.sqrt(4 + 9 + 16)
        assert errors[0] == pytest.approx(expected_error)

    def test_auto_binning(self):
        obs = Observable(name="pt", compute_fn=lambda ev: ev)
        values = np.linspace(0, 100, 50)
        counts, errors, edges = obs.histogram(values)
        assert len(counts) > 0


# -- Uncertainty band --------------------------------------------------------

class TestUncertaintyBand:
    def test_requires_replicas(self):
        result = _make_result()
        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100, 200])
        events = _make_events()
        with pytest.raises(ValueError, match="No replicas"):
            obs.uncertainty_band(events, result)

    def test_shape(self):
        result = _make_result()
        rng = np.random.RandomState(99)
        for i in range(5):
            result.add_replica(f"r{i}", rng.rand(200))

        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 25, 50, 100, 200])
        events = _make_events()

        nominal, stat_err, edges = obs.uncertainty_band(events, result)
        assert len(nominal) == 4
        assert len(stat_err) == 4
        assert len(edges) == 5

    def test_identical_replicas_give_zero_error(self):
        result = _make_result()
        for i in range(3):
            result.add_replica(f"r{i}", result.nominal().copy())

        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100, 200])
        events = _make_events()
        _, stat_err, _ = obs.uncertainty_band(events, result)
        np.testing.assert_array_almost_equal(stat_err, 0)

    def test_convenience_wrapper(self):
        result = _make_result()
        rng = np.random.RandomState(7)
        for i in range(3):
            result.add_replica(f"r{i}", rng.rand(200))

        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100, 200])
        events = _make_events()

        a = obs.uncertainty_band(events, result)
        b = uncertainty_band(result, obs, events)
        np.testing.assert_array_equal(a[0], b[0])


# -- Systematic variations ---------------------------------------------------

class TestSystematicVariations:
    def test_requires_systematics(self):
        result = _make_result()
        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100])
        events = _make_events()
        with pytest.raises(ValueError, match="No systematics"):
            obs.systematic_variations(events, result)

    def test_returns_all_systematics(self):
        result = _make_result()
        rng = np.random.RandomState(5)
        result.add_systematic("JET_JER_up", rng.rand(200))
        result.add_systematic("JET_JER_down", rng.rand(200))

        obs = Observable(name="pt", compute_fn=lambda ev: ev["pt"], binning=[0, 50, 100, 200])
        events = _make_events()

        variations = obs.systematic_variations(events, result)
        assert set(variations.keys()) == {"JET_JER_up", "JET_JER_down"}
        for name, (counts, errors, edges) in variations.items():
            assert len(counts) == 3


# -- Serialization -----------------------------------------------------------

class TestObservableSerialization:
    def test_to_dict_excludes_compute_fn(self):
        obs = Observable(
            name="jet_pt",
            compute_fn=lambda ev: ev["pt"],
            units="GeV",
            binning=[20, 50, 100],
        )
        d = obs.to_dict()
        assert "compute_fn" not in d
        assert d["name"] == "jet_pt"
        assert d["units"] == "GeV"

    def test_from_dict_roundtrip(self):
        obs = Observable(
            name="jet_eta",
            expression="pseudorapidity",
            units="",
            binning=[-2.5, -1.0, 0, 1.0, 2.5],
            phase_space={"abs_eta_max": 2.5},
        )
        d = obs.to_dict()
        loaded = Observable.from_dict(d)
        assert loaded.name == "jet_eta"
        assert loaded.binning == [-2.5, -1.0, 0, 1.0, 2.5]
        assert loaded.phase_space == {"abs_eta_max": 2.5}

    def test_json_roundtrip(self, tmp_path):
        obs = Observable(
            name="jet_mass",
            expression="jet invariant mass",
            units="GeV",
            binning=[0, 20, 40, 80, 120],
        )
        path = str(tmp_path / "obs.json")
        obs.to_json(path)
        loaded = Observable.from_json(path)
        assert loaded.name == "jet_mass"
        assert loaded.units == "GeV"

    def test_json_is_valid(self):
        obs = Observable(name="jet_pt", units="GeV")
        s = obs.to_json()
        parsed = json.loads(s)
        assert parsed["name"] == "jet_pt"

    def test_inject_compute_fn_on_load(self):
        obs = Observable(name="pt", units="GeV", binning=[0, 50, 100])
        d = obs.to_dict()
        fn = lambda ev: ev["pt"]
        loaded = Observable.from_dict(d, compute_fn=fn)
        events = _make_events()
        counts, _, _ = loaded.histogram(events)
        assert len(counts) == 2
