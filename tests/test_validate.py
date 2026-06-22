#Tests for validate.py
#Run from the OmniFold root directory:
#pytest tests/test_validate.py -v

import numpy as np
import pytest

from validate import (
    check_finite,
    check_extreme_weights,
    effective_sample_size,
    check_effective_sample_size,
    check_convergence,
    check_normalization,
    validate_weights,
    ValidationReport,
)


def _make_weights(n_iter, n_events, fill=1.0):
    """Build a weight array of shape (n_iter, 2, n_events)."""
    return np.full((n_iter, 2, n_events), fill, dtype=np.float64)


#check_finite 

class TestCheckFinite:
    def test_clean(self):
        r = check_finite(_make_weights(3, 100))
        assert r.passed

    def test_nan_detected(self):
        w = _make_weights(3, 100)
        w[1, 0, 50] = np.nan
        r = check_finite(w)
        assert not r.passed
        assert "1 NaN" in r.detail

    def test_inf_detected(self):
        w = _make_weights(3, 100)
        w[2, 1, 0] = np.inf
        w[2, 1, 1] = -np.inf
        r = check_finite(w)
        assert not r.passed
        assert "2 inf" in r.detail


#check_extreme_weights

class TestCheckExtremeWeights:
    def test_uniform_passes(self):
        r = check_extreme_weights(_make_weights(3, 1000))
        assert r.passed

    def test_single_outlier_fails(self):
        w = _make_weights(3, 1000)
        w[-1, 1, 0] = 1e7
        r = check_extreme_weights(w, max_ratio=1000.0)
        assert not r.passed

    def test_moderate_variation_passes(self):
        """Exponential weights have a long tail but max/median stays bounded."""
        rng = np.random.default_rng(42)
        w = _make_weights(3, 5000)
        w[-1, 1, :] = rng.exponential(1.0, 5000)
        r = check_extreme_weights(w, max_ratio=1000.0)
        assert r.passed


#effective_sample_size

class TestEffectiveSampleSize:
    def test_uniform(self):
        """Unit weights: ESS should equal n."""
        ess = effective_sample_size(np.ones(1000))
        assert ess == pytest.approx(1000.0)

    def test_single_nonzero(self):
        """Only one event carries weight: ESS = 1."""
        w = np.zeros(1000)
        w[0] = 1.0
        assert effective_sample_size(w) == pytest.approx(1.0)

    def test_two_equal(self):
        """Two events with equal weight: ESS = 2."""
        w = np.zeros(100)
        w[0] = 5.0
        w[1] = 5.0
        assert effective_sample_size(w) == pytest.approx(2.0)

    def test_all_zero(self):
        assert effective_sample_size(np.zeros(10)) == 0.0


#check_effective_sample_size

class TestCheckESS:
    def test_uniform_passes(self):
        r = check_effective_sample_size(_make_weights(3, 1000))
        assert r.passed

    def test_concentrated_fails(self):
        w = _make_weights(3, 1000, fill=0.0)
        w[-1, 1, 0] = 1.0
        r = check_effective_sample_size(w, min_fraction=0.01)
        assert not r.passed


#check_convergence

class TestCheckConvergence:
    def test_identical_iterations(self):
        r = check_convergence(_make_weights(3, 100))
        assert r.passed

    def test_single_iteration_skipped(self):
        r = check_convergence(_make_weights(1, 100))
        assert r.passed
        assert "skipped" in r.detail

    def test_large_change_fails(self):
        w = _make_weights(3, 100)
        w[-1, 1, :] = 10.0
        r = check_convergence(w, rtol=0.05)
        assert not r.passed


#check_normalization

class TestCheckNormalization:
    def test_unit_weights_pass(self):
        r = check_normalization(_make_weights(3, 100))
        assert r.passed

    def test_huge_mean_fails(self):
        w = _make_weights(3, 100, fill=100.0)
        r = check_normalization(w, atol=10.0)
        assert not r.passed

    def test_tiny_mean_fails(self):
        w = _make_weights(3, 100, fill=0.001)
        r = check_normalization(w, atol=10.0)
        assert not r.passed


#validate_weights

class TestValidateWeights:
    def test_clean_passes(self):
        report = validate_weights(_make_weights(4, 500))
        assert report.passed
        assert len(report.checks) == 5

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="iterations, 2, n_events"):
            validate_weights(np.ones((4, 3, 100)))

    def test_summary_contains_status(self):
        report = validate_weights(_make_weights(4, 500))
        assert "PASSED" in report.summary()

    def test_nan_makes_report_fail(self):
        w = _make_weights(4, 500)
        w[2, 0, 100] = np.nan
        report = validate_weights(w)
        assert not report.passed
        failed = [c for c in report.checks if not c.passed]
        assert any(c.name == "finite_weights" for c in failed)
