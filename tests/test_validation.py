"""Tests for the OmniFold validation framework."""

import numpy as np
import pytest

from omnifold.schema import OmniFoldMetadata
from omnifold.validation import (
    CheckResult,
    ValidationReport,
    validate,
)
from omnifold.weights import OmniFoldWeights


# -- Helpers -----------------------------------------------------------------

def _make_converged_result(n_iter=4, n_events=500, seed=42):
    """Create a synthetic result that passes all checks.

    Weights are drawn from a near-unit distribution and decay across
    iterations to simulate convergence.
    """
    rng = np.random.RandomState(seed)
    base = rng.lognormal(0, 0.1, n_events)  # mean ~1, no negatives
    w = np.empty((n_iter, 2, n_events))
    for i in range(n_iter):
        scale = 1.0 + 0.05 * (n_iter - i - 1)  # converges toward 1
        w[i, 0, :] = base * scale + rng.normal(0, 0.01, n_events)
        w[i, 1, :] = base * scale + rng.normal(0, 0.005, n_events)
    # ensure positivity
    w = np.abs(w)
    # normalize so mean push weight ~ 1
    w[:, 1, :] /= w[:, 1, :].mean(axis=1, keepdims=True)
    meta = OmniFoldMetadata(iterations=n_iter)
    return OmniFoldWeights(w, meta)


# -- CheckResult -------------------------------------------------------------

class TestCheckResult:
    def test_str_pass(self):
        c = CheckResult("finite_weights", True)
        assert "[PASS]" in str(c)

    def test_str_fail(self):
        c = CheckResult("finite_weights", False, "2 non-finite values")
        assert "[FAIL]" in str(c)
        assert "2 non-finite values" in str(c)


# -- ValidationReport --------------------------------------------------------

class TestValidationReport:
    def test_passed_all_pass(self):
        report = ValidationReport(checks=[
            CheckResult("a", True),
            CheckResult("b", True),
        ])
        assert report.passed

    def test_failed_one_fail(self):
        report = ValidationReport(checks=[
            CheckResult("a", True),
            CheckResult("b", False, "bad"),
        ])
        assert not report.passed

    def test_failed_checks_list(self):
        report = ValidationReport(checks=[
            CheckResult("a", True),
            CheckResult("b", False),
            CheckResult("c", False),
        ])
        assert len(report.failed_checks) == 2

    def test_assert_passed_raises(self):
        report = ValidationReport(checks=[CheckResult("a", False, "oops")])
        with pytest.raises(ValueError, match="Validation failed"):
            report.assert_passed()

    def test_assert_passed_does_not_raise(self):
        report = ValidationReport(checks=[CheckResult("a", True)])
        report.assert_passed()  # should not raise

    def test_str_contains_overall(self):
        report = ValidationReport(checks=[CheckResult("a", True)])
        assert "PASSED" in str(report)


# -- validate() function -----------------------------------------------------

class TestValidate:
    def test_good_result_passes(self):
        result = _make_converged_result()
        report = validate(result)
        # Converged, positive, finite, normalized -- should pass
        for check in report.checks:
            assert check.passed, f"Expected PASS for '{check.name}': {check.message}"

    def test_nan_weights_fail_finite(self):
        result = _make_converged_result()
        result._weights[-1, 1, 0] = np.nan
        report = validate(result)
        names = {c.name for c in report.failed_checks}
        assert "finite_weights" in names

    def test_inf_weights_fail_finite(self):
        result = _make_converged_result()
        result._weights[-1, 1, 5] = np.inf
        report = validate(result)
        names = {c.name for c in report.failed_checks}
        assert "finite_weights" in names

    def test_negative_weights_fail_positive(self):
        result = _make_converged_result()
        result._weights[-1, 1, :5] = -1.0
        report = validate(result)
        names = {c.name for c in report.failed_checks}
        assert "positive_weights" in names

    def test_unnormalized_weights_fail(self):
        rng = np.random.RandomState(0)
        n_events = 200
        w = rng.rand(3, 2, n_events) * 100  # mean >> 1
        meta = OmniFoldMetadata(iterations=3)
        result = OmniFoldWeights(w, meta)
        report = validate(result, normalization_tolerance=0.05)
        names = {c.name for c in report.failed_checks}
        assert "normalization" in names

    def test_extreme_weights_fail_range(self):
        result = _make_converged_result()
        result._weights[-1, 1, 0] = 2e7
        report = validate(result, max_weight=1e6)
        names = {c.name for c in report.failed_checks}
        assert "weight_range" in names

    def test_single_iteration_skips_convergence(self):
        rng = np.random.RandomState(0)
        w = rng.lognormal(0, 0.01, (1, 2, 100))
        w = np.abs(w)
        w[:, 1, :] /= w[:, 1, :].mean()
        meta = OmniFoldMetadata(iterations=1)
        result = OmniFoldWeights(w, meta)
        report = validate(result)
        conv = next(c for c in report.checks if c.name == "convergence")
        assert conv.passed  # skipped = pass

    def test_non_converged_fails(self):
        """Weights that change a lot between iterations should fail convergence."""
        rng = np.random.RandomState(77)
        n_events = 300
        w = np.empty((4, 2, n_events))
        # push weights change 50% each iteration -- clearly not converged
        w[0, 1, :] = rng.rand(n_events) * 2
        w[1, 1, :] = w[0, 1, :] * 2.0
        w[2, 1, :] = w[1, 1, :] * 2.0
        w[3, 1, :] = w[2, 1, :] * 2.0
        w[:, 0, :] = w[:, 1, :]  # pull same as push
        meta = OmniFoldMetadata(iterations=4)
        result = OmniFoldWeights(w, meta)
        report = validate(result, convergence_tolerance=0.05)
        names = {c.name for c in report.failed_checks}
        assert "convergence" in names

    def test_returns_validation_report(self):
        result = _make_converged_result()
        report = validate(result)
        from omnifold.validation import ValidationReport
        assert isinstance(report, ValidationReport)

    def test_report_has_all_standard_checks(self):
        result = _make_converged_result()
        report = validate(result)
        check_names = {c.name for c in report.checks}
        assert "finite_weights" in check_names
        assert "positive_weights" in check_names
        assert "normalization" in check_names
        assert "weight_range" in check_names
        assert "convergence" in check_names
