"""Validation framework for published OmniFold results.

Provides a set of checks that verify a weight container is physically
sensible before archiving or sharing results.  All checks are collected
into a :class:`ValidationReport` so every problem is surfaced at once
rather than raising on the first failure.

Example
-------
::

    import omnifold as of

    result = of.load("my_result/")
    report = of.validate(result)

    print(report)          # human-readable summary
    assert report.passed   # raises if any check failed
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


# -- Check result ------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    message: str = ""

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        suffix = f": {self.message}" if self.message else ""
        return f"  [{status}] {self.name}{suffix}"


# -- Validation report -------------------------------------------------------

@dataclass
class ValidationReport:
    """Collection of :class:`CheckResult` objects from a validation run."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self):
        """``True`` if every check passed."""
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self):
        """List of checks that did not pass."""
        return [c for c in self.checks if not c.passed]

    def __str__(self):
        lines = ["OmniFold Validation Report"]
        lines.append("=" * 40)
        for c in self.checks:
            lines.append(str(c))
        lines.append("=" * 40)
        overall = "PASSED" if self.passed else f"FAILED ({len(self.failed_checks)} check(s))"
        lines.append(f"Overall: {overall}")
        return "\n".join(lines)

    def assert_passed(self):
        """Raise ``ValueError`` if any check failed."""
        if not self.passed:
            msgs = "; ".join(c.name for c in self.failed_checks)
            raise ValueError(f"Validation failed: {msgs}")


# -- Individual checks -------------------------------------------------------

def _check_finite(weights_array, label="nominal"):
    """All weights must be finite (no NaN or Inf)."""
    n_bad = int(np.sum(~np.isfinite(weights_array)))
    if n_bad:
        return CheckResult(
            "finite_weights",
            False,
            f"{n_bad} non-finite value(s) in {label} weights",
        )
    return CheckResult("finite_weights", True)


def _check_positive(weights_array, label="nominal"):
    """All weights must be non-negative."""
    n_neg = int(np.sum(weights_array < 0))
    if n_neg:
        return CheckResult(
            "positive_weights",
            False,
            f"{n_neg} negative value(s) in {label} weights",
        )
    return CheckResult("positive_weights", True)


def _check_normalization(weights_array, tolerance=0.05):
    """Mean weight must be close to 1.0 (unit normalization).

    A large deviation suggests the weights were not properly normalised
    or the wrong step/iteration was extracted.
    """
    w = weights_array[np.isfinite(weights_array)]
    if len(w) == 0:
        return CheckResult(
            "normalization",
            False,
            "all weights are non-finite, cannot check normalization",
        )
    mean = float(np.mean(w))
    if abs(mean - 1.0) > tolerance:
        return CheckResult(
            "normalization",
            False,
            f"mean weight = {mean:.4f}, expected ~1.0 (tolerance {tolerance})",
        )
    return CheckResult(
        "normalization",
        True,
        f"mean weight = {mean:.4f}",
    )


def _check_convergence(result, window=2, tolerance=0.05):
    """Check that weights stabilize over the last *window* iterations.

    Computes the mean relative change in push weights between consecutive
    iterations over the final *window* iterations.  If the change exceeds
    *tolerance*, the run may not have converged.
    """
    if result.n_iterations < window + 1:
        return CheckResult(
            "convergence",
            True,
            f"only {result.n_iterations} iteration(s); convergence check skipped",
        )

    changes = []
    for i in range(result.n_iterations - window, result.n_iterations):
        w_prev = result.get_weights(i - 1, "push")
        w_curr = result.get_weights(i, "push")
        mask = np.isfinite(w_prev) & np.isfinite(w_curr) & (np.abs(w_prev) > 1e-10)
        if mask.sum() == 0:
            continue
        rel_change = float(np.mean(np.abs(w_curr[mask] - w_prev[mask]) / np.abs(w_prev[mask])))
        changes.append(rel_change)

    if not changes:
        return CheckResult("convergence", True, "insufficient finite weights to check")

    max_change = float(max(changes))
    if max_change > tolerance:
        return CheckResult(
            "convergence",
            False,
            f"max mean relative weight change over last {window} iteration(s) = "
            f"{max_change:.4f} (tolerance {tolerance}); consider more iterations",
        )
    return CheckResult(
        "convergence",
        True,
        f"max mean relative weight change = {max_change:.4f}",
    )


def _check_weight_range(weights_array, max_weight=1e6, label="nominal"):
    """Warn if any weight is extremely large (potential numerical instability)."""
    max_val = float(np.nanmax(np.abs(weights_array)))
    if max_val > max_weight:
        return CheckResult(
            "weight_range",
            False,
            f"max |weight| = {max_val:.2e} in {label} (threshold {max_weight:.0e}); "
            "consider weight clipping or more training epochs",
        )
    return CheckResult(
        "weight_range",
        True,
        f"max |weight| = {max_val:.2e}",
    )


# -- Public API --------------------------------------------------------------

def validate(result, normalization_tolerance=0.05, convergence_tolerance=0.05,
             convergence_window=2, max_weight=1e6):
    """Run all standard validation checks on an :class:`OmniFoldWeights` object.

    Parameters
    ----------
    result : OmniFoldWeights
        Published result to validate.
    normalization_tolerance : float
        Allowed deviation of mean(nominal weights) from 1.0.
    convergence_tolerance : float
        Max allowed mean relative weight change over the last
        *convergence_window* iterations.
    convergence_window : int
        Number of final iterations to inspect for convergence.
    max_weight : float
        Threshold above which individual weights are flagged.

    Returns
    -------
    ValidationReport
        Contains pass/fail for each check plus an overall verdict.
    """
    nominal = result.nominal()
    checks = [
        _check_finite(nominal),
        _check_positive(nominal),
        _check_normalization(nominal, tolerance=normalization_tolerance),
        _check_weight_range(nominal, max_weight=max_weight),
        _check_convergence(result, window=convergence_window,
                           tolerance=convergence_tolerance),
    ]
    return ValidationReport(checks=checks)
