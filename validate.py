"""
Diagnostic checks for OmniFold weight arrays

The weight array returned by omnifold() has shape (iterations, 2, n_events)
where axis 1 indexes pull (step=0) and push (step=1) weights. These
functions check the weights for common pathologies like non-finite values,
extreme outliers, low effective sample size, poor convergence between
iterations and normalization drift.

No TensorFlow dependency, works on the raw numpy array.

To use:
    weights = omnifold.omnifold(theta0, theta_unknown_S, iterations, model)
    from validate import validate_weights
    report = validate_weights(weights)
    print(report)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    #Outcome of a single validation check
    name: str
    passed: bool
    detail: str

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"[{tag}] {self.name}: {self.detail}"


@dataclass
class ValidationReport:
    #Collected results from validate_weights() function
    checks: list = field(default_factory=list)

    @property
    def passed(self):
        return all(c.passed for c in self.checks)

    def summary(self):
        lines = [str(c) for c in self.checks]
        n_passed = sum(c.passed for c in self.checks)
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"--- {status} ({n_passed}/{len(self.checks)} checks) ---")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


#individual checks-->

def check_finite(weights):
    #Check for NaN or inf values anywhere in the weight array
    n_nan = int(np.sum(np.isnan(weights)))
    n_inf = int(np.sum(np.isinf(weights)))
    total = n_nan + n_inf
    passed = total == 0
    detail = f"{n_nan} NaN, {n_inf} inf out of {weights.size} values"
    return CheckResult("finite_weights", passed, detail)


def check_extreme_weights(weights, max_ratio=1000.0):
    #Flag events where the weight exceeds max_ratio * median.
    #Examines the final-iteration push weights (weights[-1, 1, :]).
    #A high max/median ratio means a handful of events dominate the
    #reweighted distribution, which inflates statistical uncertainty
    #and can bias downstream observables.
    w = weights[-1, 1, :]
    w_pos = w[w > 0]
    if len(w_pos) == 0:
        return CheckResult("extreme_weights", False, "no positive weights found")
    median = np.median(w_pos)
    if median == 0:
        return CheckResult("extreme_weights", False, "median weight is zero")
    ratio = float(np.max(w_pos) / median)
    n_extreme = int(np.sum(w_pos > max_ratio * median))
    passed = n_extreme == 0
    detail = f"max/median = {ratio:.1f}, {n_extreme} event(s) above {max_ratio:.0f}x median"
    return CheckResult("extreme_weights", passed, detail)


def effective_sample_size(w):
    #ESS = (sum w)^2 / sum(w^2). Equals n for uniform weights.
    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)
    if sum_w2 == 0:
        return 0.0
    return float(sum_w ** 2 / sum_w2)



#Check that ESS of final push weights is at least min_fraction * n.
#ESS below 1% of the original sample size usually means the reweighting
#is dominated by a few events and the result is statistically unreliable.
def check_effective_sample_size(weights, min_fraction=0.01):
    w = weights[-1, 1, :]
    n = len(w)
    ess = effective_sample_size(w)
    fraction = ess / n if n > 0 else 0.0
    passed = fraction >= min_fraction
    detail = f"ESS = {ess:.0f} ({fraction:.1%} of {n} events, threshold {min_fraction:.1%})"
    return CheckResult("effective_sample_size", passed, detail)


#compare push weights between the last two iterations.
#this simply calculates the mean absolute relative change.
#Atleast 2 iterations required; with only 1 the check is skipped.
def check_convergence(weights, rtol=0.05):
    n_iter = weights.shape[0]
    if n_iter < 2:
        return CheckResult("convergence", True, "skipped (single iteration)")

    w_prev = weights[-2, 1, :]
    w_last = weights[-1, 1, :]
    denom = np.maximum(np.abs(w_prev), 1e-10)
    rel_change = float(np.mean(np.abs(w_last - w_prev) / denom))
    passed = rel_change <= rtol
    detail = f"mean |dw/w| = {rel_change:.4f} (threshold {rtol})"
    return CheckResult("convergence", passed, detail)

#Check that the mean final push weight is order-of-magnitude reasonable.
#For a reweighting that preserves the event yield, mean(w) should be O(1).
#A value far from 1 suggests normalization drift from poor classifier calibration or too few iterations.
def check_normalization(weights, atol=10.0):
    w = weights[-1, 1, :]
    mean_w = float(np.mean(w))
    passed = (1.0 / atol) <= mean_w <= atol
    detail = f"mean weight = {mean_w:.4f} (allowed range [{1/atol:.2f}, {atol:.1f}])"
    return CheckResult("normalization", passed, detail)


# Aggregate-->

def validate_weights(weights, max_ratio=1000.0, min_ess_fraction=0.01,
                     convergence_rtol=0.05, normalization_atol=10.0):
    """Run all checks on an OmniFold weight array.

    Parameters:
    ------
    weights : ndarray, shape (iterations, 2, n_events)
        weight array as returned by omnifold.omnifold().
    max_ratio : float
        Threshold for extreme weight detection
    min_ess_fraction : float
        Minimum ESS as a fraction of n_events
    convergence_rtol : float
        Maximum mean relative weight change between last two iterations
    normalization_atol : float
        allowed deviation of mean weight from 1.

    Returns
    -------
    ValidationReport
    """
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError(
            f"expected shape (iterations, 2, n_events), got {weights.shape}"
        )

    report = ValidationReport()
    report.checks.append(check_finite(weights))
    report.checks.append(check_extreme_weights(weights, max_ratio))
    report.checks.append(check_effective_sample_size(weights, min_ess_fraction))
    report.checks.append(check_convergence(weights, convergence_rtol))
    report.checks.append(check_normalization(weights, normalization_atol))
    return report
