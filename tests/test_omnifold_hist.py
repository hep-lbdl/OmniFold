import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from spec.weighted_histogram import compute_weighted_histogram, plot_weighted_histogram


def test_basic_histogram_computation_matches_numpy():
    """Basic correctness: weighted bin contents should match numpy.histogram."""
    values = np.array([0.2, 0.7, 1.2, 1.9])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    bins = np.array([0.0, 1.0, 2.0])

    result = compute_weighted_histogram(values, weights, bins=bins)

    expected_hist, expected_edges = np.histogram(values, bins=bins, weights=weights)
    expected_sumw2, _ = np.histogram(values, bins=bins, weights=weights**2)

    np.testing.assert_allclose(result["hist"], expected_hist)
    np.testing.assert_allclose(result["edges"], expected_edges)
    np.testing.assert_allclose(result["uncertainty"], np.sqrt(expected_sumw2))
    np.testing.assert_allclose(result["centers"], np.array([0.5, 1.5]))


def test_weighted_sum_conservation_for_event_yields():
    """Physics sanity check: histogram sum should preserve total event weight."""
    values = np.array([0.1, 0.2, 0.8, 1.2, 1.8])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0])

    result = compute_weighted_histogram(
        values=values,
        weights=weights,
        bins=np.array([0.0, 1.0, 2.0]),
        density=False,
    )

    assert np.isclose(np.sum(result["hist"]), np.sum(weights))


def test_density_integrates_to_one():
    """Density mode should integrate to 1 for normalized probability-like views."""
    values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    weights = np.ones_like(values)
    edges = np.array([0.0, 0.5, 1.0])

    result = compute_weighted_histogram(
        values=values,
        weights=weights,
        bins=edges,
        density=True,
    )

    integral = np.sum(result["hist"] * np.diff(result["edges"]))
    assert np.isclose(integral, 1.0, rtol=1e-12, atol=1e-12)


def test_nan_and_inf_entries_are_filtered():
    """HEP ntuples often contain bad entries; they should be safely ignored."""
    values = np.array([0.1, 0.3, np.nan, 0.7, 0.9])
    weights = np.array([1.0, 2.0, 3.0, np.inf, 4.0])

    result = compute_weighted_histogram(
        values=values,
        weights=weights,
        bins=np.array([0.0, 0.5, 1.0]),
    )

    # Surviving finite pairs are (0.1,1.0), (0.3,2.0), (0.9,4.0)
    np.testing.assert_allclose(result["hist"], np.array([3.0, 4.0]))
    np.testing.assert_allclose(
        result["uncertainty"],
        np.array([np.sqrt(1.0**2 + 2.0**2), np.sqrt(4.0**2)]),
    )


def test_mismatched_array_shapes_raise_value_error():
    """Shape mismatch must fail early to avoid silently wrong physics plots."""
    values = np.array([0.1, 0.2, 0.3])
    weights = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        compute_weighted_histogram(values, weights, bins=2)


def test_empty_input_arrays_raise_value_error():
    """Empty arrays should not produce empty plots that look valid."""
    with pytest.raises(ValueError, match="empty"):
        compute_weighted_histogram(np.array([]), np.array([]), bins=10)


def test_all_entries_filtered_raises_value_error():
    """If all entries are non-finite, analysis code should fail clearly."""
    values = np.array([np.nan, np.inf])
    weights = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="No finite entries"):
        compute_weighted_histogram(values, weights, bins=10)


def test_auto_binning_returns_valid_edges():
    """Auto-binning is useful when analysts do fast exploratory scans."""
    rng = np.random.default_rng(123)
    values = rng.normal(loc=0.0, scale=1.0, size=500)
    weights = np.ones_like(values)

    result = compute_weighted_histogram(values, weights, bins="auto")
    assert result["edges"].ndim == 1
    assert result["edges"].size >= 2
    assert np.all(np.diff(result["edges"]) > 0.0)
    assert result["hist"].shape[0] == result["edges"].size - 1


def test_legacy_tuple_unpacking_is_preserved():
    """Backward compatibility: older code unpacks 3 return arrays."""
    values = np.array([0.1, 0.2, 0.3])
    weights = np.array([1.0, 2.0, 3.0])

    hist, edges, uncertainty = compute_weighted_histogram(values, weights, bins=3)
    assert hist.shape[0] == 3
    assert edges.shape[0] == 4
    assert uncertainty.shape[0] == 3


def test_plot_weighted_histogram_returns_plot_and_arrays():
    """Plot helper should remain stable for notebooks and scripts."""
    values = np.array([0.2, 0.4, 0.6, 1.2])
    weights = np.array([1.0, 1.5, 0.5, 2.0])

    fig, ax, hist, edges, uncertainty = plot_weighted_histogram(
        values=values,
        weights=weights,
        bins=np.array([0.0, 1.0, 2.0]),
        density=False,
        label="nominal",
    )

    assert fig is ax.figure
    assert hist.shape == (2,)
    assert edges.shape == (3,)
    assert uncertainty.shape == (2,)
    assert len(ax.patches) >= 1
    fig.clf()


def test_plot_weighted_histogram_density_mode_runs():
    """Exercise plotting density path to prevent regressions in normalized views."""
    values = np.array([0.1, 0.2, 0.4, 0.5, 0.9, 1.1])
    weights = np.ones_like(values)

    fig, ax, hist, edges, uncertainty = plot_weighted_histogram(
        values=values,
        weights=weights,
        bins=5,
        density=True,
    )

    assert fig is ax.figure
    assert hist.size == edges.size - 1
    assert uncertainty.size == hist.size
    fig.clf()
