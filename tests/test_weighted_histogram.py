def test_negative_weights_are_handled():
    """Some MC samples have negative weights from matrix-element reweighting."""
    import numpy as np
    from spec.weighted_histogram import compute_weighted_histogram

    values = np.array([0.1, 0.5, 0.9])
    weights = np.array([-1.0, 2.0, -0.5])

    result = compute_weighted_histogram(values, weights, bins=np.array([0.0, 1.0]))

    assert np.isfinite(result["hist"]).all()
