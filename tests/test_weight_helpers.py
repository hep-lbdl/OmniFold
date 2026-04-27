from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import omnifold as of


def test_unfolded_weights_selects_step2_weights():
    weights = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ])

    np.testing.assert_array_equal(of.unfolded_weights(weights), np.array([10.0, 11.0, 12.0]))
    np.testing.assert_array_equal(of.unfolded_weights(weights, iteration=0), np.array([4.0, 5.0, 6.0]))


def test_unfolded_weights_can_normalize_to_event_count():
    weights = np.array([[[1.0, 1.0, 1.0], [2.0, 3.0, 5.0]]])

    unfolded = of.unfolded_weights(weights, normalize=True)

    np.testing.assert_allclose(np.sum(unfolded), 3.0)
    np.testing.assert_allclose(unfolded, np.array([0.6, 0.9, 1.5]))


def test_normalize_weights_can_use_target_sum():
    weights = np.array([2.0, 3.0, 5.0])

    normalized = of.normalize_weights(weights, target_sum=20.0)

    np.testing.assert_allclose(np.sum(normalized), 20.0)
    np.testing.assert_allclose(normalized, np.array([4.0, 6.0, 10.0]))


def test_unfolded_weights_rejects_wrong_shape():
    weights = np.ones((2, 3))

    try:
        of.unfolded_weights(weights)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("unfolded_weights should reject arrays without (iterations, 2, events) shape.")
