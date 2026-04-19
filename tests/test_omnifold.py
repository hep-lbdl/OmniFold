from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Lambda
from keras.models import Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import omnifold as of


def _build_model(seed=123):
    tf.keras.utils.set_random_seed(seed)
    inputs = Input((1,))
    hidden = Dense(8, activation="relu")(inputs)
    outputs = Dense(1, activation="sigmoid")(hidden)
    return Model(inputs=inputs, outputs=outputs)


def _make_dataset(n_events=128, seed=7):
    rng = np.random.default_rng(seed)

    theta0_G = rng.normal(0.2, 0.8, n_events)
    theta0_S = theta0_G + rng.normal(0.0, 0.5, n_events)
    theta_unknown_G = rng.normal(0.0, 1.0, n_events)
    theta_unknown_S = theta_unknown_G + rng.normal(0.0, 0.5, n_events)

    theta0 = np.stack([theta0_G, theta0_S], axis=1)
    return theta0, theta_unknown_S


def test_reweight_clips_saturated_predictions():
    inputs = Input((1,))
    outputs = Lambda(lambda x: tf.constant([[0.0], [1.0], [0.5]], dtype=tf.float32))(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    weights = of.reweight(np.array([0.0, 1.0, 2.0], dtype=np.float32), model)

    assert np.all(np.isfinite(weights))
    assert weights[0] > 0.0
    assert weights[1] < 1e7
    np.testing.assert_allclose(weights[2], 1.0, rtol=1e-6, atol=1e-6)


def test_fit_classifier_uses_sample_weight_and_reports_valid_accuracy():
    model = _build_model(seed=321)
    features = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
    labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    sample_weight = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64)

    trained_model, history = of._fit_classifier(
        model,
        features,
        labels,
        sample_weight,
        batch_size=2,
        epochs=1,
        validation_fraction=0.5,
        verbose=0,
        random_state=5,
    )

    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
    assert all(0.0 <= value <= 1.0 for value in history.history["accuracy"])
    assert all(0.0 <= value <= 1.0 for value in history.history["val_accuracy"])
    assert trained_model.predict(features.reshape(-1, 1), verbose=0).shape == (4, 1)


def test_omnifold_is_reproducible_and_does_not_mutate_input_model():
    theta0, theta_unknown_S = _make_dataset()
    model = _build_model(seed=11)
    original_weights = [weight.copy() for weight in model.get_weights()]

    weights_one = of.omnifold(
        theta0,
        theta_unknown_S,
        2,
        model,
        random_state=23,
        epochs=2,
        step1_batch_size=64,
        step2_batch_size=64,
    )
    weights_two = of.omnifold(
        theta0,
        theta_unknown_S,
        2,
        model,
        random_state=23,
        epochs=2,
        step1_batch_size=64,
        step2_batch_size=64,
    )

    np.testing.assert_allclose(weights_one, weights_two, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(weights_one))

    for before, after in zip(original_weights, model.get_weights()):
        np.testing.assert_allclose(before, after, rtol=1e-7, atol=1e-7)


def test_fake_events_are_excluded_from_step2_updates():
    theta0, theta_unknown_S = _make_dataset(n_events=96, seed=17)
    pass_truth = np.ones(len(theta0), dtype=bool)
    pass_reco = np.ones(len(theta0), dtype=bool)
    fake_indices = np.array([0, 1, 2, 3, 4, 5])
    pass_truth[fake_indices] = False

    weights = of.omnifold(
        theta0,
        theta_unknown_S,
        2,
        _build_model(seed=19),
        pass_truth=pass_truth,
        pass_reco=pass_reco,
        random_state=29,
        epochs=2,
        step1_batch_size=64,
        step2_batch_size=64,
    )

    np.testing.assert_allclose(weights[-1, 1, fake_indices], np.ones(len(fake_indices)))
    assert np.all(np.isfinite(weights))
