import warnings

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


DEFAULT_EPS = 1e-6
DEFAULT_CLIP_WARNING_THRESHOLD = 0.01
DEFAULT_FILTER_WARNING_THRESHOLD = 0.2


def _as_model_input(events):
    events = np.asarray(events)
    if events.ndim == 1:
        return events.reshape(-1, 1)
    return events


def _as_boolean_mask(mask, length, name):
    if mask is None:
        return np.ones(length, dtype=bool)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), received {mask.shape}.")
    return mask


def _assert_finite_weights(weights, name):
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"Non-finite weights encountered in {name}.")


def _warn_if_many_filtered(mask, name, threshold):
    removed_fraction = 1.0 - np.mean(mask.astype(np.float64))
    if removed_fraction >= threshold:
        warnings.warn(
            f"{name} removed {removed_fraction:.1%} of events via masking.",
            RuntimeWarning,
            stacklevel=2,
        )


def _clone_optimizer(model):
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return tf.keras.optimizers.Adam()
    return tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(optimizer))


def _step_seed(random_state, iteration, step):
    if random_state is None:
        return None
    return int(random_state) + (2 * iteration) + step


def clone_model_with_weights(model):
    """Clone a Keras model and copy its current weights without mutating the input model."""

    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    return cloned_model


def _fit_classifier(
    reference_model,
    features,
    labels,
    sample_weight,
    *,
    batch_size,
    epochs,
    validation_fraction,
    verbose,
    random_state,
):
    """Fit a fresh clone of the reference model using standard labels and sample weights."""

    features = _as_model_input(features)
    labels = np.asarray(labels, dtype=np.float32)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if len(features) != len(labels) or len(labels) != len(sample_weight):
        raise ValueError("Features, labels, and sample weights must have the same length.")

    if len(np.unique(labels)) != 2:
        raise ValueError("Each OmniFold classifier step requires both binary classes to be present.")

    if random_state is not None:
        tf.keras.utils.set_random_seed(int(random_state))

    model = clone_model_with_weights(reference_model)
    model.compile(loss="binary_crossentropy", optimizer=_clone_optimizer(reference_model), metrics=["accuracy"])

    fit_kwargs = {
        "epochs": epochs,
        "batch_size": min(batch_size, len(features)),
        "verbose": verbose,
        "shuffle": False,
    }

    if validation_fraction:
        split_kwargs = {"test_size": validation_fraction, "random_state": random_state}
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) == 2 and np.all(counts >= 2):
            split_kwargs["stratify"] = labels

        split = train_test_split(features, labels, sample_weight, **split_kwargs)
        X_train, X_test, Y_train, Y_test, w_train, w_test = split
        history = model.fit(
            X_train,
            Y_train,
            sample_weight=w_train,
            validation_data=(X_test, Y_test, w_test),
            **fit_kwargs,
        )
    else:
        history = model.fit(features, labels, sample_weight=sample_weight, **fit_kwargs)

    return model, history


def reweight(
    events,
    model,
    batch_size=10000,
    *,
    eps=DEFAULT_EPS,
    clip_warning_threshold=DEFAULT_CLIP_WARNING_THRESHOLD,
):
    """Compute finite density-ratio weights from classifier probabilities."""

    events = _as_model_input(events)
    f = np.asarray(model.predict(events, batch_size=batch_size, verbose=0), dtype=np.float64)

    clipped = (f <= eps) | (f >= 1.0 - eps)
    clipped_fraction = np.mean(clipped.astype(np.float64))
    if clipped_fraction >= clip_warning_threshold:
        warnings.warn(
            f"Classifier output clipping affected {clipped_fraction:.1%} of events in reweight().",
            RuntimeWarning,
            stacklevel=2,
        )

    f = np.clip(f, eps, 1.0 - eps)
    weights = f / (1.0 - f)
    _assert_finite_weights(weights, "reweight()")
    return np.squeeze(weights)


def omnifold(
    theta0,
    theta_unknown_S,
    iterations,
    model,
    verbose=0,
    *,
    pass_truth=None,
    pass_reco=None,
    random_state=None,
    epochs=20,
    step1_batch_size=10000,
    step2_batch_size=2000,
    validation_fraction=0.25,
    clip_warning_threshold=DEFAULT_CLIP_WARNING_THRESHOLD,
    filter_warning_threshold=DEFAULT_FILTER_WARNING_THRESHOLD,
):
    """Run OmniFold without mutating the input model.

    The provided `model` is treated as a template: each Step 1 and Step 2 classifier
    starts from identical cloned weights. Missing truth/reco information is handled via
    explicit `pass_truth` and `pass_reco` masks rather than sentinel feature values.
    Events masked out of a step keep the prior weight of 1 for that step.
    """

    theta0 = np.asarray(theta0)
    theta_unknown_S = _as_model_input(theta_unknown_S)

    if theta0.ndim < 2 or theta0.shape[1] != 2:
        raise ValueError("theta0 must have shape (n_events, 2, ...) with generator and detector views.")

    n_events = len(theta0)
    theta0_G = _as_model_input(theta0[:, 0])
    theta0_S = _as_model_input(theta0[:, 1])

    pass_truth = _as_boolean_mask(pass_truth, n_events, "pass_truth")
    pass_reco = _as_boolean_mask(pass_reco, n_events, "pass_reco")

    if not np.any(pass_truth):
        raise ValueError("At least one event must pass the truth-level mask.")
    if not np.any(pass_reco):
        raise ValueError("At least one event must pass the reco-level mask.")
    if len(theta_unknown_S) == 0:
        raise ValueError("theta_unknown_S must contain at least one detector-level event.")

    _warn_if_many_filtered(pass_reco, "Step 1 reco masking", filter_warning_threshold)
    _warn_if_many_filtered(pass_truth, "Step 2 truth masking", filter_warning_threshold)

    weights = np.empty(shape=(iterations, 2, n_events), dtype=np.float64)

    weights_pull = np.ones(n_events, dtype=np.float64)
    weights_push = np.ones(n_events, dtype=np.float64)

    labels0_step1 = np.zeros(np.count_nonzero(pass_reco), dtype=np.float32)
    labels_unknown_step1 = np.ones(len(theta_unknown_S), dtype=np.float32)
    labels0_step2 = np.zeros(np.count_nonzero(pass_truth), dtype=np.float32)
    labels_unknown_step2 = np.ones(np.count_nonzero(pass_truth), dtype=np.float32)

    for i in range(iterations):
        if verbose > 0:
            print("\nITERATION: {}\n".format(i + 1))
            print("STEP 1\n")

        xvals_1 = np.concatenate((theta0_S[pass_reco], theta_unknown_S), axis=0)
        yvals_1 = np.concatenate((labels0_step1, labels_unknown_step1), axis=0)
        weights_1 = np.concatenate((weights_push[pass_reco], np.ones(len(theta_unknown_S))), axis=0)

        model_step1, _ = _fit_classifier(
            model,
            xvals_1,
            yvals_1,
            weights_1,
            batch_size=step1_batch_size,
            epochs=epochs,
            validation_fraction=validation_fraction,
            verbose=verbose,
            random_state=_step_seed(random_state, i, 0),
        )

        weights_pull = np.ones(n_events, dtype=np.float64)
        weights_pull[pass_reco] = weights_push[pass_reco] * reweight(
            theta0_S[pass_reco],
            model_step1,
            batch_size=step1_batch_size,
            clip_warning_threshold=clip_warning_threshold,
        )
        _assert_finite_weights(weights_pull, f"weights_pull at iteration {i + 1}")
        weights[i, 0, :] = weights_pull

        if verbose > 0:
            print("\nSTEP 2\n")

        xvals_2 = np.concatenate((theta0_G[pass_truth], theta0_G[pass_truth]), axis=0)
        yvals_2 = np.concatenate((labels0_step2, labels_unknown_step2), axis=0)
        weights_2 = np.concatenate((np.ones(np.count_nonzero(pass_truth)), weights_pull[pass_truth]), axis=0)

        model_step2, _ = _fit_classifier(
            model,
            xvals_2,
            yvals_2,
            weights_2,
            batch_size=step2_batch_size,
            epochs=epochs,
            validation_fraction=validation_fraction,
            verbose=verbose,
            random_state=_step_seed(random_state, i, 1),
        )

        weights_push = np.ones(n_events, dtype=np.float64)
        weights_push[pass_truth] = reweight(
            theta0_G[pass_truth],
            model_step2,
            batch_size=step2_batch_size,
            clip_warning_threshold=clip_warning_threshold,
        )
        _assert_finite_weights(weights_push, f"weights_push at iteration {i + 1}")
        weights[i, 1, :] = weights_push

    return weights
