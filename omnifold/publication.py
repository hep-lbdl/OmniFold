"""High-level API for publishing and loading OmniFold results."""

import os

import numpy as np

from omnifold.schema import OmniFoldMetadata
from omnifold.weights import OmniFoldWeights


def publish(weights, output_dir, metadata=None, weight_format="hdf5"):
    """Publish OmniFold results in a standardized, self-contained format.

    Creates a directory containing:

    * ``metadata.json`` -- schema-compliant metadata.
    * ``weights.h5`` *or* ``weights.parquet`` -- per-event weights.

    Parameters
    ----------
    weights : numpy array or :class:`OmniFoldWeights`
        Shape ``(iterations, 2, n_events)`` if a raw array.
    output_dir : str
        Path to the publication directory (created if needed).
    metadata : OmniFoldMetadata, optional
        Required when *weights* is a raw numpy array.
    weight_format : {"hdf5", "parquet"}
        Serialization backend for the weight file.

    Returns
    -------
    str
        The *output_dir* path.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(weights, OmniFoldWeights):
        weights = OmniFoldWeights(np.asarray(weights), metadata)

    # Ensure the format field is consistent with what we write
    weights.metadata.weight_format = weight_format
    weights.metadata.validate()

    # -- write metadata --------------------------------------------------
    weights.metadata.to_json(os.path.join(output_dir, "metadata.json"))

    # -- write weights ---------------------------------------------------
    if weight_format == "hdf5":
        weights.to_hdf5(os.path.join(output_dir, "weights.h5"))
    elif weight_format == "parquet":
        weights.to_parquet(os.path.join(output_dir, "weights.parquet"))
    else:
        raise ValueError(f"Unsupported weight_format: {weight_format}")

    return output_dir


def load(path):
    """Load a published OmniFold result.

    Accepts a publication directory, or a direct path to a ``.h5`` /
    ``.parquet`` file.

    Returns
    -------
    OmniFoldWeights
    """
    # Direct file path
    if path.endswith((".h5", ".hdf5")):
        return OmniFoldWeights.from_hdf5(path)
    if path.endswith(".parquet"):
        return OmniFoldWeights.from_parquet(path)

    # Directory -- read metadata to learn the format
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        metadata = OmniFoldMetadata.from_json(meta_path)
        fmt = metadata.weight_format
    else:
        fmt = None  # auto-detect

    h5_path = os.path.join(path, "weights.h5")
    pq_path = os.path.join(path, "weights.parquet")

    if fmt == "hdf5" or (fmt is None and os.path.exists(h5_path)):
        return OmniFoldWeights.from_hdf5(h5_path)
    if fmt == "parquet" or (fmt is None and os.path.exists(pq_path)):
        return OmniFoldWeights.from_parquet(pq_path)

    raise FileNotFoundError(
        f"No weight file found in {path}. "
        "Expected weights.h5 or weights.parquet"
    )
