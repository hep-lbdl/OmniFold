"""Container for OmniFold per-event weights with serialization."""

import json

import numpy as np

from omnifold.schema import OmniFoldMetadata


class OmniFoldWeights:
    """Container wrapping the raw weight array with metadata.

    Parameters
    ----------
    weights : array-like, shape (iterations, 2, n_events)
        Axis 0 = iteration index, axis 1 = step (0 = pull, 1 = push).
    metadata : OmniFoldMetadata, optional
        If *None* a default instance is created with ``iterations``
        inferred from the array.
    """

    def __init__(self, weights, metadata=None):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 3 or weights.shape[1] != 2:
            raise ValueError(
                "weights must have shape (iterations, 2, n_events), "
                f"got {weights.shape}"
            )
        self._weights = weights
        self._metadata = metadata or OmniFoldMetadata(
            iterations=weights.shape[0]
        )

    # -- Properties ------------------------------------------------------

    @property
    def weights(self):
        """Raw weight array, shape (iterations, 2, n_events)."""
        return self._weights

    @property
    def metadata(self):
        """Associated :class:`OmniFoldMetadata`."""
        return self._metadata

    @property
    def n_iterations(self):
        return self._weights.shape[0]

    @property
    def n_events(self):
        return self._weights.shape[2]

    # -- Accessors -------------------------------------------------------

    def nominal(self):
        """Final-iteration push weights (the main unfolding result)."""
        return self._weights[-1, 1, :]

    def get_weights(self, iteration=-1, step="push"):
        """Retrieve weights for a given iteration and step.

        Parameters
        ----------
        iteration : int
            Iteration index (supports negative indexing).
        step : {"pull", "push"}
            ``"pull"`` = step 1 (sim -> data),
            ``"push"`` = step 2 (gen -> reweighted gen).
        """
        step_idx = {"pull": 0, "push": 1}[step]
        return self._weights[iteration, step_idx, :]

    # -- HDF5 I/O --------------------------------------------------------

    def to_hdf5(self, path):
        """Save weights and metadata to an HDF5 file.

        Requires *h5py*: ``pip install omnifold[hdf5]``
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 export. "
                "Install with: pip install omnifold[hdf5]"
            )

        with h5py.File(path, "w") as f:
            # Structured per-iteration groups for human inspection
            wg = f.create_group("weights")
            for i in range(self.n_iterations):
                ig = wg.create_group(f"iteration_{i}")
                ig.create_dataset("pull", data=self._weights[i, 0, :])
                ig.create_dataset("push", data=self._weights[i, 1, :])

            # Bulk array for programmatic loading
            f.create_dataset("weights_array", data=self._weights)

            # Metadata stored as a JSON attribute
            f.attrs["metadata"] = self._metadata.to_json()

    @classmethod
    def from_hdf5(cls, path):
        """Load weights and metadata from an HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 import. "
                "Install with: pip install omnifold[hdf5]"
            )

        with h5py.File(path, "r") as f:
            weights = f["weights_array"][:]
            metadata = OmniFoldMetadata.from_dict(
                json.loads(f.attrs["metadata"])
            )
        return cls(weights, metadata)

    # -- Parquet I/O -----------------------------------------------------

    def to_parquet(self, path):
        """Save weights to a Parquet file with embedded metadata.

        Requires *pyarrow*: ``pip install omnifold[parquet]``
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export. "
                "Install with: pip install omnifold[parquet]"
            )

        # One column per (step, iteration) pair
        columns = {}
        for i in range(self.n_iterations):
            columns[f"pull_iter{i}"] = self._weights[i, 0, :]
            columns[f"push_iter{i}"] = self._weights[i, 1, :]

        table = pa.table(columns)
        table = table.replace_schema_metadata({
            b"omnifold_metadata": self._metadata.to_json().encode(),
        })
        pq.write_table(table, path)

    @classmethod
    def from_parquet(cls, path):
        """Load weights and metadata from a Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet import. "
                "Install with: pip install omnifold[parquet]"
            )

        table = pq.read_table(path)
        schema_meta = table.schema.metadata
        if schema_meta is None or b"omnifold_metadata" not in schema_meta:
            raise ValueError("No omnifold_metadata found in Parquet file")

        metadata = OmniFoldMetadata.from_dict(
            json.loads(schema_meta[b"omnifold_metadata"].decode())
        )

        n_iter = metadata.iterations
        n_events = len(table)
        weights = np.empty((n_iter, 2, n_events))
        for i in range(n_iter):
            weights[i, 0, :] = table.column(f"pull_iter{i}").to_numpy()
            weights[i, 1, :] = table.column(f"push_iter{i}").to_numpy()

        return cls(weights, metadata)

    # -- Dunder ----------------------------------------------------------

    def __repr__(self):
        return (
            f"OmniFoldWeights(iterations={self.n_iterations}, "
            f"n_events={self.n_events})"
        )
