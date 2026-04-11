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
        # Replicas and systematics: name -> ndarray of shape (n_events,)
        self._replicas = {}
        self._systematics = {}

    # -- Properties ----------------------------------------------------------

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

    # -- Accessors -----------------------------------------------------------

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

    # -- Replicas and systematics --------------------------------------------

    def add_replica(self, name, weights_array):
        """Add a statistical replica or bootstrap variation.

        Parameters
        ----------
        name : str
            Identifier, e.g. ``"ensemble_0"`` or ``"bootstrap_3"``.
        weights_array : array-like, shape (n_events,)
            Per-event weights for this replica (final-iteration push).
        """
        w = np.asarray(weights_array, dtype=np.float64)
        if w.shape != (self.n_events,):
            raise ValueError(
                f"replica '{name}' must have shape ({self.n_events},), "
                f"got {w.shape}"
            )
        self._replicas[name] = w

    def add_systematic(self, name, weights_array):
        """Add a systematic variation weight set.

        Parameters
        ----------
        name : str
            Identifier following ATLAS convention, e.g.
            ``"JET_JER_up"`` or ``"JET_JER_down"``.
        weights_array : array-like, shape (n_events,)
            Per-event weights for this systematic variation.
        """
        w = np.asarray(weights_array, dtype=np.float64)
        if w.shape != (self.n_events,):
            raise ValueError(
                f"systematic '{name}' must have shape ({self.n_events},), "
                f"got {w.shape}"
            )
        self._systematics[name] = w

    def replica(self, name):
        """Return weights for a named replica."""
        if name not in self._replicas:
            raise KeyError(f"No replica named '{name}'")
        return self._replicas[name]

    def systematic(self, name):
        """Return weights for a named systematic variation."""
        if name not in self._systematics:
            raise KeyError(f"No systematic named '{name}'")
        return self._systematics[name]

    def replicas(self):
        """Iterate over ``(name, weights)`` pairs for all replicas."""
        return iter(self._replicas.items())

    def replica_names(self):
        """List of replica names."""
        return list(self._replicas.keys())

    def systematic_names(self):
        """List of systematic variation names."""
        return list(self._systematics.keys())

    def n_replicas(self):
        """Number of replicas stored."""
        return len(self._replicas)

    def n_systematics(self):
        """Number of systematic variations stored."""
        return len(self._systematics)

    # -- HDF5 I/O ------------------------------------------------------------

    def to_hdf5(self, path):
        """Save weights, replicas, systematics, and metadata to HDF5.

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

            # Replicas
            if self._replicas:
                rg = f.create_group("replicas")
                for name, w in self._replicas.items():
                    rg.create_dataset(name, data=w)

            # Systematics
            if self._systematics:
                sg = f.create_group("systematics")
                for name, w in self._systematics.items():
                    sg.create_dataset(name, data=w)

            # Metadata stored as a JSON attribute
            f.attrs["metadata"] = self._metadata.to_json()

    @classmethod
    def from_hdf5(cls, path):
        """Load weights, replicas, systematics, and metadata from HDF5."""
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
            obj = cls(weights, metadata)

            if "replicas" in f:
                for name, ds in f["replicas"].items():
                    obj.add_replica(name, ds[:])

            if "systematics" in f:
                for name, ds in f["systematics"].items():
                    obj.add_systematic(name, ds[:])

        return obj

    # -- Parquet I/O ---------------------------------------------------------

    def to_parquet(self, path):
        """Save weights, replicas, systematics to Parquet with metadata.

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

        # Replica columns: prefix "replica__"
        for name, w in self._replicas.items():
            columns[f"replica__{name}"] = w

        # Systematic columns: prefix "systematic__"
        for name, w in self._systematics.items():
            columns[f"systematic__{name}"] = w

        table = pa.table(columns)
        table = table.replace_schema_metadata({
            b"omnifold_metadata": self._metadata.to_json().encode(),
        })
        pq.write_table(table, path)

    @classmethod
    def from_parquet(cls, path):
        """Load weights, replicas, systematics, and metadata from Parquet."""
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

        obj = cls(weights, metadata)

        for col in table.column_names:
            if col.startswith("replica__"):
                name = col[len("replica__"):]
                obj.add_replica(name, table.column(col).to_numpy())
            elif col.startswith("systematic__"):
                name = col[len("systematic__"):]
                obj.add_systematic(name, table.column(col).to_numpy())

        return obj

    # -- Dunder --------------------------------------------------------------

    def __repr__(self):
        extras = []
        if self._replicas:
            extras.append(f"n_replicas={len(self._replicas)}")
        if self._systematics:
            extras.append(f"n_systematics={len(self._systematics)}")
        suffix = (", " + ", ".join(extras)) if extras else ""
        return (
            f"OmniFoldWeights(iterations={self.n_iterations}, "
            f"n_events={self.n_events}{suffix})"
        )
