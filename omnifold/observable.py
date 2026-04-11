"""Observable definitions for reinterpretation of OmniFold results.

An Observable encodes how to compute a physics quantity from event data
and how to bin it into a histogram. Storing observable definitions
alongside published weights makes results fully reproducible and
reinterpretable: anyone can load the weights and recompute any observable
without having the original analysis code.

Example
-------
::

    from omnifold.observable import Observable
    import omnifold as of

    result = of.load("atlas_zjets_published/")

    jet_pt = Observable(
        name="jet_pt",
        expression="leading jet transverse momentum",
        compute_fn=lambda events: events["jet_pt"],
        units="GeV",
        binning=[20, 40, 60, 80, 100, 150, 200, 300],
    )

    counts, errors, edges = jet_pt.histogram(events, weights=result.nominal())
    nominal, stat_err = jet_pt.uncertainty_band(events, result)
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

import numpy as np


@dataclass
class Observable:
    """Definition of a physics observable with histogram computation.

    Parameters
    ----------
    name : str
        Short identifier, e.g. ``"jet_pt"``.
    expression : str
        Human-readable definition, e.g.
        ``"leading-jet transverse momentum"``.
    compute_fn : callable, optional
        ``f(events) -> array-like`` mapping an event array/dict to
        per-event values.  Must be provided to call :meth:`histogram`
        or :meth:`uncertainty_band`.
    units : str
        Physical units, e.g. ``"GeV"``.
    binning : list of float, optional
        Bin edges passed to ``numpy.histogram``.
    phase_space : dict, optional
        Selection cuts documented alongside the observable definition,
        e.g. ``{"jet_pt_min": 20, "abs_eta_max": 2.5}``.
    """

    name: str
    expression: str = ""
    compute_fn: Optional[Callable] = field(default=None, repr=False, compare=False)
    units: str = ""
    binning: Optional[List[float]] = None
    phase_space: Optional[dict] = None

    # -- Histogram computation -----------------------------------------------

    def histogram(self, events, weights=None):
        """Compute a weighted histogram of this observable.

        Parameters
        ----------
        events : array-like or dict
            Event data passed directly to :attr:`compute_fn`.
        weights : array-like, optional
            Per-event weights.  If *None*, all events are weighted equally.

        Returns
        -------
        counts : ndarray
            Weighted bin counts.
        errors : ndarray
            Per-bin statistical errors (``sqrt(sum(w^2))`` per bin).
        edges : ndarray
            Bin edges (length = ``len(counts) + 1``).
        """
        if self.compute_fn is None:
            raise ValueError(
                f"Observable '{self.name}' has no compute_fn; "
                "cannot compute histogram."
            )
        values = np.asarray(self.compute_fn(events), dtype=np.float64)

        if weights is None:
            weights = np.ones(len(values))
        weights = np.asarray(weights, dtype=np.float64)

        # Filter NaN / Inf from both values and weights
        mask = np.isfinite(values) & np.isfinite(weights)
        values = values[mask]
        w = weights[mask]

        if self.binning is not None:
            bins = self.binning
        else:
            # numpy rejects 'auto' when weights are provided; compute
            # bin edges from unweighted data first, then reuse them.
            _, bins = np.histogram(values, bins="auto")

        counts, edges = np.histogram(values, bins=bins, weights=w)
        errors = np.sqrt(
            np.histogram(values, bins=edges, weights=w ** 2)[0]
        )
        return counts, errors, edges

    def uncertainty_band(self, events, result):
        """Compute nominal histogram plus statistical uncertainty from replicas.

        Uses the standard deviation across all stored replicas as the
        per-bin statistical uncertainty.

        Parameters
        ----------
        events : array-like or dict
            Event data.
        result : OmniFoldWeights
            Published result containing nominal weights and replicas.

        Returns
        -------
        nominal_counts : ndarray
            Histogram using nominal weights.
        stat_error : ndarray
            Per-bin uncertainty from replica spread (std across replicas).
        edges : ndarray
            Bin edges.

        Raises
        ------
        ValueError
            If *result* has no replicas.
        """
        if result.n_replicas() == 0:
            raise ValueError(
                "No replicas found in result; cannot compute uncertainty band. "
                "Add replicas with result.add_replica(name, weights_array)."
            )

        nominal_counts, _, edges = self.histogram(events, result.nominal())

        replica_hists = []
        for _, w in result.replicas():
            counts, _, _ = self.histogram(events, w)
            replica_hists.append(counts)

        replica_hists = np.array(replica_hists)
        stat_error = np.std(replica_hists, axis=0)

        return nominal_counts, stat_error, edges

    def systematic_variations(self, events, result):
        """Compute histograms for all systematic variations.

        Parameters
        ----------
        events : array-like or dict
            Event data.
        result : OmniFoldWeights
            Published result containing systematic weight sets.

        Returns
        -------
        dict
            Mapping ``{systematic_name: (counts, errors, edges)}``.

        Raises
        ------
        ValueError
            If *result* has no systematics.
        """
        if result.n_systematics() == 0:
            raise ValueError(
                "No systematics found in result. "
                "Add them with result.add_systematic(name, weights_array)."
            )
        out = {}
        for name in result.systematic_names():
            w = result.systematic(name)
            out[name] = self.histogram(events, w)
        return out

    # -- Serialization -------------------------------------------------------

    def to_dict(self):
        """Serialize to a plain dict (JSON-compatible, excludes compute_fn)."""
        d = asdict(self)
        d.pop("compute_fn", None)  # not serializable
        return d

    def to_json(self, path=None):
        """Serialize to JSON, optionally writing to *path*."""
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d, compute_fn=None):
        """Reconstruct from a dict, optionally injecting a *compute_fn*."""
        d = d.copy()
        d.pop("compute_fn", None)
        return cls(compute_fn=compute_fn, **d)

    @classmethod
    def from_json(cls, path, compute_fn=None):
        """Load from a JSON file, optionally injecting a *compute_fn*."""
        with open(path) as f:
            return cls.from_dict(json.load(f), compute_fn=compute_fn)


# -- Convenience functions ---------------------------------------------------

def uncertainty_band(result, observable, events):
    """Convenience wrapper for :meth:`Observable.uncertainty_band`.

    Parameters
    ----------
    result : OmniFoldWeights
    observable : Observable
    events : array-like or dict

    Returns
    -------
    nominal_counts, stat_error, edges
    """
    return observable.uncertainty_band(events, result)
