"""End-to-end plotting example for OmniFold publication sanity checks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec.weighted_histogram import plot_weighted_histogram


def main() -> None:
    df = pd.read_hdf(PROJECT_ROOT / "data/multifold.h5", "df")

    values = df["pT_ll"].to_numpy()
    weights = df["weights_nominal"].to_numpy()

    # Trim extreme tails for a stable, publication-style preview plot.
    upper = float(np.nanquantile(values, 0.995))

    fig, ax, hist, _, _ = plot_weighted_histogram(
        values=values,
        weights=weights,
        bins=50,
        hist_range=(0.0, upper),
        density=False,
        label="multifold nominal",
        xlabel=r"$p_{T}^{\ell\ell}$ [GeV]",
    )

    if not np.all(np.isfinite(hist)):
        raise ValueError("Histogram contains non-finite values.")

    ax.set_title("Weighted $p_{T}^{\\ell\\ell}$ Distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "examples/example_histogram.png", dpi=160)
    fig.clf()


if __name__ == "__main__":
    main()
