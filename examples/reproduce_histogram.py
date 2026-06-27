"""Reproduce a weighted histogram from an OmniFold publication package."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omnifold_publication import closure_test, load_package
from spec.weighted_histogram import compute_weighted_histogram


def main() -> None:
    package_dir = PROJECT_ROOT / "artifacts/demo_nominal"
    if not package_dir.exists():
        raise SystemExit(
            "Package not found. Run `python3 examples/package_roundtrip.py` first."
        )

    pkg = load_package(package_dir)
    metadata = pkg.metadata()

    observable = metadata["observables"][0]["name"]
    df = pkg.load_events(columns=[observable])
    weights = pkg.get_weights(variation="nominal")
    bins = np.linspace(0.0, 200.0, 26)

    result = compute_weighted_histogram(df[observable], weights, bins=bins)
    closure = closure_test(package_dir, observable=observable, bins=bins)

    print(f"Package: {package_dir}")
    print(f"Observable: {observable}")
    print(f"Nominal histogram bins: {len(result['hist'])}")
    print(f"Closure relative difference: {closure['relative_difference']:.3e}")

    systematics = pkg.list_systematics()
    if systematics:
        variation = systematics[0]
        uncertainty = pkg.get_uncertainty(variation)
        print(f"Example systematic: {variation}")
        print(f"Mean per-event weight shift: {np.mean(uncertainty):.6g}")


if __name__ == "__main__":
    main()
