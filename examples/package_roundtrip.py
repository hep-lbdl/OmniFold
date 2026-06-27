"""Create and verify a minimal OmniFold publication package."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omnifold_publication import (
    ensure_valid_package,
    load_package,
    write_package,
)
from omnifold_publication.writer import DEFAULT_EVENT_COUNT, DEFAULT_INPUT_PATH
from spec.weighted_histogram import compute_weighted_histogram


def main() -> None:
    package_dir = write_package()
    ensure_valid_package(package_dir)

    pkg = load_package(package_dir)
    metadata = pkg.metadata()
    packaged_df = pkg.load_events()
    direct_df = pd.read_hdf(DEFAULT_INPUT_PATH, "df").iloc[:DEFAULT_EVENT_COUNT]

    observable = metadata["observables"][0]["name"]
    bins = np.linspace(0.0, 200.0, 26)
    packaged_result = compute_weighted_histogram(
        packaged_df[observable].to_numpy(),
        pkg.get_weights(kind="nominal"),
        bins=bins,
    )
    direct_result = compute_weighted_histogram(
        direct_df[observable].to_numpy(),
        direct_df[metadata["weights"]["nominal"]].to_numpy(),
        bins=bins,
    )

    np.testing.assert_allclose(packaged_result["hist"], direct_result["hist"])
    np.testing.assert_allclose(
        packaged_result["uncertainty"],
        direct_result["uncertainty"],
    )

    print(f"Package created at {Path(package_dir).resolve()}")
    print(f"Validated {metadata['publication']['event_count']} events")
    print(f"Histogram roundtrip check passed for {observable}")


if __name__ == "__main__":
    main()
