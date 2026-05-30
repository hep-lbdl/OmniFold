from __future__ import annotations

import numpy as np
import pandas as pd

from omnifold_publication import (
    ensure_valid_package,
    load_package,
    write_package,
)
from spec.weighted_histogram import compute_weighted_histogram


def test_package_roundtrip_matches_direct_histogram(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    ensure_valid_package(package_dir)

    pkg = load_package(package_dir)
    metadata = pkg.metadata()
    packaged_df = pkg.load_events()
    direct_df = pd.read_hdf(source_hdf, "df").iloc[:6]

    observable = metadata["observables"][0]["name"]
    bins = np.linspace(0.0, 200.0, 26)

    packaged_result = compute_weighted_histogram(
        packaged_df[observable].to_numpy(),
        pkg.get_weights(),
        bins=bins,
    )
    direct_result = compute_weighted_histogram(
        direct_df[observable].to_numpy(),
        direct_df[metadata["weights"]["nominal"]].to_numpy(),
        bins=bins,
    )

    np.testing.assert_allclose(packaged_result["hist"], direct_result["hist"])
    np.testing.assert_allclose(packaged_result["edges"], direct_result["edges"])
    np.testing.assert_allclose(
        packaged_result["uncertainty"],
        direct_result["uncertainty"],
    )


def test_function_api_remains_available(tmp_path, source_hdf):
    from omnifold_publication import get_weights, load_events, load_metadata

    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    metadata = load_metadata(package_dir)
    events = load_events(package_dir, columns=["pT_ll", metadata["weights"]["nominal"]])
    weights = get_weights(events, metadata)

    assert events.shape[0] == len(weights)


def test_systematics_and_iteration_weights_are_metadata_driven(tmp_path, source_hdf):
    package_dir = write_package(
        input_path=source_hdf,
        output_dir=tmp_path / "demo_nominal",
        event_count=6,
    )
    pkg = load_package(package_dir)
    source = pd.read_hdf(source_hdf, "df")

    assert pkg.list_systematics() == ["replica"]
    np.testing.assert_allclose(
        pkg.get_weights(variation="replica"),
        source["weights_ensemble_0"].to_numpy(),
    )
    np.testing.assert_allclose(
        pkg.get_weights(iteration=0, step="step1"),
        source["weights_iter0_step1"].to_numpy(),
    )
    np.testing.assert_allclose(
        pkg.get_uncertainty("replica"),
        np.abs(source["weights_ensemble_0"] - source["weights_nominal"]),
    )
