"""Write the OmniFold publication package to Parquet."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .exceptions import PackageWriteError


DEFAULT_INPUT_PATH = Path("data/multifold.h5")
DEFAULT_METADATA_SOURCE = Path("spec/metadata.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts/demo_nominal")
DEFAULT_EVENT_COUNT = 10_000
PRIMARY_OBSERVABLE = "pT_ll"
EXTRA_OBSERVABLE = "pT_l1"
EVENT_ID_COLUMN = "event_id"
BASE_WEIGHT_COLUMN = "weight_mc"
NOMINAL_WEIGHT_COLUMN = "weights_nominal"
REPLICA_PREFIXES = ("weights_ensemble_", "weights_bootstrap_mc_")
FORMAT_VERSION = "0.2"


def _compute_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""

    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_source_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise PackageWriteError("Source metadata must be a mapping.")
    return data


def _find_replica_column(columns: list[str]) -> str | None:
    for prefix in REPLICA_PREFIXES:
        for column in columns:
            if column.startswith(prefix):
                return column
    return None


def _discover_iteration_weights(columns: list[str]) -> list[dict[str, Any]]:
    """Find step1/step2 iteration weights when the source file provides them."""

    patterns = (
        re.compile(r"^weights_(step[12])_(?:iter|iteration)_?(\d+)$"),
        re.compile(r"^weights_(?:iter|iteration)_?(\d+)_(step[12])$"),
    )
    by_iteration: dict[int, dict[str, dict[str, str]]] = {}

    for column in columns:
        for pattern in patterns:
            match = pattern.match(column)
            if match is None:
                continue
            first, second = match.groups()
            if first.startswith("step"):
                step = first
                iteration = int(second)
            else:
                iteration = int(first)
                step = second
            by_iteration.setdefault(iteration, {})[step] = {"column": column}
            break

    return [
        {"iteration": iteration, **steps}
        for iteration, steps in sorted(by_iteration.items())
    ]


def _build_systematics(replica_column: str | None) -> dict[str, dict[str, str]]:
    if replica_column is None:
        return {}
    return {
        "replica": {
            "column": replica_column,
            "type": "ensemble",
            "combination": "absolute_difference_from_nominal",
        }
    }


def _filter_observables(
    source_metadata: dict[str, Any],
    selected_names: list[str],
) -> list[dict[str, Any]]:
    observables = source_metadata.get("observables", [])
    if not isinstance(observables, list):
        return [{"name": name} for name in selected_names]

    filtered = [
        observable
        for observable in observables
        if isinstance(observable, dict) and observable.get("name") in selected_names
    ]
    if filtered:
        return filtered
    return [{"name": name} for name in selected_names]


def _build_package_metadata(
    source_metadata: dict[str, Any],
    observable_names: list[str],
    selected_columns: list[str],
    replica_column: str | None,
    iteration_weights: list[dict[str, Any]],
    event_count: int,
    nominal_sumw: float,
    input_path: Path,
    has_event_id: bool,
) -> dict[str, Any]:
    weights: dict[str, Any] = {
        "nominal": NOMINAL_WEIGHT_COLUMN,
        "base_mc_weight": BASE_WEIGHT_COLUMN,
    }
    if replica_column is not None:
        weights["replica"] = replica_column
    if iteration_weights:
        weights["iterations"] = iteration_weights

    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dataset": source_metadata.get("dataset", {}),
        "observables": _filter_observables(source_metadata, observable_names),
        "weights": weights,
        "systematics": _build_systematics(replica_column),
        "normalization": {
            "mode": "shape",
            "base_weight_column": BASE_WEIGHT_COLUMN,
            "nominal_weight_column": NOMINAL_WEIGHT_COLUMN,
            "expected_nominal_sumw": nominal_sumw,
            "tolerance": 1.0e-8,
        },
        "publication": {
            "format": "parquet",
            "events_file": "events.parquet",
            "event_count": event_count,
            "columns": selected_columns,
            "source_file": input_path.as_posix(),
            "event_alignment": {
                "method": "column" if has_event_id else "row_order",
                "column": EVENT_ID_COLUMN if has_event_id else None,
            },
        },
    }
    return metadata


def write_package(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    metadata_source: str | Path = DEFAULT_METADATA_SOURCE,
    event_count: int = DEFAULT_EVENT_COUNT,
    observables: list[str] | None = None,
) -> Path:
    """Create a minimal Parquet-backed publication package."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise PackageWriteError(
            f"Input file not found: {input_path}. "
            f"Make sure data/multifold.h5 is present locally."
        )

    output_dir = Path(output_dir)
    metadata_source = Path(metadata_source)

    df = pd.read_hdf(input_path, "df").iloc[:event_count].copy()
    source_columns = list(df.columns)
    replica_column = _find_replica_column(source_columns)
    iteration_weights = _discover_iteration_weights(source_columns)

    observable_names = observables or [PRIMARY_OBSERVABLE, EXTRA_OBSERVABLE]
    selected_columns = [
        *observable_names,
        BASE_WEIGHT_COLUMN,
        NOMINAL_WEIGHT_COLUMN,
    ]
    if EVENT_ID_COLUMN in df.columns:
        selected_columns.append(EVENT_ID_COLUMN)
    if replica_column is not None:
        selected_columns.append(replica_column)
    for iteration in iteration_weights:
        for step in ("step1", "step2"):
            step_spec = iteration.get(step)
            if isinstance(step_spec, dict):
                selected_columns.append(step_spec["column"])

    selected_columns = list(dict.fromkeys(selected_columns))

    package_df = df.loc[:, selected_columns]
    package_event_count = int(len(package_df))
    nominal_sumw = float(package_df[NOMINAL_WEIGHT_COLUMN].to_numpy(dtype=float).sum())
    source_metadata = _load_source_metadata(metadata_source)
    package_metadata = _build_package_metadata(
        source_metadata=source_metadata,
        observable_names=observable_names,
        selected_columns=selected_columns,
        replica_column=replica_column,
        iteration_weights=iteration_weights,
        event_count=package_event_count,
        nominal_sumw=nominal_sumw,
        input_path=input_path,
        has_event_id=EVENT_ID_COLUMN in package_df.columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.parquet"
    package_df.to_parquet(events_path, index=False)
    package_metadata["publication"]["checksum_sha256"] = _compute_checksum(events_path)
    with (output_dir / "metadata.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(package_metadata, stream, sort_keys=False)

    return output_dir
