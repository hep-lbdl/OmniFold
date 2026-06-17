"""Validation helpers for the OmniFold publication package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reader import (
    SUPPORTED_FORMAT_VERSIONS,
    list_systematics,
    load_metadata,
    resolve_weight_column,
)


REQUIRED_METADATA_KEYS = ("format_version", "observables", "weights", "publication")
REQUIRED_PUBLICATION_KEYS = ("format", "events_file", "event_count", "columns")
REQUIRED_WEIGHT_KEYS = ("nominal", "base_mc_weight")


def _weight_columns(metadata: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for variation in ["nominal", *list_systematics(metadata)]:
        try:
            columns.append(resolve_weight_column(metadata, variation=variation))
        except KeyError:
            continue

    weights = metadata.get("weights", {})
    if isinstance(weights, dict):
        try:
            columns.append(resolve_weight_column(metadata, variation="base_mc_weight"))
        except KeyError:
            base = weights.get("base_mc_weight")
            if isinstance(base, str):
                columns.append(base)

        for entry in weights.get("iterations", []):
            for step in ("step1", "step2"):
                step_spec = entry.get(step)
                if isinstance(step_spec, str):
                    columns.append(step_spec)
                elif isinstance(step_spec, dict) and isinstance(
                    step_spec.get("column"), str
                ):
                    columns.append(step_spec["column"])

    return list(dict.fromkeys(columns))


def _required_columns(metadata: dict[str, Any]) -> set[str]:
    publication = metadata.get("publication", {})
    required_columns = set(publication.get("columns", []))

    for observable in metadata.get("observables", []):
        if isinstance(observable, dict) and "name" in observable:
            required_columns.add(observable["name"])

    required_columns.update(_weight_columns(metadata))

    alignment = publication.get("event_alignment", {})
    if isinstance(alignment, dict) and alignment.get("method") == "column":
        column = alignment.get("column")
        if isinstance(column, str):
            required_columns.add(column)

    return required_columns


def _validate_format_version(metadata: dict[str, Any]) -> list[str]:
    version = metadata.get("format_version")
    if not isinstance(version, str):
        return ["format_version must be a string, for example '0.2'."]
    if version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FORMAT_VERSIONS))
        return [
            f"Unsupported format_version {version!r}; supported versions: {supported}."
        ]
    return []


def validate_event_alignment(df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    """Validate the event-alignment contract declared by package metadata."""

    publication = metadata.get("publication", {})
    alignment = publication.get("event_alignment", {"method": "row_order"})
    if not isinstance(alignment, dict):
        return ["publication.event_alignment must be a mapping."]

    method = alignment.get("method", "row_order")
    if method == "row_order":
        return []
    if method != "column":
        return [f"Unsupported event alignment method: {method!r}."]

    column = alignment.get("column")
    if not isinstance(column, str):
        return [
            "Column-based event alignment requires publication.event_alignment.column."
        ]
    if column not in df.columns:
        return [f"Event alignment column {column!r} is missing from events.parquet."]

    errors: list[str] = []
    if df[column].isna().any():
        errors.append(f"Event alignment column {column!r} contains null values.")
    if df[column].duplicated().any():
        errors.append(f"Event alignment column {column!r} contains duplicate values.")
    return errors


def validate_weight_lengths(df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    """Check that declared weight columns are present, finite, and row-aligned."""

    errors: list[str] = []
    event_count = len(df)
    for column in _weight_columns(metadata):
        if column not in df.columns:
            errors.append(
                f"Declared weight column {column!r} is missing from events.parquet."
            )
            continue
        if len(df[column]) != event_count:
            errors.append(
                f"Declared weight column {column!r} has length {len(df[column])}; "
                f"expected {event_count}."
            )
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            errors.append(f"Declared weight column {column!r} contains non-finite values.")
    return errors


def validate_normalization(
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[str]:
    """Validate normalization metadata against the stored event table."""

    normalization = metadata.get("normalization")
    if not isinstance(normalization, dict):
        return []

    column = normalization.get("nominal_weight_column")
    if not isinstance(column, str):
        try:
            column = resolve_weight_column(metadata, variation="nominal")
        except KeyError:
            return ["normalization.nominal_weight_column is missing."]

    if column not in df.columns:
        return [f"Normalization weight column {column!r} is missing from events.parquet."]

    weights = df[column].to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        return [f"Normalization weight column {column!r} contains non-finite values."]

    errors: list[str] = []
    expected = normalization.get("expected_nominal_sumw")
    if expected is not None:
        actual = float(weights.sum())
        tolerance = float(normalization.get("tolerance", 1.0e-8))
        allowed = tolerance * max(1.0, abs(float(expected)))
        if abs(actual - float(expected)) > allowed:
            errors.append(
                "nominal weight sum mismatch: "
                f"metadata={float(expected):.12g} actual={actual:.12g}"
            )

    return errors


def validate_package(path: str | Path) -> list[str]:
    """Return a list of validation errors for the package at ``path``."""

    package_dir = Path(path)
    metadata_path = package_dir / "metadata.yaml"
    errors: list[str] = []

    if not package_dir.exists():
        return [f"Package directory does not exist: {package_dir}"]
    if not metadata_path.exists():
        errors.append(f"Missing metadata file: {metadata_path}")
    if errors:
        return errors

    metadata = load_metadata(metadata_path)
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            errors.append(f"Missing metadata key: {key}")

    publication = metadata.get("publication", {})
    weights = metadata.get("weights", {})

    for key in REQUIRED_PUBLICATION_KEYS:
        if key not in publication:
            errors.append(f"Missing publication key: {key}")
    for key in REQUIRED_WEIGHT_KEYS:
        if key not in weights:
            errors.append(f"Missing weights key: {key}")

    if errors:
        return errors

    errors.extend(_validate_format_version(metadata))

    if publication["format"] != "parquet":
        errors.append("publication.format must be 'parquet'.")

    events_path = package_dir / publication["events_file"]
    if not events_path.exists():
        errors.append(f"Missing events file: {events_path}")

    if errors:
        return errors

    df = pd.read_parquet(events_path)
    required_columns = _required_columns(metadata)

    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        errors.append(
            "Missing required columns in events.parquet: "
            + ", ".join(missing_columns)
        )

    actual_event_count = int(len(df))
    if actual_event_count != int(publication["event_count"]):
        errors.append(
            "event_count mismatch: "
            f"metadata={publication['event_count']} actual={actual_event_count}"
        )

    errors.extend(validate_event_alignment(df, metadata))
    errors.extend(validate_weight_lengths(df, metadata))
    errors.extend(validate_normalization(df, metadata))

    return errors


def ensure_valid_package(path: str | Path) -> None:
    """Raise ``ValueError`` if the package is invalid."""

    errors = validate_package(path)
    if errors:
        raise ValueError("\n".join(errors))


def closure_test(
    path: str | Path,
    observable: str,
    bins: int | np.ndarray = 50,
    variation: str = "nominal",
) -> dict[str, Any]:
    """Compute a simple histogram closure summary for a packaged observable."""

    from .reader import load_package

    package = load_package(path)
    weights = package.get_weights(variation=variation)
    df = package.load_events(columns=[observable])
    if observable not in df.columns:
        raise KeyError(f"Observable {observable!r} is not present in the event table.")

    hist, edges = np.histogram(
        df[observable].to_numpy(dtype=float),
        bins=bins,
        weights=weights,
    )
    sumw = float(weights.sum())
    hist_sumw = float(hist.sum())
    denominator = max(1.0, abs(sumw))

    return {
        "hist": hist,
        "edges": edges,
        "sumw": sumw,
        "hist_sumw": hist_sumw,
        "relative_difference": abs(hist_sumw - sumw) / denominator,
    }
