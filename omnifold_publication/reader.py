"""Read the OmniFold publication package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SUPPORTED_FORMAT_VERSIONS = {"0.1", "0.2"}
DEFAULT_HDF_KEY = "df"


def _resolve_metadata_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "metadata.yaml" if path.is_dir() else path


def _column_from_spec(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and isinstance(spec.get("column"), str):
        return spec["column"]
    raise KeyError(f"Weight specification does not define a column: {spec!r}")


def ensure_supported_format_version(metadata: dict[str, Any]) -> None:
    """Raise a clear error when package metadata uses an unsupported version."""

    version = metadata.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FORMAT_VERSIONS))
        raise ValueError(
            f"Unsupported format_version {version!r}; supported versions: {supported}."
        )


def load_metadata(path: str | Path, enforce_version: bool = False) -> dict[str, Any]:
    """Load package metadata from a package directory or metadata file path."""

    metadata_path = _resolve_metadata_path(path)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Package metadata must be a mapping.")
    if enforce_version:
        ensure_supported_format_version(metadata)
    return metadata


def _resolve_data_path(base_path: Path, data_path: str) -> Path:
    data = Path(data_path)
    if data.is_absolute():
        return data
    return base_path / data


def load_events(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load event data declared by metadata or directly from an event file."""

    path = Path(path)
    if path.is_dir():
        metadata = load_metadata(path)
        files = metadata.get("files", {})
        nominal_file = files.get("nominal", {}) if isinstance(files, dict) else {}
        if isinstance(nominal_file, dict) and "path" in nominal_file:
            events_path = _resolve_data_path(path, nominal_file["path"])
            return pd.read_hdf(events_path, key=DEFAULT_HDF_KEY, columns=columns)

        events_file = metadata["publication"]["events_file"]
        events_path = path / events_file
        return pd.read_parquet(events_path, columns=columns)
    else:
        events_path = path
        if events_path.suffix in {".h5", ".hdf5"}:
            return pd.read_hdf(events_path, key=DEFAULT_HDF_KEY, columns=columns)
        return pd.read_parquet(events_path, columns=columns)


def list_systematics(metadata: dict[str, Any]) -> list[str]:
    """Return systematic variation names declared in package metadata."""

    file_block = metadata.get("files", {})
    files = file_block.get("systematics", []) if isinstance(file_block, dict) else []
    if isinstance(files, list) and files:
        return sorted(
            Path(systematic["path"]).stem
            for systematic in files
            if isinstance(systematic, dict) and "path" in systematic
        )

    systematics = metadata.get("systematics", {})
    if isinstance(systematics, dict) and systematics:
        return sorted(systematics)

    weights = metadata.get("weights", {})
    if isinstance(weights, dict) and "replica" in weights:
        return ["replica"]
    return []


def resolve_weight_column(
    metadata: dict[str, Any],
    variation: str = "nominal",
    iteration: int | None = None,
    step: str | None = None,
) -> str:
    """Resolve a metadata-declared weight selection to a concrete column name."""

    weights = metadata.get("weights", {})
    if not isinstance(weights, dict):
        raise KeyError("Metadata key 'weights' must be a mapping.")

    if iteration is not None or step is not None:
        if iteration is None or step is None:
            raise KeyError("Both iteration and step are required for iteration weights.")
        if step not in {"step1", "step2"}:
            raise KeyError("Iteration step must be 'step1' or 'step2'.")
        for entry in weights.get("iterations", []):
            if entry.get("iteration") == iteration and step in entry:
                return _column_from_spec(entry[step])
        raise KeyError(
            f"No weight column declared for iteration={iteration}, step={step!r}."
        )

    if variation == "nominal":
        return _column_from_spec(weights["nominal"])

    if variation in weights and isinstance(weights[variation], str):
        return weights[variation]

    raise KeyError(f"Unknown metadata-declared weight variation: {variation}")


def get_weights(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    variation: str = "nominal",
    iteration: int | None = None,
    step: str | None = None,
):
    """Return the requested weight array from the loaded event table."""

    column = resolve_weight_column(
        metadata,
        variation=variation,
        iteration=iteration,
        step=step,
    )
    if column not in df.columns:
        raise KeyError(f"Weight column {column!r} is not present in the event table.")
    return df[column].to_numpy()


def get_uncertainty(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    variation: str,
) -> np.ndarray:
    """Return per-event absolute difference between a variation and nominal weights."""

    nominal = get_weights(df, metadata, variation="nominal")
    varied = get_weights(df, metadata, variation=variation)
    return np.abs(varied - nominal)


class OmniFoldPackage:
    """Thin wrapper matching the proposal-facing package API."""

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        self._metadata = load_metadata(self.package_dir, enforce_version=True)

    def load_events(self, columns: list[str] | None = None) -> pd.DataFrame:
        return load_events(self.package_dir, columns=columns)

    def list_systematics(self) -> list[str]:
        return list_systematics(self._metadata)

    def get_weights(
        self,
        kind: str = "nominal",
        variation: str | None = None,
        iteration: int | None = None,
        step: str | None = None,
    ):
        selection = variation or kind
        column = resolve_weight_column(
            self._metadata,
            variation=selection,
            iteration=iteration,
            step=step,
        )
        df = self.load_events(columns=[column])
        return get_weights(
            df,
            self._metadata,
            variation=selection,
            iteration=iteration,
            step=step,
        )

    def get_uncertainty(self, variation: str) -> np.ndarray:
        nominal_column = resolve_weight_column(self._metadata, variation="nominal")
        variation_column = resolve_weight_column(self._metadata, variation=variation)
        columns = list(dict.fromkeys([nominal_column, variation_column]))
        df = self.load_events(columns=columns)
        return get_uncertainty(df, self._metadata, variation=variation)

    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def validate(self) -> None:
        from .validation import ensure_valid_package

        ensure_valid_package(self.package_dir)


def load_package(package_dir: str | Path) -> OmniFoldPackage:
    """Load a publication package and return the proposal-style wrapper."""

    return OmniFoldPackage(package_dir)
