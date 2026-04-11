"""Metadata schema for standardized OmniFold result publication."""

import json
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone


def _filter_to_fields(cls, d):
    """Filter a dict to only keys that match dataclass fields."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class ModelInfo:
    """Model architecture and training configuration."""
    architecture: str = ""
    optimizer: str = "Adam"
    epochs_step1: int = 20
    epochs_step2: int = 20
    batch_size_step1: int = 10000
    batch_size_step2: int = 2000


@dataclass
class DatasetInfo:
    """Dataset provenance information."""
    name: str = ""
    n_events_sim: int = 0
    n_events_data: int = 0
    provenance: str = ""
    description: str = ""


@dataclass
class OmniFoldMetadata:
    """Complete metadata for an OmniFold publication.

    Captures all information needed to understand, reproduce, and
    reinterpret a published OmniFold unfolding result.
    """
    schema_version: str = "1.0"
    omnifold_version: str = ""
    iterations: int = 0
    model: ModelInfo = field(default_factory=ModelInfo)
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    weight_format: str = "hdf5"
    normalization: str = "unit"
    created_at: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.omnifold_version:
            from omnifold._version import __version__
            self.omnifold_version = __version__
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # -- Serialization ---------------------------------------------------

    def to_dict(self):
        """Convert to a plain dict (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Reconstruct from a plain dict.

        Unknown keys are silently ignored for forward compatibility.
        """
        d = d.copy()
        model_data = d.pop("model", {})
        dataset_data = d.pop("dataset", {})
        model = ModelInfo(**_filter_to_fields(ModelInfo, model_data))
        dataset = DatasetInfo(**_filter_to_fields(DatasetInfo, dataset_data))
        return cls(
            model=model,
            dataset=dataset,
            **_filter_to_fields(cls, d),
        )

    def to_json(self, path=None):
        """Serialize to JSON string, optionally writing to a file."""
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, path):
        """Load metadata from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # -- Validation ------------------------------------------------------

    def validate(self):
        """Check that required fields are consistent.

        Raises ``ValueError`` on the first problem found.
        Returns ``True`` when valid.
        """
        errors = []
        if self.iterations <= 0:
            errors.append("iterations must be positive")
        if self.weight_format not in ("hdf5", "parquet"):
            errors.append(f"unsupported weight_format: {self.weight_format}")
        if self.normalization not in ("unit", "cross_section", "none"):
            errors.append(f"unsupported normalization: {self.normalization}")
        if errors:
            raise ValueError(
                "Schema validation failed: " + "; ".join(errors)
            )
        return True
