"""OmniFold: iterative unfolding with machine learning.

Publication layer (weights, schema, publication, observable, validation)
is always available.  Core algorithm (omnifold, reweight) requires TensorFlow.
"""

from omnifold._version import __version__
from omnifold.weights import OmniFoldWeights
from omnifold.schema import OmniFoldMetadata, ModelInfo, DatasetInfo
from omnifold.publication import publish, load
from omnifold.observable import Observable, uncertainty_band
from omnifold.validation import validate, ValidationReport, CheckResult

# Core algorithm -- requires TensorFlow, so import is optional.
# Users who only need to load/apply published weights can use the
# package without TensorFlow installed.
try:
    from omnifold.core import omnifold, reweight, weighted_binary_crossentropy
except ImportError:
    pass

__all__ = [
    "__version__",
    # Core algorithm (TF-dependent)
    "omnifold",
    "reweight",
    "weighted_binary_crossentropy",
    # Weight container
    "OmniFoldWeights",
    # Schema
    "OmniFoldMetadata",
    "ModelInfo",
    "DatasetInfo",
    # Publication API
    "publish",
    "load",
    # Observables & reinterpretation
    "Observable",
    "uncertainty_band",
    # Validation
    "validate",
    "ValidationReport",
    "CheckResult",
]
