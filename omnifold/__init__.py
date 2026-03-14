"""OmniFold: iterative unfolding with machine learning.

Publication layer (weights, schema, publication) is always available.
Core algorithm (omnifold, reweight) requires TensorFlow.
"""

from omnifold._version import __version__
from omnifold.weights import OmniFoldWeights
from omnifold.schema import OmniFoldMetadata, ModelInfo, DatasetInfo
from omnifold.publication import publish, load

# Core algorithm -- requires TensorFlow, so import is optional.
# Users who only need to load/apply published weights can use the
# package without TensorFlow installed.
try:
    from omnifold.core import omnifold, reweight, weighted_binary_crossentropy
except ImportError:
    pass

__all__ = [
    "__version__",
    "omnifold",
    "reweight",
    "weighted_binary_crossentropy",
    "OmniFoldWeights",
    "OmniFoldMetadata",
    "ModelInfo",
    "DatasetInfo",
    "publish",
    "load",
]
