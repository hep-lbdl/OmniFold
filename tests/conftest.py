from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def source_hdf(tmp_path):
    """Small HDF5 source file with nominal, systematic, and iteration weights."""

    path = tmp_path / "source_multifold.h5"
    df = pd.DataFrame(
        {
            "event_id": np.arange(6),
            "pT_ll": np.array([10.0, 20.0, 35.0, 50.0, 80.0, 120.0]),
            "pT_l1": np.array([25.0, 28.0, 32.0, 40.0, 45.0, 60.0]),
            "weight_mc": np.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8]),
            "weights_nominal": np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3]),
            "weights_ensemble_0": np.array([1.1, 1.1, 0.9, 1.0, 1.0, 1.2]),
            "weights_iter0_step1": np.array([0.9, 1.0, 1.1, 1.0, 0.95, 1.05]),
            "weights_iter0_step2": np.array([1.0, 1.1, 1.0, 1.2, 1.0, 0.9]),
        }
    )
    df.to_hdf(path, key="df", mode="w")
    return path
