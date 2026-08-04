from __future__ import annotations

import numpy as np

from zod_driveformer.dynamics.data import DynamicsRoleArrays
from zod_driveformer.dynamics.experiment import dynamics_metrics


def test_dynamics_metrics_are_per_sample_then_averaged() -> None:
    target = np.zeros((2, 3, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[0, :, 0] = 1.0
    prediction[1, :, 0] = 3.0
    arrays = DynamicsRoleArrays(
        states=np.zeros((2, 2, 1), dtype=np.float32),
        state_valid_mask=np.ones((2, 2, 1), dtype=np.bool_),
        target=target,
        target_valid_mask=np.ones((2, 3), dtype=np.bool_),
        group_index=np.asarray([0, 1], dtype=np.int32),
        sample_digest=np.asarray([b"a", b"b"]),
    )
    summary, rows = dynamics_metrics(prediction, arrays, miss_threshold_m=2.0)
    assert summary == {"ade_m": 2.0, "fde_m": 2.0, "miss_2m": 0.5}
    np.testing.assert_allclose(rows["ade_m"], [1.0, 3.0])
