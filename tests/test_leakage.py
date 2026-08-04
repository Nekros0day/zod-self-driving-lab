from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zod_driveformer.data import (
    Split,
    TrainOnlyNormalizer,
    assert_causal,
    assert_disjoint_recordings,
    build_recording_windows,
    make_recording_splits,
    make_synthetic_recording,
)


def test_future_input_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-causal"):
        assert_causal([0.0, 1.0, 2.0001], 2.0)


def test_window_validation_rejects_a_future_camera_frame() -> None:
    recording = make_synthetic_recording()
    window = build_recording_windows(recording)[0]
    leaked_times = list(window.camera_timestamps)
    leaked_times[-1] = window.t0 + 0.01
    with pytest.raises(ValueError, match="camera_timestamps"):
        replace(window, camera_timestamps=tuple(leaked_times))


def test_recordings_cannot_overlap_across_any_split() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        assert_disjoint_recordings(
            {
                "train": ["sequence-a", "sequence-b"],
                "validation": ["sequence-c"],
                "calibration": ["sequence-b"],
                "test": ["sequence-d"],
            }
        )


def test_windows_from_one_recording_share_one_partition() -> None:
    recording_ids = [f"recording-{index}" for index in range(20)]
    splits = make_recording_splits(recording_ids, seed=7)
    assignment = splits.by_recording()
    repeated_window_ids = np.repeat(recording_ids, 50)
    for recording_id in recording_ids:
        labels = {assignment[item] for item in repeated_window_ids if item == recording_id}
        assert len(labels) == 1
    assert set(splits.calibration).isdisjoint(splits.test)


@pytest.mark.parametrize("split", [Split.VALIDATION, Split.CALIBRATION, Split.TEST])
def test_normalizer_refuses_non_train_fit(split: Split) -> None:
    with pytest.raises(ValueError, match="only be fit on train"):
        TrainOnlyNormalizer().fit([[1.0], [2.0]], split=split)


def test_window_state_and_camera_provenance_never_passes_t0() -> None:
    recording = make_synthetic_recording(motion="left_turn")
    for window in build_recording_windows(recording):
        assert max(window.camera_timestamps) <= window.t0
        assert max(window.state_left_timestamps) <= window.t0
        assert max(window.state_right_timestamps) <= window.t0
        assert window.reference_pose_timestamp <= window.t0
        assert min(window.target_pose_timestamps) > window.t0
