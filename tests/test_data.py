from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from zod_driveformer.data import (
    Manifest,
    PoseSeries,
    Split,
    TrainOnlyNormalizer,
    WindowConfig,
    ZODSequenceAdapter,
    build_partitioned_windows,
    build_recording_windows,
    build_window_index,
    extract_trajectory_target,
    fit_normalizer_from_recordings,
    make_recording_splits,
    make_synthetic_adapter,
    make_synthetic_recording,
    manifest_hash,
    match_timestamps,
    read_manifest_jsonl,
    resample_window_state,
    stable_hash,
    write_manifest_jsonl,
)
from zod_driveformer.data.alignment import interpolate_timeseries, validate_timestamps
from zod_driveformer.data.zod_adapter import (
    _normalize_accelerator_ratio,
    _turn_indicator_flags,
    _turn_indicator_numeric,
)
from zod_driveformer.geometry import (
    from_rotation_translation,
    from_yaw_translation,
    future_trajectory,
)


def test_zod_turn_indicator_numeric_contract() -> None:
    numeric = _turn_indicator_numeric(
        np.asarray([0, 1, 2, "left", "right", "hazard", None], dtype=object)
    )
    left, right = _turn_indicator_flags(numeric)
    np.testing.assert_allclose(numeric[:6], [0, 1, 2, 1, 2, 3])
    np.testing.assert_allclose(left[:6], [0, 1, 0, 1, 0, 1])
    np.testing.assert_allclose(right[:6], [0, 0, 1, 0, 1, 1])
    assert np.isnan(left[-1]) and np.isnan(right[-1])


def test_zod_accelerator_normalization_handles_both_release_encodings() -> None:
    percentage, percentage_encoding = _normalize_accelerator_ratio(
        np.asarray([0.0, 40.15234375, 100.0])
    )
    ratio, ratio_encoding = _normalize_accelerator_ratio(
        np.asarray([0.0, 0.4015234375, 0.7615234375])
    )

    np.testing.assert_allclose(percentage, [0.0, 0.4015234375, 1.0])
    np.testing.assert_allclose(ratio, [0.0, 0.4015234375, 0.7615234375])
    assert percentage_encoding == "percentage_0_100"
    assert ratio_encoding == "ratio_0_1"


def test_zod_accelerator_normalization_fails_closed_when_scale_is_ambiguous() -> None:
    with pytest.raises(ValueError, match="complete parent-drive stream"):
        _normalize_accelerator_ratio(np.asarray([0.0, 1.0 / 256.0, 0.5]))


@pytest.mark.parametrize(
    "values, message",
    [
        (np.asarray([0.0, np.nan, 0.5]), "non-finite"),
        (np.asarray([-0.1, 0.5]), "negative"),
        (np.asarray([0.0, 101.0]), "0..100"),
    ],
)
def test_zod_accelerator_normalization_rejects_invalid_streams(
    values: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _normalize_accelerator_ratio(values)


def test_zod_yaw_rate_does_not_hold_across_an_unbounded_gap() -> None:
    adapter = ZODSequenceAdapter(dataset={}, recording_ids=(), yaw_rate_max_age_seconds=0.05)
    sequence = SimpleNamespace(
        oxts=SimpleNamespace(
            timestamps=np.asarray([0.0]),
            angular_rates=np.asarray([[0.0, 0.0, -30.0]]),
        )
    )
    aligned = adapter._yaw_rate(sequence, np.asarray([0.0, 0.04, 0.20]))
    np.testing.assert_allclose(aligned[:2], np.deg2rad([30.0, 30.0]))
    assert np.isnan(aligned[2])


def test_zod_yaw_rate_converts_oxts_degrees_and_down_axis_sign() -> None:
    adapter = ZODSequenceAdapter(dataset={}, recording_ids=())
    sequence = SimpleNamespace(
        oxts=SimpleNamespace(
            timestamps=np.asarray([0.0, 0.1, 0.2]),
            # OxTS +z points down: positive values are right turns in its
            # x-forward/y-right frame, opposite our positive-y left turns.
            angular_rates=np.asarray(
                [
                    [0.0, 0.0, 90.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -45.0],
                ]
            ),
        )
    )

    converted = adapter._yaw_rate(sequence, np.asarray([0.0, 0.1, 0.2]))

    np.testing.assert_allclose(converted, [-np.pi / 2.0, 0.0, np.pi / 4.0])


@pytest.mark.parametrize("name", ["control_max_age_seconds", "yaw_rate_max_age_seconds"])
def test_zod_adapter_rejects_invalid_source_age(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        ZODSequenceAdapter(dataset={}, recording_ids=(), **{name: -0.1})


def test_nearest_and_causal_timestamp_alignment_are_explicit() -> None:
    source = np.array([0.0, 1.0, 2.0])
    query = np.array([0.5, 1.5])
    nearest = match_timestamps(source, query)
    np.testing.assert_array_equal(nearest.indices, [0, 1])  # ties choose earlier
    causal = match_timestamps(source, np.array([0.25, 1.75]), causal=True)
    np.testing.assert_array_equal(causal.indices, [0, 1])
    assert np.all(causal.source_timestamps <= causal.query_timestamps)
    too_old = match_timestamps(source, np.array([0.4, 1.4]), max_delta=0.2, causal=True)
    np.testing.assert_array_equal(too_old.indices, [-1, -1])
    assert np.isnan(too_old.source_timestamps).all()
    repeated_unsorted = match_timestamps(source, np.array([1.75, 0.25, 1.75]), causal=True)
    np.testing.assert_array_equal(repeated_unsorted.indices, [1, 0, 1])


def test_timestamp_validation_rejects_duplicates_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_timestamps([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        validate_timestamps([0.0, np.nan])


def test_interpolation_handles_masks_gaps_and_wrapped_yaw() -> None:
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[0.0, np.deg2rad(179.0)], [10.0, np.deg2rad(-179.0)], [20.0, 0.0]])
    result = interpolate_timeseries(times, values, [0.5, 1.5], angle_columns=(1,), max_gap=1.0)
    np.testing.assert_allclose(result.values[:, 0], [5.0, 15.0])
    assert abs(abs(result.values[0, 1]) - np.pi) < 1e-12
    masked = interpolate_timeseries(
        times,
        values,
        [0.5],
        valid_mask=np.array([[True, True], [False, True], [True, True]]),
    )
    assert np.isnan(masked.values[0, 0])
    assert masked.valid[0, 1]


def test_default_synthetic_window_has_causal_inputs_and_known_target() -> None:
    recording = make_synthetic_recording(speed_mps=8.0)
    windows = build_recording_windows(recording)
    assert windows
    window = windows[0]
    assert len(window.camera_indices) == WindowConfig().camera_frames
    assert len(window.state_query_timestamps) == WindowConfig().state_samples
    assert len(window.target_pose_indices) == WindowConfig().target_samples
    assert max(window.camera_timestamps) <= window.t0
    assert max(window.state_right_timestamps) <= window.t0
    state = resample_window_state(window, recording.vehicle_state)
    assert state.valid.all()
    np.testing.assert_allclose(state.values[:, 0], 8.0)
    np.testing.assert_allclose(state.values[:, -1], 0.05)
    target = extract_trajectory_target(window, recording.ego_poses)
    np.testing.assert_allclose(target[:, 0], np.arange(1, 31) * 0.8, atol=1e-10)
    np.testing.assert_allclose(target[:, 1], 0.0, atol=1e-10)


def _nonplanar_pose_at(timestamp: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    angles = 0.8 * np.asarray(timestamp, dtype=np.float64)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotations = np.asarray(
        [
            np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
            for angle in angles
        ]
    )
    translations = np.column_stack((2.0 * timestamp, -timestamp, 0.5 * timestamp))
    return from_rotation_translation(rotations, translations)


def test_irregular_pose_labels_are_interpolated_at_exact_query_times() -> None:
    pose_times = np.array([0.0, 0.29, 0.41, 0.53, 0.61, 0.69, 0.80])
    pose_matrices = _nonplanar_pose_at(pose_times)
    config = WindowConfig(
        history_seconds=0.2,
        future_seconds=0.3,
        camera_frames=2,
        state_hz=10.0,
        target_hz=10.0,
        stride_seconds=0.1,
        camera_max_delta=0.01,
        state_max_gap=0.11,
        pose_max_delta=0.08,
    )
    windows = build_window_index(
        "irregular",
        [0.149, 0.35],
        [0.15, 0.25, 0.35],
        pose_times,
        config=config,
        candidate_timestamps=[0.35],
        pose_world_from_ego=pose_matrices,
    )
    assert len(windows) == 1
    window = windows[0]
    assert window.reference_pose_timestamp == pytest.approx(0.29)
    assert window.reference_pose_right_timestamp == pytest.approx(0.41)
    assert window.target_pose_timestamps == pytest.approx((0.41, 0.53, 0.61))
    assert window.target_pose_right_timestamps == pytest.approx((0.53, 0.61, 0.69))

    exact_times = np.array([0.35, 0.45, 0.55, 0.65])
    exact_poses = _nonplanar_pose_at(exact_times)
    expected = future_trajectory(exact_poses[0], exact_poses[1:])
    actual = extract_trajectory_target(window, PoseSeries(pose_times, pose_matrices))
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(window.frozen_target_xy, expected, atol=1e-10)
    record = window.to_record()
    assert record["frozen_target_valid_mask"] == (True, True, True)
    assert "frozen_target_xy" in record


def test_pose_brackets_enforce_tolerance_on_both_sides() -> None:
    pose_times = np.array([0.0, 0.29, 0.41, 0.53, 0.61, 0.69, 0.80])
    windows = build_window_index(
        "irregular",
        [0.149, 0.35],
        [0.15, 0.25, 0.35],
        pose_times,
        config=WindowConfig(
            history_seconds=0.2,
            future_seconds=0.3,
            camera_frames=2,
            state_hz=10.0,
            target_hz=10.0,
            stride_seconds=0.1,
            camera_max_delta=0.01,
            state_max_gap=0.11,
            pose_max_delta=0.05,
        ),
        candidate_timestamps=[0.35],
        pose_world_from_ego=_nonplanar_pose_at(pose_times),
    )
    assert windows == ()  # t0 is 60 ms after its left pose, outside 50 ms tolerance.


def test_frozen_target_detects_raw_pose_content_drift() -> None:
    recording = make_synthetic_recording(duration_seconds=7.0)
    window = build_recording_windows(recording)[0]
    changed = recording.ego_poses.world_from_ego.copy()
    changed[changed.shape[0] // 2 :, 0, 3] += 0.5
    drifted = PoseSeries(recording.ego_poses.timestamps, changed)
    with pytest.raises(ValueError, match="frozen manifest label"):
        extract_trajectory_target(window, drifted)


def test_synthetic_adapter_exposes_all_motion_cases_and_rgb_frames() -> None:
    adapter = make_synthetic_adapter()
    assert len(adapter.recording_ids()) == 4
    for recording_id in adapter.recording_ids():
        recording = adapter.load_recording(recording_id)
        frame = adapter.load_camera_frame(recording_id, 0)
        assert frame.shape == (64, 96, 3)
        assert frame.dtype == np.uint8
        assert recording.metadata["synthetic"] is True


def test_windows_are_built_only_inside_preassigned_recording_partitions() -> None:
    adapter = make_synthetic_adapter()
    splits = make_recording_splits(adapter.recording_ids(), seed=5)
    partitions = build_partitioned_windows(adapter, splits)
    for split_name, windows in partitions.items():
        allowed = set(splits.groups()[split_name])
        assert windows
        assert {window.recording_id for window in windows} <= allowed


def test_group_split_is_stable_order_independent_and_includes_calibration() -> None:
    identifiers = [f"sequence-{index:03d}" for index in range(40)]
    first = make_recording_splits(identifiers, seed=11)
    second = make_recording_splits(reversed(identifiers), seed=11)
    assert first == second
    assert first.digest == second.digest
    assert len(first.train) == 28
    assert len(first.validation) == 4
    assert len(first.calibration) == 2
    assert len(first.test) == 6
    all_groups = [set(group) for group in first.groups().values()]
    assert len(set.union(*all_groups)) == len(identifiers)
    assert sum(len(group) for group in all_groups) == len(identifiers)


def test_train_only_normalizer_uses_masks_and_round_trips() -> None:
    values = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    normalizer = TrainOnlyNormalizer().fit(values, recording_ids=["train-a"])
    np.testing.assert_allclose(normalizer.mean_, [3.0, 20.0])
    np.testing.assert_allclose(normalizer.count_, [3, 2])
    transformed, valid = normalizer.transform_with_mask(values)
    assert transformed[1, 1] == 0.0
    assert not valid[1, 1]
    restored = normalizer.inverse_transform(normalizer.transform(values))
    np.testing.assert_allclose(restored, values, equal_nan=True)
    loaded = TrainOnlyNormalizer.from_dict(normalizer.to_dict())
    assert loaded.digest == normalizer.digest


def test_fit_normalizer_from_recordings_excludes_non_train_values() -> None:
    values = {
        "train": np.array([[0.0], [2.0]]),
        "validation": np.array([[10000.0]]),
        "calibration": np.array([[-10000.0]]),
    }
    splits = {
        "train": Split.TRAIN,
        "validation": Split.VALIDATION,
        "calibration": Split.CALIBRATION,
    }
    normalizer = fit_normalizer_from_recordings(values, splits)
    np.testing.assert_allclose(normalizer.mean_, [1.0])
    assert normalizer.fitted_recording_ids == ("train",)


def test_manifest_hash_is_stable_and_record_order_independent(tmp_path: Path) -> None:
    records = ({"recording_id": "b", "t0": 2.0}, {"t0": 1.0, "recording_id": "a"})
    assert manifest_hash(records) == manifest_hash(reversed(records))
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    manifest = Manifest(records=records, metadata={"split_seed": 2026}, version="1")
    destination = tmp_path / "manifest.jsonl"
    digest = write_manifest_jsonl(manifest, destination)
    loaded = read_manifest_jsonl(destination)
    assert digest == manifest.digest == loaded.digest


@dataclass
class _FakeFrame:
    time: float

    def read(self) -> np.ndarray:
        return np.full((4, 5, 3), round(self.time * 10), dtype=np.uint8)


class _FakeInfo:
    id = "fake-sequence"
    start_time = 0.0
    end_time = 0.3
    keyframe_time = 0.1
    camera_frames = {"front_blur": [_FakeFrame(0.0), _FakeFrame(0.1), _FakeFrame(0.2)]}


class _FakeOxts:
    timestamps = np.array([0.0, 0.1, 0.2])
    poses = from_yaw_translation(np.zeros(3), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    angular_rates = np.zeros((3, 3))


class _FakeVehicleData:
    ego_vehicle_data = {
        "timestamp": np.array([0, 100_000_000, 200_000_000]),
        "lon_vel": np.array([10.0, 10.0, 10.0]),
        "lon_acc": np.zeros(3),
    }
    ego_vehicle_controls = {
        "timestamp": np.array([0, 100_000_000, 200_000_000]),
        "steering_angle": np.array([0.1, 0.2, 0.3]),
        "acc_pedal": np.array([40.0, 50.0, 60.0]),
        "brake_pedal_pressed": np.array([False, False, True]),
        "turn_indicator": np.array([None, "left", "hazard"], dtype=object),
    }
    # A conflicting legacy field proves the official field takes precedence.
    controls = {
        "timestamp": np.array([0, 100_000_000, 200_000_000]),
        "steering_angle": np.array([9.0, 9.0, 9.0]),
    }


class _FakeSequence:
    info = _FakeInfo()
    oxts = _FakeOxts()
    vehicle_data = _FakeVehicleData()


class _FakeDataset(dict[str, object]):
    def get_all_ids(self) -> list[str]:
        return list(self)


def test_optional_zod_adapter_uses_causal_frame_selection_without_sdk() -> None:
    adapter = ZODSequenceAdapter(
        dataset=_FakeDataset({"fake-sequence": _FakeSequence()}),
        camera_stream_key="front_blur",
    )
    assert adapter.recording_ids() == ("fake-sequence",)
    recording = adapter.load_recording("fake-sequence")
    np.testing.assert_allclose(recording.camera_timestamps, [0.0, 0.1, 0.2])
    assert recording.vehicle_state.channels == (
        "speed_mps",
        "acceleration_mps2",
        "yaw_rate_rps",
        "steering_rad",
        "accelerator_ratio",
        "brake_pressed",
        "turn_indicator_left",
        "turn_indicator_right",
        "delta_t",
    )
    assert recording.vehicle_state.values.shape == (3, 9)
    np.testing.assert_allclose(recording.vehicle_state.values[:, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(recording.vehicle_state.values[:, 4], [0.4, 0.5, 0.6])
    assert recording.metadata["accelerator_raw_encoding"] == "percentage_0_100"
    np.testing.assert_allclose(
        recording.vehicle_state.values[:, 6], [np.nan, 1.0, 1.0], equal_nan=True
    )
    np.testing.assert_allclose(
        recording.vehicle_state.values[:, 7], [np.nan, 0.0, 1.0], equal_nan=True
    )
    assert not recording.vehicle_state.valid[0, 6:8].any()
    assert adapter.camera_index_at_or_before("fake-sequence", 0.15) == 1
    assert adapter.load_camera_frame("fake-sequence", 1)[0, 0, 0] == 1
    assert adapter._sequence_cache
    assert adapter._camera_order_cache
    adapter.clear_runtime_cache()
    assert not adapter._sequence_cache
    assert not adapter._camera_order_cache
    assert isinstance(pickle.dumps(adapter), bytes)


def test_optional_zod_adapter_accepts_legacy_controls_fallback() -> None:
    legacy_controls = {
        "timestamp": np.array([0, 100_000_000, 200_000_000]),
        "steering_angle": np.array([-0.1, -0.2, -0.3]),
    }
    vehicle = SimpleNamespace(
        ego_vehicle_data=_FakeVehicleData.ego_vehicle_data,
        controls=legacy_controls,
    )
    sequence = SimpleNamespace(
        info=_FakeInfo(),
        oxts=_FakeOxts(),
        vehicle_data=vehicle,
    )
    adapter = ZODSequenceAdapter(
        dataset={"fake-sequence": sequence},
        recording_ids=["fake-sequence"],
        camera_stream_key="front_blur",
    )
    recording = adapter.load_recording("fake-sequence")
    np.testing.assert_allclose(recording.vehicle_state.values[:, 3], [-0.1, -0.2, -0.3])


def test_zod_adapter_crops_parent_drive_vehicle_rows_to_pose_span() -> None:
    ego = {
        "timestamp": np.array([-10_000_000_000, 0, 100_000_000, 200_000_000, 10_000_000_000]),
        "lon_vel": np.array([99.0, 10.0, 11.0, 12.0, 99.0]),
        "lon_acc": np.zeros(5),
    }
    sequence = SimpleNamespace(
        info=_FakeInfo(),
        oxts=_FakeOxts(),
        vehicle_data=SimpleNamespace(ego_vehicle_data=ego),
    )
    adapter = ZODSequenceAdapter(
        dataset={"fake-sequence": sequence},
        recording_ids=["fake-sequence"],
        camera_stream_key="front_blur",
    )
    recording = adapter.load_recording("fake-sequence")
    np.testing.assert_allclose(recording.vehicle_state.timestamps, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(recording.vehicle_state.values[:, 0], [10.0, 11.0, 12.0])
    assert recording.metadata["vehicle_state_crop"] == "inclusive OXTS pose span"


def test_zod_camera_decode_uses_sorted_manifest_index() -> None:
    info = SimpleNamespace(
        id="fake-sequence",
        start_time=0.0,
        end_time=0.3,
        keyframe_time=0.1,
        camera_frames={"front_blur": [_FakeFrame(0.2), _FakeFrame(0.0), _FakeFrame(0.1)]},
    )
    sequence = SimpleNamespace(
        info=info,
        oxts=_FakeOxts(),
        vehicle_data=_FakeVehicleData(),
    )
    adapter = ZODSequenceAdapter(
        dataset={"fake-sequence": sequence},
        recording_ids=["fake-sequence"],
        camera_stream_key="front_blur",
    )
    np.testing.assert_allclose(
        adapter.load_recording("fake-sequence").camera_timestamps,
        [0.0, 0.1, 0.2],
    )
    assert adapter.load_camera_frame("fake-sequence", 0)[0, 0, 0] == 0
    assert adapter.load_camera_frame("fake-sequence", 2)[0, 0, 0] == 2


def test_zod_camera_decode_recovers_windows_tar_timestamp_name(tmp_path: Path) -> None:
    extracted_path = tmp_path / "frame_2026-07-23T12_34_56.000000Z.jpg"
    Image.fromarray(np.full((4, 5, 3), 73, dtype=np.uint8)).save(extracted_path)

    class WindowsTarFrame:
        time = 0.0
        filepath = str(tmp_path / "frame_2026-07-23T12:34:56.000000Z.jpg")

        @staticmethod
        def read() -> np.ndarray:
            raise OSError(22, "Invalid argument")

    info = SimpleNamespace(
        id="fake-sequence",
        start_time=0.0,
        end_time=0.3,
        keyframe_time=0.1,
        camera_frames={"front_blur": [WindowsTarFrame()]},
    )
    sequence = SimpleNamespace(
        info=info,
        oxts=_FakeOxts(),
        vehicle_data=_FakeVehicleData(),
    )
    adapter = ZODSequenceAdapter(
        dataset={"fake-sequence": sequence},
        recording_ids=["fake-sequence"],
        camera_stream_key="front_blur",
    )
    decoded = adapter.load_camera_frame("fake-sequence", 0)
    assert decoded.shape == (4, 5, 3)
    np.testing.assert_allclose(decoded, 73, atol=2)
