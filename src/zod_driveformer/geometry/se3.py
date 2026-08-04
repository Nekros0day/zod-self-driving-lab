"""Small, NumPy-first SE(3) helpers used by the data pipeline.

A pose named ``world_from_ego`` maps coordinates expressed in the ego frame
into world coordinates.  This naming convention is intentionally verbose:
mixing up ``world_from_ego`` and ``ego_from_world`` produces plausible-looking
but incorrect trajectories.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .angles import interpolate_yaw, wrap_yaw

FloatArray: TypeAlias = NDArray[np.float64]


def _as_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_transform(
    transform: ArrayLike,
    *,
    atol: float = 1e-7,
    check_rotation: bool = True,
) -> FloatArray:
    """Validate and return one or a batch of homogeneous SE(3) transforms.

    Accepted shapes are ``(4, 4)`` and ``(..., 4, 4)``.  The returned array is
    a float64 NumPy view/copy.  Rotation matrices must be orthonormal with
    determinant +1 unless ``check_rotation`` is disabled.
    """

    matrix = _as_float_array(transform, name="transform")
    if matrix.ndim < 2 or matrix.shape[-2:] != (4, 4):
        raise ValueError(f"transform must have shape (..., 4, 4); got {matrix.shape}")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(matrix[..., 3, :], expected_bottom, atol=atol, rtol=0.0):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    if check_rotation:
        rotation = matrix[..., :3, :3]
        gram = np.matmul(np.swapaxes(rotation, -1, -2), rotation)
        identity_matrix = np.eye(3, dtype=np.float64)
        if not np.allclose(gram, identity_matrix, atol=atol, rtol=0.0):
            raise ValueError("transform rotation must be orthonormal")
        determinant = np.linalg.det(rotation)
        if not np.allclose(determinant, 1.0, atol=atol, rtol=0.0):
            raise ValueError("transform rotation must have determinant +1")
    return matrix


def identity(batch_shape: int | Sequence[int] = ()) -> FloatArray:
    """Return an identity transform, optionally with a leading batch shape."""

    shape: tuple[int, ...]
    if isinstance(batch_shape, int):
        if batch_shape < 0:
            raise ValueError("batch size cannot be negative")
        shape = (batch_shape,)
    else:
        shape = tuple(batch_shape)
        if any(size < 0 for size in shape):
            raise ValueError("batch dimensions cannot be negative")
    return np.broadcast_to(np.eye(4, dtype=np.float64), shape + (4, 4)).copy()


def from_rotation_translation(
    rotation: ArrayLike,
    translation: ArrayLike,
    *,
    validate: bool = True,
) -> FloatArray:
    """Construct homogeneous transforms from rotation and translation arrays."""

    rotation_array = _as_float_array(rotation, name="rotation")
    translation_array = _as_float_array(translation, name="translation")
    if rotation_array.ndim < 2 or rotation_array.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (..., 3, 3)")
    if translation_array.ndim < 1 or translation_array.shape[-1] != 3:
        raise ValueError("translation must have shape (..., 3)")
    batch_shape = np.broadcast_shapes(rotation_array.shape[:-2], translation_array.shape[:-1])
    result = identity(batch_shape)
    result[..., :3, :3] = np.broadcast_to(rotation_array, batch_shape + (3, 3))
    result[..., :3, 3] = np.broadcast_to(translation_array, batch_shape + (3,))
    if validate:
        validate_transform(result)
    return result


def rotation_from_yaw(yaw: ArrayLike) -> FloatArray:
    """Return z-axis rotation matrices for scalar or batched yaw angles."""

    angle = _as_float_array(yaw, name="yaw")
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.zeros(angle.shape + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


def from_yaw_translation(yaw: ArrayLike, translation: ArrayLike) -> FloatArray:
    """Construct planar z-up poses, allowing arbitrary z translation."""

    return from_rotation_translation(rotation_from_yaw(yaw), translation)


def compose(*transforms: ArrayLike, validate: bool = True) -> FloatArray:
    """Compose transforms from left to right.

    ``compose(a_from_b, b_from_c)`` returns ``a_from_c``.
    NumPy broadcasting applies to leading batch dimensions.
    """

    if not transforms:
        return identity()
    result = (
        validate_transform(transforms[0])
        if validate
        else np.asarray(transforms[0], dtype=np.float64)
    )
    for transform in transforms[1:]:
        next_transform = (
            validate_transform(transform) if validate else np.asarray(transform, dtype=np.float64)
        )
        result = np.matmul(result, next_transform)
    if validate:
        validate_transform(result)
    return result


def inverse(transform: ArrayLike, *, validate: bool = True) -> FloatArray:
    """Invert one or a batch of rigid transforms analytically."""

    matrix = validate_transform(transform) if validate else np.asarray(transform, dtype=np.float64)
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    inverse_rotation = np.swapaxes(rotation, -1, -2)
    inverse_translation = -np.einsum("...ij,...j->...i", inverse_rotation, translation)
    return from_rotation_translation(inverse_rotation, inverse_translation, validate=validate)


def relative_transform(
    world_from_reference: ArrayLike,
    world_from_target: ArrayLike,
    *,
    validate: bool = True,
) -> FloatArray:
    """Express ``target`` in ``reference`` coordinates.

    Mathematically this is ``inverse(T_world_reference) @ T_world_target``.
    The translation is therefore the target-frame origin expressed in the
    reference frame, exactly the label required by the blueprint.
    """

    return compose(
        inverse(world_from_reference, validate=validate),
        world_from_target,
        validate=validate,
    )


def transform_points(transform: ArrayLike, points: ArrayLike) -> FloatArray:
    """Apply a transform to points with shape ``(..., 3)`` or ``(..., N, 3)``.

    Common supported cases are a single transform with any point array, one
    point per batched transform, and one point cloud per batched transform.
    """

    matrix = validate_transform(transform)
    point_array = _as_float_array(points, name="points")
    if point_array.ndim < 1 or point_array.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    if matrix.ndim == 2:
        return np.asarray(np.matmul(point_array, rotation.T) + translation, dtype=np.float64)
    if point_array.ndim == matrix.ndim - 1:
        return np.asarray(
            np.einsum("...ij,...j->...i", rotation, point_array) + translation,
            dtype=np.float64,
        )
    if point_array.ndim == matrix.ndim:
        return np.asarray(
            np.einsum("...ij,...nj->...ni", rotation, point_array) + translation[..., None, :],
            dtype=np.float64,
        )
    raise ValueError("batched transforms require one point or one point cloud per transform")


def yaw_from_transform(transform: ArrayLike) -> float | FloatArray:
    """Extract planar yaw from a z-up transform."""

    matrix = validate_transform(transform)
    yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    wrapped = wrap_yaw(yaw)
    if np.asarray(wrapped).ndim == 0:
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float64)


def interpolate_planar(
    start: ArrayLike,
    end: ArrayLike,
    fraction: ArrayLike,
) -> FloatArray:
    """Interpolate planar poses using linear translation and shortest-arc yaw.

    This helper is intentionally planar.  It is appropriate for resampling
    ego-trajectory labels; it is not a general SO(3) SLERP implementation.
    """

    start_matrix = validate_transform(start)
    end_matrix = validate_transform(end)
    alpha = _as_float_array(fraction, name="fraction")
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("fraction must lie in [0, 1]")
    batch_shape = np.broadcast_shapes(start_matrix.shape[:-2], end_matrix.shape[:-2], alpha.shape)
    start_translation = np.broadcast_to(start_matrix[..., :3, 3], batch_shape + (3,))
    end_translation = np.broadcast_to(end_matrix[..., :3, 3], batch_shape + (3,))
    alpha_expanded = np.broadcast_to(alpha, batch_shape)[..., None]
    translation = start_translation + alpha_expanded * (end_translation - start_translation)
    yaw = interpolate_yaw(
        np.broadcast_to(yaw_from_transform(start_matrix), batch_shape),
        np.broadcast_to(yaw_from_transform(end_matrix), batch_shape),
        np.broadcast_to(alpha, batch_shape),
    )
    return from_yaw_translation(yaw, translation)


def _rotation_to_quaternion(rotation: FloatArray) -> FloatArray:
    """Convert proper rotation matrices to unit ``(w, x, y, z)`` quaternions.

    The branch-by-largest-diagonal algorithm remains stable close to 180 degree
    rotations, where the more compact trace-only formula becomes ill-conditioned.
    This is kept private because quaternion convention is an implementation detail
    of :func:`interpolate_se3` rather than part of the project's pose contract.
    """

    flattened = np.asarray(rotation, dtype=np.float64).reshape(-1, 3, 3)
    quaternions = np.empty((flattened.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(flattened):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif matrix[1, 1] > matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
        quaternions[index] = quaternion / np.linalg.norm(quaternion)
    return quaternions.reshape(rotation.shape[:-2] + (4,))


def _quaternion_to_rotation(quaternion: FloatArray) -> FloatArray:
    normalized = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(normalized, -1, 0)
    rotation = np.empty(normalized.shape[:-1] + (3, 3), dtype=np.float64)
    rotation[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[..., 0, 1] = 2.0 * (x * y - z * w)
    rotation[..., 0, 2] = 2.0 * (x * z + y * w)
    rotation[..., 1, 0] = 2.0 * (x * y + z * w)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[..., 1, 2] = 2.0 * (y * z - x * w)
    rotation[..., 2, 0] = 2.0 * (x * z - y * w)
    rotation[..., 2, 1] = 2.0 * (y * z + x * w)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation


def interpolate_se3(
    start: ArrayLike,
    end: ArrayLike,
    fraction: ArrayLike,
) -> FloatArray:
    """Interpolate full SE(3) poses with linear translation and SO(3) SLERP.

    ``fraction`` may be scalar or batched and broadcasts with the leading pose
    dimensions. Quaternion signs are aligned before interpolation so the
    shortest rotation arc is used. Near-identical rotations fall back to a
    normalized linear interpolation to avoid division by a tiny sine.

    This interpolates a time-stamped pose *at* a query instant; it does not
    infer motion between coordinate frames and does not extrapolate. Fractions
    outside the closed interval ``[0, 1]`` are therefore rejected.
    """

    start_matrix = validate_transform(start)
    end_matrix = validate_transform(end)
    alpha = _as_float_array(fraction, name="fraction")
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("fraction must lie in [0, 1]")
    batch_shape = np.broadcast_shapes(start_matrix.shape[:-2], end_matrix.shape[:-2], alpha.shape)
    start_broadcast = np.broadcast_to(start_matrix, batch_shape + (4, 4))
    end_broadcast = np.broadcast_to(end_matrix, batch_shape + (4, 4))
    alpha_broadcast = np.broadcast_to(alpha, batch_shape)

    start_quaternion = _rotation_to_quaternion(start_broadcast[..., :3, :3])
    end_quaternion = _rotation_to_quaternion(end_broadcast[..., :3, :3])
    dot = np.sum(start_quaternion * end_quaternion, axis=-1)
    end_quaternion = np.where((dot < 0.0)[..., None], -end_quaternion, end_quaternion)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    linear = dot > 0.9995
    theta = np.arccos(dot)
    sine_theta = np.sin(theta)
    safe_denominator = np.where(linear, 1.0, sine_theta)
    start_weight = np.sin((1.0 - alpha_broadcast) * theta) / safe_denominator
    end_weight = np.sin(alpha_broadcast * theta) / safe_denominator
    slerped = start_weight[..., None] * start_quaternion + end_weight[..., None] * end_quaternion
    linearly_interpolated = (1.0 - alpha_broadcast)[..., None] * start_quaternion + alpha_broadcast[
        ..., None
    ] * end_quaternion
    quaternion = np.where(linear[..., None], linearly_interpolated, slerped)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)

    start_translation = start_broadcast[..., :3, 3]
    end_translation = end_broadcast[..., :3, 3]
    translation = start_translation + alpha_broadcast[..., None] * (
        end_translation - start_translation
    )
    return from_rotation_translation(_quaternion_to_rotation(quaternion), translation)


# Familiar aliases used in robotics code and teaching material.
invert = inverse
make_transform = from_rotation_translation
