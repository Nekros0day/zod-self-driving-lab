"""Trajectory accuracy, multimodality and kinematic metrics."""

from .kinematics import (
    KinematicLimits,
    acceleration,
    acceleration_vectors,
    accelerations,
    curvature,
    curvatures,
    jerk,
    jerk_vectors,
    jerks,
    kinematic_violation_rates,
    speed,
    speeds,
    velocity_vectors,
)
from .trajectory import (
    ade,
    average_displacement_error,
    displacement_errors,
    fde,
    final_displacement_error,
    horizonwise_l2,
    min_ade,
    min_ade_k,
    min_fde,
    min_fde_k,
    miss_rate,
    mode_entropy,
    path_diversity,
    top1_ade,
    top1_fde,
)

__all__ = [
    "KinematicLimits",
    "acceleration",
    "acceleration_vectors",
    "accelerations",
    "ade",
    "average_displacement_error",
    "curvature",
    "curvatures",
    "displacement_errors",
    "fde",
    "final_displacement_error",
    "horizonwise_l2",
    "jerk",
    "jerk_vectors",
    "jerks",
    "kinematic_violation_rates",
    "min_ade",
    "min_ade_k",
    "min_fde",
    "min_fde_k",
    "miss_rate",
    "mode_entropy",
    "path_diversity",
    "speed",
    "speeds",
    "top1_ade",
    "top1_fde",
    "velocity_vectors",
]

# Verbose aliases read naturally in report/evaluation code.
compute_ade = average_displacement_error
compute_fde = final_displacement_error
kinematic_metrics = kinematic_violation_rates
__all__ += ["compute_ade", "compute_fde", "kinematic_metrics"]
