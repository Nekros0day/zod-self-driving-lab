# Frozen benchmark results

## Decision summary

- **Promote temporal FNO for state-based trajectory forecasting.** It ties the
  best accuracy, has a reliable −0.131 m ADE improvement over B2, and is much
  faster than either ODE.
- **Retain hybrid NeuralODE as the interpretable physics study.** It improves B2
  reliably but does not outperform the unconstrained alternatives.
- **Promote ResNet-18 U-Net for affordance segmentation.** It captures nearly
  all Fourier U-Net accuracy at one quarter of the parameters and lower latency.
- **Keep Fourier U-Net as a negative complexity control.** Its direct score
  difference from U-Net is compatible with zero.
- **Keep SFA3D as a pinned BEV transfer baseline, not a promoted ZOD detector.**
  Vehicle localization is useful, while pedestrian and cyclist recall is zero.

## Dynamics

Three-seed means on 2,549 windows / 72 recording groups:

| Model | ADE | FDE | Miss >2m | Parameters | Median latency |
|---|---:|---:|---:|---:|---:|
| B2 state MLP | 0.6732 | 1.7082 | 0.2776 | 112,444 | 0.196 ms |
| Hybrid NeuralODE | 0.5507 | 1.5245 | 0.2438 | 108,674 | 35.956 ms |
| NeuralODE | 0.5435 | 1.5007 | 0.2367 | 108,162 | 24.450 ms |
| Temporal FNO | **0.5424** | **1.4864** | **0.2346** | 1,230,242 | **2.689 ms** |

Paired ADE differences versus B2:

| Candidate − B2 | Estimate | 95% recording bootstrap |
|---|---:|---:|
| Hybrid NeuralODE | −0.1226 | [−0.1475, −0.0971] |
| NeuralODE | −0.1297 | [−0.1550, −0.1046] |
| Temporal FNO | **−0.1308** | **[−0.1546, −0.1080]** |

## Segmentation

Three-seed global metrics on the fresh 51-recording test role:

| Model | Road IoU | Lane IoU | Lane tolerant F1 | Score | Parameters | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| DeepLabV3-MobileNet | 0.8087 | 0.1871 | 0.6542 | 0.7314 | 11.0M | 3.198 ms |
| ResNet-18 U-Net | **0.8579** | 0.4663 | 0.8609 | 0.8594 | **14.4M** | **2.932 ms** |
| Fourier U-Net | 0.8564 | **0.4892** | **0.8651** | **0.8608** | 56.8M | 4.995 ms |

Per-image paired score differences:

| Comparison | Estimate | 95% recording bootstrap |
|---|---:|---:|
| U-Net − DeepLab | +0.1355 | [+0.1165, +0.1560] |
| Fourier U-Net − DeepLab | +0.1365 | [+0.1179, +0.1577] |
| Fourier U-Net − U-Net | **+0.0010** | **[−0.0072, +0.0094]** |

## LiDAR BEV transfer diagnostic

Fixed KITTI-pretrained SFA3D evaluation on all 12 ZOD Frames mini keyframes
(104 in-range dynamic labels), with class-consistent oriented IoU ≥ 0.5:

| Class | Precision | Recall | F1 | Matched IoU | Center error |
|---|---:|---:|---:|---:|---:|
| All | 0.8085 | 0.3654 | 0.5033 | 0.7183 | 0.2935 m |
| Vehicle | **0.8636** | **0.5278** | **0.6552** | **0.7183** | **0.2935 m** |
| Pedestrian | 0.0000 | 0.0000 | 0.0000 | — | — |
| Cyclist | 0.0000 | 0.0000 | 0.0000 | — | — |

The model receives no ZOD fine-tuning and uses a fixed confidence threshold.
These numbers diagnose transfer behavior on a very small subset; they are not
treated as a statistically powered detector comparison.

## Evidence files

- `v4_dynamics_test.json`: seed-level metrics, grouped intervals, latency, and
  checkpoint hashes.
- `v4_segmentation_test.json`: seed thresholds, global metrics, paired
  improvements, latency, and checkpoint hashes.
- `benchmark_summary.json`: consolidated public evidence and learning curves.
- `bev_detection_mini.json`: pinned source/checkpoint identity, fixed protocol,
  aggregate per-class metrics, and latency for the transfer diagnostic.
- `figures/`: aggregate plots only; no ZOD image or mask is redistributed.
