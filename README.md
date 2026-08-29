# ZOD Self-Driving Lab

### Deep learning with the Zenseact Open Dataset

I built this as a personal learning project to explore how deep learning,
geometry, and physics-based models can solve practical self-driving problems
with the Zenseact Open Dataset (ZOD). I focus on four questions:

1. Can continuous-time or operator-learning models forecast three seconds of
   ego motion better than a strong state MLP?
2. Can an efficient encoder–decoder segment the road surface and thin lane
   markings from a front-camera keyframe?
3. Can ZOD fine-tuning and calibrated camera–LiDAR fusion produce a useful
   top-down detector for vehicles and vulnerable road users?
4. Can one camera network share semantics, metric depth, and 2-D detection—and
   can uneven non-IID vehicle clients improve it through federated fine-tuning?

The trajectory and segmentation studies have sealed three-seed tests: a
temporal Fourier Neural Operator (FNO) reduces trajectory ADE from **0.673 m to
0.542 m**, and a ResNet-18 U-Net raises camera-segmentation score from **0.731
to 0.859**. The BEV study starts from a failed 12-frame KITTI→ZOD transfer, then
uses protected recording splits to fine-tune SFA3D and add camera semantics.
The final hybrid reaches AP@0.30 of **0.616 vehicle, 0.530 pedestrian, and 0.328
cyclist**, while a vehicle pass-through rule prevents camera fusion from
degrading the LiDAR vehicle branch.

The camera multi-task extension finds a more nuanced result. Hard sharing
collapses thin-lane segmentation, while a 19.8M-parameter split-decoder model
retains one shared encoder and restores task-specific spatial decoding. It uses
about 54% fewer parameters than three separate networks. Four simulated
collection-car clients then test pooled tuning, uniform averaging,
sample-weighted model-delta FedAvg, FedProx, head-only FedAvg, and gradient-only
FedSGD. The local-optimizer methods retain round zero, while sample-weighted
FedSGD selects round 7 and improves the composite test score from **0.339 to
0.344**, mainly through better sparse metric depth.

![Model input, architecture, and output overview](reports/figures/model_architecture_overview.png)

![Trajectory benchmark](reports/figures/dynamics_test_ade.png)

![Camera, LiDAR-only BEV, fused BEV, and labels](reports/figures/bev_v2_fusion_comparison.png)

![ZOD camera–LiDAR fusion comparison](reports/figures/bev_v2_fusion_comparison.gif)

![Camera multi-task inference gallery](reports/figures/multitask_camera_inference.gif)

[Open the camera multi-task inference video (MP4)](reports/figures/multitask_camera_inference.mp4)

## Headline results

### Three-second ego-trajectory forecasting

| Model | Test ADE ↓ | Test FDE ↓ | Miss rate ↓ | Batch-1 GPU latency |
|---|---:|---:|---:|---:|
| Constant velocity | 1.036 | 2.732 | 0.463 | — |
| CTRV | 0.849 | 2.268 | 0.380 | — |
| Frozen B2 state MLP | 0.673 | 1.708 | 0.278 | 0.20 ms |
| Hybrid physics NeuralODE | 0.551 | 1.525 | 0.244 | 35.96 ms |
| NeuralODE | 0.544 | 1.501 | 0.237 | 24.45 ms |
| **Temporal FNO** | **0.542** | **1.486** | **0.235** | **2.69 ms** |

Temporal FNO minus B2 is **−0.131 m ADE**, with a 95% recording-bootstrap
interval of **[−0.155, −0.108] m**. The hybrid ODE is scientifically useful but
does not beat the generic ODE; hand-specified kinematics stabilize the model,
yet they also restrict the learned vector field. FNO provides essentially the
same accuracy as NeuralODE at about one ninth of its latency.

![Held-out camera trajectories](reports/figures/dynamics_camera_predictions.gif)

The animation projects ground truth, CTRV, and the three learned paths through
the calibrated front camera. It is a qualitative interpretation layer: the
trajectory networks receive only the causal 21-step vehicle-state history, not
the camera image. Each learned curve is the mean of three frozen-seed paths. A
four-scene comparison is available as a
[full-resolution static figure](reports/figures/dynamics_camera_predictions.png).

### Road and lane segmentation

| Model | Road IoU ↑ | Strict lane IoU ↑ | Lane tolerant F1 ↑ | Score ↑ | Params | Latency |
|---|---:|---:|---:|---:|---:|---:|
| DeepLabV3-MobileNet | 0.809 | 0.187 | 0.654 | 0.731 | 11.0M | 3.20 ms |
| **ResNet-18 U-Net** | **0.858** | 0.466 | 0.861 | 0.859 | **14.4M** | **2.93 ms** |
| Fourier U-Net | 0.856 | **0.489** | **0.865** | **0.861** | 56.8M | 4.99 ms |

Both U-Nets improve the per-image score over DeepLab by about **+0.136**, with
95% intervals fully above zero. Fourier U-Net minus ordinary U-Net is only
**+0.0010**, interval **[−0.0072, +0.0094]**. I therefore promote the ordinary
U-Net: the Fourier bottleneck is an informative control, not an efficiency win.

![Segmentation benchmark](reports/figures/segmentation_test_metrics.png)

![Held-out road and lane segmentation](reports/figures/segmentation_model_comparison.gif)

Each animation frame keeps the RGB input and ground truth beside all three
frozen model outputs. Cyan marks road and magenta marks lane. The fixed sample
rules include score quantiles and model-disagreement cases rather than only
attractive scenes; predictions use seed 2026 and validation-fitted thresholds.
The first three scenes are also available as a
[full-resolution static montage](reports/figures/segmentation_model_comparison.png).

### LiDAR BEV object detection and tracking

![Promoted BEV perception architecture](reports/figures/bev_v2_pipeline.png)

| Sealed AP at oriented IoU ≥ 0.30 | Vehicle | Pedestrian | Cyclist |
|---|---:|---:|---:|
| Unmodified KITTI SFA3D | 0.361 | 0.366 | 0.007 |
| ZOD fine-tuned SFA3D, one sweep | **0.616** | 0.501 | 0.156 |
| **Hybrid camera–LiDAR fusion** | **0.616** | **0.530** | **0.328** |
| PointPillars, trained from scratch | 0.018 | 0.000 | 0.000 |
| CenterPoint head, trained from scratch | 0.000 | 0.000 | 0.000 |

The protected cohort contains 70 training, 16 validation, and 30 sealed test
ZOD Sequence recordings. Every recording contributes only its annotated central
keyframe; all mini recordings are excluded. SFA3D is initialized from the
pinned KITTI checkpoint, first trains only its heads, then progressively
unfreezes the feature pyramid and backbone. Class-balanced sampling counters the
much lower pedestrian and cyclist frequency.

![BEV detector and fusion AP](reports/figures/bev_v2_test_ap.png)

The promoted detector uses one current LiDAR sweep. A five-sweep detector input
looked attractive but reduced validation macro-F1 because ego compensation
cannot remove trails from independently moving objects. Five compensated sweeps
are instead used only to estimate metric depth for camera detections. Unmatched
camera pedestrians and cyclists may supplement LiDAR; vehicle boxes and scores
remain a strict LiDAR pass-through.

![Confidence-ranked BEV precision–recall curves](reports/figures/bev_v2_pr_curves.png)

Evaluation includes 101-point AP at IoU 0.30/0.50/0.70, precision–recall curves,
confidence calibration, center/yaw/size error, and near/mid/far range slices.
PointPillars and CenterPoint are retained only as small-data controls: training
from scratch on 70 recordings overfits and is not competitive with transfer
learning. The earlier 12-frame mini transfer remains a historical smoke test,
not the headline result.

### Multi-task camera perception and federated fine-tuning

![Multi-task and federated-learning pipeline](reports/figures/multitask_federated_pipeline.png)

One 192×320 front image feeds a shared ResNet-18 encoder. The model predicts
19-class Cityscapes teacher semantics, native ZOD road/lane affordances, sparse-
LiDAR-supervised monocular depth, and native ZOD 2-D object centers. Teacher
mIoU measures distillation fidelity—not ZOD semantic ground-truth accuracy.

| Frozen camera model | Params | Teacher mIoU ↑ | Lane tolerant F1 ↑ | Depth δ<1.25 ↑ | 2-D mAP50 ↑ | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Separate task controls | 43.4M total | 0.228 | 0.630 | 0.354 | **0.034** | 9.1 ms total |
| Hard-shared, equal loss | **14.5M** | 0.187 | 0.000 | **0.387** | 0.029 | **3.1 ms** |
| **Split decoder, equal loss** | **19.8M** | 0.225 | **0.705** | 0.358 | 0.025 | 4.7 ms |

The table compares each multi-task output with the corresponding single-task
control. Hard sharing is the negative-transfer control: its shared decoder
loses the thin lane task, and measured encoder gradients show segmentation–depth
conflict. Independent task decoders recover lane detail while preserving a
single encoder pass. Learned uncertainty weighting is also retained, but equal
weighting wins the frozen validation rule by a small margin.

The centralized training role is divided into 220 initial samples and four
collection-car clients with **65/40/25/15** additional samples. Their LiDAR-
depth coverage is **22/11/6/2**, and their country/object distributions differ.
Sample-weighted FedAvg assigns weights **0.448/0.276/0.172/0.103**; uniform
averaging is the control that overweights the smallest client.

![Central and federated camera benchmark](reports/figures/multitask_federated_benchmark.png)

Pooled fine-tuning, uniform averaging, model-delta FedAvg, FedProx, and encoder-
frozen FedAvg remain below the centralized validation score of **0.372**.
Gradient-only FedSGD behaves differently: a validation-only learning-rate sweep
selects \(\eta=0.003\) and round 7 at **0.377**. Its test score rises from
**0.339 to 0.344**, with depth \(\delta_1\) improving from **0.359 to 0.382**;
detection mAP50 slips from 0.025 to 0.023, so this is a measured multi-task
trade rather than an across-the-board win.

FedAvg now transmits explicit model deltas and FedSGD transmits mean parameter
gradients. Dense deltas and gradients are still approximately 75.5 MiB per
client for this FP32 model. This is a federated-optimization mechanics study,
not a privacy claim: there is no secure aggregation, differential privacy, or
adversarial-server evaluation. The pooled control deliberately centralizes its
samples and is not described as federated.

## What is unusual about the project

- **True multiple shooting.** NeuralODE training solves three shorter initial
  value problems, penalizes boundary discontinuities, and retains an
  uninterrupted full-rollout objective. Future-derived shooting states exist
  only inside training loss code.
- **Physics is an ablation, not decoration.** The hybrid state is
  \([x,y,\psi,v,r]\); \(\dot x=v\cos\psi\), \(\dot y=v\sin\psi\), and
  \(\dot\psi=r\) are exact. Only bounded acceleration and yaw acceleration are
  learned.
- **The test roles were sealed.** Dynamics kept the original 72-recording test
  role. Segmentation preserves the old validation role and creates a fresh
  51-recording test subset exclusively from previously unobserved training
  examples.
- **Thin structures get the right metric.** Lane markings are reported with
  strict IoU and a three-pixel tolerance F1; thresholds are fitted independently
  for road and lane using validation only.
- **Negative complexity results remain visible.** Fourier U-Net is larger and
  slower without a reliable aggregate gain. The repository keeps that finding,
  but does not keep the old implementation sprawl that led nowhere.
- **Cross-domain failure drives the next experiment.** The frozen mini transfer
  misses every vulnerable road user; protected ZOD fine-tuning and class-gated
  camera fusion turn that failure into measurable pedestrian/cyclist AP.
- **Metric geometry stays explicit.** ZOD sensor calibration, ego-frame axes,
  oriented polygon IoU, Kalman state transitions, and every raster convention
  are implemented and tested rather than buried inside a visualization.
- **Federated updates must earn promotion.** Client quantity, domain, and label
  availability are deliberately non-IID; pooled tuning, model-delta FedAvg,
  FedProx, head-only aggregation, and gradient-only FedSGD all compete with
  round zero rather than assuming a fleet update must help.

## Learn the project in order

The executed notebooks are the main teaching surface:

| Notebook | What it teaches |
|---|---|
| `00_project_map.ipynb` | Raw-to-evidence workflow, tensor contracts, experiment ledger, and measured claims |
| `01_geometry_splits_and_baselines.ipynb` | Signal synchronization, missingness, train-only normalization, local SE(2), group splits, CV, and CTRV |
| `02_neural_ode_and_multiple_shooting.ipynb` | GRU context, continuous states, RK4, multiple-shooting supervision, losses, and hybrid vehicle physics |
| `03_fourier_operators.ipynb` | DFT intuition, spectral convolution tensors, causal query grids, temporal FNO, padding, and accuracy–latency trade-offs |
| `04_road_lane_segmentation.ipynb` | Polygon rasterization, resize/augmentation rules, normalized batches, U-Net skips, imbalance, thresholds, and thin-lane metrics |
| `05_lidar_bev_detection_and_tracking.ipynb` | Point-cloud validation, SE(3), BEV rasterization, heatmap targets, temporal sweeps, transfer learning, fusion, AP/calibration, and tracking |
| `06_project_synthesis.ipynb` | A personal end-to-end reconstruction of the data paths, comparisons, failure analysis, and implementation map |
| `07_multitask_federated_perception.ipynb` | Pseudo-label provenance, LiDAR depth projection, center targets, multi-task losses, gradient conflict, model-delta FedAvg, gradient-only FedSGD, communication, and privacy boundaries |

The mathematical reference is [methods.md](docs/methods.md), the exact data and
evaluation contract is [data_and_evaluation.md](docs/data_and_evaluation.md),
and the conclusions I use when returning to the work are in
[project_learning_review.md](docs/project_learning_review.md).
The new camera/fleet experiment has its own literature and claim record in
[multitask_federated_learning.md](docs/multitask_federated_learning.md).

## Reproduce

Create an environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[all]"
```

ZOD access must be requested from Zenseact. The repository never contains or
redistributes raw data. The benchmark pipeline is intentionally explicit:

```powershell
# 1. Materialize train/validation tensors outside the repository.
.venv\Scripts\python scripts\build_v4_dynamics_cache.py `
  --data-root D:\datasets\zod `
  --manifest-dir D:\private\zod-parent-manifest `
  --reference-checkpoint D:\private\b2-seed-2026-best.pt `
  --output D:\datasets\zod-v4-private\dynamics_selection

# 2. Train all three models × three seeds on CUDA.
.venv\Scripts\python scripts\train_v4_dynamics.py `
  --cache D:\datasets\zod-v4-private\dynamics_selection `
  --output D:\datasets\zod-v4-private\dynamics_runs `
  --device cuda

# Segmentation has analogous build/train/evaluate scripts.
```

The BEV study additionally uses the official MIT-licensed SFA3D source and
checkpoint outside this repository. Pinning the recorded commit avoids silently
changing the starting model:

```powershell
git clone https://github.com/maudzung/SFA3D.git D:\datasets\zod-sfa3d
git -C D:\datasets\zod-sfa3d checkout 0e2f0b63dc4090bd6c08e15505f11d764390087c

.venv\Scripts\python scripts\cache_bev_training_data.py `
  --zod-root D:\datasets\zod `
  --subset sequences `
  --zod-version full `
  --private-roles D:\private\zod-bev-roles.json `
  --output-dir D:\datasets\zod-bev-cache `
  --sweeps 1

.venv\Scripts\python scripts\train_bev_detectors.py `
  --model sfa3d `
  --cache-root D:\datasets\zod-bev-cache `
  --output-dir D:\private\zod-bev-models `
  --sfa3d-root D:\datasets\zod-sfa3d `
  --sfa3d-checkpoint D:\datasets\zod-sfa3d\checkpoints\fpn_resnet_18\fpn_resnet_18_epoch_300.pth `
  --device cuda
```

The camera multi-task/federated study follows the same external-data boundary:

```powershell
.venv\Scripts\python scripts\prepare_multitask_federated_roles.py `
  --source-manifest D:\private\segmentation_manifest.csv `
  --zod-root D:\datasets\zod `
  --private-output D:\private\multitask_roles.csv

.venv\Scripts\python scripts\build_multitask_federated_cache.py `
  --private-manifest D:\private\multitask_roles.csv `
  --zod-root D:\datasets\zod `
  --output D:\datasets\zod-multitask-cache

.venv\Scripts\python scripts\train_multitask_central.py `
  --cache-root D:\datasets\zod-multitask-cache `
  --output D:\private\multitask-models

.venv\Scripts\python scripts\run_multitask_federated.py `
  --cache-root D:\datasets\zod-multitask-cache `
  --checkpoint D:\private\multitask-models\split_equal\best.pt `
  --output D:\private\federated-models `
  --method fedsgd --rounds 12 --learning-rate 0.003
```

The public evidence is in [RESULTS.md](reports/RESULTS.md) and the machine-readable
[benchmark summary](reports/benchmark_summary.json). Exact private paths are
arguments, never committed configuration.

## Scope and limitations

This is an offline trajectory, segmentation, and perception laboratory—not an
end-to-end driving policy. It does not perform navigation, collision-aware path
planning, behavior prediction, or closed-loop control. The trajectory model uses
vehicle state rather than camera or BEV features. Segmentation has only 489
labeled keyframes and 51 final test recordings. The BEV cohort is larger than
the original mini diagnostic but still only 116 locally complete annotated
Sequence recordings; a full-Frames confirmation remains future work. The
statistically reliable findings are the gains over B2 and DeepLab—not the tiny
differences between FNO and NeuralODE or between the two U-Nets.
The federated clients are simulated sequentially on one machine; model updates
are not protected by secure aggregation or differential privacy. The broad
scene-semantic output is teacher distillation, so its mIoU is agreement with
SegFormer rather than native ZOD semantic accuracy.

## ZOD attribution

ZOD is © 2022 Zenseact AB and is licensed under
[CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Dataset terms and
attribution remain governed by the [official ZOD license](https://zod.zenseact.com/license/).

> For this dataset, Zenseact AB has taken all reasonable measures to remove all
> personally identifiable information, including faces and license plates. To
> the extent that you like to request removal of specific images from the
> dataset, please contact privacy@zenseact.com.

The displayed derivatives and their terms are listed in the complete
[visual-asset notice](reports/figures/ZOD_ASSET_NOTICE.md).
