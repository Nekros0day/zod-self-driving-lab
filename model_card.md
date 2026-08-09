# Model card

## Intended use

I use this repository as an educational offline laboratory for causal
ego-trajectory forecasting, road/lane segmentation, and LiDAR bird's-eye-view
perception. It is not a driving policy, safety monitor, or production
perception stack.

## Promoted models

### Temporal FNO trajectory forecaster

- Input: normalized 21×9 state history, feature-validity mask, and future time
  queries.
- Reference: causal CTRV trajectory.
- Learned output: 30×2 residual through four width-96 spectral blocks with 16
  temporal modes.
- Test: 0.5424 m ADE, 1.4864 m FDE, 0.2346 miss rate.
- Cost: 1.23M parameters; 2.689 ms median batch-1 GPU latency.
- Reason for promotion: reliable improvement over B2 with ODE-level accuracy
  and much lower latency than either ODE.

### ResNet-18 U-Net road/lane segmenter

- Input: one ImageNet-normalized 512×288 RGB keyframe.
- Output: independent road and lane logits.
- Training: weighted binary cross entropy and validation-fitted per-class
  thresholds.
- Test: 0.8579 road IoU, 0.4663 strict lane IoU, 0.8609 lane tolerant F1.
- Cost: 14.37M parameters; 2.932 ms median batch-1 GPU latency.
- Reason for promotion: statistically reliable improvement over DeepLab and a
  better efficiency frontier than Fourier U-Net.

### Hybrid LiDAR-camera BEV detector

- LiDAR branch: single-sweep 608 x 608 intensity/height/density raster with
  SFA3D initialized from KITTI and progressively fine-tuned on ZOD.
- Camera branch: Faster R-CNN semantics lifted to ego coordinates with five
  motion-compensated LiDAR sweeps and calibrated projection.
- Fusion: camera-only Pedestrian/Cyclist proposals may supplement LiDAR;
  Vehicle boxes and confidence pass through unchanged.
- Protected test AP@0.30: 0.616 Vehicle, 0.529 Pedestrian, 0.327 Cyclist.
- Reason for promotion: large gains over unmodified transfer for Vehicle and
  Cyclist, plus exact preservation of the stronger LiDAR vehicle branch.

## Retained research controls

- Generic NeuralODE demonstrates true multiple-shooting continuous dynamics.
- Hybrid NeuralODE preserves planar kinematics and tests physical inductive bias.
- Fourier U-Net tests global spectral mixing, but its score difference from
  U-Net is compatible with zero and its cost is substantially higher.
- Unmodified SFA3D is retained as the cross-domain starting control.
- PointPillars and pillar-CenterPoint are retained as from-scratch small-data
  controls; neither is competitive with transferred SFA3D on 70 train recordings.
- Five-sweep detector input is retained as a temporal control because moving
  object trails reduce validation performance relative to one sweep.
- The constant-velocity Kalman tracker is a transparent qualitative sequence
  component; no frame-by-frame MOT ground truth is available for that demo.

## Evaluation

Dynamics and segmentation use seeds 2026–2028. Their checkpoints and thresholds
are selected on validation, with recording-group bootstrap intervals. BEV uses
recording-disjoint 70/16/30 roles; checkpoint, sweep count, and confidence
thresholds are frozen from validation before the sealed test.

## Limitations and risks

- State-only trajectory input cannot react to an unseen obstacle or traffic light.
- Camera segmentation is single-frame and does not model temporal consistency.
- Thin-lane tolerance can hide small offsets, so strict IoU is also mandatory.
- Aggregate performance does not guarantee rare-condition or geographic safety.
- The benchmark hardware latency is not an embedded deployment measurement.
- The bounded BEV cohort is not the complete ZOD Frames release and contains
  relatively few vulnerable-road-user instances.
- COCO camera categories and sparse projected depth can fail under occlusion or
  category shift. The BEV result is not a production safety guarantee.
- The Kalman sequence remains qualitative rather than a labeled MOT benchmark.
