# Frozen experiment protocol

This records the separation between fitting, validation selection, and final
evaluation. Results are documented separately so the rules are not rewritten
around an attractive test outcome.

## Dynamics track

- Input: 21 causal samples over two seconds, nine normalized vehicle-state
  channels, and explicit validity masks.
- Output: 30 ego-local `(x, y)` points over three seconds at 10 Hz.
- Roles: 315/73/72 recordings for train/validation/test.
- Models: generic NeuralODE, physics-constrained NeuralODE, temporal FNO; CV,
  CTRV, and frozen B2 MLP references.
- Seeds: 2026, 2027, 2028.
- Selection: lowest validation ADE per seed.
- Uncertainty: mean per-sample metric over seeds, then 2,000 recording-group
  bootstrap draws.

## Segmentation track

- Input: one 512 x 288 front-camera keyframe.
- Output: overlapping road and lane-marking logits.
- Roles: 365/73/51 recordings for train/validation/fresh test.
- Models: DeepLabV3-MobileNetV3-Large, ResNet-18 U-Net, and Fourier U-Net.
- Seeds: 2026, 2027, 2028.
- Loss: channel-weighted multilabel BCE.
- Selection: best validation score, then independent validation thresholds for
  road and lane.
- Test uncertainty: paired per-image differences with recording bootstrap.

## LiDAR-camera BEV track

- Dataset: annotated ZOD Sequences central keyframes with complete image, LiDAR,
  calibration, ego-motion, and label data; mini recordings excluded.
- Roles: 70 train, 16 validation, 30 sealed test recordings. IDs remain private
  and sets are hash-bound.
- Spatial support: 0-50 m forward, +/-25 m lateral, -1-3 m vertical, 608 x 608.
- Classes: Vehicle, Pedestrian, and Cyclist after ZOD label mapping.
- Transfer model: external SFA3D FPN-ResNet-18 initialized from a pinned KITTI
  checkpoint, then fine-tuned in stages: prediction heads, deeper feature path,
  and finally the complete network.
- Native controls: a compact PointPillars anchor detector and pillar-CenterPoint
  detector, both trained from scratch under the same roles.
- Sampling: class-balanced training sampler increases rare-user exposure.
- Temporal comparison: one motion-compensated sweep versus five. Sweep count is
  selected on validation; one sweep wins for the detector.
- Camera branch: COCO-pretrained Faster R-CNN supplies semantic boxes. Calibrated
  Kannala-Brandt projection associates LiDAR foreground points, whose robust
  depth lifts camera boxes into ego coordinates.
- Fusion: match same-class metric boxes; supplement unmatched camera proposals
  only for Pedestrian/Cyclist. Vehicle boxes and confidence pass through exactly.
- Selection: early stopping, sweep count, and three operating thresholds use
  train/validation only.
- Test metrics: 101-point AP and PR at oriented IoU 0.30/0.50/0.70, fixed-point
  precision/recall/F1, geometry errors, range slices, ECE, and Brier score.
- Tracking: constant-velocity Kalman tracking remains qualitative because the
  selected central-keyframe benchmark is not a contiguous labeled MOT stream.

## Publication boundary

Raw ZOD assets, masks, point clouds, recording IDs, tensor caches, checkpoints,
and per-frame predictions stay outside Git. Public files contain aggregate
counts, hashes, metrics, environment information, and attributed qualitative
derivatives only.
