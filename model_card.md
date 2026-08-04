# Model card

## Intended use

I use this repository as an educational offline benchmark for causal ego-trajectory
forecasting and road/lane segmentation. It is not a driving policy,
safety monitor, or production perception stack.

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

## Retained research controls

- Generic NeuralODE demonstrates true multiple-shooting continuous dynamics.
- Hybrid NeuralODE preserves planar kinematics and tests physical inductive bias.
- Fourier U-Net tests global spectral mixing, but its score difference from
  U-Net is compatible with zero and its cost is substantially higher.

## Evaluation

All learned results use seeds 2026–2028. Checkpoints are selected on validation.
Segmentation thresholds are fitted on validation. Confidence intervals resample
complete recordings after averaging aligned per-sample metrics over seeds.

## Limitations and risks

- State-only trajectory input cannot react to an unseen obstacle or traffic light.
- Camera segmentation is single-frame and does not model temporal consistency.
- Thin-lane tolerance can hide small offsets, so strict IoU is also mandatory.
- Aggregate performance does not guarantee rare-condition or geographic safety.
- The benchmark hardware latency is not an embedded deployment measurement.
