# Frozen experiment protocol

This document records the contract used before final-test evaluation. It is
kept separate from the results so that the rules cannot be rewritten around a
preferred outcome.

## Dynamics track

- Input: 21 causal samples over two seconds, with nine normalized vehicle-state
  channels and an explicit feature-validity mask.
- Output: 30 local-frame \((x,y)\) samples at 10 Hz, covering three seconds.
- Split: the unchanged parent recording roles; 11,208 train windows from 315
  recordings, 2,600 validation windows from 73 recordings, and a sealed 2,549
  window / 72 recording test role.
- Selection: lowest validation ADE independently for each seed.
- Models: generic NeuralODE, physics-constrained NeuralODE, and temporal FNO.
- Seeds: 2026, 2027, 2028.
- Training: AdamW, cosine decay, 40 epochs, batch 2,048, CUDA AMP, and gradient
  clipping. In-memory tensor batches use no worker processes on Windows.
- Test uncertainty: average each per-sample metric over seeds, then bootstrap
  complete recordings 2,000 times.
- References: CV, CTRV, and the three frozen B2 state-MLP checkpoints.

The test cache was created only after all nine model checkpoints were frozen.
Evaluation revision 2 corrected seed aggregation from prediction ensembling to
the preregistered mean of per-sample metrics. No model, hyperparameter, or
checkpoint changed after that correction.

## Segmentation track

- Input: one 512×288 front-camera keyframe.
- Output: overlapping road and lane-marking logits.
- Dataset: 489 labeled recordings.
- Split repair: old validation remains validation; a deterministic,
  country-stratified 15% subset of old train becomes a fresh final test; old
  test joins training because its aggregate metrics were previously observed.
- Final roles: 365 train, 73 validation, 51 sealed test.
- Models: DeepLabV3-MobileNetV3-Large, ResNet-18 U-Net, and the same U-Net with
  two low-mode Fourier blocks at the bottleneck.
- Seeds: 2026, 2027, 2028.
- Selection: checkpoint by validation score at threshold 0.5; then independently
  tune road and lane thresholds on validation only.
- Metrics: road IoU, strict lane IoU, lane F1 with three-pixel tolerance, and
  \(S=\tfrac12(\mathrm{road\ IoU}+\mathrm{lane\ tolerant\ F1})\).
- Precision: real-valued models use AMP. Fourier U-Net uses FP32 because CUDA
  gradient scaling cannot unscale complex spectral-weight gradients.
- Test uncertainty: average each per-image metric over seeds, then bootstrap
  complete recordings 2,000 times.

## Publication boundary

Raw ZOD assets, masks, manifests with identifiers, tensor caches, checkpoints,
and per-sample metrics stay outside the repository. Public reports contain only
aggregate counts, hashes, metrics, intervals, and environment information.
