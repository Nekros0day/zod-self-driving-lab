# Data and evaluation contract

## Access and storage

ZOD access is granted by Zenseact. I do not redistribute images, annotations,
metadata, tensors, or download locators. Every licensed derivative is required
to resolve outside the repository. Public cache receipts bind private files by
SHA-256 while exposing only aggregate shapes and counts.

## Dynamics roles

| Role | Recordings | Windows | Used for |
|---|---:|---:|---|
| Train | 315 | 11,208 | Parameter fitting and shooting boundaries |
| Validation | 73 | 2,600 | Epoch/checkpoint selection |
| Test | 72 | 2,549 | One frozen evaluation |

The normalizer is inherited from the original train role. Test data never
refits means or scales. Sample and split membership are checksum-bound.

## Segmentation roles

The earlier standalone segmentation project had already exposed aggregate test
metrics. Reusing that test would bias a new architecture benchmark. I therefore:

1. preserve all 73 old validation recordings;
2. deterministically select 51 country-stratified recordings from old train as
   the new final test;
3. combine remaining old train with the 73 previously observed test examples,
   yielding 365 training recordings;
4. retrain every model and seed from ImageNet initialization.

Tiny-country strata with fewer than three examples are retained in training.
The split assignment hash is public; raw recording IDs remain private.

## Selection and calibration

- Dynamics checkpoints minimize validation ADE.
- Segmentation checkpoints maximize the fixed-threshold validation score.
- Road and lane thresholds are then selected independently on a fixed validation
  grid from 0.30 to 0.75.
- Test metrics do not affect epochs, thresholds, hyperparameters, or model
  promotion.

## BEV transfer diagnostic

The BEV experiment uses all 12 annotated keyframes in ZOD Frames mini and an
unchanged KITTI-pretrained SFA3D checkpoint. It is neither trained nor selected
on ZOD. The confidence threshold (0.20), top-K limit (50), spatial crop, class
map, and oriented-IoU match threshold (0.50) are fixed in code.

Targets are non-unclear Vehicle, Pedestrian, and VulnerableVehicle boxes with
centers inside the front raster. Boxes are converted from their annotation
frame to ego coordinates before evaluation. The public report contains only
aggregate counts, scores, source/checkpoint hashes, and timing; no recording
identifiers or per-frame predictions are retained.

Because the mini subset has only 12 frames, this experiment is a reproducible
smoke/domain-shift diagnostic. It is not assigned the inferential status of the
sealed trajectory and segmentation benchmarks.

## Metrics

Trajectory ADE averages Euclidean distance over 30 horizons; FDE uses the last
valid horizon; miss rate is the fraction with FDE above two metres.

Segmentation reports global pixel confusion metrics and per-image metrics.
The composite score is half road IoU plus half lane tolerant F1. Model-pair
confidence intervals use per-image differences, which preserves pairing.

BEV detection reports class-consistent precision, recall, F1, mean matched
oriented IoU, and matched center error. The sequence GIF has no corresponding
framewise target stream and is therefore explicitly qualitative.

## Integrity notes

The first trajectory evaluator revision averaged seed predictions, creating an
implicit ensemble. This was inconsistent with the preregistered seed-reduction
rule. Revision 2 recomputed the report as the arithmetic mean of each
per-sample metric. No checkpoint or configuration changed. The correction is
recorded in the machine-readable test report rather than hidden.
