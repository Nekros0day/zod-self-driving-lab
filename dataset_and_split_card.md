# Dataset and split card

## Sources

This project uses the authorized Zenseact Open Dataset (ZOD) Sequences and
Frames releases, including official road/lane and 3-D object annotations. No raw
licensed sample is stored in the repository. Attributed qualitative derivatives
are listed in the visual-asset notice. Dataset access, attribution, privacy, and
redistribution are governed by Zenseact's terms.

## Dynamics samples

One sample contains 21 causal vehicle-state queries from \(t_0-2\) s through
\(t_0\), plus 30 local-frame future points from \(t_0+0.1\) through
\(t_0+3.0\) s. State features are speed, acceleration, yaw rate, steering,
accelerator, brake, left/right indicators, and delta time, each with validity.

| Role | Recordings | Windows |
|---|---:|---:|
| Train | 315 | 11,208 |
| Validation | 73 | 2,600 |
| Calibration (legacy B2 only) | 24 | 854 |
| Test | 72 | 2,549 |

All windows from one recording share one role. Train-only normalization and role
memberships are inherited unchanged from the corrected parent benchmark.

## Segmentation samples

One sample is a 512×288 front-camera keyframe with overlapping binary ego-road
and lane-marking masks. There are 489 complete-recording samples.

| Role | Recordings | Construction |
|---|---:|---|
| Train | 365 | Remaining old train plus previously observed test |
| Validation | 73 | Previous validation preserved exactly |
| Test | 51 | Fresh country-stratified subset of previous train |

The split was redesigned because metrics on the previous test role had already
been observed. Test selection uses seeded SHA-256 ranking within country; strata
with fewer than three examples remain in train.

## BEV transfer samples

The transfer diagnostic uses all 12 annotated keyframes in ZOD Frames mini. One
sample contains a motion-compensated LiDAR sweep, calibration, and eligible
dynamic 3-D boxes. There are 104 non-unclear in-range targets after mapping
Vehicle, Pedestrian, and VulnerableVehicle to the three detector classes. No
training or model selection is performed on these frames.

A separate 20-second ZOD sequence supplies only the qualitative tracker GIF. It
is not used to compute detection or MOT metrics.

## Public/private boundary

Public cache receipts contain counts, shapes, hashes, and byte totals. External
private artifacts contain exact tensors, image/mask paths, recording IDs,
checkpoints, and per-sample metrics. Source paths are command-line arguments and
never serialized into public reports.

## Known coverage limitations

- Dynamics describes the corrected 484-recording subset, not all ZOD.
- Segmentation has only 51 final test keyframes and sparse rare-country coverage.
- No closed-loop control or interaction with other road users is evaluated.
- Country, weather, and collection-car shifts are not separately powered.
- The 12-frame BEV diagnostic is too small for general detector conclusions;
  its fixed KITTI checkpoint has zero vulnerable-road-user true positives.
