# Data and evaluation contract

## Access and storage

ZOD access is granted by Zenseact. I do not redistribute images, annotations,
metadata, tensors, identifiers, or download locators. Licensed source data,
caches, and checkpoints resolve outside the repository. Public receipts bind
private ID sets by SHA-256 and expose only aggregate counts.

## Dynamics roles

| Role | Recordings | Windows | Used for |
|---|---:|---:|---|
| Train | 315 | 11,208 | Fitting and shooting boundaries |
| Validation | 73 | 2,600 | Checkpoint selection |
| Test | 72 | 2,549 | One frozen evaluation |

The normalizer is fitted on train only. Sample and role membership are
checksum-bound. Test never changes normalization, epochs, or model choices.

## Segmentation roles

The earlier standalone project had exposed aggregate test metrics, so that test
could not remain final. I preserved its 73 validation recordings, selected a
fresh country-stratified 51-recording test role from old train, and trained on
the remaining 365 recordings. Every architecture and seed was retrained from
the same ImageNet starting conditions.

## BEV roles

The final BEV experiment uses annotated central keyframes from locally complete
ZOD Sequences recordings. A recording is eligible only when image, LiDAR,
calibration, ego motion, and official 3-D object labels are all available. The
two mini recordings are excluded. A fixed seed produced these disjoint roles:

| Role | Recordings | Vehicle labels | Pedestrian labels | Cyclist labels |
|---|---:|---:|---:|---:|
| Train | 70 | 852 | 50 | 28 |
| Validation | 16 | 220 | 53 | 24 |
| Sealed test | 30 | 351 | 31 | 18 |

Counts above describe metadata labels before range, visibility, and dynamic-class
filtering. Private recording IDs are not published; their set hashes and the
selection policy are stored in `reports/bev_protected_roles.json`.

The available cohort is bounded by local sensor completeness. It is substantially
stronger than the original 12-frame smoke test, but it is not the full ZOD Frames
benchmark. A larger Frames confirmation remains future work.

## Selection and calibration

- Dynamics checkpoints minimize validation ADE.
- Segmentation checkpoints maximize validation score; road/lane thresholds are
  independently selected on validation.
- BEV training uses class-balanced sampling and early stopping on validation
  loss. The SFA3D sweep count and operating confidence are selected on validation.
- Camera, LiDAR, and fusion confidence thresholds are independently selected on
  validation. The sealed test does not influence them.
- Five-sweep detector input lost to one sweep on validation and is retained as a
  negative temporal-control result.

## Metrics

Trajectory ADE averages Euclidean distance over 30 horizons; FDE uses the last
valid horizon; miss rate is the fraction with FDE above two metres.

Segmentation reports global pixel metrics and per-image metrics. The promotion
score is half road IoU plus half three-pixel-tolerant lane F1. Model-pair
intervals use paired per-image differences.

BEV detection uses confidence-ranked 101-point AP with class-consistent,
one-to-one oriented box matching at IoU 0.30, 0.50, and 0.70. The report also
contains precision-recall curves, center/yaw/size error, 0-20/20-35/35-50 m
range slices, and expected calibration error/Brier score. A fixed confidence
operating point is reported separately from AP.

## Integrity notes

The trajectory evaluator's first revision averaged seed predictions, creating
an unintended ensemble. Revision 2 recomputed the result as the preregistered
mean of per-sample metrics without changing checkpoints or configurations.

The BEV mini transfer diagnostic remains useful historical evidence, but it no
longer supports the promoted claim. The final claim comes from the protected
70/16/30 recording study.
