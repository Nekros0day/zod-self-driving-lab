# Project learning review

This is the short version I use when I return to the project after some time
away. The notebooks contain the derivations and runnable examples; this page
records the conclusions those experiments changed or confirmed for me.

## What I built

I developed four leakage-audited learning tracks on ZOD. For three-second ego
forecasting, temporal FNO reached 0.542 m ADE and 2.69 ms latency, narrowly
ahead of two multiple-shooting NeuralODEs. For camera affordances, ResNet-18
U-Net raised thin-lane tolerant F1 from 0.654 to 0.861; a much larger Fourier
variant did not establish a useful improvement. For BEV perception, I moved
from a 12-frame transfer diagnostic to protected 70/16/30 recording roles,
fine-tuned SFA3D, tested PointPillars and CenterPoint controls, and fused camera
semantics with calibrated LiDAR depth. The hybrid preserves vehicle AP@0.30 at
0.616 and reaches pedestrian/cyclist AP of 0.529/0.327.

The fourth track asks whether camera tasks and vehicle clients should share
parameters. A split-decoder model shares one ResNet-18 encoder across semantic
segmentation, road/lane affordances, sparse metric depth, and 2-D detection. It
uses 19.80M parameters instead of 43.44M for three separate networks, while
avoiding the lane collapse observed with a hard-shared decoder. Four uneven,
non-IID collection-car clients then provide a controlled FedAvg/FedProx study.
Local-optimizer deltas do not beat the centralized round-zero model, but
gradient-only FedSGD selects round 7: validation rises from 0.3715 to 0.3765
and the composite test score from 0.3392 to 0.3445. The gain is mostly sparse
depth, while detection AP decreases slightly.

## Lessons I want to retain

### Multiple shooting is training structure, not extra inference information

Multiple shooting replaces one difficult long training IVP with shorter
shared-parameter segments and a boundary-continuity penalty. Target-derived
boundary states exist only during training. Evaluation remains one causal
rollout from observed history.

### Physics is a useful prior, not an automatic winner

Exact kinematics improve structure and interpretation, but biased yaw rate and
unmodeled road or driver effects persist through a constrained flow. The hybrid
experiment is useful even though it did not become the promoted trajectory
model.

### FNO is an operator on a known temporal grid

The input is masked state history followed by known future-time query rows—not
future measurements. Low temporal modes provide global mixing and a CTRV
residual supplies sensible initial behavior. Its main result here is an
accuracy-latency engineering choice, not universal superiority to NeuralODE.

### Thin structures require geometric evaluation

A one-pixel shift can destroy strict IoU for a lane marking. Tolerant F1 measures
geometric proximity while strict IoU remains visible so that an excessively
thick mask cannot hide. Thresholds are selected on validation and frozen before
test.

### U-Net skips mattered more than a Fourier bottleneck

Both U-Net variants reliably improve the retrained DeepLab reference. The
Fourier U-Net's direct score difference from ordinary U-Net is uncertain while
its parameter cost is about four times larger, so ordinary U-Net is the cleaner
choice.

### Transfer dominated from-scratch architecture choice in bounded BEV data

PointPillars and CenterPoint controls were trained from scratch on 70 recordings
and overfit. Fine-tuned SFA3D transferred useful 3-D features. This is evidence
about the cohort and initialization, not a claim that those architecture
families are weak.

### More sweeps do not automatically mean better detection

Ego-motion compensation aligns static structure but not independently moving
actors. Five sweeps add density and also smear objects; validation selected one
sweep for detection. A longer accumulation still helped stabilize sparse depth
for camera proposals.

### Camera and LiDAR contribute different information

The camera recognizes object semantics. Calibrated LiDAR supplies metric depth
and BEV geometry. Class-gated fusion leaves the vehicle branch unchanged while
supplementing vulnerable-road-user proposals. The protected test improvement is
bounded evidence, not a safety claim.

### Sharing an encoder does not mean every spatial decoder should be shared

The hard-shared camera model is efficient, but thin-lane F1 falls to zero and
segmentation/depth encoder gradients conflict. Independent task decoders retain
one shared encoder pass while giving dense semantics, depth, and center
heatmaps different reconstruction paths. The result is a practical compromise,
not a claim that it dominates each single-task specialist.

### Federated aggregation needs a round-zero option

Sample-weighted FedAvg correctly gives more influence to clients with more
local samples, while uniform averaging overweights the smallest client.
FedProx constrains local drift, but neither rule guarantees improvement under
strong domain and label-availability shift. Selecting against the untouched
central model prevents a fleet update from being promoted merely because it
completed.

### A gradient message and a locally optimized delta are different algorithms

FedSGD evaluates every client's mean gradient at the same global checkpoint;
FedAvg performs local optimizer steps before returning a model delta. Averaging
complete client weights and averaging their deltas are algebraically equivalent
when their starting checkpoint is shared. In this experiment the small global
FedSGD steps improve validation while local AdamW deltas drift. Both messages
remain model-sized and potentially information-bearing unless compression and
privacy mechanisms are added.

## Boundaries of the project

- The trajectory model forecasts ego motion; it does not navigate or react to
  future obstacles.
- Segmentation does not currently feed the trajectory model.
- FNO is not proven significantly more accurate than NeuralODE.
- Fourier U-Net is not established as better than ordinary U-Net.
- The bounded BEV cohort is not a production-scale safety evaluation.
- The qualitative Kalman animation demonstrates mechanics, not MOT quality.
- Broad scene semantics are SegFormer teacher labels; their mIoU measures
  distillation agreement rather than native ZOD ground-truth accuracy.
- The federated experiment is a sequential single-machine simulation without
  secure aggregation, differential privacy, or a privacy guarantee.

## Experiments I would run next

- repeat the benchmarks on a larger official ZOD cohort;
- fine-tune an image detector on native ZOD categories;
- add object-aware motion compensation or learned multi-sweep fusion;
- benchmark a pretrained full-scale CenterPoint or BEVFusion model;
- quantify geographic, weather, distance, and calibration shift;
- evaluate tracking on a contiguous labeled MOT benchmark.
- expand each federated client before comparing SCAFFOLD, FedBN, partial
  participation, and privacy-preserving aggregation.
