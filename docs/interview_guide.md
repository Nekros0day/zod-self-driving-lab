# Interview guide

## The 30-second version

I built three leakage-audited learning tracks on ZOD. For three-second ego
forecasting, a temporal FNO achieved 0.542 m ADE and 2.69 ms latency, narrowly
beating two multiple-shooting NeuralODEs. For camera affordances, ResNet-18 U-Net
raised thin-lane tolerant F1 from 0.654 to 0.861; a four-times-larger Fourier
variant did not establish a meaningful improvement. For BEV perception, I moved
from a 12-frame transfer diagnostic to protected 70/16/30 recording roles,
fine-tuned SFA3D, tested PointPillars and CenterPoint controls, and fused camera
semantics with calibrated LiDAR depth. The hybrid preserves vehicle AP@0.30 at
0.616 and raises pedestrian/cyclist AP to 0.529/0.327.

## Questions I should be ready for

### Why multiple shooting?

It replaces one long, difficult training IVP with shorter shared-parameter
segments and penalizes boundary disagreement. Target-derived boundaries exist
only in the loss; inference remains one causal rollout.

### Why did physics not win?

Exact kinematics improve interpretability, but biased yaw rate and unmodeled
road/driver effects persist through the constrained flow. Physics is a useful
prior, not an automatic accuracy guarantee.

### Why can an FNO model a trajectory?

An FNO maps one discretized function to another. Here the input is masked state
history plus future-time queries. Low temporal modes provide global mixing and
a CTRV residual supplies physical boundary behavior.

### Why use tolerant lane F1?

A one-pixel shift can destroy strict IoU for a thin marking. Tolerant F1 measures
geometric proximity; strict IoU remains visible so thickness cannot hide.

### Why not promote Fourier U-Net?

Its +0.0010 score difference has a confidence interval crossing zero while its
parameters rise from 14.4M to 56.8M. Promotion considers uncertainty and cost.

### What does the BEV detector receive and predict?

It receives one calibrated ego-frame LiDAR sweep rasterized as intensity,
height, and density. SFA3D predicts class centers, offsets, size, vertical
position, and yaw components, decoded into oriented metric 3-D boxes.

### Why did one sweep beat five?

Ego compensation aligns static structure, not independently moving actors.
Five scans increase density but smear objects. Validation selected one sweep for
detection. Five still stabilize foreground depth for camera proposals.

### How does camera-LiDAR fusion work?

Faster R-CNN recognizes image objects. Calibrated projection places LiDAR points
inside each camera box, robust depth lifts it into ego coordinates, and
same-class metric proposals are associated. Camera-only pedestrian/cyclist
proposals may supplement LiDAR; Vehicle passes through unchanged.

### Why did PointPillars and CenterPoint fail?

They were trained from scratch on 70 recordings. The cohort is too small to
learn robust low-level 3-D features and rare classes. This demonstrates the
value of transfer, not that those architecture families are generally poor.

### Is the BEV improvement convincing?

It comes from a protected 30-recording test, not tuning the old 12-frame mini
sample. Fusion raises pedestrian AP@0.30 from 0.501 to 0.529 and cyclist AP from
0.156 to 0.327 while exactly preserving vehicle output. Rare-user support is
still limited, so this is a bounded study rather than a safety claim.

### What would I do next?

- repeat the benchmark on a larger official ZOD Frames cohort;
- fine-tune an image detector on native ZOD categories;
- add object-aware temporal compensation or learned multi-sweep fusion;
- benchmark pretrained full-scale CenterPoint or BEVFusion;
- quantify geographic, weather, distance, and calibration shift;
- evaluate tracking on a contiguous labeled MOT benchmark.

## Claims I should not make

- This is not an end-to-end driving or navigation system.
- The trajectory forecast does not react to future obstacles or route intent.
- Segmentation does not yet feed the trajectory model.
- FNO is not established as significantly more accurate than NeuralODE.
- Fourier U-Net is not established as better than ordinary U-Net.
- The bounded BEV cohort is not a production-scale safety evaluation.
- The qualitative Kalman animation is not a MOT benchmark.
