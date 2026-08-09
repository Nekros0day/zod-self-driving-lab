# Interview guide

## The 30-second version

I built two leakage-audited ZOD benchmarks plus a calibrated LiDAR perception
study. For three-second ego forecasting, I
compared a true multiple-shooting NeuralODE, a kinematic hybrid ODE, and a
temporal FNO against CV, CTRV, and a frozen state MLP. All three new models
reliably improved ADE; temporal FNO was best at 0.542 m and 2.69 ms. For camera
affordances, a ResNet-18 U-Net improved thin-lane tolerant F1 from 0.654 to
0.861. A Fourier U-Net tied it but required four times the parameters, so I
promoted the ordinary U-Net.
I then built a 608×608 LiDAR BEV detector and Kalman tracker. A frozen
KITTI-trained SFA3D checkpoint transferred reasonably to matched vehicles
(0.718 IoU and 0.293 m center error), but produced zero pedestrian/cyclist true
positives on ZOD mini, exposing the domain gap instead of hiding it.

## Questions I should be ready for

### Why multiple shooting?

Backpropagating through a long nonlinear flow can amplify solver and gradient
errors. Multiple shooting replaces one difficult training IVP with shorter
ones and explicitly penalizes their boundary mismatch. It does not make
inference easier: the final model still rolls out once from the observed state.

### Is using target-derived shooting states leakage?

Not if they exist only inside the training objective. They are analogous to
teacher-forced intermediate supervision. Validation and test `forward()` calls
receive only causal history; tests enforce that API boundary.

### Why did physics not win?

The exact kinematics reduce the hypothesis space and yield interpretable
motion, but errors in current yaw rate or the unmodeled road/driver dynamics can
persist through the constrained flow. The generic ODE can learn compensating
lateral dynamics. The result argues for physics as a useful prior, not an
automatic performance guarantee.

### Why can FNO work on a trajectory rather than a PDE?

An FNO learns a mapping between discretized functions. Here the function is a
masked state history plus future time queries. That is legitimate operator
learning, although the short non-periodic interval is not the canonical PDE
setting. I treated it as an empirical control and used low modes plus a CTRV
residual to manage boundary behavior.

### Why is lane tolerant F1 necessary?

One-pixel shifts can destroy strict IoU for a thin line despite nearly correct
geometry. Tolerant F1 asks whether predicted and target lines lie within a small
pixel radius. I still report strict lane IoU so a thick, imprecise mask cannot
hide behind tolerance.

### Why not promote Fourier U-Net?

Its score advantage over U-Net is +0.0010 with a confidence interval crossing
zero, while parameters rise from 14.4M to 56.8M and latency from 2.93 to 4.99
ms. Promotion should consider uncertainty and deployment cost, not only the
third decimal place.

### What does the BEV detector receive and predict?

It receives a motion-compensated LiDAR sweep in the calibrated ZOD ego frame,
rasterized as intensity, top-height, and log-density. SFA3D predicts center
heatmaps plus offset, size, direction, and vertical heads. The adapter decodes
oriented ground-plane boxes in metres; a separate constant-velocity Kalman
filter estimates track velocity over time.

### Why is the BEV result useful if vulnerable users fail?

It validates the complete sensor-to-metric-box path and establishes a pinned,
reproducible vehicle baseline. More importantly, the per-class evaluation shows
that a KITTI checkpoint cannot be presented as a general ZOD detector. The next
credible step is ZOD-native training with a protected split or camera–LiDAR
fusion, not choosing a favorable threshold on the 12 mini frames.

### What would I do next?

- train and evaluate a ZOD-native BEV detector on protected recording splits;
- test camera–LiDAR fusion for pedestrians and cyclists;
- add temporally labeled camera frames rather than single keyframes;
- test calibration under country, weather, and collection-car shift;
- distill the FNO trajectory model if edge latency becomes restrictive;
- expand segmentation labels before making stronger rare-condition claims.

## Claims I should not make

- This is not an end-to-end driving system.
- Segmentation does not yet improve the trajectory model.
- FNO is not significantly more accurate than NeuralODE.
- Fourier U-Net is not established as better than U-Net.
- A 51-recording segmentation test cannot establish broad geographic safety.
- The 12-frame BEV transfer diagnostic is not a detector or tracking benchmark.
