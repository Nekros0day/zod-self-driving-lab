# Methods and mathematics

## 1. Coordinate frame and trajectory targets

Every future pose is expressed in the ego frame at the prediction anchor. If
\(T^w_t\in SE(3)\) maps the vehicle frame at time \(t\) into world coordinates,

\[
T^0_t=(T^w_0)^{-1}T^w_t, \qquad \mathbf y_t=[T^0_t]_{x,y}.
\]

The model therefore predicts relative motion, not absolute map position. Forward
is positive \(x\), left is positive \(y\), and the future begins at 0.1 s.

## 2. Trajectory references

Constant velocity uses \(\hat x(t)=vt,\ \hat y(t)=0\). Constant turn-rate and
velocity (CTRV) uses

\[
\hat x(t)=\frac{v}{r}\sin(rt),\qquad
\hat y(t)=\frac{v}{r}(1-\cos(rt)),
\]

with the continuous straight-line limit as \(|r|\to0\). B2 flattens state
history and validity masks into a compact MLP that predicts all 30 future points.

## 3. NeuralODEs

The generic state is \(z=[x,y,v_x,v_y]\). A causal GRU encodes normalized
history, missingness, and relative time into context \(c\). The vector field is

\[
\dot z=[v_x,v_y,a_{\theta,x}(z,c,t),a_{\theta,y}(z,c,t)].
\]

The hybrid state is \(z=[x,y,\psi,v,r]\). It preserves vehicle kinematics

\[
\dot x=v\cos\psi,\quad \dot y=v\sin\psi,\quad \dot\psi=r,
\]

and learns only bounded longitudinal acceleration \(\dot v\) and yaw
acceleration \(\dot r\). Both use explicit fourth-order Runge-Kutta integration:

\[
z_{n+1}=z_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\]

## 4. Multiple shooting

The 30-step future is split into three ten-step training IVPs. If
\(\Phi_\theta\) is the RK4 flow and \(\tilde z_j\) is a target-derived boundary
used only by the training objective,

\[
\mathcal L=\lambda_s\frac13\sum_{j=0}^{2}
\operatorname{ADE}(\Phi_\theta(\tilde z_j),Y_j)
+\lambda_c\frac12\sum_{j=0}^{1}
\lVert\Phi_\theta(\tilde z_j)_{end}-\tilde z_{j+1}\rVert^2
+\lambda_f\operatorname{ADE}(\Phi_\theta(z_0),Y).
\]

Inference never receives a future boundary and always performs one causal
rollout. Multiple shooting stabilizes training; it is not extra test input.

## 5. Temporal Fourier Neural Operator

The FNO treats observed history plus future queries as a one-dimensional
function grid. Each residual block applies

\[
u_{l+1}(t)=u_l(t)+\sigma\left(Wu_l(t)+
\mathcal F^{-1}(R_l\mathcal F(u_l))(t)\right),
\]

where \(R_l\) learns only the lowest 16 temporal modes. Future entries contain
time queries and the repeated causal anchor, never targets. The output is a
residual around CTRV, with zero initialization reproducing the physics reference.

## 6. Road and lane segmentation

Road and lane are overlapping labels, so each channel uses weighted binary
cross entropy rather than softmax:

\[
\mathcal L=-\sum_c w_cy_c\log\sigma(\ell_c)
 +(1-y_c)\log(1-\sigma(\ell_c)).
\]

The U-Net decoder concatenates ResNet-18 encoder skips to restore high-frequency
spatial evidence. Fourier U-Net adds two 2-D low-mode spectral blocks at the
bottleneck. Lane tolerant precision dilates the target before matching;
tolerant recall dilates the prediction. Strict lane IoU is reported alongside
it so thick predictions cannot exploit tolerance.

## 7. LiDAR-to-BEV representation and temporal alignment

Each LiDAR return is deskewed, transformed by the calibrated sensor-to-ego
extrinsic, and moved into the keyframe ego frame. For a point measured at
\(t_i\),

\[
p_i^{e_0}=(T^w_{e_0})^{-1}T^w_{e_i}T^{e_i}_{L}p_i^L.
\]

The crop is \(x\in[0,50]\) m, \(y\in[-25,25]\) m, \(z\in[-1,3]\) m. Its
608 x 608 cells store robust intensity, top height, and log-density

\[
d=\min\left(1,\frac{\log(1+n)}{\log 64}\right).
\]

Five compensated sweeps align static surfaces but smear independently moving
actors. Validation therefore selected one detector sweep. Five sweeps are used
only to stabilize camera-object foreground depth.

## 8. BEV targets, architectures, and loss

At output cell \((u,v)\), each object creates a class Gaussian center target.
Regression targets exist only at centers: sub-cell offset, dimensions, vertical
center, and \((\sin\psi,\cos\psi)\). The objective is

\[
\mathcal L=\lambda_h\mathcal L_{focal}(H,H^*)+
M\left(\lambda_o\lVert o-o^*\rVert_1+
\lambda_d\lVert d-d^*\rVert_1+
\lambda_z\lVert z-z^*\rVert_1+
\lambda_r\lVert r-r^*\rVert_1\right),
\]

where \(M\) selects valid centers. SFA3D uses an FPN-ResNet-18 BEV backbone and
center heads. Training starts from a pinned KITTI checkpoint and progressively
unfreezes heads, deeper features, then the full network. Class-balanced sampling
increases rare pedestrian/cyclist exposure.

PointPillars and CenterPoint are native controls. For a vertical pillar \(P\),
the point encoder is conceptually

\[
f_P=\max_{i\in P}\phi([p_i,p_i-\bar p_P,p_i-c_P]).
\]

PointPillars applies anchor classification/regression; the CenterPoint control
uses anchor-free heatmaps. Both overfit when initialized from scratch on only 70
recordings, demonstrating the value of transfer rather than a general weakness
of those architecture families.

## 9. Camera depth lifting and class-gated fusion

Faster R-CNN supplies image-space semantics. The calibrated Kannala-Brandt model
projects an ego ray through

\[
r(\theta)=\theta+k_1\theta^3+k_2\theta^5+k_3\theta^7+k_4\theta^9.
\]

Projected LiDAR points inside a camera box provide a robust foreground depth;
the corresponding ray is lifted back to an ego ground location. Same-class
metric proposals are associated. Unmatched camera proposals supplement only
Pedestrian and Cyclist. Vehicle geometry and confidence pass through LiDAR
unchanged, making vehicle AP preservation an explicit invariant.

## 10. Oriented evaluation and calibration

Rotated length/width corners form an oriented footprint. Convex clipping gives
intersection area and

\[
IoU_{BEV}=\frac{|P\cap G|}{|P|+|G|-|P\cap G|}.
\]

Confidence-ranked evaluation uses class-consistent one-to-one matches and
101-point interpolated AP at IoU 0.30, 0.50, and 0.70. Fixed operating points
also report precision/recall/F1, center/yaw/size errors, near/mid/far slices,
expected calibration error, and Brier score. All thresholds use validation.

## 11. Constant-velocity Kalman tracking

Each track has state \([x,y,v_x,v_y]^\top\). For elapsed time \(\Delta t\),

\[
F=\begin{bmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix},
\quad s^-=Fs,\quad P^-=FPF^\top+Q.
\]

Position observations use the standard Kalman gain. Association is class-aware
greedy nearest neighbor under a metric gate. This teaching tracker is not a
learned or quantitatively validated MOT system.

## 12. Statistical unit

Overlapping windows from one recording are correlated. Dynamics intervals
therefore resample complete recording groups after averaging seed metrics per
sample. Segmentation uses paired image differences and recording resampling.
The bounded BEV test reports complete confidence-ranked detection metrics; its
rare-class support remains too small for broad safety claims.

Primary references: [Neural ODE](https://papers.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html),
[Differentiable Multiple Shooting](https://proceedings.neurips.cc/paper/2021/file/89b9c689a57b82e59074c6ba09aa394d-Paper.pdf),
[FNO](https://arxiv.org/abs/2010.08895), [U-Net](https://arxiv.org/abs/1505.04597),
and the external [SFA3D implementation](https://github.com/maudzung/SFA3D).
