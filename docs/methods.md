# Methods and mathematics

## 1. Coordinate frame and targets

Every future pose is expressed in the ego frame at the prediction anchor. If
\(T^w_t\in SE(3)\) maps the vehicle frame at time \(t\) into world coordinates,
then the relative transform is

\[
T^0_t=(T^w_0)^{-1}T^w_t,
\qquad
\mathbf y_t = [T^0_t]_{x,y}.
\]

This removes absolute map position and heading. At the anchor, forward is
positive \(x\), left is positive \(y\), and the target begins at \(t=0.1\) s.

## 2. Baselines

Constant velocity uses

\[
\hat x(t)=vt,\qquad \hat y(t)=0.
\]

Constant turn-rate and velocity (CTRV) uses current speed \(v\) and yaw rate
\(r\):

\[
\hat x(t)=\frac{v}{r}\sin(rt),\qquad
\hat y(t)=\frac{v}{r}(1-\cos(rt)).
\]

The implementation switches continuously to the straight limit as
\(|r|\to0\). B2 flattens values and missingness masks from the state history and
maps them through a compact MLP to all 30 future points.

## 3. NeuralODE

The generic state is \(z=[x,y,v_x,v_y]\). A causal GRU encodes normalized
history values, missingness, and relative time into context \(c\). The vector
field is

\[
\dot z = [v_x,v_y,a_{\theta,x}(z,c,t),a_{\theta,y}(z,c,t)],
\]

where learned acceleration is bounded by a `tanh`. A fixed fourth-order
Runge–Kutta step makes the numerical method explicit:

\[
z_{n+1}=z_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\]

The hybrid state is \(z=[x,y,\psi,v,r]\). It preserves

\[
\dot x=v\cos\psi,\quad
\dot y=v\sin\psi,\quad
\dot\psi=r,
\]

and learns only bounded longitudinal acceleration \(\dot v\) and yaw
acceleration \(\dot r\). Current acceleration provides a decaying causal prior.

## 4. Multiple shooting

Long ODE rollouts can produce unstable gradients. I split the 30-step training
future into three ten-step initial-value problems. If \(\Phi_\theta\) is the RK4
flow and \(\tilde z_j\) is a target-derived boundary state used only in training,

\[
\mathcal L =
\lambda_s\frac13\sum_{j=0}^{2}
  \operatorname{ADE}(\Phi_\theta(\tilde z_j),Y_j)
+\lambda_c\frac12\sum_{j=0}^{1}
  \|\Phi_\theta(\tilde z_j)_{\mathrm{end}}-\tilde z_{j+1}\|^2
+\lambda_f\operatorname{ADE}(\Phi_\theta(z_0),Y).
\]

The three equal shooting intervals share one batched RK4 solve. This changes
kernel-launch efficiency, not the objective. Inference never receives a future
boundary and always performs one uninterrupted rollout.

## 5. Temporal Fourier Neural Operator

The FNO treats observed history plus future query times as a one-dimensional
function grid. Each block applies

\[
u_{l+1}(t)=u_l(t)+\sigma\left(Wu_l(t)+
\mathcal F^{-1}(R_l\cdot\mathcal F(u_l))(t)\right),
\]

where \(R_l\) learns only the lowest 16 temporal modes. Future grid entries
contain time queries and the repeated causal anchor, never target values. The
head predicts a residual around CTRV. Zero initialization makes the initial
operator exactly reproduce that physical reference.

## 6. Segmentation

Road and lane are overlapping binary labels, so the loss is channel-wise
weighted binary cross entropy rather than softmax cross entropy:

\[
\mathcal L=-\sum_c w_c y_c\log\sigma(\ell_c)
 +(1-y_c)\log(1-\sigma(\ell_c)).
\]

The U-Net decoder upsamples ResNet-18 features and concatenates encoder skips.
This restores high-frequency spatial evidence lost by the bottleneck. Fourier
U-Net adds two 2-D spectral residual blocks to the 9×16 bottleneck, learning a
small rectangle of positive and negative frequency modes.

Strict lane IoU is intentionally harsh for one-pixel-wide structures. Tolerant
precision dilates the target before matching predictions; tolerant recall
dilates predictions before matching the target. Their harmonic mean measures
whether the predicted line lies within three pixels without allowing unlimited
thickness.

## 7. LiDAR-to-BEV representation

ZOD LiDAR returns are motion-compensated to the keyframe time and transformed
into the ego frame with the calibrated homogeneous extrinsic. The project uses
(x) forward, (y) left, and (z) up. Points inside a 50 m × 50 m front crop
are quantized to a 608 × 608 raster. In each cell, the highest return supplies
top height and robustly scaled intensity. If a cell contains (n) returns, its
density is

\[
d=\min\left(1,\frac{\log(1+n)}{\log 64}\right).
\]

The three detector channels are intensity, height, and density. This is a lossy
projection: it preserves metric ground-plane structure but discards most
within-cell vertical detail.

## 8. Center-based oriented detection

The external SFA3D FPN-ResNet-18 checkpoint predicts a three-class center
heatmap plus sub-cell offset, direction, vertical center, and dimensions. The
adapter decodes peaks and maps pixel locations and sizes back into metres in the
ZOD ego frame. No SFA3D code or weights are vendored, and the fixed transfer
benchmark performs no ZOD fitting or threshold calibration.

An oriented footprint is formed by rotating local length/width corners by yaw.
Intersection is computed by convex polygon clipping, followed by

\[
IoU_{BEV}=\frac{|P\cap G|}{|P|+|G|-|P\cap G|}.
\]

Evaluation greedily chooses the highest-IoU class-consistent one-to-one matches
at (IoU\ge0.5).

## 9. Constant-velocity Kalman tracking

Each track has state ([x,y,v_x,v_y]^\top). With elapsed time (Delta t),

\[
F=\begin{bmatrix}
1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1
\end{bmatrix},\quad
s^- = Fs,\quad P^-=FPF^\top+Q.
\]

Position observations use the standard Kalman gain
(K=P^-H^\top(HP^-H^\top+R)^{-1}). Association is class-aware greedy nearest
neighbor under a metric gate. This deliberately small tracker makes the state
transition and uncertainty update inspectable; it is not a learned MOT model.

## 10. Statistical unit

Overlapping trajectory windows from one recording are correlated. Resampling
individual windows would overstate certainty. Metrics are first averaged over
seeds per sample, and the bootstrap resamples whole recording groups. For
segmentation, each keyframe corresponds to one complete recording, so the 51
test images are also the 51 bootstrap groups.

Primary references: the original [Neural ODE paper](https://papers.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html),
[Differentiable Multiple Shooting Layers](https://proceedings.neurips.cc/paper/2021/file/89b9c689a57b82e59074c6ba09aa394d-Paper.pdf),
[Fourier Neural Operator](https://arxiv.org/abs/2010.08895), and
[U-Net](https://arxiv.org/abs/1505.04597). BEV detection follows the external
[SFA3D implementation](https://github.com/maudzung/SFA3D), pinned by commit in
the public transfer report.
