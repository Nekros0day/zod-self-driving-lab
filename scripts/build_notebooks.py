"""Build and execute the focused educational notebook sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import nbformat as nbf
from nbclient import NotebookClient


def md(text: str) -> Any:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> Any:
    return nbf.v4.new_code_cell(text.strip())


SETUP = """
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
plt.style.use("seaborn-v0_8-whitegrid")
summary = json.loads((ROOT / "reports" / "benchmark_summary.json").read_text())
"""


def notebook(title: str, cells: list[Any]) -> Any:
    value = nbf.v4.new_notebook(
        cells=[
            md(
                f"# {title}\n\n"
                "*The full dataset remains outside Git. These notebooks combine synthetic worked "
                "examples, aggregate evidence, and small attributed qualitative panels.*"
            ),
            *cells,
        ]
    )
    value.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    value.metadata["language_info"] = {"name": "python", "version": "3"}
    return value


def project_map() -> Any:
    return notebook(
        "00 — Project map, claims, and evidence",
        [
            md(
                """
## What I am trying to learn

This project has three deliberately separated learning tracks. The dynamics track
maps **causal vehicle state history → future ego path**. The segmentation track
maps **one front-camera image → overlapping road and lane masks**. The BEV track
maps **a calibrated LiDAR sweep → oriented dynamic-object footprints and temporal
tracks**. Keeping the targets separate prevents an attractive but unsupported
claim that one perception output already improves trajectory prediction.

The scientific hierarchy is:

1. define coordinate, split, and leakage contracts;
2. freeze model families and seeds;
3. select checkpoints and thresholds on validation only;
4. evaluate a sealed test role;
5. bootstrap complete recordings rather than correlated windows.
"""
            ),
            md(
                r"""
## End-to-end map and tensor contracts

The three tracks share an experimental method, not a learned feature path.

| Track | Input | Shape | Output | Shape |
|---|---|---:|---|---:|
| Dynamics | normalized state values + validity mask | $B\times21\times9$ each | anchor-local $(x,y)$ future | $B\times30\times2$ |
| Segmentation | RGB front image | $B\times3\times288\times512$ | road/lane logits | $B\times2\times288\times512$ |
| BEV perception | intensity/height/density raster | $B\times3\times608\times608$ | class, center, size, yaw | variable detections |

The history covers $[-2,0]$ s at 10 Hz and the trajectory target covers
$(0,3]$ s. The two segmentation channels are independent because a lane
marking can also be part of the ego-road surface.
"""
            ),
            code(
                """
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
boxes = [
    (0.3, 3.8, 2.0, 1.0, "ZOD state\\nhistory"),
    (3.0, 3.8, 2.3, 1.0, "causal tensor\\n21 × 9 + mask"),
    (6.0, 3.8, 2.3, 1.0, "B2 / ODE / FNO"),
    (9.2, 3.8, 2.3, 1.0, "30 × 2 local\\ntrajectory"),
    (0.3, 1.2, 2.0, 1.0, "ZOD RGB\\nkeyframe"),
    (3.0, 1.2, 2.3, 1.0, "3 × 288 × 512\\nimage"),
    (6.0, 1.2, 2.3, 1.0, "DeepLab / U-Net"),
    (9.2, 1.2, 2.3, 1.0, "2 overlapping\\nprobability maps"),
]
for x, y, w, h, label in boxes:
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=.08", fc="#eaf2f8", ec="#2471a3"))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10)
for y in (4.3, 1.7):
    for x1, x2 in ((2.3,3.0),(5.3,6.0),(8.3,9.2)):
        ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle="->",mutation_scale=14,color="#555"))
ax.text(6, 5.45, "trajectory forecasting", ha="center", weight="bold", fontsize=12)
ax.text(6, 0.35, "road/lane segmentation", ha="center", weight="bold", fontsize=12)
plt.show()
"""
            ),
            code(SETUP),
            code(
                """
dyn = summary["dynamics"]
seg = summary["segmentation"]
print(f"Dynamics test: {dyn['sample_count']:,} windows / {dyn['recording_group_count']} recordings")
print(f"Segmentation test: {seg['sample_count']} keyframes / {seg['recording_group_count']} recordings")
print("Public evidence status:", summary["status"])
"""
            ),
            md(
                """
## What the frozen models actually produce

The first panel projects trajectory cases onto their calibrated front-camera
frames. Learned curves average the three frozen seeds, while the camera remains
context rather than model input. The second panel fixes seed 2026 and places
segmentation targets beside all three model outputs. Together they connect the
tensor contracts above to outputs I can inspect.
"""
            ),
            md(
                """
![Held-out camera trajectories](../reports/figures/dynamics_camera_predictions.png)

![Held-out segmentation outputs](../reports/figures/segmentation_model_comparison.png)
"""
            ),
            md(
                """
## Read a result as a claim with boundaries

“Temporal FNO is best” is too vague. The defensible version is:

> On the frozen 2,549-window ZOD test role, temporal FNO has the lowest
> three-seed mean ADE (0.542 m). Its paired ADE improvement over B2 is −0.131 m,
> and a 95% recording-level bootstrap interval excludes zero.

That sentence states the dataset role, metric, seed reduction, comparison, and
uncertainty unit. It does **not** claim safety, closed-loop performance, or
superiority under every distribution shift.
"""
            ),
            md(
                r"""
## Metrics in equations

For sample $i$, horizon step $t$, target $y_{i,t}$, and prediction
$\hat y_{i,t}$,

\[
\operatorname{ADE}_i=\frac{1}{T}\sum_{t=1}^{T}
\lVert\hat y_{i,t}-y_{i,t}\rVert_2,
\qquad
\operatorname{FDE}_i=\lVert\hat y_{i,T}-y_{i,T}\rVert_2.
\]

The miss indicator is $\mathbb 1[\operatorname{FDE}_i>2\text{ m}]$. For
segmentation channel $c$, strict intersection-over-union is

\[
\operatorname{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c}.
\]

The selection score is
$S=\tfrac12(\operatorname{IoU}_{road}+F1^{tol}_{lane})$. These definitions
matter because a ranking is only meaningful after the metric is fixed.
"""
            ),
            code(
                """
rows = []
for name, entry in dyn["models"].items():
    rows.append({
        "model": name,
        "ADE_m": entry["metrics_across_seeds"]["ade_m"]["mean"],
        "FDE_m": entry["metrics_across_seeds"]["fde_m"]["mean"],
        "latency_ms": entry["latency"]["batch_1_median_ms"],
        "parameters": entry["parameters"],
    })
pd.DataFrame(rows).sort_values("ADE_m").round(4)
"""
            ),
            code(
                """
rows = []
for name, entry in seg["models"].items():
    metrics = entry["global_pixel_metrics_across_seeds"]
    rows.append({
        "model": name,
        "road_IoU": metrics["road_iou"]["mean"],
        "lane_IoU": metrics["lane_iou"]["mean"],
        "lane_tolerant_F1": metrics["lane_tolerant_f1"]["mean"],
        "score": metrics["selection_score"]["mean"],
        "parameters_M": entry["parameters"] / 1e6,
    })
pd.DataFrame(rows).sort_values("score", ascending=False).round(4)
"""
            ),
            code(
                """
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
dyn_frame = pd.DataFrame([
    (name, entry["metrics_across_seeds"]["ade_m"]["mean"])
    for name, entry in dyn["models"].items()
], columns=["model", "ADE"])
seg_frame = pd.DataFrame([
    (name, entry["global_pixel_metrics_across_seeds"]["selection_score"]["mean"])
    for name, entry in seg["models"].items()
], columns=["model", "score"])
ax[0].barh(dyn_frame.model, dyn_frame.ADE, color="#2874a6")
ax[0].invert_yaxis(); ax[0].set_xlabel("ADE (m), lower is better")
ax[1].barh(seg_frame.model, seg_frame.score, color="#229954")
ax[1].invert_yaxis(); ax[1].set_xlabel("segmentation score, higher is better")
fig.suptitle("Frozen test evidence: two tasks, two metric directions")
plt.tight_layout(); plt.show()
"""
            ),
            md(
                """
## Evidence integrity

The JSON reports include checkpoint hashes and cache hashes. Hashes do not prove
that a model is good; they prove that the exact evaluated artifact is identifiable.
This matters when nine training runs share names and directory structures.
"""
            ),
            code(
                """
import hashlib
for filename in ("v4_dynamics_test.json", "v4_segmentation_test.json"):
    payload = (ROOT / "reports" / filename).read_bytes()
    print(filename, hashlib.sha256(payload).hexdigest()[:20] + "…")
"""
            ),
            md(
                """
## Takeaways

- The strongest trajectory claim is the **paired improvement over B2**, not the
  tiny FNO–NeuralODE ranking.
- The strongest segmentation claim is the **U-Net family improvement over
  DeepLab**, not the Fourier–ordinary U-Net ranking.
- Model promotion includes latency and parameter cost.
- Negative complexity findings are useful when they change the final decision.

### Check my understanding

1. Why would averaging predictions across seeds be a different estimator from
   averaging each seed's ADE?  
2. Which test examples may influence threshold choice?  
3. Why can the three tracks be presented in one project without claiming an
   end-to-end driving system?

Answers: ensembling changes predictions before a nonlinear metric; no test
example may select a threshold; and the tracks share data discipline and an
autonomous-driving setting but remain experimentally independent.
"""
            ),
        ],
    )


def geometry() -> Any:
    return notebook(
        "01 — Geometry, causal splits, and physics baselines",
        [
            code(SETUP),
            md(
                r"""
## Learning objectives and one supervised sample

After this notebook I should be able to derive the anchor-local target, explain
every input channel, implement CV/CTRV, and defend recording-level splits.

One sample contains 21 rows at 10 Hz. Its nine causal channels are speed $v$,
longitudinal acceleration $a$, yaw rate $r$, steering angle $\delta$,
accelerator ratio, brake flag, left/right indicators, and observation age
$\Delta t$. A Boolean mask with the same shape records freshness and finiteness.
The output contains 30 future $(x,y)$ positions at 10 Hz.
"""
            ),
            code(
                """
history_t = np.arange(-20, 1) * .1
future_t = np.arange(1, 31) * .1
fig, ax = plt.subplots(figsize=(11, 2.6))
ax.scatter(history_t, np.zeros_like(history_t), label="observed state", s=35)
ax.scatter(future_t, np.ones_like(future_t), label="trajectory target", s=28)
ax.axvline(0, color="black", lw=1.2); ax.text(.03, .48, "anchor $t=0$")
ax.set_yticks([0,1], ["21 × 9 input", "30 × 2 target"])
ax.set_xlabel("time relative to prediction anchor (s)")
ax.set_title("Causal window: nothing right of zero enters the input")
ax.legend(loc="upper center", ncol=2); plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## From world poses to a local learning target

Let $T^w_t$ map the vehicle frame at time $t$ into the world. I anchor every
future at $t=0$:

\[
T^0_t=(T^w_0)^{-1}T^w_t,
\qquad \mathbf y_t=[T^0_t]_{x,y}.
\]

The inverse is essential. Subtracting world translations alone would leave the
target rotated by global heading. Local coordinates make “forward” comparable
across Sweden, Germany, and France. In 2-D, $T=[R\;p;\;0\;1]$ and
$T^{-1}=[R^\top\;-R^\top p;\;0\;1]$, so
$p^0_t=R_0^\top(p^w_t-p^w_0)$: translate to the anchor and undo its heading.
"""
            ),
            code(
                """
def pose2d(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])

t = np.linspace(0, 3, 31)
world = np.stack([20 + 8*t, 50 + 0.7*t**2], axis=1)
yaw0 = np.deg2rad(35)
anchor = pose2d(*world[0], yaw0)
homogeneous = np.c_[world, np.ones(len(world))]
local = (np.linalg.inv(anchor) @ homogeneous.T).T[:, :2]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(world[:,0], world[:,1], marker='.')
ax[0].set_title("World trajectory")
ax[0].axis('equal')
ax[1].plot(local[:,0], local[:,1], marker='.')
ax[1].axhline(0, color='k', lw=.7)
ax[1].set_title("Anchor-local trajectory")
ax[1].set_xlabel("forward x (m)"); ax[1].set_ylabel("left y (m)")
ax[1].axis('equal'); plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Normalization and missingness are different information

Train-only statistics give

\[
\tilde x_{t,j}=\frac{x_{t,j}-\mu_j}{s_j}.
\]

Invalid entries are filled with zero *after normalization* and accompanied by
$m_{t,j}\in\{0,1\}$. Thus zero means “equal to the train mean” when $m=1$,
or “missing” when $m=0$. Fitting $\mu_j,s_j$ on validation or test would leak
their feature distribution into training.
"""
            ),
            code(
                """
rng = np.random.default_rng(3)
raw = rng.normal([12.0, 0.0, 0.02], [3.0, 1.2, .08], size=(21,3))
valid = np.ones_like(raw, dtype=bool); valid[7:11, 1] = False; valid[15, 2] = False
mean = np.nanmean(np.where(valid, raw, np.nan), axis=0)
scale = np.nanstd(np.where(valid, raw, np.nan), axis=0)
normalized = np.where(valid, (raw-mean)/scale, 0.0)
fig, ax = plt.subplots(1,2,figsize=(11,3.5), sharex=True)
ax[0].plot(history_t, normalized); ax[0].set_title("filled normalized values")
ax[0].legend(["speed", "accel", "yaw rate"], ncol=3, fontsize=8)
ax[1].imshow(valid.T, aspect="auto", cmap="Greys", extent=[-2,0,2.5,-.5])
ax[1].set_yticks([0,1,2], ["speed","accel","yaw rate"]); ax[1].set_title("validity mask")
for axis in ax: axis.set_xlabel("history time (s)")
plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## CV and CTRV

Constant velocity predicts $(vt,0)$. Constant turn-rate and velocity integrates
heading $\psi(t)=rt$:

\[
x(t)=\frac vr\sin(rt),\qquad
y(t)=\frac vr(1-\cos(rt)).
\]

As $r\to0$, these equations approach CV. A robust implementation
uses the straight limit near zero instead of dividing by a tiny yaw rate.
"""
            ),
            code(
                """
from zod_driveformer.models.baselines import constant_velocity, constant_turn_rate_velocity
speed, yaw_rate = 12.0, 0.18
cv = constant_velocity(speed, future_steps=30, dt=.1)
ctrv = constant_turn_rate_velocity(speed, yaw_rate, future_steps=30, dt=.1)
plt.figure(figsize=(7,4))
plt.plot(cv[:,0], cv[:,1], label='CV')
plt.plot(ctrv[:,0], ctrv[:,1], label='CTRV')
plt.axis('equal'); plt.xlabel('forward x (m)'); plt.ylabel('left y (m)')
plt.title('Same causal state, different physical assumption'); plt.legend(); plt.show()
"""
            ),
            md(
                r"""
## Why splits use complete recordings

Two neighboring windows can share nearly every state sample and future target.
Randomly splitting windows would put near-duplicates on both sides. The model
would appear to generalize while recognizing recording-specific dynamics.

If $g(i)$ is the recording containing window $i$, then for every pair of roles
$a\ne b$ the contract is
$\{g(i):i\in\mathcal D_a\}\cap\{g(i):i\in\mathcal D_b\}=\varnothing$.
This is stronger than merely having distinct row indices.
"""
            ),
            code(
                """
rng = np.random.default_rng(7)
recordings = np.repeat(np.arange(12), rng.integers(20, 45, size=12))
random_test = rng.random(len(recordings)) < .2
leaked_groups = set(recordings[random_test]) & set(recordings[~random_test])
group_test = np.isin(recordings, [1, 6, 10])
clean_overlap = set(recordings[group_test]) & set(recordings[~group_test])
print("Random-window split leaks recording groups:", sorted(leaked_groups))
print("Complete-recording split overlap:", clean_overlap)
"""
            ),
            md(
                r"""
## From point errors to ADE, FDE, and miss rate

Let $e_t=\lVert\hat y_t-y_t\rVert_2$. Then

\[
\operatorname{ADE}=\frac1T\sum_{t=1}^T e_t,\qquad
\operatorname{FDE}=e_T,\qquad
\operatorname{miss}=\mathbb 1[e_T>2\text{ m}].
\]

ADE rewards the whole path while FDE emphasizes the horizon endpoint. Neither
metric establishes closed-loop control stability.
"""
            ),
            code(
                """
target = ctrv
biased = ctrv + np.c_[.05*np.arange(1,31), .015*np.arange(1,31)**1.25]
errors = np.linalg.norm(biased-target, axis=1)
print(f"ADE={errors.mean():.3f} m, FDE={errors[-1]:.3f} m, miss={errors[-1] > 2.0}")
plt.figure(figsize=(8,3.5)); plt.plot(future_t, errors, marker="o", ms=3)
plt.axhline(2, color="crimson", ls="--", label="miss threshold")
plt.xlabel("future time (s)"); plt.ylabel("Euclidean error (m)")
plt.title("Error can grow strongly near the horizon endpoint")
plt.legend(); plt.tight_layout(); plt.show()
"""
            ),
            md(
                """
## Actual baseline lesson

CTRV improves CV from 1.036 m to 0.849 m ADE, so yaw rate contains useful
causal information. B2 improves further to 0.673 m by learning from two seconds
of state history. The new models must beat this learned reference, not merely CV.

### Check my understanding

- The anchor inverse makes a north-facing straight path point along local $+x$.
- The mask distinguishes a missing value from a valid value at the train mean.
- CTRV is a stronger turning baseline because it uses causal yaw rate.
"""
            ),
        ],
    )


def neural_ode() -> Any:
    return notebook(
        "02 — NeuralODE, RK4, and multiple shooting",
        [
            code(SETUP),
            md(
                r"""
## Learning objectives and architecture

I want to understand what is continuous, what remains discrete, how RK4
produces predictions, why multiple shooting is a *training loss*, and what the
physics hybrid constrains.

```text
states B×21×9 + mask ──> GRU context c B×128 ──┐
current physical state ──> ODE initial state z0 ├─> RK4, 30 × 0.1 s ─> B×30×2
time features [t/H, sin(πt/H), cos(πt/H)] ─────┘
```

The history encoder is discrete because observations arrive on a fixed grid.
The future decoder is continuous because it defines $\dot z=f_\theta(z,c,t)$
and can in principle be queried using another integration step size.
"""
            ),
            code(
                """
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(12,3.5)); ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis("off")
nodes=[(.3,2.4,2.1,.9,"21×9 states\\n+ validity"),(3.0,2.4,2.0,.9,"masked GRU\\ncontext 128"),
       (3.0,.7,2.0,.9,"physical anchor\\nstate"),(6.0,1.55,2.0,.9,"vector field\\n$f_\\theta(z,c,t)$"),
       (9.0,1.55,2.2,.9,"30 RK4 steps\\ntrajectory")]
for x,y,w,h,label in nodes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",fc="#f4ecf7",ec="#7d3c98"))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center")
for start,end in [((2.4,2.85),(3,2.85)),((5,2.85),(6,2.2)),((5,1.15),(6,1.9)),((8,2),(9,2))]:
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=14,color="#555"))
ax.text(7,3.25,"time features",ha="center",fontsize=9); ax.annotate("",(7,2.45),(7,3.05),arrowprops={"arrowstyle":"->"})
ax.set_title("NeuralODE forecasting path: targets are not model inputs",weight="bold")
plt.show()
"""
            ),
            md(
                r"""
## Generic continuous hidden dynamics

A NeuralODE defines a derivative rather than a fixed sequence of layers:

\[
\frac{d\mathbf z}{dt}=f_\theta(\mathbf z,t,\mathbf c),
\qquad \mathbf z(0)=\mathbf z_0.
\]

The generic state is $\mathbf z=[x,y,v_x,v_y]$. Its derivative is

\[
\dot{\mathbf z}=[v_x,v_y,a_{\theta,x},a_{\theta,y}],\qquad
\mathbf a_\theta=10\tanh(\operatorname{MLP}([\mathbf z,\mathbf c,\tau(t)])).
\]

The `tanh` bound prevents arbitrarily large learned acceleration. The context
$\mathbf c$ comes only from normalized history values, validity, and relative
history time. The initial state is $[0,0,v_0,0]$ in the anchor frame.
"""
            ),
            code(
                """
from zod_driveformer.dynamics.models import NeuralODEForecaster
teaching_ode = NeuralODEForecaster(
    state_dim=9, history_steps=21, future_steps=30, context_dim=32,
    step_seconds=.1, normalizer_mean=[0]*9, normalizer_scale=[1]*9,
    hidden_dim=48,
)
fake_states = torch.zeros(4,21,9); fake_states[:,-1,0] = 8.0
fake_mask = torch.ones_like(fake_states, dtype=torch.bool)
with torch.no_grad(): fake_output = teaching_ode(fake_states, fake_mask)
print("input:", tuple(fake_states.shape), "mask:", tuple(fake_mask.shape))
print("output:", tuple(fake_output.shape), "parameters:", sum(p.numel() for p in teaching_ode.parameters()))
"""
            ),
            md(
                r"""
## RK4 by hand

For step $h$, fourth-order Runge–Kutta samples the vector field four times:

\[
\begin{aligned}
k_1&=f(z_n,t_n),\\
k_2&=f(z_n+\tfrac h2k_1,t_n+\tfrac h2),\\
k_3&=f(z_n+\tfrac h2k_2,t_n+\tfrac h2),\\
k_4&=f(z_n+hk_3,t_n+h),\\
z_{n+1}&=z_n+\tfrac h6(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

Its local truncation error is $O(h^5)$ and accumulated global error is
$O(h^4)$ when the field is smooth. Gradients flow through all arithmetic and
all 30 solver steps because this is an ordinary differentiable PyTorch graph.
"""
            ),
            code(
                """
def rk4(field, z0, dt, steps):
    z = np.asarray(z0, dtype=float)
    values = [z.copy()]
    for i in range(steps):
        t = i * dt
        k1 = field(z, t)
        k2 = field(z + .5*dt*k1, t + .5*dt)
        k3 = field(z + .5*dt*k2, t + .5*dt)
        k4 = field(z + dt*k3, t + dt)
        z = z + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        values.append(z.copy())
    return np.asarray(values)

field = lambda z, t: np.array([z[1], -z[0]])
dt = .1
solution = rk4(field, [1, 0], dt, 100)
t = np.arange(101)*dt
exact = np.c_[np.cos(t), -np.sin(t)]
print("maximum state error:", np.abs(solution-exact).max())
fig, ax = plt.subplots(1,2,figsize=(11,4))
ax[0].plot(t, solution[:,0], label='RK4'); ax[0].plot(t, exact[:,0], '--', label='exact')
ax[0].legend(); ax[0].set_title('RK4 against known solution'); ax[0].set_xlabel('time')
ax[1].plot(solution[:,0], solution[:,1], label='RK4 phase path')
ax[1].axis('equal'); ax[1].set_title('State-space trajectory'); ax[1].legend()
plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Multiple shooting

One long nonlinear rollout can create poorly conditioned gradients. I train
three ten-step initial-value problems and penalize disagreement at their
boundaries. Boundary states are derived from training targets only. The model's
public `forward` method cannot accept them.

\[
\mathcal L=\underbrace{\mathcal L_{shoot}}_{\lambda_s=1}
+0.25\underbrace{\mathcal L_{continuity}}_{\text{boundary state}}
+0.5\underbrace{\mathcal L_{full}}_{\text{one inference-like rollout}}.
\]

For boundaries $b=[0,10,20,30]$,

\[
\mathcal L_{shoot}=\frac13\sum_{j=0}^{2}
\operatorname{ADE}(\Phi_\theta(\tilde z_j,10),Y_{b_j:b_{j+1}}).
\]

$\tilde z_0=z_0$; later $\tilde z_j$ reconstruct position and velocity (or
heading/speed/yaw rate) from the training target. Continuity compares the
predicted terminal state with the next reconstructed state. The implementation
batches the three equal-length IVPs into one larger GPU solve; this changes
kernel efficiency, not the mathematical objective.

The full term matters: without it, individually good short segments may still
drift when inference starts only at $z_0$.
"""
            ),
            code(
                """
fig, ax = plt.subplots(figsize=(11,3))
colors=["#2e86c1","#28b463","#ca6f1e"]
for j,(a,b) in enumerate(zip([0,10,20],[10,20,30])):
    ax.plot(np.arange(a,b+1)*.1, np.full(b-a+1,j), lw=8, color=colors[j], solid_capstyle='round')
    ax.scatter([a*.1],[j],s=110,facecolor='white',edgecolor=colors[j],zorder=3)
    ax.text((a+b)*.05,j+.18,f"IVP {j+1}: steps {a}–{b}",ha='center')
ax.plot(np.arange(31)*.1, np.full(31,3.25), color='black', lw=2, label='full rollout term')
for x in (1,2): ax.axvline(x,color='crimson',ls='--',alpha=.6)
ax.set_yticks([0,1,2,3.25],["shoot 1","shoot 2","shoot 3","full"])
ax.set_xlabel("future time (s)"); ax.set_title("Multiple shooting is training-only intermediate supervision")
ax.set_ylim(-.5,3.8); plt.tight_layout(); plt.show()
"""
            ),
            code(
                """
# Gradient growth in the scalar ODE z' = theta*z.
theta = torch.tensor(0.7, requires_grad=True)
z0 = torch.tensor(1.0)
times = torch.arange(1, 31) * .1
prediction = z0 * torch.exp(theta * times)
target = z0 * torch.exp(torch.tensor(.5) * times)
full_loss = ((prediction-target)**2).mean()
full_grad, = torch.autograd.grad(full_loss, theta, retain_graph=True)

short_losses = []
for start in (0, 10, 20):
    local_t = torch.arange(1, 11) * .1
    boundary = z0 * torch.exp(torch.tensor(.5) * (start*.1))
    local_prediction = boundary * torch.exp(theta * local_t)
    local_target = boundary * torch.exp(torch.tensor(.5) * local_t)
    short_losses.append(((local_prediction-local_target)**2).mean())
shooting_loss = torch.stack(short_losses).mean()
shooting_grad, = torch.autograd.grad(shooting_loss, theta)
print(f"full-horizon gradient: {full_grad.item():.3f}")
print(f"mean short-IVP gradient: {shooting_grad.item():.3f}")
"""
            ),
            md(
                r"""
## Hybrid vehicle physics

The hybrid uses $z=[x,y,\psi,v,r]$ and makes three derivatives exact:

\[
\dot x=v\cos\psi,\quad \dot y=v\sin\psi,\quad \dot\psi=r,
\quad \dot v=a,\quad \dot r=\alpha_r.
\]

The network learns bounded corrections for $a$ and $\alpha_r$. Current
longitudinal acceleration provides a causal prior that decays as $e^{-t}$.
This improves interpretability but can preserve biased yaw-rate measurements.
Physics narrows the hypothesis space; it does not guarantee better data fit.

| Model | ODE state | Exact derivatives | Learned derivatives |
|---|---|---|---|
| Generic | $[x,y,v_x,v_y]$ | $\dot x=v_x,\dot y=v_y$ | $\dot v_x,\dot v_y$ |
| Hybrid | $[x,y,\psi,v,r]$ | kinematic $x,y,\psi$ | $\dot v,\dot r$ |
"""
            ),
            code(
                """
from zod_driveformer.dynamics.models import HybridPhysicsNeuralODE
hybrid = HybridPhysicsNeuralODE(
    state_dim=9, history_steps=21, future_steps=30, context_dim=32,
    step_seconds=.1, normalizer_mean=[0]*9, normalizer_scale=[1]*9, hidden_dim=48,
)
state = torch.tensor([[0.,0.,0.,10.,.2]])
context = torch.zeros(1,32); controls = torch.tensor([[1.,.2,.05]])
derivative = hybrid.vector_field(state, context, 0.0, controls)
pd.DataFrame(derivative.detach().numpy(), columns=["dx/dt","dy/dt","dpsi/dt","dv/dt","dr/dt"]).round(3)
"""
            ),
            md(
                r"""
## What each loss sees

The shooting loss may read future target-derived boundary states during
*training only*. `model.forward(states, valid_mask)` accepts no target argument.
Validation and test call exactly that forward signature and obtain one rollout.
This is supervised learning, not leakage: labels guide optimization, but test
labels never enter model selection or prediction.
"""
            ),
            code(
                """
models = summary["dynamics"]["models"]
table = []
for name in ("b2_state_mlp", "hybrid_neural_ode", "neural_ode", "temporal_fno"):
    e = models[name]
    table.append([name, e["metrics_across_seeds"]["ade_m"]["mean"],
                  e["latency"]["batch_1_median_ms"], e["parameters"]])
pd.DataFrame(table, columns=["model","ADE_m","latency_ms","parameters"]).round(4)
"""
            ),
            code(
                """
from IPython.display import Image, display
display(Image(filename=str(ROOT/'reports/figures/dynamics_accuracy_latency.png')))
"""
            ),
            md(
                """
## From a scalar metric back to a path

ADE compresses thirty two-dimensional errors into one number. Projecting the
paths through the calibrated fisheye camera restores the scene geometry: I can
see where models agree early and how endpoint or turn error accumulates. This is
diagnostic context only; every forecasting network remains state-only.
"""
            ),
            md(
                """
![Camera-projected trajectories](../reports/figures/dynamics_camera_predictions.png)
"""
            ),
            md(
                """
## Interpretation

NeuralODE and hybrid ODE reliably beat B2, supporting the continuous-dynamics
hypothesis on this test role. The generic ODE is slightly better than the
physics hybrid. Temporal FNO has only 0.0011 m lower ADE than NeuralODE—far too
small to sell as a substantive accuracy victory—but is about nine times faster
at batch 1, so it is the engineering winner.

### Check my understanding

1. RK4 is the numerical solver; $f_\theta$ is the learned vector field.
2. Multiple shooting changes training supervision, not the inference API.
3. The hybrid is interpretable because three state derivatives are exact.
4. A physics prior can lose when its assumptions or measured controls are biased.
"""
            ),
        ],
    )


def fourier() -> Any:
    return notebook(
        "03 — Fourier operators and trajectory forecasting",
        [
            code(SETUP),
            md(
                r"""
## Learning objectives and input grid

I want to derive the spectral convolution, see how a function-to-function map
becomes tensors, audit causality, and interpret the accuracy–latency decision.

The operator grid concatenates 21 observed history locations and 30 future
query locations. At each of the 51 positions, the lifted feature vector contains

\[
[\text{values}_{9},\ \text{validity}_{9},\ \text{current}_{9},\
t/H,\ \sin(\pi t/H),\ \cos(\pi t/H),\ \mathbb 1_{observed}],
\]

for $3(9)+4=31$ channels. Future `values` and `validity` are zero; the repeated
current state and time/query flag are causal and known.
"""
            ),
            code(
                """
grid_t = np.r_[np.arange(-20,1)*.1, np.arange(1,31)*.1]
observed = grid_t <= 0
feature_map = np.vstack([
    observed.astype(float),
    (~observed).astype(float),
    grid_t / np.abs(grid_t).max(),
    np.sin(np.pi*grid_t/np.abs(grid_t).max()),
    np.cos(np.pi*grid_t/np.abs(grid_t).max()),
])
plt.figure(figsize=(11,3.2)); plt.imshow(feature_map, aspect='auto', cmap='coolwarm',
    extent=[grid_t.min(),grid_t.max(),4.5,-.5])
plt.yticks(range(5),['observed','future query','t/H','sin','cos'])
plt.axvline(0,color='black'); plt.xlabel('relative time (s)')
plt.title('Known query features over the combined operator grid'); plt.colorbar(); plt.show()
"""
            ),
            md(
                r"""
## The operator-learning view

A normal neural network maps finite vectors to finite vectors. An operator maps
one function to another. Here the input function is a masked state history on a
time grid; the output function is a future trajectory on query times.

An FNO block transforms along time, multiplies selected complex Fourier modes,
returns to the time domain, and adds a learned pointwise path:

\[
u_{l+1}=u_l+\sigma\left(Wu_l+\mathcal F^{-1}(R\mathcal F(u_l))\right).
\]

For $N$ grid points, the discrete Fourier transform is

\[
\hat u_k=\sum_{n=0}^{N-1}u_n e^{-2\pi i kn/N},\qquad
u_n=\frac1N\sum_{k=0}^{N-1}\hat u_k e^{2\pi i kn/N}.
\]

$R_k\in\mathbb C^{C_{in}\times C_{out}}$ mixes channels only for retained low
modes. The $1\times1$ convolution $W$ preserves a local path, while the spectral
path gives a global receptive field in a single block.
"""
            ),
            code(
                """
t = np.linspace(0, 2*np.pi, 256, endpoint=False)
signal = np.sin(2*t) + .45*np.sin(11*t) + .20*np.random.default_rng(4).normal(size=len(t))
spectrum = np.fft.rfft(signal, norm='ortho')
low = spectrum.copy(); low[8:] = 0
restored = np.fft.irfft(low, n=len(t), norm='ortho')
fig, ax = plt.subplots(1,2,figsize=(11,4))
ax[0].plot(t, signal, alpha=.55, label='input'); ax[0].plot(t, restored, lw=2, label='low modes')
ax[0].legend(); ax[0].set_title('Low modes preserve global structure')
ax[1].stem(np.abs(spectrum), basefmt=' '); ax[1].set_xlim(0,20); ax[1].set_title('Fourier amplitudes');
plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Architecture, shapes, and residual parameterization

```text
$B\times51\times31$ ─ lift 31→96 ─ transpose ─ 4 Fourier blocks
                    ─ remove 8 padded steps ─ project 96→96→2
                    ─ keep 30 future rows ─ add CTRV ─ $B\times30\times2$
```

With $C=96$ and $M=16$ modes, each complex spectral layer learns approximately
$2C^2M$ real numbers, plus its local $C^2$ mixing. Spectral capacity therefore
grows as $O(C^2M)$ rather than $O(N^2)$ pairwise time interactions.

## Causality despite global mixing

Fourier convolution is global over the supplied grid. Causality therefore comes
from **what is supplied**, not from a triangular attention mask. Future slots
contain only query time, an observed/future flag, missingness zeros, and the
repeated current state. No future vehicle state or target enters the operator.

The head predicts a residual around CTRV. Its last layer is initialized to zero,
so the untrained network exactly reproduces the physical reference.
"""
            ),
            code(
                """
from zod_driveformer.dynamics.models import TemporalFNOForecaster
model = TemporalFNOForecaster(
    state_dim=9, history_steps=21, future_steps=30, step_seconds=.1,
    normalizer_mean=[0]*9, normalizer_scale=[1]*9,
    width=32, modes=8, blocks=2, padding_steps=4,
)
states = torch.zeros(2,21,9); states[:,-1,0] = 10.0
with torch.no_grad(): trajectory = model(states)
print("trajectory shape:", tuple(trajectory.shape))
print("untrained final x at 10 m/s:", trajectory[0,-1,0].item(), "m")
print("parameters in teaching-size model:", sum(p.numel() for p in model.parameters()))
"""
            ),
            code(
                """
# Verify the zero-initialized residual starts exactly at the CTRV reference.
from zod_driveformer.models.baselines import constant_turn_rate_velocity
physical_reference = constant_turn_rate_velocity(
    torch.tensor([10.,10.]), torch.tensor([0.,0.]), future_steps=30, dt=.1
)
print("maximum initialization residual:", (trajectory-physical_reference).abs().max().item())
plt.figure(figsize=(7,3.5)); plt.plot(trajectory[0,:,0], trajectory[0,:,1], label='FNO at initialization')
plt.plot(physical_reference[0,:,0], physical_reference[0,:,1], '--', label='CTRV reference')
plt.axis('equal'); plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.legend();
plt.title('Residual learning begins from a causal physical baseline'); plt.show()
"""
            ),
            md(
                r"""
## Why padding exists

FFT assumes periodic boundaries: the last time point wraps to the first. A
driving history is not periodic. Zero-padding before spectral blocks increases
the distance between the physical boundaries, reducing wrap-around artifacts;
the padded tail is removed before prediction.

The padding is eight grid steps in the frozen model. It does not create future
information: all added entries are deterministic zeros.
"""
            ),
            code(
                """
step = np.r_[np.zeros(32), np.ones(19)]
spec = np.fft.rfft(step)
unfiltered = np.fft.irfft(spec, n=len(step))
filtered_spec = spec.copy(); filtered_spec[10:] = 0
filtered = np.fft.irfft(filtered_spec, n=len(step))
padded = np.pad(step, (0,8)); padded_spec=np.fft.rfft(padded); padded_spec[10:]=0
padded_filtered=np.fft.irfft(padded_spec,n=len(padded))[:len(step)]
plt.figure(figsize=(9,3.5)); plt.plot(step,label='non-periodic feature',lw=2)
plt.plot(filtered,label='low-mode reconstruction'); plt.plot(padded_filtered,label='with tail padding')
plt.axvline(31.5,color='black',ls=':'); plt.legend(); plt.title('Boundary behavior under spectral truncation')
plt.xlabel('operator-grid index'); plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Training loss and model selection

Temporal FNO is not an ODE and therefore does not use multiple shooting. It is
trained directly with full-horizon trajectory loss

\[
\mathcal L_{FNO}=\frac1{BT}\sum_{i=1}^{B}\sum_{t=1}^{T}
\lVert\hat y_{i,t}-y_{i,t}\rVert_2.
\]

The checkpoint with minimum validation ADE is selected independently for each
seed. Test metrics are computed only after selection is frozen.
"""
            ),
            code(
                """
entry = summary["dynamics"]["paired_difference_candidate_minus_b2"]["temporal_fno"]["delta_ade_m"]
print(f"FNO − B2 ADE: {entry['estimate']:+.4f} m")
print(f"95% grouped interval: [{entry['lower']:+.4f}, {entry['upper']:+.4f}] m")
"""
            ),
            code(
                """
models = summary['dynamics']['models']
comparison=[]
for name in ('b2_state_mlp','hybrid_neural_ode','neural_ode','temporal_fno'):
    e=models[name]
    comparison.append((name,e['metrics_across_seeds']['ade_m']['mean'],
                       e['latency']['batch_1_median_ms'],e['parameters']/1e6))
frame=pd.DataFrame(comparison,columns=['model','ADE_m','latency_ms','parameters_M'])
fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(frame.latency_ms,frame.ADE_m,s=80+100*frame.parameters_M)
for row in frame.itertuples(): ax.annotate(row.model,(row.latency_ms,row.ADE_m),xytext=(4,4),textcoords='offset points')
ax.set_xscale('log'); ax.set_xlabel('batch-1 GPU latency (ms, log scale)'); ax.set_ylabel('test ADE (m)')
ax.set_title('Accuracy–latency–size trade-off (marker area follows parameters)'); plt.tight_layout(); plt.show()
"""
            ),
            md(
                """
## Inspecting operator output in trajectory space

The red FNO curve below is not a frequency-domain diagnostic: it is the decoded
physical output in anchor-local metres, projected only for display. Comparing it
with ODE paths and ground truth checks that similar ADE is not hiding a different
failure mode. The learned curves average all three frozen seeds; the benchmark
itself averages per-seed metrics rather than scoring this visual ensemble.
"""
            ),
            md(
                """
![Camera-projected trajectory montage](../reports/figures/dynamics_camera_predictions.png)

![Animated held-out trajectory cases](../reports/figures/dynamics_camera_predictions.gif)
"""
            ),
            md(
                """
## A careful conclusion

The project does not establish that FNO is universally superior to ODE models.
FNO's ADE is 0.0011 m below NeuralODE, a negligible raw gap without a direct
superiority claim. The evidence establishes a stronger engineering fact: FNO
preserves ODE-level accuracy while reducing batch-1 latency from 24.45 ms to
2.69 ms on the same GPU.

### Check my understanding

- Global Fourier mixing is causal here because future slots contain queries,
  not future measurements.
- The zero-initialized residual head makes the initial model equal CTRV.
- Padding mitigates FFT wrap-around but does not change the physical grid.
- “Best” means the promoted accuracy–latency choice, not universal dominance.
"""
            ),
        ],
    )


def segmentation() -> Any:
    return notebook(
        "04 — Road and lane segmentation",
        [
            code(SETUP),
            md(
                r"""
## Learning objectives and sample contract

I want to understand label topology, preprocessing, weighted multilabel loss,
encoder–decoder shapes, thin-structure metrics, threshold calibration, and why
the largest model is not promoted.

| Tensor | Shape | Meaning |
|---|---:|---|
| image | $B\times3\times288\times512$ | ImageNet-normalized RGB |
| target | $B\times2\times288\times512$ | binary road and lane masks |
| logits | $B\times2\times288\times512$ | unconstrained model scores |
| probability | same | $p=\sigma(\ell)$ independently per channel |

Training augmentation applies the same horizontal flip to image and masks,
but color jitter only to RGB. Images use bilinear resize; discrete masks use
nearest-neighbor resize to avoid inventing fractional labels.
"""
            ),
            code(
                """
# Synthetic overlapping labels: lane pixels are also valid road pixels.
H,W=96,160
yy,xx=np.mgrid[:H,:W]
road=(yy > 42 + .002*(xx-80)**2) & (np.abs(xx-80) < (yy-30)*1.25)
lane=np.zeros((H,W),bool)
for offset in (-28,0,28):
    center=80 + offset*(yy/95)
    lane |= road & (np.abs(xx-center)<1.4) & ((yy//10)%2==0)
fig,ax=plt.subplots(1,3,figsize=(12,3.3))
ax[0].imshow(road,cmap='Greens'); ax[0].set_title('road target')
ax[1].imshow(lane,cmap='Oranges'); ax[1].set_title('lane target')
ax[2].imshow(road,cmap='Greens'); ax[2].imshow(lane,cmap='Oranges',alpha=.9); ax[2].set_title('overlap is valid')
for a in ax:a.axis('off')
plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Why this is multilabel segmentation

A lane marking lies on the ego road, so road and lane labels overlap. A two-class
softmax would incorrectly force mutual exclusion. The model instead emits two
independent logits and uses weighted binary cross entropy. Lane positives receive
a larger weight because they occupy very few pixels.

For channel $c$ and pixel $q$,

\[
\mathcal L_{BCE}=-\frac1N\sum_{q,c}
\left[w_c^+ y_{qc}\log\sigma(\ell_{qc})+
(1-y_{qc})\log(1-\sigma(\ell_{qc}))\right],
\]

with $(w^+_{road},w^+_{lane})=(1.5,12)$. The preregistered experiment uses no
Dice-loss term; validation in the source study favored weighted BCE alone.
"""
            ),
            code(
                """
logits=torch.linspace(-7,7,200)
positive_loss=-12*torch.log(torch.sigmoid(logits))
negative_loss=-torch.log(1-torch.sigmoid(logits))
plt.figure(figsize=(8,3.5)); plt.plot(logits,positive_loss,label='lane positive, weight 12')
plt.plot(logits,negative_loss,label='negative, weight 1'); plt.ylim(0,30)
plt.xlabel('logit'); plt.ylabel('per-pixel BCE contribution'); plt.legend()
plt.title('Class weighting amplifies rare positive mistakes'); plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Strict versus tolerant thin-line metrics

For a one-pixel structure, a one-pixel translation can make strict intersection
almost zero. Tolerant precision matches predictions against a dilated target;
tolerant recall matches targets against a dilated prediction. The radius is
three pixels at 512×288.

\[
P_{tol}=\frac{|P\cap D_r(Y)|}{|P|},\qquad
R_{tol}=\frac{|Y\cap D_r(P)|}{|Y|},\qquad
F1_{tol}=\frac{2P_{tol}R_{tol}}{P_{tol}+R_{tol}}.
\]

Strict $IoU=TP/(TP+FP+FN)$ remains visible so an excessively thick prediction
cannot hide behind the tolerance radius.
"""
            ),
            code(
                """
from torch.nn import functional as F
H,W = 48,96
target = torch.zeros(1,1,H,W); prediction = torch.zeros_like(target)
target[:,:,10:42,46] = 1
prediction[:,:,10:42,48] = 1  # correct shape, two-pixel offset
intersection = (target*prediction).sum()
strict_iou = intersection / ((target+prediction)>0).sum()
radius=3; kernel=2*radius+1
dilated_target=F.max_pool2d(target,kernel,1,radius)
dilated_prediction=F.max_pool2d(prediction,kernel,1,radius)
precision=(prediction*dilated_target).sum()/prediction.sum()
recall=(target*dilated_prediction).sum()/target.sum()
tolerant_f1=2*precision*recall/(precision+recall)
print(f"strict IoU={strict_iou:.3f}, tolerant F1={tolerant_f1:.3f}")
plt.figure(figsize=(8,3));
plt.imshow(target[0,0], cmap='Blues', alpha=.7); plt.imshow(prediction[0,0], cmap='Reds', alpha=.6)
plt.title('Blue target and red prediction: geometry is close, strict overlap is zero'); plt.axis('off'); plt.show()
"""
            ),
            md(
                r"""
## DeepLab reference and U-Net architecture

My earlier standalone study used MobileNetV3–DeepLabV3. Atrous convolutions
provide broad semantic context, but this compact decoder does not expose a full
symmetric skip hierarchy. I use it here as the reference for the U-Net study.

The replacement uses a pretrained ResNet-18 encoder:

| Stage | Channels | Spatial size for 288×512 |
|---|---:|---:|
| stem skip | 64 | $144\times256$ |
| layer1 skip | 64 | $72\times128$ |
| layer2 skip | 128 | $36\times64$ |
| layer3 skip | 256 | $18\times32$ |
| layer4 bottleneck | 512 | $9\times16$ |

The decoder bilinearly upsamples, concatenates the matching encoder skip, and
applies two $3\times3$ convolution–batchnorm–ReLU blocks at each level.

## Why U-Net helps

Deep encoders build semantics but reduce spatial resolution. U-Net concatenates
high-resolution encoder features into each decoder level, allowing the model to
recover exact boundaries and narrow lane markings. The result is especially
clear in strict lane IoU: 0.187 for DeepLab versus 0.466 for U-Net.
"""
            ),
            code(
                """
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(12,5)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
levels=[('stem',64,4.8),('L1',64,4.0),('L2',128,3.2),('L3',256,2.4),('L4',512,1.6)]
for i,(name,ch,y) in enumerate(levels):
    x=.7+i*1.25; ax.add_patch(FancyBboxPatch((x,y),1,.55,boxstyle='round,pad=.05',fc='#d6eaf8',ec='#2874a6'))
    ax.text(x+.5,y+.28,f'{name}\\n{ch}ch',ha='center',va='center',fontsize=8)
    if i: ax.add_patch(FancyArrowPatch((x-.25,y+.55),(x,y+.28),arrowstyle='->',mutation_scale=12))
decoder=[('up3',256,2.4),('up2',128,3.2),('up1',64,4.0),('up0',48,4.8)]
for i,(name,ch,y) in enumerate(decoder):
    x=7.0+i*1.15; ax.add_patch(FancyBboxPatch((x,y),1,.55,boxstyle='round,pad=.05',fc='#d5f5e3',ec='#239b56'))
    ax.text(x+.5,y+.28,f'{name}\\n{ch}ch',ha='center',va='center',fontsize=8)
    if i: ax.add_patch(FancyArrowPatch((x-1.15,y-.25),(x,y+.25),arrowstyle='->',mutation_scale=12))
for i in range(4):
    sx=.7+(3-i)*1.25+.5; dx=7.0+i*1.15+.5; y=2.4+i*.8
    ax.add_patch(FancyArrowPatch((sx,y+.55),(dx,y+.55),arrowstyle='->',ls='--',color='#8e44ad',mutation_scale=12))
ax.add_patch(FancyArrowPatch((5.7,1.9),(7,2.65),arrowstyle='->',mutation_scale=12))
ax.text(6.1,5.7,'encoder skips preserve spatial detail',ha='center',weight='bold')
ax.text(11.1,5.2,'→ 2 logits',fontsize=10); plt.show()
"""
            ),
            code(
                """
from zod_driveformer.segmentation.models import build_segmentation_model
models = {
    'U-Net': build_segmentation_model('resnet18_unet', pretrained=False),
    'Fourier U-Net': build_segmentation_model('resnet18_fourier_unet', pretrained=False,
        config={'spectral_modes_height':5,'spectral_modes_width':8,'spectral_blocks':2}),
}
for name, model in models.items():
    image=torch.zeros(1,3,96,160)
    with torch.no_grad(): output=model(image)
    print(name, f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters", "output", tuple(output.shape))
del models
"""
            ),
            md(
                r"""
## What the Fourier bottleneck tests

The Fourier variant keeps the same encoder, skips, decoder, data, and loss, but
adds global low-frequency mixing at the 9×16 bottleneck. This is a controlled
test of whether global spectral context adds value after U-Net already supplies
local multiscale evidence.

Each of two residual blocks computes

\[
u' = u+\operatorname{GELU}(\operatorname{GN}(
\mathcal F^{-1}(R\mathcal F(u))+W_{1\times1}u)).
\]

It retains 5 vertical and 8 horizontal modes with separate learned weights for
positive and negative vertical frequencies. Complex weights explain much of
the increase from 14.4M to 56.8M parameters. This candidate trained in FP32
because PyTorch gradient scaling cannot unscale `ComplexFloat` gradients; the
real-valued models used AMP.
"""
            ),
            md(
                r"""
## Training, selection, and threshold calibration

All three architectures use AdamW, cosine learning-rate decay, three seeds,
batch size 8, and at most 24 epochs. The pretrained encoder is frozen for the
first two epochs. The best epoch maximizes validation

\[
S=\tfrac12(IoU_{road}+F1^{tol}_{lane}).
\]

After checkpoint selection, road and lane thresholds are independently chosen
from $\{0.30,0.35,\ldots,0.75\}$ on validation only. The decomposable score
allows road threshold to maximize road IoU and lane threshold to maximize lane
tolerant F1. Test masks influence neither choice.
"""
            ),
            code(
                """
from IPython.display import Image, display
display(Image(filename=str(ROOT/'reports/figures/segmentation_test_metrics.png')))
"""
            ),
            md(
                """
## Reading the predicted masks

Cyan is the thresholded road channel and magenta is the thresholded lane
channel. Lane pixels can lie inside road pixels, so the overlay deliberately
shows overlap rather than forcing a mutually exclusive class map. Fixed score
quantiles plus disagreement cases avoid selecting only attractive examples.
All columns use the frozen seed-2026 checkpoints.
"""
            ),
            md(
                """
![Segmentation model montage](../reports/figures/segmentation_model_comparison.png)

![Animated held-out segmentation cases](../reports/figures/segmentation_model_comparison.gif)
"""
            ),
            code(
                """
pair = summary["fourier_unet_minus_unet_per_image"]["delta_selection_score"]
print(f"Fourier U-Net − U-Net score: {pair['estimate']:+.4f}")
print(f"95% paired interval: [{pair['lower']:+.4f}, {pair['upper']:+.4f}]")
"""
            ),
            code(
                """
seg=summary['segmentation']['models']
rows=[]
for name,e in seg.items():
    m=e['global_pixel_metrics_across_seeds']
    rows.append((name,m['road_iou']['mean'],m['lane_iou']['mean'],m['lane_tolerant_f1']['mean'],
                 m['selection_score']['mean'],e['parameters']/1e6,e['latency']['batch_1_median_ms']))
frame=pd.DataFrame(rows,columns=['model','road_IoU','lane_IoU','lane_tol_F1','score','params_M','latency_ms'])
fig,ax=plt.subplots(1,2,figsize=(12,4.3))
frame.set_index('model')[['road_IoU','lane_IoU','lane_tol_F1']].plot.bar(ax=ax[0],rot=15)
ax[0].set_ylim(0,1); ax[0].set_title('Accuracy components')
ax[1].scatter(frame.params_M,frame.score,s=120,c=frame.latency_ms,cmap='viridis')
for row in frame.itertuples(): ax[1].annotate(row.model,(row.params_M,row.score),xytext=(4,4),textcoords='offset points')
ax[1].set_xlabel('parameters (millions)'); ax[1].set_ylabel('selection score'); ax[1].set_title('Complexity versus score (color = latency)')
plt.tight_layout(); plt.show()
"""
            ),
            md(
                """
## Model choice

Fourier U-Net has the highest raw strict lane IoU and score, but its direct
score improvement over ordinary U-Net is only +0.0010 with interval
[−0.0072,+0.0094]. That is not evidence of a reliable win. Both U-Nets improve
the original DeepLab family reliably by about +0.136 score. Because Fourier
U-Net uses 56.8M rather than 14.4M parameters and is slower, ordinary U-Net is
the promoted model.

### Check my understanding

1. Use sigmoid, not channel softmax, because road and lane overlap.
2. Use nearest-neighbor mask resize because labels are categorical.
3. Keep strict IoU beside tolerant F1 so tolerance cannot reward thick masks.
4. Select thresholds on validation; touching test thresholds invalidates the test.
5. “Fourier has the highest decimal” is not “Fourier is demonstrably better.”
"""
            ),
        ],
    )


def bev_perception() -> Any:
    return notebook(
        "05 — LiDAR BEV detection and temporal tracking",
        [
            code(SETUP),
            md(
                """
## Learning objective and system boundary

This track turns one calibrated ZOD LiDAR sweep into a metric bird's-eye view
(BEV), predicts oriented objects, and links detections through time. It answers
**what occupies the space around the ego vehicle?** It does not choose a route,
plan around an obstacle, or control steering and braking.

```text
raw LiDAR → motion compensation → LiDAR-to-ego SE(3) → 3-channel BEV
          → SFA3D center detector → metric oriented boxes → Kalman tracks
```

The detector is the official MIT-licensed SFA3D FPN-ResNet-18 checkpoint trained
on KITTI, kept outside this repository. The ZOD adapter, rasterizer, evaluator,
tracker, and visualizer are implemented here. This is therefore a fixed
**cross-dataset transfer diagnostic**, not a claim of ZOD-trained performance.
"""
            ),
            md(
                r"""
## Coordinate frames and calibration

ZOD ego coordinates are right-handed: (x) forward, (y) left, (z) up.
Raw LiDAR points are first motion-compensated to the keyframe time and then
transformed with the calibrated homogeneous pose

\[
\tilde{\mathbf p}_{ego}=T^{ego}_{lidar}\tilde{\mathbf p}_{lidar},
\qquad \tilde{\mathbf p}=[x,y,z,1]^\top.
\]

The BEV covers (x\in[0,50]) m, (y\in[-25,25]) m, and
(z\in[-1,3]) m. Raster row increases forward; column increases left. The
renderer rotates the array so forward appears at the top while preserving the
metric convention in every model and evaluation calculation.
"""
            ),
            code(
                """
from zod_driveformer.bev import BEVConfig, build_bev_layers

rng=np.random.default_rng(7)
road=np.column_stack((rng.uniform(0,50,7000), rng.normal(0,4,7000), rng.normal(-.8,.04,7000)))
car=np.column_stack((rng.normal(18,1.8,900), rng.normal(-3,0.8,900), rng.uniform(-.6,1.2,900)))
points=np.vstack((road,car)).astype('float32')
intensity=np.r_[rng.uniform(20,90,len(road)),rng.uniform(90,220,len(car))]
demo_cfg=BEVConfig(height=240,width=240)
demo_layers=build_bev_layers(points,intensity,demo_cfg)
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
for axis,layer,title in zip(ax,[demo_layers.intensity,demo_layers.height,demo_layers.density],
                            ['robust intensity','top height','log density']):
    axis.imshow(np.rot90(layer,2),cmap='viridis'); axis.set_title(title); axis.axis('off')
plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Rasterization math

For cell resolution (H\times W), a point is quantized as

\[
r=\left\lfloor H\frac{x-x_{min}}{x_{max}-x_{min}}\right\rfloor,
\quad
c=\left\lfloor W\frac{y-y_{min}}{y_{max}-y_{min}}\right\rfloor.
\]

Within each cell the highest return supplies normalized height
(h=(z-z_{min})/(z_{max}-z_{min})) and robustly percentile-scaled intensity.
All returns contribute to density

\[
d=\min\left(1,\frac{\log(1+n)}{\log 64}\right).
\]

This compresses an unordered point set into three translation-aligned image
channels. It is fast and CNN-friendly, but vertical detail and multiple surfaces
inside one cell are deliberately discarded.
"""
            ),
            md(
                r"""
## Detector architecture: center-based oriented boxes

SFA3D uses an FPN-ResNet-18 backbone and several dense heads on a downsampled
BEV feature map:

| Head | Meaning | Decoded output |
|---|---|---|
| `hm_cen` | per-class center heatmap | class and confidence |
| `cen_offset` | sub-cell center correction | continuous (x,y) |
| `direction` | sine/cosine-like orientation pair | yaw (psi) |
| `dim` | object dimensions | length and width |
| `z_coor` | vertical center | retained by SFA3D, outside 2-D metric here |

The heatmap formulation avoids proposing thousands of anchors. Peak decoding
selects at most 50 centers above the frozen 0.20 confidence threshold. Pixel
centers and dimensions are then scaled back into metres in the ZOD ego frame.
"""
            ),
            code(
                """
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(12,4)); ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis('off')
nodes=[(.2,1.45,2.0,1.0,'3 × 608 × 608\\nBEV'),(3.0,1.45,2.2,1.0,'FPN ResNet-18\\nmultiscale features'),
       (6.0,2.6,2.2,.75,'center heatmap'),(6.0,1.55,2.2,.75,'offset + yaw'),
       (6.0,.5,2.2,.75,'dimensions + z'),(9.2,1.45,2.5,1.0,'top-K decoder\\noriented boxes')]
for x,y,w,h,label in nodes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc='#e8f4fb',ec='#1677a7'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center')
ax.add_patch(FancyArrowPatch((2.2,1.95),(3,1.95),arrowstyle='->',mutation_scale=15))
for y in (2.98,1.93,.88):
    ax.add_patch(FancyArrowPatch((5.2,1.95),(6,y),arrowstyle='->',mutation_scale=13))
    ax.add_patch(FancyArrowPatch((8.2,y),(9.2,1.95),arrowstyle='->',mutation_scale=13))
ax.set_title('External frozen detector; local ZOD coordinate adapter'); plt.show()
"""
            ),
            md(
                r"""
## What training losses produced the checkpoint?

Although this project does not retrain SFA3D, understanding its outputs matters.
A CenterNet-style objective uses focal loss on the center heatmap and masked
regression losses at true object centers for offset, direction, height, and
dimensions. Schematically,

\[
\mathcal L=\mathcal L_{focal}(\hat H,H)
+\lambda_o\|\hat o-o\|_1
+\lambda_d\|\hat d-d\|_1
+\lambda_\psi\|\hat q-q\|_1
+\lambda_z\|\hat z-z\|_1.
\]

Only peaks at inference become `BEVDetection(class, x, y, length, width, yaw,
confidence)`. The adapter never retrains or silently recalibrates the checkpoint
on the 12 ZOD mini frames.
"""
            ),
            md(
                r"""
## Oriented intersection-over-union

Axis-aligned IoU would reward a box with the wrong heading. I construct each
footprint by rotating local corners with

\[
R(\psi)=\begin{bmatrix}\cos\psi&-\sin\psi\\\sin\psi&\cos\psi\end{bmatrix},
\]

clip the two convex polygons, and report

\[
IoU_{BEV}=\frac{|P\cap G|}{|P|+|G|-|P\cap G|}.
\]

Predictions and labels are greedily matched one-to-one, class consistently, in
descending IoU order at (IoU\ge0.5). Precision measures false alarms; recall
measures missed annotated objects; matched center error measures localization.
"""
            ),
            code(
                """
from zod_driveformer.bev import BEVDetection, oriented_bev_iou
from zod_driveformer.bev.evaluation import _corners
truth=BEVDetection('Vehicle',15,0,4.6,1.9,.15)
candidates=[BEVDetection('Vehicle',15+e,0,4.6,1.9,.15+a) for e,a in [(0,0),(1,0),(0,.5)]]
fig,axes=plt.subplots(1,3,figsize=(12,3.4))
for ax,pred in zip(axes,candidates):
    for box,color,label in [(truth,'#2ecc71','truth'),(pred,'#00bfff','prediction')]:
        corners=_corners(box); closed=np.vstack((corners,corners[0])); ax.plot(closed[:,1],closed[:,0],color=color,lw=2,label=label)
    ax.set_aspect('equal'); ax.set_xlim(-4,4); ax.set_ylim(11,19); ax.set_title(f'IoU = {oriented_bev_iou(pred,truth):.3f}')
axes[0].legend(); plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Tracking: state, prediction, update

Each class-aware track uses state (mathbf s=[x,y,v_x,v_y]^\top). For elapsed
time (Delta t), constant-velocity prediction is

\[
\mathbf s^-_k=F\mathbf s_{k-1},\quad
P^-_k=FP_{k-1}F^\top+Q,
\quad
F=\begin{bmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix}.
\]

For measured center (mathbf z=[x,y]^\top), the Kalman update is

\[
K=P^-H^\top(HP^-H^\top+R)^{-1},\quad
\mathbf s=\mathbf s^-+K(\mathbf z-H\mathbf s^-).
\]

Class-consistent nearest-neighbor association is gated at 4 m. A track needs two
hits before display and is deleted after two misses in the animation. This is a
transparent teaching tracker, not a claim of state-of-the-art MOT accuracy.
"""
            ),
            code(
                """
from zod_driveformer.bev import MultiObjectTracker
tracker=MultiObjectTracker(minimum_hits=1,maximum_misses=2)
track_xy=[]
for t in np.arange(0,4,.2):
    noisy=BEVDetection('Vehicle',10+3*t+rng.normal(0,.25),2+rng.normal(0,.25),4.5,1.9,0,0.8)
    estimate=tracker.step([noisy],dt=.2)[0]; track_xy.append((estimate.x_m,estimate.y_m,estimate.velocity_x_mps))
track_xy=np.asarray(track_xy)
fig,ax=plt.subplots(figsize=(8,3.5)); ax.plot(10+3*np.arange(0,4,.2),np.full(20,2),label='latent path',lw=3)
ax.plot(track_xy[:,0],track_xy[:,1],'o-',label='Kalman estimate'); ax.set_aspect('equal'); ax.set_xlabel('forward x (m)'); ax.set_ylabel('left y (m)'); ax.legend(); plt.show()
"""
            ),
            md(
                """
## Fixed evaluation contract

- Data: all 12 annotated frames in ZOD Frames mini.
- Targets: non-unclear Vehicle, Pedestrian, and VulnerableVehicle 3-D boxes,
  converted into ego-frame Vehicle, Pedestrian, and Cyclist footprints.
- Scope: target centers inside the 50 m × 50 m front raster.
- Frozen model: SFA3D FPN-ResNet-18 at a pinned source commit and checkpoint hash.
- No ZOD fine-tuning, confidence sweep, per-frame persistence, or attractive-case
  selection affects the reported metrics.

This small subset can expose domain-transfer failure; it cannot establish broad
detector quality. The qualitative montage uses the first three sorted mini
frames, a deterministic rule fixed independently of outcomes.
"""
            ),
            code(
                """
bev_report=json.loads((ROOT/'reports/bev_detection_mini.json').read_text())
evaluation=bev_report['evaluation']; rows=[]
for name,metrics in [('All',evaluation['all']),*evaluation['by_class'].items()]:
    rows.append([name,metrics['precision'],metrics['recall'],metrics['f1'],metrics['mean_matched_iou'],metrics['mean_center_error_m']])
bev_table=pd.DataFrame(rows,columns=['class','precision','recall','F1','matched IoU','center error (m)'])
bev_table.round(3)
"""
            ),
            code(
                """
plot=bev_table.set_index('class')[['precision','recall','F1']]
ax=plot.plot.bar(figsize=(9,4),color=['#38bdf8','#f59e0b','#8b5cf6'])
ax.set_ylim(0,1); ax.set_ylabel('score'); ax.set_title('Fixed KITTI → ZOD mini transfer diagnostic')
ax.legend(loc='upper right'); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
"""
            ),
            md(
                """
## Reading the result honestly

Vehicle localization transfers surprisingly well when a match exists: vehicle
precision is 0.864, recall 0.528, matched IoU 0.718, and matched-center error
0.293 m. Overall F1 is 0.503 across 104 targets.

The failure is just as important: pedestrian and cyclist true positives are
zero. KITTI-to-ZOD differences in class appearance, density, sensor setup, and
decoding are too large for this fixed checkpoint. A Tesla-like picture is not
evidence of a safe perception stack. The appropriate next experiment is a
proper ZOD train/validation/test detector benchmark (and possibly camera–LiDAR
fusion), not threshold tuning on these 12 evaluation frames.
"""
            ),
            code(
                """
from IPython.display import Image as NotebookImage, display
display(NotebookImage(filename=str(ROOT/'reports/figures/bev_detection_mini.png')))
"""
            ),
            md(
                """
The green rectangles above are ZOD ground-truth footprints; cyan, orange, and
purple are model predictions. The front-camera row is context only—the LiDAR
detector never receives RGB. The point-cloud background makes occlusion and
sparse returns visible rather than presenting boxes on an unexplained blank map.
"""
            ),
            code(
                """
display(NotebookImage(filename=str(ROOT/'reports/figures/bev_tracking.png')))
"""
            ),
            md(
                """
## Reproduction and review checklist

1. Keep ZOD, the pinned SFA3D source, and its checkpoint outside Git.
2. Rebuild `reports/bev_detection_mini.json` with
   `scripts/evaluate_bev_detection.py`; all paths are explicit CLI arguments.
3. Rebuild the montage/GIF with `scripts/build_bev_visuals.py`.
4. Verify the checkpoint SHA-256 and external source commit in the report.
5. Inspect oriented-IoU, coordinate-transform, raster, and tracker tests.
6. Do not call the animated sequence a tracking benchmark: it is a qualitative
   demonstration because that sequence has no frame-by-frame 3-D box labels.

The complete animation is embedded in the README. Raw point clouds, images,
annotations, checkpoint weights, identifiers, and per-frame predictions remain
outside the repository.
"""
            ),
        ],
    )


def capstone() -> Any:
    return notebook(
        "06 — Failure analysis and interview capstone",
        [
            code(SETUP),
            md(
                """
## How to use this capstone

This notebook is a compact oral-exam pass over the project. Before reading each
answer, I should explain the input, output, inductive bias, loss, selection rule,
test unit, result, limitation, and next experiment in my own words.

## The project in one diagram

```text
ZOD sequences ── causal state windows ──┬─ CV / CTRV
                                       ├─ frozen B2 MLP
                                       ├─ NeuralODE + multiple shooting
                                       ├─ hybrid physics NeuralODE
                                       └─ temporal FNO ──► 30-point trajectory

ZOD keyframes ── road/lane masks ──────┬─ DeepLabV3-MobileNet
                                       ├─ ResNet-18 U-Net
                                       └─ Fourier U-Net ──► overlapping masks

ZOD LiDAR ── calibrated BEV ────────────┬─ SFA3D oriented boxes
                                       └─ Kalman association ──► object tracks
```
"""
            ),
            md(
                r"""
## Experimental lifecycle

1. **Define:** causal tensors, coordinate frame, model families, seeds, metrics.
2. **Fit:** learn parameters on train; target-derived shooting states stay here.
3. **Select:** choose epochs and segmentation thresholds on validation.
4. **Seal:** evaluate each selected seed once on test.
5. **Infer:** compute paired differences and resample recording groups.
6. **Decide:** promote a model using accuracy, uncertainty, latency, and size.

For paired sample loss $d_i=m_A(i)-m_B(i)$, grouping by recording $g$ prevents
overlapping windows from masquerading as independent evidence. A bootstrap draw
samples recordings with replacement and includes all their windows:

\[
\hat\Delta^{*(b)}=\frac{1}{N_b}\sum_{g\in G^{*(b)}}\sum_{i:g(i)=g}d_i.
\]

The 2.5th and 97.5th percentiles of 2,000 draws form the reported 95% interval.
For ADE differences, negative favors the candidate; for segmentation score,
positive favors the candidate.
"""
            ),
            md(
                """
## Evidence matrix

| Question | Evidence | Defensible answer |
|---|---|---|
| Does continuous modeling help B2? | Three candidates; paired recording CIs | Yes |
| Does exact kinematic bias win? | Hybrid vs generic ODE | No |
| Does FNO preserve accuracy at lower latency? | ADE and GPU latency | Yes |
| Do U-Net skips improve thin lanes? | DeepLab vs both U-Nets | Yes |
| Does Fourier U-Net beat U-Net? | Direct paired CI and efficiency | Not established |
| Does a KITTI LiDAR detector transfer to ZOD? | Fixed 12-frame mini diagnostic | Vehicles partly; vulnerable users do not |
"""
            ),
            code(
                """
dyn_pairs = summary['dynamics']['paired_difference_candidate_minus_b2']
rows=[]
for model, metrics in dyn_pairs.items():
    ci=metrics['delta_ade_m']; rows.append([model,ci['estimate'],ci['lower'],ci['upper']])
pd.DataFrame(rows,columns=['candidate','delta_ADE','CI_low','CI_high']).round(4)
"""
            ),
            code(
                """
forest=[]
for model,metrics in dyn_pairs.items():
    ci=metrics['delta_ade_m']; forest.append((f'{model} − B2',ci['estimate'],ci['lower'],ci['upper'],'ADE (m)'))
against_deeplab = summary['segmentation']['paired_difference_candidate_minus_deeplab']
for key,label in [('resnet18_unet','U-Net − DeepLab'),
                  ('resnet18_fourier_unet','Fourier U-Net − DeepLab')]:
    ci=against_deeplab[key]['delta_selection_score']
    forest.append((label,ci['estimate'],ci['lower'],ci['upper'],'score'))
ci=summary['fourier_unet_minus_unet_per_image']['delta_selection_score']
forest.append(('Fourier U-Net − U-Net',ci['estimate'],ci['lower'],ci['upper'],'score'))
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,unit,better in zip(axes,['ADE (m)','score'],['negative favors candidate','positive favors candidate']):
    subset=[x for x in forest if x[4]==unit]
    y=np.arange(len(subset)); est=np.array([x[1] for x in subset]); lo=np.array([x[2] for x in subset]); hi=np.array([x[3] for x in subset])
    ax.errorbar(est,y,xerr=[est-lo,hi-est],fmt='o',capsize=4)
    ax.axvline(0,color='black',lw=1); ax.set_yticks(y,[x[0] for x in subset]); ax.invert_yaxis()
    ax.set_xlabel(f'difference; {better}'); ax.set_title(f'Paired 95% intervals: {unit}')
plt.tight_layout(); plt.show()
"""
            ),
            md(
                """
## Failure modes I would inspect next

### Dynamics

- turns with biased or delayed yaw rate;
- braking transitions, where constant-command priors fail;
- long straight highway windows that dominate aggregate metrics;
- invalid or stale control channels;
- country and collection-car shifts.

### Segmentation

- worn or occluded lane markings;
- road boundaries under snow, glare, or construction;
- predictions that exploit a thick tolerance band;
- country-specific markings absent from the small test role;
- threshold drift under camera exposure changes.

For each slice I would report support, the same frozen metric, a paired model
difference where possible, and representative *private* examples outside Git.
Slicing after seeing test results is exploratory diagnosis, not a new confirmatory
claim; a subsequent test role would be needed to confirm a discovered subgroup.
"""
            ),
            code(
                """
# Synthetic example of why aggregate metrics need slice support.
slices=pd.DataFrame({
    'slice':['straight','turn','braking','stale controls'],
    'windows':[1800,420,250,79],
    'ADE_m':[.46,.71,.82,1.08],
})
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
ax[0].bar(slices['slice'],slices['ADE_m'],color='#5dade2'); ax[0].tick_params(axis='x',rotation=20); ax[0].set_ylabel('ADE (m)')
ax[1].bar(slices['slice'],slices['windows'],color='#f5b041'); ax[1].tick_params(axis='x',rotation=20); ax[1].set_ylabel('support')
fig.suptitle('Illustration only: a high-error slice may have little support'); plt.tight_layout(); plt.show()
"""
            ),
            md(
                r"""
## Architecture and loss flash cards

| Model | Inductive bias | Objective | Main limitation |
|---|---|---|---|
| B2 MLP | finite vector mapping | trajectory ADE | no explicit temporal operator/physics |
| NeuralODE | continuous second-order field | shooting + continuity + full ADE | sequential solver latency |
| Hybrid ODE | exact kinematic core | same multiple-shooting loss | biased physics/control assumptions |
| Temporal FNO | global low-mode temporal mixing + CTRV residual | full ADE | fixed-grid spectral design |
| DeepLab | atrous semantic context | weighted multilabel BCE | weaker thin-detail recovery here |
| U-Net | multiscale encoder skips | weighted multilabel BCE | larger than DeepLab reference |
| Fourier U-Net | U-Net + global spectral bottleneck | same BCE | 4× parameters, no reliable score gain |
| SFA3D | BEV center heatmaps + box heads | external focal + regression losses | KITTI→ZOD class/domain shift |
| Kalman tracker | constant-velocity Gaussian state | recursive filtering, no learned loss | simple nearest-neighbor association |

Promotion is not determined by training loss. The chosen checkpoint is judged by
validation, then its final claim uses sealed test metrics and grouped uncertainty.
"""
            ),
            md(
                """
## Interview drill

Pause before expanding each answer.

1. **Why is target-derived multiple shooting not test leakage?**  
   It is training-only intermediate supervision. `forward()` never accepts a
   boundary target, and evaluation performs one causal rollout.

2. **Why bootstrap recordings?**  
   Overlapping windows from one recording are correlated. Window bootstrap
   would pretend the effective sample size is much larger than it is.

3. **Why report both strict lane IoU and tolerant F1?**  
   Tolerance reflects geometric usefulness for thin lines; strict IoU prevents
   a thick imprecise mask from looking perfect.

4. **Why not claim Fourier U-Net wins?**  
   Its paired interval crosses zero and its compute cost is much higher.

5. **What is the cleanest next experiment?**  
   Train a ZOD-native BEV detector with a protected split, then test whether
   camera–LiDAR fusion repairs the vulnerable-road-user transfer gap.

6. **Why is temporal FNO called best if its NeuralODE gap is tiny?**  
   It is the best measured and promoted engineering point: essentially tied
   accuracy with much lower latency. I do not claim statistically proven FNO
   accuracy superiority over NeuralODE.

7. **Did Fourier U-Net improve the original project?**  
   The U-Net family clearly improved the retrained DeepLab reference on a fresh
   test role. Fourier U-Net has the highest raw score, but its direct advantage
   over ordinary U-Net is uncertain and computationally expensive.

8. **Is the BEV animation evidence of a good pedestrian detector?**
   No. The fixed transfer diagnostic has zero pedestrian and cyclist true
   positives. It demonstrates calibrated geometry, decoding, and temporal
   state estimation—not a safety or multi-object-tracking claim.
"""
            ),
            code(
                """
# A compact promotion scorecard generated from frozen evidence.
scorecard = pd.DataFrame([
    ["Temporal FNO", True, True, "promote"],
    ["NeuralODE", True, False, "retain as continuous-time study"],
    ["Hybrid NeuralODE", True, False, "retain as physics study"],
    ["ResNet-18 U-Net", True, True, "promote"],
    ["Fourier U-Net", False, False, "retain as complexity control"],
    ["SFA3D KITTI→ZOD", False, True, "retain as transfer baseline"],
], columns=["model","reliable_gain","efficient_frontier","decision"])
scorecard
"""
            ),
            md(
                """
## Visual recall: architecture to output

These panels are the fastest way to rehearse the project aloud: name the input
tensor, explain the model's inductive bias, point to its physical output, then
connect the visible behavior back to the sealed quantitative result.
"""
            ),
            md(
                """
![Model architecture overview](../reports/figures/model_architecture_overview.png)

![Camera-projected trajectory outputs](../reports/figures/dynamics_camera_predictions.png)

![Segmentation outputs](../reports/figures/segmentation_model_comparison.png)

![LiDAR BEV detection](../reports/figures/bev_detection_mini.png)
"""
            ),
            code(
                """
# Data-safe repository completeness check used during a review.
required=[
    'README.md','docs/methods.md','docs/data_and_evaluation.md','docs/interview_guide.md',
    'reports/v4_dynamics_test.json','reports/v4_segmentation_test.json',
    'reports/bev_detection_mini.json',
    'src/zod_driveformer/dynamics/models.py','src/zod_driveformer/segmentation/models.py',
    'src/zod_driveformer/bev/representation.py','src/zod_driveformer/bev/tracking.py',
]
pd.DataFrame([(name,(ROOT/name).exists()) for name in required],columns=['artifact','present'])
"""
            ),
            md(
                """
## Final perspective

The most valuable result is not a particular architecture. It is a disciplined
way to turn exploratory modeling into claims that survive scrutiny: protect the
test role, preserve pairing, make leakage boundaries executable, report costs,
and allow a sophisticated model to lose when its gain is not reliable.

The concise project story is: **continuous and spectral dynamics reliably beat
the strong state MLP; temporal FNO is the best accuracy–latency choice; U-Net
skips reliably repair thin-lane segmentation; extra Fourier capacity in the
U-Net bottleneck does not justify its cost; calibrated LiDAR BEV provides a
working vehicle-transfer baseline while exposing a serious vulnerable-road-user
domain gap.**
"""
            ),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks"))
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    notebooks = {
        "00_project_map.ipynb": project_map(),
        "01_geometry_splits_and_baselines.ipynb": geometry(),
        "02_neural_ode_and_multiple_shooting.ipynb": neural_ode(),
        "03_fourier_operators.ipynb": fourier(),
        "04_road_lane_segmentation.ipynb": segmentation(),
        "05_lidar_bev_detection_and_tracking.ipynb": bev_perception(),
        "06_interview_capstone.ipynb": capstone(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    legacy_capstone = args.output / "05_interview_capstone.ipynb"
    if legacy_capstone.exists():
        legacy_capstone.unlink()
    root = Path.cwd()
    for filename, value in notebooks.items():
        if not args.no_execute:
            NotebookClient(
                value,
                timeout=300,
                kernel_name="python3",
                resources={"metadata": {"path": str(root)}},
            ).execute()
        nbf.write(value, args.output / filename)
        print(f"wrote {filename} ({len(value.cells)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
