# Multi-task and federated camera-perception study

## Question

I use this experiment to study two related engineering questions:

1. How much computation can three camera tasks share before negative transfer
   outweighs the efficiency gain?
2. After centralized training, can uneven and non-IID vehicle cohorts improve a
   global model through federated fine-tuning without moving raw samples?

The input is one blurred ZOD front-camera keyframe. The outputs are scene
semantics and native road/lane affordances, monocular metric depth, and native
2-D object centers/boxes.

## Research basis

The experiment is intentionally small, but its controls follow established
work:

- [Kendall, Gal, and Cipolla (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)
  learn homoscedastic uncertainty weights for semantic, instance, and depth
  objectives with different scales.
- [PAD-Net (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html)
  studies shared scene parsing, depth, surface normals, and contours.
- [MTAN (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html)
  separates shared features from task-specific attention.
- [PCGrad (NeurIPS 2020)](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
  identifies conflicting task gradients and projects one away from another.
- [QuadroNet (WACV 2021)](https://openaccess.thecvf.com/content/WACV2021/html/Goel_QuadroNet_Multi-Task_Learning_for_Real-Time_Semantic_Depth_Aware_Instance_Segmentation_WACV_2021_paper.html)
  is a directly relevant autonomous-driving example with detection, semantic
  and instance segmentation, and monocular depth in one network.
- [FedAvg (AISTATS 2017)](https://proceedings.mlr.press/v54/mcmahan17a.html)
  aggregates client models in proportion to their local sample counts.
- [FedProx (MLSys 2020)](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
  adds a proximal penalty to reduce drift under statistical and systems
  heterogeneity.
- [SCAFFOLD (ICML 2020)](https://proceedings.mlr.press/v119/karimireddy20a.html)
  explains client drift under non-IID local updates and corrects it with control
  variates. I study the drift diagnosis here but do not implement SCAFFOLD.
- [FedBN (ICLR 2021)](https://openreview.net/pdf?id=6YEQUn0QICG)
  keeps normalization local under feature shift, including the autonomous-
  driving example of highway versus city scenery. It motivates the feature-
  shift analysis, although this study keeps one deployable global model.

The semantic pseudo-label teacher is the published
[NVIDIA SegFormer-B0 Cityscapes checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024).
Teacher agreement is reported as distillation fidelity, never as ZOD semantic
ground-truth accuracy.

## Label provenance

| Output | Supervision | Interpretation |
|---|---|---|
| 19-class scene semantics | frozen Cityscapes SegFormer pseudo-label | teacher agreement only |
| ego-road and lane masks | native ZOD polygons | ZOD affordance accuracy |
| metric depth | calibrated ZOD LiDAR projected into the camera | sparse-pixel depth accuracy |
| vehicle, pedestrian, cyclist centers | native ZOD 2-D annotations | ZOD detection accuracy |

The 489 recording-level samples are split into 220 centralized training
examples, four collection-car clients with 65/40/25/15 examples, 73 validation
examples, and 51 test examples. Only 95 training and 26 validation/test examples
have locally downloaded LiDAR, so depth loss is masked where supervision is
absent. This is label-availability heterogeneity in addition to client quantity
and geographic/vehicle feature shift.

## Architectures

The single-task controls train three complete 14.48M-parameter networks. Hard
sharing uses one encoder and one decoder with small task heads. The selected
split-decoder model shares the ResNet-18 encoder but has independent semantic,
depth, and detection decoders. It has 19.80M parameters, approximately 54% fewer
than deploying the three single-task controls together.

Hard sharing collapses the lane objective and produces negative encoder-gradient
cosines for segmentation versus depth. Split decoders recover thin spatial
detail while retaining a shared visual representation. Equal loss weighting
slightly beats learned uncertainty weighting on validation, so equal weighting
is selected even though the more elaborate method is retained as a control.

## Federated objective

The implementation separates three data/communication arrangements:

- **pooled retraining** centralizes every client sample and is therefore a
  non-federated oracle;
- **FedSGD** keeps samples local and sends the mean parameter gradient computed
  at the current global checkpoint;
- **FedAvg/FedProx** perform local optimizer steps and send the resulting model
  delta rather than raw samples or complete client datasets.

FedSGD computes

\[
g_k=\nabla F_k(w^t), \qquad
w^{t+1}=w^t-\eta\sum_{k=1}^{K}\frac{n_k}{n}g_k.
\]

For FedAvg I explicitly encode the client message as
\(\Delta_k=w_k^{t+1}-w^t\) and reconstruct

\[
w^{t+1}=w^t+\sum_{k=1}^{K}\frac{n_k}{n}\Delta_k.
\]

This is algebraically identical to averaging complete client weights when all
clients start from the same \(w^t\). A gradient or dense delta still has one
value per trainable parameter, so it avoids raw-data transfer but does not by
itself reduce message size or guarantee privacy.

For client \(k\), local sample count \(n_k\), and \(n=\sum_k n_k\), sample-
weighted FedAvg uses

\[
w^{t+1}=\sum_{k=1}^{K}\frac{n_k}{n}w_k^{t+1}.
\]

The four weights are therefore 0.448, 0.276, 0.172, and 0.103. Uniform averaging
assigns 0.25 to every client and intentionally overweights the 15-sample client
relative to the 65-sample client. FedProx changes each local objective to

\[
\min_w F_k(w)+\frac{\mu}{2}\|w-w^t\|_2^2,
\]

with \(\mu=10^{-3}\). I also retain pooled fine-tuning as an oracle that moves
all client data into one loader and therefore violates the federated data
boundary.

## Result and decision

The split-decoder centralized model preserves semantic/affordance quality,
improves sparse depth over the depth-only control, and loses some 2-D detection
AP. It is the best validation-selected compromise, not the winner on every
task.

Neither the original nor low-drift local-optimizer schedule exceeds the
centralized round-zero validation score. Pooled, uniform, sample-weighted
FedAvg, FedProx, and head-only FedAvg therefore retain round zero.

FedSGD behaves differently. A validation-only sweep over server learning rates
selects \(\eta=0.003\) and round 7, raising the composite validation score from
0.3715 to 0.3765. The once-read test score increases from 0.3392 to 0.3445,
primarily because sparse-depth \(\delta_1\) rises from 0.3586 to 0.3824.
Detection mAP50 falls from 0.0252 to 0.0232, while teacher mIoU and lane F1 are
approximately stable. I promote the FedSGD checkpoint under the registered
composite selection rule, but record the per-task trade rather than claiming
that every output improves.

## Privacy boundary

This is a logical single-machine simulation. Each federated client reads only
its own directory, and aggregation receives dense model deltas or mean
gradients plus sample counts rather than raw images. The pooled control is the
deliberate exception because it centralizes samples. This is not a complete
privacy system:

- updates are visible to the simulated server;
- no secure aggregation is implemented;
- no client-level clipping or differential privacy is implemented;
- membership or gradient leakage is not evaluated;
- all clients execute sequentially on one workstation.

The correct claim is therefore *federated optimization mechanics with a raw-
sample boundary*, not privacy-preserving deployment.
