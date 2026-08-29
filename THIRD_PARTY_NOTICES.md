# Third-party notices

## Zenseact Open Dataset

The dataset-derived qualitative figures are governed by the ZOD terms and the
attribution in `reports/figures/ZOD_ASSET_NOTICE.md`. Raw ZOD data is not
redistributed.

## SFA3D

The LiDAR BEV transfer and fine-tuning experiments interoperate with the external
[SFA3D repository](https://github.com/maudzung/SFA3D) by Nguyen Mau Dung,
licensed under the MIT License. The public report pins source commit
`0e2f0b63dc4090bd6c08e15505f11d764390087c` and the checkpoint SHA-256.

SFA3D source code and checkpoint weights are not copied into this repository.
Users obtain them from the upstream project and retain its copyright and license
notices.

## SegFormer Cityscapes teacher

The camera multi-task experiment creates scene-semantic pseudo-labels with the
published
[NVIDIA SegFormer-B0 Cityscapes checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024).
Teacher weights are downloaded to the user's external model cache and are not
redistributed by this repository. Public metrics label this output as teacher
agreement rather than native ZOD semantic ground-truth accuracy.
