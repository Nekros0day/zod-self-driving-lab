"""Shared-encoder camera network for semantics, depth, and 2-D detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from zod_driveformer.segmentation.models import DoubleConv, UpBlock


@dataclass(frozen=True)
class TaskOutputs:
    semantic_logits: torch.Tensor
    affordance_logits: torch.Tensor
    log_depth: torch.Tensor
    center_logits: torch.Tensor
    center_offset: torch.Tensor
    box_size: torch.Tensor


class MultiTaskPerceptionNet(nn.Module):
    """Hard sharing with one decoder and lightweight task-specific heads.

    The dense decoder ends at half resolution and is resized for semantic and
    depth outputs. Detection uses the quarter-resolution decoder feature so its
    center heatmap retains a fixed output stride of four.
    """

    def __init__(
        self,
        *,
        semantic_classes: int = 19,
        affordance_channels: int = 2,
        detection_classes: int = 3,
        pretrained: bool = True,
        enabled_tasks: tuple[str, ...] = ("segmentation", "depth", "detection"),
    ) -> None:
        super().__init__()
        unknown = set(enabled_tasks) - {"segmentation", "depth", "detection"}
        if unknown or not enabled_tasks:
            raise ValueError(f"unsupported enabled tasks: {sorted(unknown)}")
        self.enabled_tasks = frozenset(enabled_tasks)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 64, 48)
        self.semantic_head = nn.Conv2d(48, semantic_classes, 1)
        self.affordance_head = nn.Conv2d(48, affordance_channels, 1)
        self.depth_head = nn.Sequential(DoubleConv(48, 32), nn.Conv2d(32, 1, 1))
        self.center_head = nn.Sequential(DoubleConv(64, 48), nn.Conv2d(48, detection_classes, 1))
        self.offset_head = nn.Sequential(DoubleConv(64, 32), nn.Conv2d(32, 2, 1))
        self.size_head = nn.Sequential(DoubleConv(64, 32), nn.Conv2d(32, 2, 1))
        center_output = cast(nn.Conv2d, self.center_head[-1])
        assert center_output.bias is not None
        nn.init.constant_(center_output.bias, -2.19)

    def forward(self, image: torch.Tensor) -> TaskOutputs:
        stem = self.stem(image)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        values = self.layer4(layer3)
        values = self.up3(values, layer3)
        values = self.up2(values, layer2)
        quarter = self.up1(values, layer1)
        dense = self.up0(quarter, stem)
        dense = F.interpolate(dense, size=image.shape[-2:], mode="bilinear", align_corners=False)
        semantic = self.semantic_head(dense)
        affordance = self.affordance_head(dense)
        log_depth = self.depth_head(dense)
        center = self.center_head(quarter)
        offset = self.offset_head(quarter)
        size = F.softplus(self.size_head(quarter))
        return TaskOutputs(
            semantic_logits=cast(torch.Tensor, semantic),
            affordance_logits=cast(torch.Tensor, affordance),
            log_depth=cast(torch.Tensor, log_depth),
            center_logits=cast(torch.Tensor, center),
            center_offset=cast(torch.Tensor, offset),
            box_size=cast(torch.Tensor, size),
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class SplitDecoderPerceptionNet(nn.Module):
    """Share the encoder but give geometry and semantics independent decoders."""

    def __init__(
        self,
        *,
        semantic_classes: int = 19,
        affordance_channels: int = 2,
        detection_classes: int = 3,
        pretrained: bool = True,
        enabled_tasks: tuple[str, ...] = ("segmentation", "depth", "detection"),
    ) -> None:
        super().__init__()
        self.enabled_tasks = frozenset(enabled_tasks)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.seg_up3 = UpBlock(512, 256, 256)
        self.seg_up2 = UpBlock(256, 128, 128)
        self.seg_up1 = UpBlock(128, 64, 64)
        self.seg_up0 = UpBlock(64, 64, 48)
        self.depth_up3 = UpBlock(512, 256, 256)
        self.depth_up2 = UpBlock(256, 128, 128)
        self.depth_up1 = UpBlock(128, 64, 64)
        self.depth_up0 = UpBlock(64, 64, 48)
        self.det_up3 = UpBlock(512, 256, 192)
        self.det_up2 = UpBlock(192, 128, 96)
        self.det_up1 = UpBlock(96, 64, 64)
        self.semantic_head = nn.Conv2d(48, semantic_classes, 1)
        self.affordance_head = nn.Conv2d(48, affordance_channels, 1)
        self.depth_head = nn.Sequential(DoubleConv(48, 32), nn.Conv2d(32, 1, 1))
        self.center_head = nn.Sequential(DoubleConv(64, 48), nn.Conv2d(48, detection_classes, 1))
        self.offset_head = nn.Sequential(DoubleConv(64, 32), nn.Conv2d(32, 2, 1))
        self.size_head = nn.Sequential(DoubleConv(64, 32), nn.Conv2d(32, 2, 1))
        center_output = cast(nn.Conv2d, self.center_head[-1])
        assert center_output.bias is not None
        nn.init.constant_(center_output.bias, -2.19)

    def _dense(
        self,
        values: torch.Tensor,
        layer3: torch.Tensor,
        layer2: torch.Tensor,
        layer1: torch.Tensor,
        stem: torch.Tensor,
        blocks: tuple[UpBlock, UpBlock, UpBlock, UpBlock],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        up3, up2, up1, up0 = blocks
        values = up3(values, layer3)
        values = up2(values, layer2)
        values = up1(values, layer1)
        values = up0(values, stem)
        return cast(
            torch.Tensor,
            F.interpolate(values, size=output_size, mode="bilinear", align_corners=False),
        )

    def forward(self, image: torch.Tensor) -> TaskOutputs:
        stem = self.stem(image)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        bottleneck = self.layer4(layer3)
        semantic_features = self._dense(
            bottleneck,
            layer3,
            layer2,
            layer1,
            stem,
            (self.seg_up3, self.seg_up2, self.seg_up1, self.seg_up0),
            (image.shape[-2], image.shape[-1]),
        )
        depth_features = self._dense(
            bottleneck,
            layer3,
            layer2,
            layer1,
            stem,
            (self.depth_up3, self.depth_up2, self.depth_up1, self.depth_up0),
            (image.shape[-2], image.shape[-1]),
        )
        detection_features = self.det_up1(
            self.det_up2(self.det_up3(bottleneck, layer3), layer2), layer1
        )
        return TaskOutputs(
            semantic_logits=self.semantic_head(semantic_features),
            affordance_logits=self.affordance_head(semantic_features),
            log_depth=self.depth_head(depth_features),
            center_logits=self.center_head(detection_features),
            center_offset=self.offset_head(detection_features),
            box_size=F.softplus(self.size_head(detection_features)),
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_multitask_model(
    architecture: str,
    *,
    pretrained: bool,
    enabled_tasks: tuple[str, ...],
) -> MultiTaskPerceptionNet | SplitDecoderPerceptionNet:
    if architecture == "hard_shared":
        return MultiTaskPerceptionNet(pretrained=pretrained, enabled_tasks=enabled_tasks)
    if architecture == "split_decoder":
        return SplitDecoderPerceptionNet(pretrained=pretrained, enabled_tasks=enabled_tasks)
    raise ValueError(f"unsupported multi-task architecture: {architecture}")
