"""DeepLab, U-Net, and Fourier U-Net affordance models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class DeepLabAffordance(nn.Module):
    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.network = deeplabv3_mobilenet_v3_large(weights=weights, weights_backbone=None)
        final = self.network.classifier[-1]
        self.network.classifier[-1] = nn.Conv2d(final.in_channels, 2, kernel_size=1)
        self.network.aux_classifier = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(image)["out"])


class DoubleConv(nn.Sequential):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )


class SpectralConv2d(nn.Module):
    """Complex multiplication of a bounded rectangle of 2-D Fourier modes."""

    def __init__(self, channels: int, modes_height: int, modes_width: int) -> None:
        super().__init__()
        if min(channels, modes_height, modes_width) < 1:
            raise ValueError("spectral dimensions must be positive")
        self.channels = channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        scale = 1.0 / math.sqrt(channels)
        shape = (channels, channels, modes_height, modes_width)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        original_dtype = values.dtype
        spectrum = torch.fft.rfft2(values.float(), norm="ortho")
        height = min(self.modes_height, spectrum.shape[-2] // 2)
        width = min(self.modes_width, spectrum.shape[-1])
        output = torch.zeros(
            values.shape[0],
            self.channels,
            spectrum.shape[-2],
            spectrum.shape[-1],
            dtype=spectrum.dtype,
            device=values.device,
        )
        output[:, :, :height, :width] = torch.einsum(
            "bihw,iohw->bohw",
            spectrum[:, :, :height, :width],
            self.weight_positive[:, :, :height, :width],
        )
        output[:, :, -height:, :width] = torch.einsum(
            "bihw,iohw->bohw",
            spectrum[:, :, -height:, :width],
            self.weight_negative[:, :, :height, :width],
        )
        restored = torch.fft.irfft2(output, s=values.shape[-2:], norm="ortho")
        return cast(torch.Tensor, restored.to(original_dtype))


class FourierBlock2d(nn.Module):
    def __init__(self, channels: int, modes_height: int, modes_width: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, modes_height, modes_width)
        self.local = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            values + F.gelu(self.norm(self.spectral(values) + self.local(values))),
        )


class UpBlock(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int) -> None:
        super().__init__()
        self.conv = DoubleConv(inputs + skip, outputs)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return cast(torch.Tensor, self.conv(torch.cat((values, skip), dim=1)))


class ResNet18UNet(nn.Module):
    def __init__(
        self,
        *,
        pretrained: bool = True,
        fourier: bool = False,
        spectral_modes_height: int = 5,
        spectral_modes_width: int = 8,
        spectral_blocks: int = 2,
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.operator = nn.Sequential(
            *(
                FourierBlock2d(512, spectral_modes_height, spectral_modes_width)
                for _ in range(spectral_blocks if fourier else 0)
            )
        )
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 64, 48)
        self.head = nn.Sequential(DoubleConv(48, 32), nn.Conv2d(32, 2, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stem = self.stem(image)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        values = self.operator(self.layer4(layer3))
        values = self.up3(values, layer3)
        values = self.up2(values, layer2)
        values = self.up1(values, layer1)
        values = self.up0(values, stem)
        values = F.interpolate(values, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return cast(torch.Tensor, self.head(values))


def build_segmentation_model(
    name: str,
    *,
    pretrained: bool = True,
    config: Mapping[str, Any] | None = None,
) -> nn.Module:
    key = name.strip().lower().replace("-", "_")
    if key == "deeplabv3_mobilenet_v3_large":
        return DeepLabAffordance(pretrained=pretrained)
    if key == "resnet18_unet":
        return ResNet18UNet(pretrained=pretrained)
    if key == "resnet18_fourier_unet":
        return ResNet18UNet(pretrained=pretrained, fourier=True, **dict(config or {}))
    raise ValueError(f"unsupported segmentation model: {name}")
