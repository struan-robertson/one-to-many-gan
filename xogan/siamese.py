"""Siamese judgement network."""

import torch
from torch import nn
from torchvision.ops import SqueezeExcitation

from .layers import DownSample, EqualisedConv2d, EqualisedLinear
from .utils import compile_


class DilatedResBlock(nn.Module):
    """Dilated residual blocks to take into account global context."""

    def __init__(
        self, in_channels: int, dilation_rate: int = 2, reduction_ratio: int = 4
    ):
        super().__init__()
        self.conv1 = EqualisedConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = EqualisedConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Spatial-Channel Attention (SCA)
        squeeze_channels = in_channels // reduction_ratio
        self.sca = SqueezeExcitation(in_channels, squeeze_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sca(x)
        x += residual

        return self.leaky_relu(x)


@compile_
class GlobalContextSiamese(nn.Module):
    """Produces a feature vector for comparing images."""

    def __init__(self, in_channels: int = 1, emb_dim: int = 256):
        super().__init__()

        # Initial downsampling
        self.stem = nn.Sequential(
            EqualisedConv2d(in_channels, 64, kernel_size=7, padding=3),
            DownSample(),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, padding=1),
            DownSample(),
        )

        # Dilated residual blocks for multi-scale context
        self.layer1 = nn.Sequential(
            DilatedResBlock(64, dilation_rate=1),  # Local features
            DilatedResBlock(64, dilation_rate=2),  # Medium-range context
        )
        self.layer2 = nn.Sequential(
            DilatedResBlock(128, dilation_rate=3),  # Long-range context
            DilatedResBlock(128, dilation_rate=6),  # Global features
        )

        self.downsample = nn.Sequential(
            EqualisedConv2d(64, 128, kernel_size=1), DownSample()
        )

        # Global Average Pooling (GAP) + Projection
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualisedLinear(128, emb_dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)  # [B, 64, H/4, W/4]
        residual = self.downsample(x)
        x = self.layer1(x)  # [B, 64, H/4, W/4]
        x = self.downsample(x)  # [B, 128, H/8, W/8]
        x = self.layer2(x)  # [B, 128, H/8, W/8]
        x += residual
        x = self.gap(x)  # [B, 128, 1, 1]
        x = torch.flatten(x, 1)  # [B, 128]
        return self.fc(x)  # [B, 256]
