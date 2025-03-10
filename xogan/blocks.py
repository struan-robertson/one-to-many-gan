"""Blocks from which to construct full models."""

import math

import torch
from torch import nn

from .layers import Conv2dWeightModulate, DownSample, EqualisedConv2d, EqualisedLinear


class ToRGB(nn.Module):
    """Generates an RGB image from feature maps using a 1x1 convolution."""

    def __init__(self, d_latent: int, features: int, out_channels=1):
        super().__init__()

        self.to_style = EqualisedLinear(d_latent, features, bias=1.0)

        self.conv = Conv2dWeightModulate(
            features, out_channels, kernel_size=1, demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style = self.to_style(w)

        x = self.conv(x, style)

        return self.activation(x + self.bias[None, :, None, None])


class StyleBlock(nn.Module):
    """Weight modulation and noise injection."""

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()

        self.to_style = EqualisedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor | None):
        s = self.to_style(w)
        x = self.conv(x, s)

        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise

        return self.activation(x + self.bias[None, :, None, None])


class GeneratorEncoderBlock(nn.Module):
    """No style or noise injection."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.residual = EqualisedConv2d(in_features, out_features, kernel_size=1)

        # TODO try with two conv blocks
        self.conv = EqualisedConv2d(in_features, out_features, kernel_size=3, padding=1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.down_sample = DownSample()

    def forward(self, x: torch.Tensor):
        residual = self.residual(x)

        x = self.conv(x)
        x = self.activation(x + self.bias[None, :, None, None])

        return x + residual


class GeneratorDecoderBlock(nn.Module):
    """Two StyleBlocks and an RGB output."""

    def __init__(
        self, d_latent: int, in_features: int, out_features: int, out_channels: int
    ):
        super().__init__()

        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        self.to_rgb = ToRGB(d_latent, out_features, out_channels=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: tuple[torch.Tensor | None, torch.Tensor | None],
    ):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb


class DiscriminatorBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.residual = nn.Sequential(
            DownSample(), EqualisedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualisedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualisedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_sample = DownSample()

        self.scale = 1 / math.sqrt(2)

    def forward(self, x: torch.Tensor):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    """Calculates the stddev across subgroups in a mini batch and appends as a feature."""

    def __init__(self, group_size: int = 4):
        super().__init__()

        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        if x.shape[0] % self.group_size != 0:
            raise ValueError

        grouped = x.view(self.group_size, -1)

        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)

        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)

        return torch.cat([x, std], dim=1)


# Local Variables:
# jinx-local-words: "stddev"
# End:
