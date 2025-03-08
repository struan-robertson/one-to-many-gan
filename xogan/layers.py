"""Basic layers for StyleGAN model."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class EqualisedWeight(nn.Module):
    """Based on equalised learning rate introduced in the Progressive GAN paper."""

    def __init__(self, shape: list[int]):
        super().__init__()

        # He initialisation constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))

        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualisedLinear(nn.Module):
    """Uses learning-rate equalised weights for a linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        super().__init__()

        self.weight = EqualisedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualisedConv2d(nn.Module):
    """Uses learning-rate equalised weights for a convolution layer."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int, padding: int = 0
    ):
        super().__init__()

        self.padding = padding
        self.weight = EqualisedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class Conv2dWeightModulate(nn.Module):
    """Scales the convolutional weights by the style vector and demodulates by normalising it."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        demodulate: float = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualisedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]

        weights = self.weight()[None, :, :, :, :]

        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt(
                (weights**2).sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )

            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class Smooth(nn.Module):
    """Blurs each channel."""

    def __init__(self):
        super().__init__()

        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

        kernel = torch.tensor([[kernel]], dtype=torch.float)

        kernel /= kernel.sum()

        # TODO try this with registering a buffer to see if it works as well
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)

        return x.view(b, c, h, w)


class UpSample(nn.Module):
    """Scales the image up by 2x and then smooths."""

    def __init__(self):
        super().__init__()

        # TODO investigate how this is different than F.interpolate
        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class DownSample(nn.Module):
    """Smooths each feature and then scales down by 2x."""

    def __init__(self):
        super().__init__()

        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)

        return F.interpolate(
            x, (x.shape[2] // 2, x.shape[3] // 2), mode="bilinear", align_corners=False
        )
