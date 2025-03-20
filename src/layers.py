"""Basic layers for StyleGAN model."""

import math
from typing import cast

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

        self.in_features = in_features
        self.out_features = out_features

        self.weight = EqualisedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.zeros(out_features) + bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"


class EqualisedConv2d(nn.Module):
    """Uses learning-rate equalised weights for a convolution layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        *,
        use_bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if isinstance(kernel_size, int):
            kernel_height, kernel_width = (kernel_size, kernel_size)
        else:
            kernel_height, kernel_width = kernel_size

        self.weight = EqualisedWeight(
            [out_features, in_features, kernel_height, kernel_width]
        )

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor):
        if self.use_bias:
            x = F.conv2d(
                x,
                self.weight(),
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        else:
            x = F.conv2d(
                x,
                self.weight(),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation}"
        )


class Conv2dWeightModulate(nn.Module):
    """Scales the convolutional weights by the style vector and demodulates by normalising it."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        w_dim: int,
        padding: int,
        *,
        use_bias: bool = False,
        demodulate: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = padding
        self.weight = EqualisedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        self.eps = eps
        self.use_bias = use_bias

        self.to_style = EqualisedLinear(
            w_dim, in_features, bias=1
        )  # Account for style vectors with zero values

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        b, _, height, width = x.shape

        s = self.to_style(w)

        s = s[:, None, :, None, None]

        weights = self.weight()[None, :, :, :, :]

        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt(
                (weights**2).sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )

            weights = weights * sigma_inv

        x = x.reshape(1, -1, height, width)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Bit of an ugly hack to account for the fact that external padding
        # is applied before we calculate the height and width.
        # Not the end of the world as I only use nn.ReflectionPad2d(1)
        if self.padding == 0:
            height -= 2
            width -= 2

        x = x.reshape(-1, self.out_features, height, width)

        if self.use_bias:
            x = x + self.bias[None, :, None, None]

        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f"demodulate={self.demodulate}, padding={self.padding}, eps={self.eps}"
        )


class Smooth(nn.Module):
    """Blurs each channel."""

    def __init__(self):
        super().__init__()

        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

        kernel = torch.tensor([[kernel]], dtype=torch.float)

        kernel /= kernel.sum()

        self.register_buffer("kernel", kernel)
        self.kernel = cast(torch.Tensor, self.kernel)
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

        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class DownSample(nn.Module):
    """Smooths each feature and then scales down by 2x."""

    def __init__(self, *, smooth=True):
        super().__init__()

        self.smooth_map = smooth
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        if self.smooth_map:
            x = self.smooth(x)

        return F.interpolate(
            x, (x.shape[2] // 2, x.shape[3] // 2), mode="bilinear", align_corners=False
        )
