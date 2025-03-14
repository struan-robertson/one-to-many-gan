"""AdaIN Santa model from https://github.com/Mid-Push/santa/."""

from typing import Literal

import torch
from torch import nn

from .layers import (
    Conv2dWeightModulate,
    DownSample,
    EqualisedConv2d,
    EqualisedLinear,
    UpSample,
)
from .utils import compile_


@compile_
class ResnetBlock(nn.Module):
    """Define a Resnet block."""

    def __init__(
        self,
        dim: int,
        padding_type: Literal["reflect", "replicate", "zero"],
        *,
        use_dropout: bool = False,
        use_bias: bool = False,
    ):
        super().__init__()

        conv_block = []
        p = 0

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            err = f"padding {padding_type} is not implemented"
            raise NotImplementedError(err)

        conv_block += [
            EqualisedConv2d(dim, dim, kernel_size=3, padding=p, use_bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            err = f"padding {padding_type} is not implemented"
            raise NotImplementedError(err)
        conv_block += [
            EqualisedConv2d(dim, dim, kernel_size=3, padding=p, use_bias=use_bias),
            nn.InstanceNorm2d(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor):
        return x + self.conv_block(x)  # Residual connection


class ModulatedResnetBlock(nn.Module):
    """Resnet block with weight modulation."""

    def __init__(
        self,
        dim: int,
        w_dim: int,
        *,
        use_dropout: bool = False,
        use_bias: bool = False,
    ):
        super().__init__()

        conv_block = []
        conv_block += [
            Conv2dWeightModulate(
                dim, dim, w_dim=w_dim, kernel_size=3, use_bias=use_bias
            ),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
            Conv2dWeightModulate(
                dim, dim, w_dim=w_dim, kernel_size=3, use_bias=use_bias
            ),
        ]

        self.conv_block = nn.ModuleList(conv_block)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        residual = x

        for block in self.conv_block:
            x = block(x, w) if isinstance(block, Conv2dWeightModulate) else block(x)

        return residual + x


@compile_
class Generator(nn.Module):
    """Resnet generator consisting of Resnet blocks between downsampling/upsampling operations."""

    def __init__(self, input_nc: int = 1, *, w_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            EqualisedConv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            EqualisedConv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            DownSample(),
            EqualisedConv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            DownSample(),
            ResnetBlock(256, "reflect"),
            ResnetBlock(256, "reflect"),
            ResnetBlock(256, "reflect"),
        )

        self.decoder = nn.ModuleList(
            [
                ModulatedResnetBlock(256, w_dim=w_dim),
                ModulatedResnetBlock(256, w_dim=w_dim),
                ModulatedResnetBlock(256, w_dim=w_dim),
                ModulatedResnetBlock(256, w_dim=w_dim),
                UpSample(),
                Conv2dWeightModulate(256, 128, w_dim=w_dim, kernel_size=3),
                nn.ReLU(inplace=True),
                UpSample(),
                Conv2dWeightModulate(128, 64, w_dim=w_dim, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                EqualisedConv2d(64, 1, kernel_size=7),
                nn.Tanh(),
            ]
        )

        self.n_style_blocks = sum(
            isinstance(m, ModulatedResnetBlock | Conv2dWeightModulate)
            for m in self.decoder
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.encoder(x)

        i = 0
        for block in self.decoder:
            if isinstance(block, (ModulatedResnetBlock, Conv2dWeightModulate)):  # noqa: UP038
                x = block(x, w[i])
                i += 1
            else:
                x = block(x)

        return x


# TODO Try discriminator as pixel gan and siamese as this
@compile_
class Discriminator(nn.Module):
    """Discriminator used in Sanata paper."""

    def __init__(
        self,
        input_nc: int,
    ):
        super().__init__()

        self.model = nn.Sequential(
            EqualisedConv2d(input_nc, 64, kernel_size=4, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(),
            EqualisedConv2d(64, 128, kernel_size=4, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(),
            EqualisedConv2d(128, 256, kernel_size=4, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(),
            EqualisedConv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualisedConv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


@compile_
class StyleExtractor(nn.Module):
    """Given an image, extract the style vector used to create it."""

    def __init__(self, input_nc: int = 1, w_dim: int = 8):
        super().__init__()

        self.model = nn.Sequential(
            EqualisedConv2d(input_nc, 64, kernel_size=4, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(),
            EqualisedConv2d(64, 128, kernel_size=4, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(),
            EqualisedConv2d(128, 256, kernel_size=4, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(),
            EqualisedConv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            EqualisedLinear(512, w_dim),
        )

    def forward(self, x):
        return self.model(x)
