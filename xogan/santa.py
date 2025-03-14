"""AdaIN Santa model from https://github.com/Mid-Push/santa/."""

from typing import Literal

import torch
from torch import nn

from .layers import Conv2dWeightModulate, DownSample, EqualisedConv2d, UpSample
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
            Conv2dWeightModulate(dim, dim, kernel_size=3, padding=p, use_bias=use_bias),
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


@compile_
class AdaINResnetGenerator(nn.Module):
    """Resnet generator consisting of Resnet blocks between downsampling/upsampling operations."""

    def __init__(self, input_nc: int = 1):
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

        self.decoder = nn.Sequential(
            ResnetBlock(256, "reflect"),  # These are the ones that need style norming
            ResnetBlock(256, "reflect"),
            ResnetBlock(256, "reflect"),
            ResnetBlock(256, "reflect"),
            UpSample(),
            EqualisedConv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            UpSample(),
            EqualisedConv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            EqualisedConv2d(64, 1, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return self.decoder(x)


# TODO Try discriminator as pixel gan and siamese as this
class AdaINDiscriminator(nn.Module):
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
