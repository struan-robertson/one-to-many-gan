"""Blocks from which to construct full models."""

import torch
from torch import nn

from .layers import Conv2dWeightModulate, EqualisedConv2d


class ResnetBlock(nn.Module):
    """Define a Resnet block."""

    def __init__(
        self,
        dim: int,
        *,
        use_bias: bool = False,
    ):
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            EqualisedConv2d(dim, dim, kernel_size=3, padding=0, use_bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            EqualisedConv2d(dim, dim, kernel_size=3, padding=0, use_bias=use_bias),
            nn.InstanceNorm2d(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor):
        return x + self.conv_block(x)


class ModulatedResnetBlock(nn.Module):
    """Resnet block with weight modulation."""

    def __init__(
        self,
        dim: int,
        w_dim: int,
        *,
        use_bias: bool = False,
    ):
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            Conv2dWeightModulate(
                dim, dim, w_dim=w_dim, kernel_size=3, padding=0, use_bias=use_bias
            ),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            Conv2dWeightModulate(
                dim, dim, w_dim=w_dim, kernel_size=3, padding=0, use_bias=use_bias
            ),
        ]

        self.conv_block = nn.ModuleList(conv_block)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        residual = x

        for block in self.conv_block:
            x = block(x, w) if isinstance(block, Conv2dWeightModulate) else block(x)

        return residual + x


# Local Variables:
# jinx-local-words: "stddev"
# End:
