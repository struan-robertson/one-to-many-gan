"""Lets try this with PyTorch instead of Jax."""

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from .blocks import (
    DiscriminatorBlock,
    GeneratorBlock,
    MiniBatchStdDev,
    StyleBlock,
    ToRGB,
)
from .layers import EqualisedConv2d, EqualisedLinear, UpSample


class MappingNetwork(nn.Module):
    """Maps from latent vector z to intermediate latent vector w."""

    def __init__(self, features: int, n_layers: int):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(EqualisedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)

        return self.net(z)


class Generator(nn.Module):
    """Generates images given intermediate latent space w."""

    def __init__(
        self,
        log_resolution: int,
        d_latent: int,
        n_features: int = 32,
        max_features: int = 512,
    ):
        super().__init__()

        features = [
            min(max_features, n_features * (2**i))
            for i in range(log_resolution - 2, -1, -1)
        ]

        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        blocks = [
            GeneratorBlock(d_latent, features[i - 1], features[i])
            for i in range(1, self.n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.up_sample = UpSample()

    def forward(
        self,
        w: torch.Tensor,
        input_noise: list[tuple[torch.Tensor | None, torch.Tensor | None]],
    ):
        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])

        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])

            rgb = self.up_sample(rgb) + rgb_new

        return rgb


class Discriminator(nn.Module):
    """Judges if an image is fake or real."""

    def __init__(
        self,
        log_resolution: int,
        n_features: int = 64,
        max_features: int = 512,
        in_channels: int = 3,
    ):
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualisedConv2d(in_channels, n_features, 1), nn.LeakyReLU(0.2, inplace=True)
        )

        features = [
            min(max_features, n_features * (2**i)) for i in range(log_resolution - 1)
        ]

        n_blocks = len(features) - 1
        blocks = [
            DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1  # Account for standard deviation feature

        self.conv = EqualisedConv2d(final_features, final_features, 3)
        self.final = EqualisedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        # Optionally normalise input image
        x = x - 0.5

        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)

        return self.final(x)
