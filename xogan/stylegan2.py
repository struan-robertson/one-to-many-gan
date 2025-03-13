"""Lets try this with PyTorch instead of Jax."""

import math

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torch._prims_common import DeviceLikeType

from .blocks import (
    DiscriminatorBlock,
    GeneratorBottleneckBlock,
    GeneratorDecoderBlock,
    GeneratorEncoderBlock,
    MiniBatchStdDev,
    StyleBlock,
    ToRGB,
)
from .layers import DownSample, EqualisedConv2d, EqualisedLinear, UpSample
from .utils import compile_


@compile_
class MappingNetwork(nn.Module):
    """Maps from latent vector z to intermediate latent vector w."""

    def __init__(self, features: int, n_layers: int, style_mixing_prob: float):
        super().__init__()

        self.d_latent = features
        self.style_mixing_prob = style_mixing_prob

        layers = []
        for _ in range(n_layers):
            layers.append(EqualisedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)

        return self.net(z)

    def get_w(
        self,
        batch_size: int,
        n_gen_blocks: int,
        device: DeviceLikeType,
        *,
        mix_styles=True,
    ):
        """Sample z randomly and get w from mapping network.

        Style mixing is also applied randomly."""
        if torch.rand(()).lt(self.style_mixing_prob) and mix_styles:
            cross_over_point = torch.randint(0, n_gen_blocks, ())

            z1 = torch.randn(batch_size, self.d_latent).to(device)
            z2 = torch.randn(batch_size, self.d_latent).to(device)

            w1 = self.forward(z1)
            w2 = self.forward(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)

        z = torch.randn(batch_size, self.d_latent).to(device)
        w = self.forward(z)

        return w[None, :, :].expand(n_gen_blocks, -1, -1)


@compile_
class Generator(nn.Module):
    """Generates images given intermediate latent space w."""

    def __init__(
        self,
        log_resolution: int,
        bottleneck_resolution: tuple[int, int],
        d_latent: int,
        n_bottleneck_blocks: int = 9,
        n_features: int = 32,
        img_channels: int = 1,
    ):
        super().__init__()

        # Number of encoding and decoding steps required to reach specified bottleneck resolution
        coding_steps = int(log_resolution - math.log2(min(bottleneck_resolution)))
        coding_features = [n_features * (2**i) for i in range(coding_steps)]
        encoder_features = coding_features
        decoder_features = coding_features[::-1]

        # Encoder/decoder + bottleneck blocks + first RGB style block
        self.n_noise_blocks = len(decoder_features)
        self.n_style_blocks = len(decoder_features) + n_bottleneck_blocks + 1

        self.from_rgb = nn.Sequential(
            EqualisedConv2d(
                in_features=img_channels, out_features=n_features, kernel_size=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        encoder_blocks = [
            GeneratorEncoderBlock(
                in_features=encoder_features[i - 1], out_features=encoder_features[i]
            )
            for i in range(1, len(encoder_features))
        ]
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.bottleneck_resolution = bottleneck_resolution
        bottleneck_blocks = [
            GeneratorBottleneckBlock(d_latent, encoder_features[-1])
        ] * (n_bottleneck_blocks - 1)
        self.bottleneck_blocks = nn.ModuleList(bottleneck_blocks)

        # First RGB output from bottleneck
        self.style_block = StyleBlock(
            d_latent, decoder_features[0], decoder_features[0]
        )
        self.to_rgb = ToRGB(d_latent, decoder_features[0], out_channels=img_channels)

        decoder_blocks = [
            GeneratorDecoderBlock(
                d_latent,
                in_features=decoder_features[i - 1],
                out_features=decoder_features[i],
                out_channels=img_channels,
            )
            for i in range(1, len(decoder_features))
        ]
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.down_sample = DownSample(smooth=False)
        self.up_sample = UpSample()

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        input_noise: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        w_iter = iter(w)
        noise_iter = iter(input_noise)
        # The last block doesn't get run
        # Encoder
        x = self.from_rgb(x)

        # Encoder
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            x = self.down_sample(x)

        # Bottleneck
        for i in range(len(self.bottleneck_blocks)):
            x = self.bottleneck_blocks[i](x, next(w_iter))

        # Decoder
        first_encoder_w = next(w_iter)
        x = self.style_block(x, first_encoder_w, next(noise_iter)[1])
        rgb = self.to_rgb(x, first_encoder_w)

        for i in range(len(self.decoder_blocks)):
            x = self.up_sample(x)
            x, rgb_new = self.decoder_blocks[i](x, next(w_iter), next(noise_iter))

            rgb = self.up_sample(rgb) + rgb_new

        return rgb

    def get_noise(
        self, batch_size: int, device: DeviceLikeType
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate noise for each generator block."""
        noise = []
        resolution = self.bottleneck_resolution

        for i in range(self.n_noise_blocks):
            n1 = (
                None
                if i == 0
                else torch.randn(
                    batch_size, 1, resolution[0], resolution[1], device=device
                )
            )
            n2 = torch.randn(batch_size, 1, resolution[0], resolution[1], device=device)

            noise.append((n1, n2))

            resolution = (resolution[0] * 2, resolution[1] * 2)

        return noise

    def generate_images(
        self,
        batch_size: int,
        x: torch.Tensor,
        w: torch.Tensor,
        device: DeviceLikeType,
    ) -> torch.Tensor:
        """Generate images using the generator."""
        noise = self.get_noise(batch_size, device)
        return self.forward(x, w, noise)


@compile_
class Discriminator(nn.Module):
    """Judges if an image is fake or real."""

    def __init__(
        self,
        log_resolution: int,
        n_features: int = 64,
        max_features: int = 512,
        in_channels: int = 1,
    ):
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualisedConv2d(
                in_features=in_channels, out_features=n_features, kernel_size=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
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

        self.conv = EqualisedConv2d(
            final_features, final_features, kernel_size=(7, 3)
        )  # 7, 3
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
