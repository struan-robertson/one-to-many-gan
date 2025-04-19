"""Different models used in overall architecture."""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Conv2dWeightModulate, ModulatedResnetBlock, ResnetBlock
from .layers import DownSample, EqualisedConv2d, EqualisedLinear, UpSample

# * Mapping Network


class MappingNetwork(nn.Module):
    """Maps from latent vector z to intermediate latent vector w."""

    def __init__(self, features: int, n_layers: int, style_mixing_prob: float):
        super().__init__()

        self.d_latent = features
        self.style_mixing_prob = style_mixing_prob

        layers = []
        for _ in range(n_layers):
            layers.append(EqualisedLinear(features, features))
            layers.append(
                nn.LeakyReLU(
                    negative_slope=0.2,
                    inplace=True,
                )
            )

        # Final layer should be a ReLU as when θ is zero the style vector should be zero
        layers[-1] = nn.ReLU(inplace=True)

        self.net = nn.Sequential(*layers)

        # Style vector when \theta=0
        shoeprint_style_vector = torch.zeros((1, 1, features), dtype=torch.float)
        self.register_buffer(
            "shoeprint_style_vector", shoeprint_style_vector, persistent=False
        )

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)

        return self.net(z)

    def get_two_w(
        self,
        batch_size: int,
        n_gen_blocks: int,
        device: torch.device,
        domain_variables: tuple[torch.Tensor, torch.Tensor],
        *,
        mix_styles=True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply two domain variables to the same style vector."""
        d1, d2 = domain_variables

        style_vector = self._get_style_vector(
            batch_size, n_gen_blocks, device, mix_styles=mix_styles
        )

        shoeprint_style_vector = cast(torch.Tensor, self.shoeprint_style_vector)
        w1 = torch.lerp(
            shoeprint_style_vector, style_vector, d1.view(1, -1, 1)
        )  # d1: [1, batch_dim, 1]
        w2 = torch.lerp(shoeprint_style_vector, style_vector, d2.view(1, -1, 1))

        return w1, w2

    def get_single_w(
        self,
        batch_size: int,
        n_gen_blocks: int,
        device: torch.device,
        domain_variable: float | torch.Tensor,
        *,
        mix_styles=True,
    ) -> torch.Tensor:
        """Apply a domain variable to a single style vector."""
        shoeprint_style_vector = cast(torch.Tensor, self.shoeprint_style_vector)

        if domain_variable == 0:
            return shoeprint_style_vector.expand(
                (n_gen_blocks, batch_size, self.d_latent)
            )

        style_vector = self._get_style_vector(
            batch_size, n_gen_blocks, device, mix_styles=mix_styles
        )

        if isinstance(domain_variable, torch.Tensor):
            # Reshape for broadcasting
            d = domain_variable.view(1, -1, 1)
        else:  # Scalar case
            d = torch.tensor(domain_variable, dtype=torch.float, device=device).view(
                1, 1, 1
            )

        return torch.lerp(shoeprint_style_vector, style_vector, d)

    def _get_style_vector(
        self,
        batch_size: int,
        n_gen_blocks: int,
        device: torch.device,
        *,
        mix_styles=True,
    ) -> torch.Tensor:
        """Sample z randomly and get style vector s from mapping network."""
        if mix_styles and torch.rand(()).lt(self.style_mixing_prob):
            cross_over_point = torch.randint(0, n_gen_blocks, ())

            z1 = torch.randn(batch_size, self.d_latent).to(device)
            z2 = torch.randn(batch_size, self.d_latent).to(device)

            s1 = self.forward(z1)
            s2 = self.forward(z2)

            s1 = s1[None, :, :].expand(cross_over_point, -1, -1)
            s2 = s2[None, :, :].expand(n_gen_blocks - cross_over_point, -1, -1)
            s = torch.cat((s1, s2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(device)
            s = self.forward(z)
            s = s[None, :, :].expand(n_gen_blocks, -1, -1)

        return s


# * Generator


class Generator(nn.Module):
    """Resnet generator consisting of Resnet blocks between downsampling/upsampling operations."""

    def __init__(
        self,
        input_nc: int,
        w_dim: int,
        image_size: tuple[int, int],
        min_latent_resolution: int,
        n_resnet_blocks: int,
        start_filters: int = 64,
    ):
        super().__init__()

        filters = start_filters
        min_image_resolution = min(image_size)
        n_downsamples = math.ceil(
            math.log2(min_image_resolution / min_latent_resolution)
        )
        n_encoder_resnet_blocks = n_resnet_blocks // 2
        n_decoder_resnet_blocks = math.ceil(n_resnet_blocks / 2)

        # Initial encoder block without downsampling
        encoder = [
            nn.ReflectionPad2d(3),
            EqualisedConv2d(input_nc, filters, kernel_size=7),
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True),
        ]

        # Downsample to specified latent resolution
        for _ in range(n_downsamples):
            encoder += [
                EqualisedConv2d(filters, filters * 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(filters * 2),
                nn.ReLU(inplace=True),
                DownSample(),
            ]
            filters *= 2

        # Encoder portion of resnet blocks
        encoder += [ResnetBlock(filters) for _ in range(n_encoder_resnet_blocks)]

        self.encoder = nn.Sequential(*encoder)

        # Decoder portion of resnet blocks
        decoder = [
            ModulatedResnetBlock(filters, w_dim=w_dim)
            for _ in range(n_decoder_resnet_blocks)
        ]

        # Upsample to image dimensions
        for _ in range(n_downsamples):
            decoder += [
                UpSample(),
                Conv2dWeightModulate(
                    filters, filters // 2, kernel_size=3, padding=1, w_dim=w_dim
                ),
                nn.ReLU(inplace=True),
            ]
            filters //= 2

        # Final decoder block without downsampling
        decoder += [
            nn.ReflectionPad2d(3),
            EqualisedConv2d(filters, input_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.decoder = nn.ModuleList(decoder)

        self.n_style_blocks = sum(
            [
                isinstance(m, ModulatedResnetBlock | Conv2dWeightModulate)
                for m in self.decoder
            ]  # Use list comprehension instead of generator for compatibility with torch.compile
        )

    def encode(self, x: torch.Tensor):
        """Encode x to latent space z."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor, w: torch.Tensor):
        """Decode from latent space z to image, using style vector w."""
        i = 0
        for layer in self.decoder:
            if isinstance(layer, ModulatedResnetBlock | Conv2dWeightModulate):
                z = layer(z, w[i])
                i += 1
            else:
                z = layer(z)

        return z

    def extract(self, z: torch.Tensor, w: torch.Tensor):
        """Return feature maps from specified layers."""
        features = []
        i = 0
        for layer in self.decoder:
            if isinstance(layer, ModulatedResnetBlock | Conv2dWeightModulate):
                z = layer(z, w[i])
                i += 1

                # Return each weight demodulation layer
                features.append(z)
                if i == self.n_style_blocks:
                    return features
            else:
                z = layer(z)

        msg = "No return layers specified."
        raise ValueError(msg)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.encode(x)
        return self.decode(x, w)


# * Discriminator


class Discriminator(nn.Module):
    """Discriminator used in Santa paper."""

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


# * Style Extractor


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
