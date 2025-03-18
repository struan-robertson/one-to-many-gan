"""Different models used in overall architecture."""

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Conv2dWeightModulate, ModulatedResnetBlock, ResnetBlock
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
        device: str | int,
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


@compile_
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
