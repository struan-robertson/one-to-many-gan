"""Generate synthetic data."""

from typing import cast

import torch
import torchvision.transforms.v2.functional as F

from one_to_many_gan.data.config import Config
from one_to_many_gan.model.builder import Generator, MappingNetwork


class GeneratorHandler:
    """Class to handle generating shoeprint and shoemark images using a pre-trained network."""

    def __init__(
        self,
        config: Config,
        device: torch.device,
    ):
        generator = (
            Generator(
                input_nc=config["data"]["image_channels"],
                s_dim=config["architecture"]["s_dim"],
                image_size=config["data"]["image_size"],
                min_latent_resolution=config["architecture"]["min_latent_resolution"],
                n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
            )
            .to(device)
            .eval()
        )

        mapping_network = (
            MappingNetwork(
                features=config["architecture"]["s_dim"],
                n_layers=config["architecture"]["mapping_network_layers"],
                style_mixing_prob=config["training"]["style_mixing_prob"],
                n_gen_blocks=generator.n_style_blocks,
            )
            .to(device)
            .eval()
        )

        generator = torch.compile(generator, fullgraph=True, mode="default")
        mapping_network = torch.compile(mapping_network, fullgraph=True, mode="default")
        generator = cast(Generator, generator)
        mapping_network = cast(MappingNetwork, mapping_network)

        checkpoint = torch.load(
            config["inference"]["checkpoint"],
            map_location=device,
        )
        generator.load_state_dict(checkpoint["generator_state_dict"])
        mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])

        for param in generator.parameters():
            param.requires_grad = False

        for param in mapping_network.parameters():
            param.requires_grad = False

        self.device = device
        self.shoeprint_norm = config["data"]["shoeprint_norm"]
        self.generator = generator
        self.mapping_network = mapping_network

    def generate(
        self,
        shoeprints: torch.Tensor,
        *,
        normalised=False,
        difficulty: float | None = None,
        style: torch.Tensor | None = None,
    ):
        if style is None:
            if difficulty is None:
                raise ValueError("Either 'style' or 'difficulty' must be provided.")

            style = self.mapping_network.get_single_s(
                batch_size=shoeprints.shape[0],
                device=self.device,
                mix_styles=False,
                domain_variable=difficulty,
            )

        if not normalised:
            shoeprints = F.normalize(shoeprints, *self.shoeprint_norm)  # pyright: ignore [reportArgumentType]

        return self.generator(shoeprints, style)

    def get_style(self, batch_size: int, difficulty: float):
        return self.mapping_network.get_single_s(  # pyright: ignore [reportFunctionMemberAccess]
            batch_size=batch_size,
            device=self.device,
            mix_styles=False,
            domain_variable=difficulty,
        )
