"""Generate synthetic data."""

import torch

from src.data.config import Config
from src.model.builder import Generator, MappingNetwork


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
                w_dim=config["architecture"]["w_dim"],
                image_size=config["data"]["image_size"],
                min_latent_resolution=config["architecture"]["min_latent_resolution"],
                n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
            )
            .to(device)
            .eval()
        )

        mapping_network = (
            MappingNetwork(
                features=config["architecture"]["w_dim"],
                n_layers=config["architecture"]["mapping_network_layers"],
                style_mixing_prob=config["training"]["style_mixing_prob"],
            )
            .to(device)
            .eval()
        )

        checkpoint = torch.load(
            config["inference"]["checkpoint"],
            map_location=device,
        )
        generator.load_state_dict(checkpoint["generator_state_dict"])
        mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])
        self.generator = generator
        self.mapping_network = mapping_network
        self.device = device

    def generate(self, shoeprints: torch.Tensor, difficulty: float):
        with torch.no_grad():
            s = self.mapping_network.get_single_w(
                batch_size=shoeprints.shape[0],
                n_gen_blocks=self.generator.n_style_blocks,
                device=self.device,
                mix_styles=False,
                domain_variable=difficulty,
            )

            return self.generator(shoeprints, s)
