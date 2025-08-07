"""Generate synthetic data."""

import itertools

import torch
from torchvision.transforms import v2

from src.data.config import Config
from src.data.datasets import ShoeDataset, dataset_transform
from src.model.builder import Generator, MappingNetwork


class GeneratorHandler:
    """Class to handle generating shoeprint and shoemark images using a pre-trained network."""

    def __init__(
        self,
        config: Config,
        device: torch.device,
        batch_size: int,
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

        transform = dataset_transform(
            config["data"]["image_size"], config["data"]["norm_mean"], config["data"]["norm_std"]
        )
        shoeprint_data = ShoeDataset(
            config["data"]["shoeprint_data_dir"], mode="train", transform=transform
        )
        shoeprint_dataloader = torch.utils.data.DataLoader(
            shoeprint_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        self.shoeprint_cycle = itertools.cycle(shoeprint_dataloader)

        self.batch_size = batch_size
        self.device = device

        self.transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=(512, 256), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation((0, 180), v2.InterpolationMode.BILINEAR, fill=256),
                v2.ColorJitter(brightness=0.5, hue=0.3),
            ]
        )

    # TODO make generator so I can call next() on it
    def generate(self, difficulty: float):
        with torch.no_grad():
            shoeprints = next(self.shoeprint_cycle).to(self.device)

            s1 = self.mapping_network.get_single_w(
                batch_size=self.batch_size,
                n_gen_blocks=self.generator.n_style_blocks,
                device=self.device,
                mix_styles=False,
                domain_variable=difficulty,
            )
            s2 = self.mapping_network.get_single_w(
                batch_size=self.batch_size,
                n_gen_blocks=self.generator.n_style_blocks,
                device=self.device,
                mix_styles=False,
                domain_variable=difficulty,
            )

            shoemarks1 = self.generator(shoeprints, s1)
            shoemarks2 = self.generator(shoeprints, s2)

            return shoeprints, shoemarks1, shoemarks2

    def generate_from_shoeprint(self, shoeprint: torch.Tensor, difficulty: float):
        with torch.no_grad():
            s = self.mapping_network.get_single_w(
                batch_size=1,
                n_gen_blocks=self.generator.n_style_blocks,
                device=self.device,
                mix_styles=False,
                domain_variable=difficulty,
            )

            return self.generator(shoeprint, s)

    def generate_from_shoeprints(self, shoeprints: torch.Tensor, difficulty: float):
        with torch.no_grad():
            s = self.mapping_network.get_single_w(
                batch_size=shoeprints.shape[0],
                n_gen_blocks=self.generator.n_style_blocks,
                device=self.device,
                mix_styles=False,
                domain_variable=difficulty,
            )

            return self.generator(shoeprints, s)
