"""Generate synthetic data."""

import itertools
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.config import load_config
from src.data.datasets import ShoeDataset, dataset_transform
from src.model.builder import Generator, MappingNetwork

rng = np.random.default_rng(10)

config = load_config("config.toml")
device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

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
    "/home/struan/Vault/University/Doctorate/GAN Training/training_runs/no_ada/models/150000.tar",
    map_location=device,
)
generator.load_state_dict(checkpoint["generator_state_dict"])
mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])

transform = dataset_transform(config["data"]["image_size"])

shoeprint_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"], mode="train", transform=transform
)

shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

shoeprint_cycle = itertools.cycle(shoeprint_dataloader)


def blend(flooring: np.ndarray, shoemark: np.ndarray):
    """Blend a flooring and shoemark image."""
    # Select scale of floor image rectangle w.r.t the shoemark
    max_scale = min(flooring.shape[0] / shoemark.shape[0], flooring.shape[1] / shoemark.shape[1])
    scale = rng.uniform(1, max_scale)

    scaled_width = math.floor(shoemark.shape[1] * scale)
    scaled_height = math.floor(shoemark.shape[0] * scale)

    # Create an all white mask
    mask = 255 * np.ones(shoemark.shape[:2], shoemark.dtype)

    # Select random co-ordinates of floor rectangle
    min_x = math.ceil(scaled_width / 2)
    max_x = flooring.shape[1] - min_x
    min_y = math.ceil(scaled_height / 2)
    max_y = flooring.shape[0] - min_y

    x = rng.integers(min_x, max_x)
    y = rng.integers(min_y, max_y)

    crop_half_width = scaled_width // 2
    crop_half_height = scaled_height // 2
    x1 = x - crop_half_width
    x2 = x + crop_half_width
    y1 = y - crop_half_height
    y2 = y + crop_half_height

    cropped_flooring = flooring[y1:y2, x1:x2]
    cropped_flooring = cv2.resize(cropped_flooring, (512, 1024), interpolation=cv2.INTER_AREA)

    # Seamless clone
    return cv2.seamlessClone(
        shoemark,
        cropped_flooring,
        mask,
        (256, 512),
        cv2.MIXED_CLONE,
    )


flooring_image_dir = Path("flooring/")
shoemark_image_dir = Path("example_shoemarks/")

flooring_images = [f for f in flooring_image_dir.iterdir() if f.is_file()]
shoemark_images = [f for f in shoemark_image_dir.iterdir() if f.is_file()]


def display(difficulty: float):
    """Display blended images as MPL figure."""
    # Read images
    rand_floor = random.choice(flooring_images)
    rand_shopeprint = next(shoeprint_cycle).to(device)

    s = mapping_network.get_single_w(
        batch_size=1,
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        mix_styles=False,
        domain_variable=difficulty,
    )

    shoemark = generator(rand_shopeprint, s)[0].permute(1, 2, 0).cpu().numpy()
    shoemark = (shoemark * 255).astype(np.uint8)
    shoemark = cv2.cvtColor(shoemark, cv2.COLOR_RGB2BGR)

    flooring = cv2.imread(str(rand_floor))
    shoemark = cv2.resize(shoemark, (512, 1024), interpolation=cv2.INTER_AREA)

    blended = blend(flooring, shoemark)

    plt.axis("off")
    plt.tight_layout()
    plt.imshow(blended)
    plt.show()
