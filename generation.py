"""Generate images into directory."""

import itertools
import random
from pathlib import Path

import torch
import torchvision
from seamless_clone import clone
from tqdm import tqdm, trange

from src.core.generate import GeneratorHandler
from src.data.config import load_config
from src.data.datasets import LabeledShoeDataset, dataset_transform

config = load_config("config.toml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = GeneratorHandler(config, device, config["inference"]["batch_size"])

random.seed(config["training"]["random_seed"])

flooring_images = list(config["generation"]["flooring_dir"].glob("*"))
flooring_images = [str(f) for f in flooring_images if f.is_file()]

transform = dataset_transform(
    config["data"]["image_size"], config["data"]["norm_mean"], config["data"]["norm_std"]
)
shoeprint_data = LabeledShoeDataset(
    config["data"]["shoeprint_data_dir"], mode="val", transform=transform
)
shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=config["inference"]["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True,
)
shoeprint_cycle = itertools.cycle(shoeprint_dataloader)


def generate(number: int, output_dir: Path | str, min_difficulty: float, max_difficulty: float):
    """Generate shoemarks and fill background with flooring images."""
    output_dir = Path(output_dir)

    steps = number // config["inference"]["batch_size"]

    epoch = 0
    for i in trange(steps):
        shoeprints, labels = next(shoeprint_cycle)

        difficulty = random.uniform(min_difficulty, max_difficulty)

        shoemarks = generator.generate_from_shoeprints(shoeprints.to(device), difficulty)
        shoemarks = shoemarks.expand(config["inference"]["batch_size"], 3, 512, 256).cpu()

        floor_images = random.sample(flooring_images, config["inference"]["batch_size"])

        # batch, channel, height, width
        shoemarks_with_background = clone(shoemarks.permute(0, 2, 3, 1).contiguous(), floor_images)

        shoemarks_with_background = shoemarks_with_background.permute(0, 3, 1, 2)

        for label, shoemark in zip(labels, shoemarks_with_background, strict=True):
            torchvision.utils.save_image(shoemark, output_dir / f"{label}_{epoch}.png")

        if i != 0 and i % len(shoeprint_dataloader) == 0:
            epoch += 1
