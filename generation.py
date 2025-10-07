"""Generate images into directory."""

import random
from pathlib import Path

import torch
import torchvision
from seamless_clone import clone
from tqdm import tqdm

from src.core.generate import GeneratorHandler
from src.data.config import load_config
from src.data.datasets import LabeledShoeDataset, dataset_transform

config = load_config("config.toml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = GeneratorHandler(config, device)

random.seed(config["training"]["random_seed"])

flooring_images = list(config["generation"]["flooring_dir"].glob("*"))
flooring_images = [str(f) for f in flooring_images if f.is_file()]

transform = dataset_transform(config["data"]["image_size"], *config["data"]["shoeprint_norm"])
shoeprint_data = LabeledShoeDataset(
    config["data"]["shoeprint_data_dir"], mode="train", transform=transform, flip_prob=0
)
shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=config["inference"]["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)


def generate(number: int, output_dir: Path | str, min_difficulty: float, max_difficulty: float):
    """Generate shoemarks and fill background with flooring images."""
    output_dir = Path(output_dir)

    work = number * len(shoeprint_dataloader) * config["inference"]["batch_size"]

    with tqdm(total=work) as pbar:
        for epoch in range(number):
            for shoeprints, labels in shoeprint_dataloader:
                difficulty = random.uniform(min_difficulty, max_difficulty)

                shoemarks = generator.generate(shoeprints.to(device), difficulty).cpu()

                floor_images = random.sample(flooring_images, shoemarks.shape[0])

                # batch, channel, height, width
                shoemarks_with_background = clone(
                    shoemarks.permute(0, 2, 3, 1).contiguous(), floor_images
                )

                shoemarks_with_background = shoemarks_with_background.permute(0, 3, 1, 2)

                for shoemark, label in zip(shoemarks_with_background, labels, strict=True):
                    torchvision.utils.save_image(
                        shoemark,
                        output_dir / f"{label}_{epoch}.png",
                    )

                pbar.update(config["inference"]["batch_size"])


if __name__ == "__main__":
    generate(40, "/home/struan/Vault/University/Doctorate/Data/Siamese/Shoemarks/train/", 0.75, 1)
