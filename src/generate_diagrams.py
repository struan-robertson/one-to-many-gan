"""Generate diagrams for the paper."""

import random
from pathlib import Path
from typing import cast

import torch
import torchvision
from tqdm import tqdm

from one_to_many_gan.core.generate import GeneratorHandler
from one_to_many_gan.data.config import load_config
from one_to_many_gan.data.datasets import ShoeDataset, dataset_transform

config = load_config("../config.toml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = GeneratorHandler(config, device)

random.seed(config["training"]["random_seed"])

transform = dataset_transform(
    config["data"]["image_size"],
    *config["data"]["shoeprint_norm"],
    random_image_flip=False,
)
shoeprint_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"],
    mode="train",
    transform=transform,
    channels=config["data"]["image_channels"],
)
shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)


def single_shoeprint(
    rows: int, columns: int, output_path: Path | str, styles: torch.Tensor | None = None
):
    """Generate a diagram of a single shoeprint translated to many shoemarks."""
    output_path = Path(output_path)

    if styles is None:
        styles = generator.get_style(rows, 1.0)

        style_steps = [(i + 1) / columns for i in range(columns)]

        styles = torch.stack([styles * style_step for style_step in style_steps])

    shoeprint = next(iter(shoeprint_dataloader)).squeeze(0)

    shoeprints = torch.stack([shoeprint] * rows).to(device)

    col_tensors = torch.stack([generator.generate(shoeprints, style=style) for style in styles])  # pyright: ignore [reportAssignmentType]

    # 1. Permute to [rows, cols, c, h, w]
    # 2. Reshape to [rows * cols, c, h, w]
    # 3. Set nrow to the original number of columns
    grid = col_tensors.permute(1, 0, 2, 3, 4)
    grid = grid.reshape(-1, config["data"]["image_channels"], *config["data"]["image_size"])

    torchvision.utils.save_image(grid, output_path, nrow=columns)

    return styles


def multiple_shoemarks(rows: int, columns: int, output_path: Path | str):
    """Generate a diagram of a single shoeprint translated to many shoemarks."""
    output_path = Path(output_path)

    style = generator.get_style(1, 1.0)
    style = style.expand(-1, rows * columns, -1)

    shoeprints = torch.stack(
        [next(iter(shoeprint_dataloader)).squeeze(0) for _ in range(rows * columns)]
    ).to(device)

    shoemarks = generator.generate(shoeprints, style=style)

    torchvision.utils.save_image(shoemarks, output_path, nrow=columns)
