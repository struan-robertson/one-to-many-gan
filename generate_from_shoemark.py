"""Fill in backgrounds on shoemark images."""

import random
from pathlib import Path

import torch
import torchvision
from seamless_clone import clone
from torchvision import transforms
from tqdm import tqdm

from src.data.config import load_config
from src.data.datasets import LabeledShoeDataset

config = load_config("config.toml")

input_dir = Path("/home/struan/Vault/University/Doctorate/Data/Siamese/Impress/No Background/train")
output_dir = Path(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Impress/No Background (substituted)/train"
)
output_dir.mkdir(exist_ok=True)


flooring_images = list(config["generation"]["flooring_dir"].glob("*"))
flooring_images = [str(f) for f in flooring_images if f.is_file()]

dataset = LabeledShoeDataset(input_dir, mode=None, transform=transforms.ToTensor(), flip_prob=0)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)

for number in tqdm(range(40)):
    for shoemarks, filenames in dataloader:
        floor_images = random.sample(flooring_images, shoemarks.shape[0])
        shoemarks_with_background = clone(shoemarks.permute(0, 2, 3, 1).contiguous(), floor_images)
        shoemarks_with_background = shoemarks_with_background.permute(0, 3, 1, 2)

        for shoemark, mark_class in zip(shoemarks_with_background, filenames, strict=True):
            torchvision.utils.save_image(shoemark, output_dir / f"{mark_class}_{number}.png")
