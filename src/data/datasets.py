"""Load datasets using torch.utils.data.Dataset."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

_dataset_mode = Literal["train", "test", "val", "is_val"]


def dataset_transform(
    image_size: tuple[int, int],
    mean: float | tuple[float, float, float],
    std: float | tuple[float, float, float],
):
    """Initialise transforms for a dataset."""
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    return transforms.Compose(transform_list)


def no_norm_transform(
    image_size: tuple[int, int],
):
    """Transform without normalising, for calculating dataset statistics."""
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]

    return transforms.Compose(transform_list)


def calculate_stats(
    loader: torch.utils.data.DataLoader,
    num_channels: int = 1,
):
    """Calculate per-channel mean and std using explicit sum of squares."""
    sum_pixels = torch.zeros(num_channels)
    sum_squares = torch.zeros(num_channels)
    total_pixels = 0

    for image in tqdm(loader):
        # Mean over batch, height and width, but not over channels

        flattened = image.flatten(start_dim=2)  # [B, C, H*W]

        # Accumulate statistics
        sum_pixels += flattened.sum(dim=(0, 2))
        sum_squares += (flattened**2).sum(dim=(0, 2))
        total_pixels += flattened.shape[0] * flattened.shape[2]

    # Final calculations
    mean = sum_pixels / total_pixels
    std = torch.sqrt((sum_squares / total_pixels) - (mean**2))

    return mean, std


class ShoeDataset(Dataset):
    """Load shoe images into RAM."""

    def __init__(
        self,
        path: Path | str,
        *,
        mode: _dataset_mode,
        transform,
        flip_prob: float = 0.5,
    ):
        path = Path(path).expanduser() / mode

        jpg_files = list(path.rglob("*.jpg"))
        png_files = list(path.rglob("*.png"))

        image_files = jpg_files + png_files

        if len(image_files) == 0:
            raise FileNotFoundError

        images = []

        for image_file in image_files:
            image = Image.open(image_file)
            image = transform(image)

            images.append(image)

        self.images = images
        self.hflipper = transforms.RandomHorizontalFlip(flip_prob)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        images = self.images[idx]
        return self.hflipper(images)


class LabeledShoeDataset(Dataset):
    """Load shoe images into RAM. Returns (image, class_id) pairs."""

    def __init__(
        self,
        path: Path | str,
        *,
        mode: _dataset_mode | None,
        transform,
        flip_prob: float = 0.5,
    ):
        path = Path(path)

        if mode:
            path = path.expanduser() / mode

        image_files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))

        if len(image_files) == 0:
            raise FileNotFoundError

        self.image_classes = defaultdict(list)

        for image_file in image_files:
            image_class = int(image_file.stem.split("_")[0])
            image = Image.open(image_file)
            self.image_classes[image_class].append(image)

        self.image_classes_keys = list(self.image_classes.keys())
        self.transform = transform
        self.hflipper = transforms.RandomHorizontalFlip(flip_prob)

    def __len__(self):
        return len(self.image_classes.keys())

    def __getitem__(self, idx: int):
        shoe_class = self.image_classes_keys[idx]
        image = random.choice(self.image_classes[shoe_class])

        image = self.transform(image)

        return (self.hflipper(image), shoe_class)  # Return two elements


class Edges2ShoesDataset(Dataset):
    """Load shoe images into RAM."""

    def __init__(
        self,
        path: Path | str,
        *,
        mode: _dataset_mode,
        transform,
        type_: Literal["edge", "shoe"],
    ):
        path = Path(path).expanduser() / mode

        jpg_files = list(path.rglob("*.jpg"))
        png_files = list(path.rglob("*.png"))

        image_files = jpg_files + png_files

        if len(image_files) == 0:
            raise FileNotFoundError

        images = []

        for image_file in image_files:
            image = Image.open(image_file)

            image = (
                image.crop((0, 0, 256, 256)) if type_ == "edge" else image.crop((256, 0, 512, 256))
            )

            image = transform(image)
            images.append(image)

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]
