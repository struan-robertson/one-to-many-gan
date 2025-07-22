"""Load datasets using torch.utils.data.Dataset."""

import random
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

_dataset_mode = Literal["train", "test", "val"]


def dataset_transform(image_size: tuple[int, int]):
    """Initialise transforms for a dataset."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4779, 0.5329, 0.5856), (0.1842, 0.1935, 0.2015)
            ),  # Calculated for the synthetic dataset
            # transforms.Normalize(0.5, 0.5), # Used for GAN
        ]
    )


def calculate_stats(loader):
    mean = 0.0
    std = 0.0
    total_samples = 0.0

    for shoeprints, shoemarks in tqdm(loader):
        shoemarks_shaped = shoemarks.view(
            -1, shoemarks.shape[2], shoemarks.shape[3], shoemarks.shape[4]
        )
        batched_tensors = torch.cat([shoeprints, shoemarks_shaped], dim=0)
        batched_tensors = batched_tensors.view(
            batched_tensors.shape[0], batched_tensors.shape[1], -1
        )
        mean += batched_tensors.mean(2).sum(0)
        std += batched_tensors.std(2).sum(0)
        total_samples += batched_tensors.shape[0]

    mean /= total_samples
    std /= total_samples

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


class LabeledCombinedDataset(Dataset):
    """Load shoeprint and shoemark images. Returns (shoeprint, (shoemarks)) tuples."""

    def __init__(
        self,
        shoeprint_path: Path | str,
        shoemark_path: Path | str,
        *,
        mode: _dataset_mode,
        transform,
        flip_prob: float = 0.5,
        sample_no: int = 5,
    ):
        shoeprint_path = Path(shoeprint_path).expanduser() / mode
        shoemark_path = Path(shoemark_path).expanduser() / mode

        shoeprint_jpg_files = list(shoeprint_path.rglob("*.jpg"))
        shoeprint_png_files = list(shoeprint_path.rglob("*.png"))
        self.shoeprint_files = shoeprint_jpg_files + shoeprint_png_files

        shoemark_jpg_files = list(shoemark_path.rglob("*.jpg"))
        shoemark_png_files = list(shoemark_path.rglob("*.png"))

        self.shoemark_files = {f.stem: f for f in shoemark_jpg_files + shoemark_png_files}
        self.transform = transform
        self.hflipper = transforms.RandomHorizontalFlip(flip_prob)
        self.sample_no = sample_no
        self.mode = mode

    def __len__(self):
        return len(self.shoeprint_files)

    def __getitem__(self, idx: int):
        shoeprint = self.shoeprint_files[idx]
        shoeprint_name = shoeprint.stem
        shoeprint_image = Image.open(shoeprint).convert("RGB")

        if self.mode == "train":
            shoemarks = random.sample(
                [
                    self.transform(Image.open(file))
                    for key, file in self.shoemark_files.items()
                    if shoeprint_name in key
                ],
                self.sample_no,
            )
            shoemarks = torch.stack(shoemarks)
        else:
            shoemarks = [file for key, file in self.shoemark_files.items() if shoeprint_name in key]
            shoemarks = self.transform(Image.open(shoemarks[0]))

        shoeprint_image = self.transform(shoeprint_image)

        return (shoeprint_image, shoemarks)


class LabeledShoeDataset(Dataset):
    """Load shoe images into RAM. Returns (image, filename) pairs."""

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

        self.image_files = jpg_files + png_files  # Store filenumerate

        if len(self.image_files) == 0:
            raise FileNotFoundError

        images = []
        for image_file in self.image_files:
            image = Image.open(image_file)
            images.append(transform(image))

        self.images = images
        self.hflipper = transforms.RandomHorizontalFlip(flip_prob)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        filename = self.image_files[idx].stem  # Extract filename
        return (self.hflipper(image), filename)  # Return two elements


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
