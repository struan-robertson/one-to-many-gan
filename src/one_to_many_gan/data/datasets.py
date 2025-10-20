"""Load datasets using torch.utils.data.Dataset."""

import random
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_dataset_mode = Literal["train", "test", "val"]


def dataset_transform(
    image_size: tuple[int, int],
    mean: float | tuple[float, float, float],
    std: float | tuple[float, float, float],
    *,
    random_image_flip: bool = True,
):
    """Initialise transforms for a dataset."""
    transform_list = [
        transforms.Resize(image_size),
        transforms.Normalize(mean, std),  # pyright: ignore [reportArgumentType]
    ]

    if random_image_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


class CyclingDataLoader:
    """Infinitely cycle dataloader with reshuffling."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


class ShoeDataset(Dataset):
    """Shoe dataset loaded entirely into RAM."""

    def __init__(
        self,
        path: Path | str,
        *,
        mode: _dataset_mode | None,
        transform,
    ):
        path = Path(path)

        if mode:
            path = path.expanduser() / mode

        image_files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))

        if len(image_files) == 0:
            raise FileNotFoundError

        self.images = []
        for image_file in image_files:
            image = transforms.ToTensor()(Image.open(image_file))
            self.images.append(image)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]

        return self.transform(image)

    def random_sample(self, n: int):
        images = random.sample(self.images, n)
        images = [self.transform(image) for image in images]

        return torch.stack(images)
