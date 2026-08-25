"""Load datasets using torch.utils.data.Dataset."""

import random
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_dataset_mode = Literal["train", "test", "val"]
_to_tensor = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)


def dataset_transform(
    image_size: tuple[int, int],
    mean: float,
    std: float,
    *,
    random_image_flip: bool = True,
):
    """Initialise transforms for a dataset."""
    transform_list = [
        transforms.Resize(image_size),
        transforms.Normalize([mean], [std]),
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
    """Shoe dataset that loads images lazily from disk."""

    def __init__(
        self,
        path: Path | str,
        *,
        mode: _dataset_mode | None,
        transform,
        channels: int = 3,
    ):
        path = Path(path)

        if mode:
            path = path.expanduser() / mode

        self.image_files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))

        if len(self.image_files) == 0:
            raise FileNotFoundError

        self.transform = transform
        self._pil_mode = "RGB" if channels == 3 else "L"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image = _to_tensor(Image.open(self.image_files[idx]).convert(self._pil_mode))

        return self.transform(image)

    def random_sample(self, n: int):
        files = random.sample(self.image_files, n)
        images = [self.transform(_to_tensor(Image.open(f).convert(self._pil_mode))) for f in files]

        return torch.stack(images)
