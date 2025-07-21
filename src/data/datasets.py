"""Load datasets using torch.utils.data.Dataset."""

from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_dataset_mode = Literal["train", "test", "val"]


def dataset_transform(image_size: tuple[int, int]):
    """Initialise transforms for a dataset."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


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
