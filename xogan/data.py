"""Load datasets using torch.utils.data.Dataset."""

from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

_dataset_mode = Literal["train", "test", "val"]


class ShoeDataset(Dataset):
    """Load shoe images into RAM."""

    def __init__(self, path: Path | str, *, mode: _dataset_mode, transform):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]
