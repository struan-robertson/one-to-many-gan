"""Generate synthetic data."""

import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(10)


def blend(flooring: np.ndarray, shoemark: np.ndarray):
    """Blend a flooring and shoemark image."""
    # Select scale of floor image rectangle w.r.t the shoemark
    max_scale = min(flooring.shape[0] / shoemark.shape[0], flooring.shape[1] / shoemark.shape[1])
    scale = rng.uniform(1, max_scale)

    scaled_width = math.floor(shoemark.shape[1] * scale)
    scaled_height = math.floor(shoemark.shape[0] * scale)

    # Create an all white mask
    mask = 255 * np.ones((512, 256), shoemark.dtype)

    # Select random co-ordinates of floor rectangle
    min_x = math.ceil(scaled_width / 2)
    max_x = flooring.shape[1] - min_x
    min_y = math.ceil(scaled_height / 2)
    max_y = flooring.shape[0] - min_y

    x = rng.integers(min_x, max_x)
    y = rng.integers(min_y, max_y)

    crop_half_width = scaled_width // 2
    crop_half_height = scaled_height // 2
    x1 = x - crop_half_width
    x2 = x + crop_half_width
    y1 = y - crop_half_height
    y2 = y + crop_half_height

    cropped_flooring = flooring[y1:y2, x1:x2]
    cropped_flooring = cv2.resize(cropped_flooring, (256, 512), interpolation=cv2.INTER_AREA)

    # Seamless clone
    return cv2.seamlessClone(
        shoemark,
        cropped_flooring,
        mask,
        (128, 256),
        cv2.MIXED_CLONE,
    )


flooring_image_dir = Path("flooring/")
shoemark_image_dir = Path("example_shoemarks/")

flooring_images = [f for f in flooring_image_dir.iterdir() if f.is_file()]
shoemark_images = [f for f in shoemark_image_dir.iterdir() if f.is_file()]


def display():
    """Display blended images as MPL figure."""
    # Read images
    rand_floor = random.choice(flooring_images)
    rand_shoemark = random.choice(shoemark_images)
    flooring = cv2.imread(str(rand_floor))
    shoemark = cv2.imread(str(rand_shoemark))

    blended = blend(flooring, shoemark)

    plt.axis("off")
    plt.tight_layout()
    plt.imshow(blended)
    plt.show()


display()
