"""Miscellaneous utility functions."""

import functools
import operator
from pathlib import Path

import torch
from matplotlib import pyplot as plt


def save_grid(step: int, images: torch.Tensor):
    """Save a grid of generated images to a file."""
    # Put colour channel as final dim
    images = images.permute(0, 2, 3, 1)

    # Scale to between 0 and 1
    images = (images - images.min()) / (images.max() - images.min())

    # Convert to numpy and move to cpu
    images_np = images.cpu().numpy()

    plt.ioff()

    _, axes = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))

    for ax, image in zip(
        functools.reduce(operator.iadd, axes.tolist(), []),
        images_np,
        strict=True,
    ):
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()

    savepath: Path = Path("./checkpoints")
    savepath.mkdir(exist_ok=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(savepath / f"{step}.png", dpi=300, bbox_inches="tight")

    plt.close()
    plt.ion()
