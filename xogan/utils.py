"""Miscellaneous utility functions."""

import functools
import operator
from pathlib import Path

import torch
from matplotlib import pyplot as plt

# I want it to be explicit if I am compiling a function that cannot be fully compiled
compile_ = functools.partial(
    torch.compile,
    fullgraph=True,
)


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


class Logger:
    """Keep track of losses/accs."""

    def __init__(self, training_steps: int):
        self.n_steps = 0
        self.training_steps = training_steps

        self.initialise_trackers()

    def initialise_trackers(self):
        self.log_disc_losses = 0.0
        self.log_disc_real_acc = 0.0
        self.log_disc_fake_acc = 0.0
        self.log_gen_loss = 0.0
        self.log_siamese_loss = 0.0
        self.log_siamese_positive_accuracy = 0.0
        self.log_siamese_negative_accuracy = 0.0
        self.log_ada_p = 0.0

    def step(self):
        self.n_steps += 1

    def print(self):
        calc_mean = lambda x: x / self.n_steps

        string = (
            f"Step: {self.n_steps}/{self.training_steps}, "
            f"Generator loss: {calc_mean(self.log_gen_loss):.6g}, "
            f"Discriminator loss: {calc_mean(self.log_disc_losses):.6g}, "
            f"D real/fake acc: {calc_mean(self.log_disc_real_acc):.6g}"
            f"/{calc_mean(self.log_disc_fake_acc):.6g}, "
            f"ADA: {calc_mean(self.log_ada_p)}, "
            f"Siamese loss: {calc_mean(self.log_siamese_loss):.6g}, "
            f"S positive/fake acc {calc_mean(self.log_siamese_positive_accuracy):.6g}"
            f"/{calc_mean(self.log_siamese_negative_accuracy):.6g}"
        )

        self.initialise_trackers()

        return string
