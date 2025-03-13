"""Miscellaneous utility functions."""

import functools
from pathlib import Path

import torch
from matplotlib import pyplot as plt

# I want it to be explicit if I am compiling a function that cannot be fully compiled
compile_ = functools.partial(
    torch.compile,
    fullgraph=True,
)


def save_grid(step: int, images: list[list[torch.Tensor]]):
    """Save a grid of generated images to a file."""

    def process_image(image: torch.Tensor):
        image = image.permute(0, 2, 3, 1)
        image = (image - image.min()) / (image.max() - image.min())
        image = image[0]

        return image.cpu().numpy()

    images_np = [[process_image(image) for image in row] for row in images]

    plt.ioff()

    _, axes = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))

    for row_idx in range(4):
        for col_idx in range(8):
            axes[row_idx, col_idx].imshow(
                images_np[col_idx][row_idx], cmap="gray"
            )  # Note that images_np is col major, and axes_np is row major
            axes[row_idx, col_idx].set_axis_off()

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
            f"ADA: {calc_mean(self.log_ada_p):.6g}, "
            f"Siamese loss: {calc_mean(self.log_siamese_loss):.6g}, "
            f"S positive/fake acc {calc_mean(self.log_siamese_positive_accuracy):.6g}"
            f"/{calc_mean(self.log_siamese_negative_accuracy):.6g}"
        )

        self.initialise_trackers()

        return string
