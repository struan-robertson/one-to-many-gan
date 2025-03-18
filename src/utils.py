"""Miscellaneous utility functions."""

import functools
import random
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
        self.print_steps = 0
        self.training_steps = training_steps

        self.initialise_trackers()

    def initialise_trackers(self):
        self.log_disc_losses = 0.0
        self.log_disc_real_acc = 0.0
        self.log_disc_fake_acc = 0.0
        self.log_gen_loss = 0.0
        self.log_style_loss = 0.0
        self.log_ada_p = 0.0

    def step(self):
        self.n_steps += 1
        self.print_steps += 1

    def print(self):
        calc_mean = lambda x: x / self.print_steps

        string = (
            f"Step: {self.n_steps}/{self.training_steps}, "
            f"D loss: {calc_mean(self.log_disc_losses):.6g}, "
            f"D real/fake acc: {calc_mean(self.log_disc_real_acc):.6g}"
            f"/{calc_mean(self.log_disc_fake_acc):.6g}, "
            f"G loss: {calc_mean(self.log_gen_loss):.6g}, "
            f"Style loss: {calc_mean(self.log_style_loss):.6g}, "
            f"ADA: {calc_mean(self.log_ada_p):.6g}, "
        )

        self.initialise_trackers()
        self.print_steps = 0

        return string


class ImageBuffer:
    """An image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    buffer_size: int
    num_imgs: int
    images: list[torch.Tensor]
    styles: list[torch.Tensor]

    def __init__(self, pool_size: int):
        self.buffer_size = pool_size

        if self.buffer_size < 1:
            raise ValueError

        self.num_imgs = 0
        self.images = []

    def __call__(self, images: torch.Tensor, styles: torch.Tensor):
        return_images = []
        return_styles = []

        for image, style in zip(images, styles, strict=True):
            # Fill buffer if it is not full
            if self.num_imgs < self.buffer_size:
                self.num_imgs += 1
                self.images.append(image)
                self.styles.append(style)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(
                        0, self.buffer_size - 1
                    )  # randint is inclusive
                    # Clone tensors as they may be used many times
                    cloned_image = self.images[random_id].clone()
                    cloned_style = self.styles[random_id].clone()
                    self.images[random_id] = image
                    self.styles[random_id] = style
                    return_images.append(cloned_image)
                    return_styles.append(cloned_style)
                else:
                    return_images.append(image)
                    return_styles.append(style)

        return torch.stack(return_images, 0), torch.stack(return_styles, 0)
