"""Miscellaneous utility functions."""

import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

# * Training


def save_grid(step: int, images: list[list[torch.Tensor]], save_path: Path | str):
    """Save a grid of generated images to a file."""
    save_path = Path(save_path)

    def process_image(image: torch.Tensor):
        image = image.permute(0, 2, 3, 1)
        image = (image - image.min()) / (image.max() - image.min())
        image = image[0]

        return image.cpu().numpy()

    images_np = [[process_image(image) for image in row] for row in images]

    plt.ioff()

    _, axes = plt.subplots(nrows=9, ncols=8, figsize=(8, 9))

    for row_idx in range(9):
        for col_idx in range(8):
            axes[row_idx, col_idx].imshow(
                images_np[col_idx][row_idx], cmap="gray"
            )  # Note that images_np is col major, and axes_np is row major
            axes[row_idx, col_idx].set_axis_off()

    save_path.mkdir(exist_ok=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path / f"{step}.png", dpi=300, bbox_inches="tight")

    plt.close()
    plt.ion()


class Logger:
    """Keep track of losses/accs."""

    def __init__(self, training_steps: int):
        self.training_steps = training_steps

        self.initialise_trackers()

    def initialise_trackers(self):
        self.log_total_disc_losses = []
        self.log_disc_real_accs = []
        self.log_disc_fake_accs = []
        self.log_total_gen_losses = []
        self.log_gan_losses = []
        self.log_idt_losses = []
        self.log_rec_losses = []
        self.log_kl_losses = []
        self.log_path_losses = []
        self.log_style_losses = []
        self.log_ada_ps = []

    def print(self, step: int):
        calc_mean = lambda x: np.mean(x)

        string = (
            f"Step: {step}/{self.training_steps}, "
            f"D loss: {calc_mean(self.log_total_disc_losses):.6g}, "
            f"D real/fake acc: {calc_mean(self.log_disc_real_accs):.6g}"
            f"/{calc_mean(self.log_disc_fake_accs):.6g}, "
            f"Total G loss: {calc_mean(self.log_total_gen_losses):.6g}, "
            f"Gan loss {calc_mean(self.log_gan_losses):.6g} "
            f"Idt loss {calc_mean(self.log_idt_losses):.6g}, "
            f"Rec loss {calc_mean(self.log_rec_losses):.6g}, "
            f"KL loss {calc_mean(self.log_kl_losses):.6g}, "
            f"Path loss {calc_mean(self.log_path_losses):.6g}, "
            f"Style loss: {calc_mean(self.log_style_losses):.6g}, "
            f"ADA: {calc_mean(self.log_ada_ps):.6g}, "
        )

        self.initialise_trackers()

        return string


# Adapted from CycleGAN
class ImageBuffer:
    """An image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    buffer_size: int
    num_imgs: int
    images: list[torch.Tensor]

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        if self.buffer_size < 1:
            raise ValueError

        self.num_imgs = 0
        self.images = []

    def __call__(self, images: torch.Tensor):
        return_images = []

        for image in images:
            image_unsqueezed = torch.unsqueeze(image.detach(), 0)
            # Fill buffer if it is not full
            if self.num_imgs < self.buffer_size:
                self.num_imgs += 1
                self.images.append(image_unsqueezed)
                return_images.append(image_unsqueezed)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(
                        0, self.buffer_size - 1
                    )  # randint is inclusive
                    # Clone tensors as they may be used many times
                    cloned_image = self.images[random_id].clone()
                    self.images[random_id] = image_unsqueezed
                    return_images.append(cloned_image)
                else:
                    return_images.append(image_unsqueezed)

        return torch.cat(return_images, 0)
