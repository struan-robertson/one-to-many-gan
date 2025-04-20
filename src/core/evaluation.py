"""Classes and methods used for evaluation."""

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torchvision
from cleanfid import fid
from matplotlib import pyplot as plt
from src.core.training import ImageBuffer
from src.data.config import Config
from src.model.builder import Discriminator, Generator, MappingNetwork, StyleExtractor
from src.model.loss import ADAp
from tqdm import tqdm, trange

# * Checkpoints

# ** Validation


def val_checkpoint(
    step: int,
    config: Config,
    device: torch.device,
    shoeprint_val_iter: Iterator[torch.Tensor],
    mapping_network: MappingNetwork,
    generator: Generator,
):
    """Calculate FID and KID scores and save to checkpoint."""
    val_checkpoint_dir = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "val"
    )
    val_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    for _ in trange(
        math.ceil(
            config["evaluation"]["n_evaluation_images"]
            / config["evaluation"]["inference_batch_size"]
        ),
        desc="Generating shoemarks",
        leave=False,
    ):
        shoeprints = next(shoeprint_val_iter).to(device)
        w = mapping_network.get_single_w(
            batch_size=config["evaluation"]["inference_batch_size"],
            n_gen_blocks=generator.n_style_blocks,
            device=device,
            mix_styles=False,
            domain_variable=1,
        )

        val_shoemarks = generator(shoeprints, w)

        for shoemark in val_shoemarks:
            torchvision.utils.save_image(shoemark, val_checkpoint_dir / f"{i}.png")
            i += 1

    shoemark_train_dir = config["data"]["shoemark_data_dir"] / "train"
    fid_score = fid.compute_fid(
        str(val_checkpoint_dir), str(shoemark_train_dir), verbose=False, device=device
    )
    kid_score = fid.compute_kid(
        str(val_checkpoint_dir), str(shoemark_train_dir), verbose=False, device=device
    )

    log = f"Step {step + 1} val | fid: {fid_score}, kid: {kid_score}"
    tqdm.write(log)

    log_checkpoint_dir = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "log"
    )
    with log_checkpoint_dir.open("a") as file:
        file.write(log + "\n")


# ** Image


def save_grid(
    images: list[list[torch.Tensor]],
    save_path: Path | str,
    grid_size: tuple[int, int],
):
    """Save a grid of generated images to a file."""
    save_path = Path(save_path)

    def process_image(image: torch.Tensor):
        image = image.permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        return image.cpu().numpy()

    images_np = [[process_image(image) for image in row] for row in images]

    plt.ioff()

    rows, cols = grid_size

    _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))

    for row_idx in range(rows):
        for col_idx in range(cols):
            axes[row_idx, col_idx].imshow(
                images_np[col_idx][row_idx], cmap="gray"
            )  # Note that images_np is col major, and axes_np is row major
            axes[row_idx, col_idx].set_axis_off()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()
    plt.ion()


def image_checkpoint(
    step: int,
    config: Config,
    device: torch.device,
    shoeprint_iter: Iterator[torch.Tensor],
    shoemark_iter: Iterator[torch.Tensor],
    mapping_network: MappingNetwork,
    generator: Generator,
    style_extractor: StyleExtractor,
):
    """Generate and save checkpoint grid images."""
    w = mapping_network.get_single_w(
        batch_size=8,
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        mix_styles=False,
        domain_variable=1,
    )

    image_checkpoint_dir = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "images"
    )
    image_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Collect enough shoeprints and shoemarks
    if config["training"]["batch_size"] < 8:
        real_shoeprint_images = [
            next(shoeprint_iter).to(device)
            for _ in range(math.ceil(8 / config["training"]["batch_size"]))
        ]
        real_shoemark_images = [
            next(shoemark_iter).to(device)
            for _ in range(math.ceil(8 / config["training"]["batch_size"]))
        ]

        real_shoeprint_images = torch.cat(real_shoeprint_images, dim=0)
        real_shoemark_images = torch.cat(real_shoemark_images, dim=0)
    else:
        real_shoeprint_images = next(shoeprint_iter).to(device)
        real_shoemark_images = next(shoemark_iter).to(device)

    real_shoeprint_images = real_shoeprint_images[:8]
    real_shoemark_images = real_shoemark_images[:8]

    shoeprint_latents = generator.encode(real_shoeprint_images)
    shoemark_latents = generator.encode(real_shoemark_images)

    translation_grid_images = []
    for column in range(8):
        column_images = [real_shoeprint_images[column]]
        column_images += [
            *generator.decode(shoeprint_latents[column].expand(8, -1, -1, -1), w)
        ]
        translation_grid_images.append(column_images)

    save_grid(
        translation_grid_images,
        image_checkpoint_dir / f"translation_{step + 1}.png",
        (9, 8),
    )

    w0 = torch.zeros(
        (
            generator.n_style_blocks,
            8,
            config["architecture"]["w_dim"],
        ),
        device=device,
    )
    reconstructed_shoeprints = generator.decode(shoeprint_latents, w0)

    real_shoemark_w = style_extractor(real_shoemark_images)
    reconstructed_shoemarks = generator.decode(
        shoemark_latents,
        real_shoemark_w.expand(generator.n_style_blocks, *real_shoemark_w.shape),
    )

    translated_shoemarks = generator.decode(
        shoeprint_latents,
        real_shoemark_w.expand(generator.n_style_blocks, *real_shoemark_w.shape),
    )

    decoding_grid = [
        [
            real_shoeprint_images[column],
            reconstructed_shoeprints[column],
            translated_shoemarks[column],
            real_shoemark_images[column],
            reconstructed_shoemarks[column],
        ]
        for column in range(8)
    ]

    save_grid(
        decoding_grid,
        image_checkpoint_dir / f"decoding_{step + 1}.png",
        (5, 8),
    )


# * Model


def model_checkpoint(
    step: int,
    config: Config,
    generator: Generator,
    discriminator: Discriminator,
    mapping_network: MappingNetwork,
    style_extractor: StyleExtractor,
    generator_optimiser: torch.optim.Optimizer,
    discriminator_optimiser: torch.optim.Optimizer,
    mapping_network_optimiser: torch.optim.Optimizer,
    style_extractor_optimiser: torch.optim.Optimizer,
    ada_p: ADAp,
    image_buffer: ImageBuffer,
):
    """Save all network training state to file."""
    models_checkpoint_dir = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "models"
    )
    models_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "generator_optim_state_dict": generator_optimiser.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "discriminator_optim_state_dict": discriminator_optimiser.state_dict(),
            "mapping_network_state_dict": mapping_network.state_dict(),
            "mapping_network_optim_state_dict": mapping_network_optimiser.state_dict(),
            "style_extractor_state_dict": style_extractor.state_dict(),
            "style_extractor_optim_state_dict": style_extractor_optimiser.state_dict(),
            "ada_p": ada_p(),
            "image_buffer_images": image_buffer.images,
            "image_buffer_size": image_buffer.buffer_size,
        },
        models_checkpoint_dir / f"{step + 1}.tar",
    )


# * Logger


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
        string = (
            f"Step: {step}/{self.training_steps}, "
            f"D loss: {np.mean(self.log_total_disc_losses):.6g}, "
            f"D real/fake acc: {np.mean(self.log_disc_real_accs):.6g}"
            f"/{np.mean(self.log_disc_fake_accs):.6g}, "
            f"Total G loss: {np.mean(self.log_total_gen_losses):.6g}, "
            f"Gan loss {np.mean(self.log_gan_losses):.6g}, "
            f"Idt loss {np.mean(self.log_idt_losses):.6g}, "
            f"Rec loss {np.mean(self.log_rec_losses):.6g}, "
            f"KL loss {np.mean(self.log_kl_losses):.6g}, "
            f"Path loss {np.mean(self.log_path_losses):.6g}, "
            f"Style loss: {np.mean(self.log_style_losses):.6g}, "
            f"ADA: {np.mean(self.log_ada_ps):.6g}, "
        )

        self.initialise_trackers()

        return string
