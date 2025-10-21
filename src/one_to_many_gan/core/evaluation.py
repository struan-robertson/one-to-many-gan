"""Classes and methods used for evaluation."""

import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from cleanfid import fid
from one_to_many_gan.core.training import ImageBuffer
from one_to_many_gan.data.config import Config
from one_to_many_gan.external.inception_score import inception_score
from one_to_many_gan.model.builder import (
    Discriminator,
    Generator,
    MappingNetwork,
    StyleExtractor,
)
from tqdm import tqdm, trange

# * Checkpoints


def write_logfile(config: Config, line: str):
    """Print a line and write it to a log file."""
    tqdm.write(line)
    checkpoint_log_file = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "log"
    )
    checkpoint_log_file.parent.mkdir(exist_ok=True)
    with checkpoint_log_file.open("a") as file:
        file.write(line + "\n")


# ** Validation


def validate_cis(
    config: Config,
    device: torch.device,
    shoeprint: torch.Tensor,
    mapping_network: MappingNetwork,
    generator: Generator,
    inception_model,
):
    """Calculate conditional inception score (CIS) for an individual shoeprint."""
    # We want to use the same shoeprint to generate multiple shoemarks
    shoeprints = shoeprint.to(device).expand(
        config["inference"]["batch_size"], -1, -1, -1
    )

    shoemark_batches = []
    for _ in range(
        math.ceil(
            config["evaluation"]["cond_is_n_evaluation_images"]
            / config["inference"]["batch_size"]
        )
    ):
        s = mapping_network.get_single_s(
            batch_size=config["inference"]["batch_size"],
            device=device,
            mix_styles=False,
            domain_variable=1,
        )

        shoemarks = generator(shoeprints, s)

        # Inception score model requires images of shape (3,299,299)
        shoemarks = F.interpolate(
            shoemarks, (299, 299), mode="bicubic", align_corners=False, antialias=True
        )
        shoemarks = shoemarks.expand(-1, 3, -1, -1)

        # Normalise to ensure [0-1] range
        def normalise(image: torch.Tensor):
            image_min = image.min()
            image_max = image.max()
            safe_range = image_max - image_min + 1e-14
            return (image - image_min) / safe_range

        shoemarks = normalise(shoemarks)

        shoemark_batches.append(shoemarks)

    top_tensors = torch.cat(shoemark_batches, dim=0)[
        : config["evaluation"]["cond_is_n_evaluation_images"]
    ]

    is_, _ = inception_score(
        top_tensors,
        inception_model,
        device=device,
        batch_size=config["inference"]["batch_size"],
        splits=1,
    )

    return is_


def validate_kid_fid(
    config: Config,
    device: torch.device,
    shoeprint_val_iter: Iterator[torch.Tensor],
    mapping_network: MappingNetwork,
    generator: Generator,
):
    """Calculate FID and KID scores and save to checkpoint."""
    # Directory to store generated shoemarks
    val_checkpoint_dir = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "val"
    )
    val_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate shoemarks and save to file
    shoemark_count = 0
    for _ in trange(
        math.ceil(
            config["evaluation"]["n_evaluation_images"]
            / config["inference"]["batch_size"]
        ),
        desc="Generating shoemarks: ",
        leave=False,
        dynamic_ncols=True,
    ):
        shoeprints = next(shoeprint_val_iter).to(device)
        s = mapping_network.get_single_s(
            batch_size=shoeprints.shape[0],
            device=device,
            mix_styles=False,
            domain_variable=1,
        )

        val_shoemarks = generator(shoeprints, s)

        for shoemark in val_shoemarks:
            torchvision.utils.save_image(
                shoemark, val_checkpoint_dir / f"{shoemark_count}.png"
            )
            shoemark_count += 1

    shoemark_train_dir = config["data"]["shoemark_data_dir"] / "train"
    fid_score = fid.compute_fid(
        str(val_checkpoint_dir), str(shoemark_train_dir), verbose=False, device=device
    )
    kid_score = fid.compute_kid(
        str(val_checkpoint_dir), str(shoemark_train_dir), verbose=False, device=device
    )

    return fid_score, kid_score


# ** Image


def create_image_checkpoint(
    step: int,
    config: Config,
    device: torch.device,
    shoeprints: torch.Tensor,
    shoemarks: torch.Tensor,
    mapping_network: MappingNetwork,
    generator: Generator,
    style_extractor: StyleExtractor,
):
    """Generate and save image checkpoints."""
    image_checkpoint_dir = (
        config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "images"
    )
    image_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = 8, 9

    # Get a style vector for each column
    s = mapping_network.get_single_s(
        batch_size=cols - 1,
        device=device,
        mix_styles=False,
        domain_variable=1,
    )

    # Load images to GPU
    gpu_shoeprints = shoeprints.to(device)
    gpu_shoemarks = shoemarks.to(device)

    # Get latent encodings
    shoeprint_latents = generator.encode(gpu_shoeprints)
    shoemark_latents = generator.encode(gpu_shoemarks)

    # Style vector at origin, denoting a shoeprint
    s0 = torch.zeros(
        (
            generator.n_style_blocks,
            cols - 1,
            config["architecture"]["s_dim"],
        ),
        device=device,
    )
    reconstructed_shoeprints = generator.decode(shoeprint_latents, s0)

    # Get the style vector of the real shoemarks
    real_shoemark_s = style_extractor(gpu_shoemarks)
    reconstructed_shoemarks = generator.decode(
        shoemark_latents,
        real_shoemark_s.expand(generator.n_style_blocks, *real_shoemark_s.shape),
    )

    # Translate from a real shoeprint to a shoemark in the style of the real shoemark
    translated_shoemarks = generator.decode(
        shoeprint_latents,
        real_shoemark_s.expand(generator.n_style_blocks, *real_shoemark_s.shape),
    )

    # ---- Translation checkpoint image

    translation_grid_images = torch.stack(
        [
            torch.stack(
                [
                    real_shoeprint_image,
                    *generator.decode(shoeprint_latent.expand(cols - 1, -1, -1, -1), s),
                ]
            )
            for real_shoeprint_image, shoeprint_latent in zip(
                gpu_shoeprints, shoeprint_latents, strict=True
            )
        ]
    )

    save_path = image_checkpoint_dir / f"translation_{step + 1}.png"
    torchvision.utils.save_image(
        translation_grid_images.view(
            rows * cols, config["data"]["image_channels"], *config["data"]["image_size"]
        ),
        save_path,
        nrow=cols,  # Number of images to be displayed for each row
        padding=2,
        normalize=True,
    )

    # ---- Style transfer checkpoint image

    rows, cols = 8, 5

    style_grid_images = torch.stack(
        [
            torch.stack(
                [
                    gpu_shoeprints[row],
                    reconstructed_shoeprints[row],
                    translated_shoemarks[row],
                    gpu_shoemarks[row],
                    reconstructed_shoemarks[row],
                ]
            )
            for row in range(rows)
        ]
    )

    save_path = image_checkpoint_dir / f"style_{step + 1}.png"
    torchvision.utils.save_image(
        style_grid_images.view(
            rows * cols, config["data"]["image_channels"], *config["data"]["image_size"]
        ),
        save_path,
        nrow=cols,
        padding=2,
        normalize=True,
    )


# * Model


def create_model_checkpoint(
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
            "image_buffer_images": image_buffer.images,
            "image_buffer_size": image_buffer.buffer_size,
        },
        models_checkpoint_dir / f"{step}.tar",
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
        )

        self.initialise_trackers()

        return string
