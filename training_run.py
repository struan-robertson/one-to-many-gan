"""Orchestrate training of model."""

import itertools
import math
from typing import cast

import numpy as np
import torch
import torch.utils.data
from ada import AdaptiveDiscriminatorAugmentation
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
from xogan.data import Edges2ShoesDataset, ShoeDataset
from xogan.loss import (
    ADAp,
    SiameseTripletLoss,
)
from xogan.santa import AdaINDiscriminator, AdaINResnetGenerator
from xogan.siamese import GlobalContextSiamese
from xogan.utils import Logger, save_grid

# * Hyperparameters

CONFIG = {
    "siamese_generator_coefficient": 0.5,
    "batch_size": 16,
    "d_latent": 8,
    "siamese_embedding_dimensions": 256,
    "image_size": (256, 128),
    "image_channels": 1,
    "mapping_network_layers": 2,
    "learning_rate": 2e-3,
    "mapping_network_learning_rate": 2e-5,  # 100x less
    "gradient_accumulate_steps": 1,
    "discriminator_steps": 1,
    "generator_steps": 1,
    "siamese_steps": 1,
    "generator_bottleneck_resolution": (16, 8),
    "generator_bottleneck_blocks": 6,  # TODO make this 9
    "discriminator_overfitting_target": 0.6,
    "ada_E": 256,  # Number of images over which to take the mean discriminator overfitting
    "ada_adjustment_size": 5.12e-4,  # Adjustment amount per image, multiplied by ada_E
    "adam_betas": (0.5, 0.99),
    "style_mixing_prob": 0.9,
    "triplet_loss_margin": 0.8,
    "training_steps": 1_000_000,
    "log_generated_interval": 500,
    "save_checkpoint_interval": 1_000,
}

# * Initialisation

# ** PyTorch
# *** Random Seed
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# *** Device

device = (
    torch.cuda.current_device()
    if torch.cuda.is_available()
    else torch.cpu.current_device()
)

# *** Config

torch.set_float32_matmul_precision("high")
torch._functorch.config.donated_buffer = False  # pyright: ignore[reportAttributeAccessIssue] # noqa: SLF001

# ** Models

discriminator = AdaINDiscriminator(input_nc=CONFIG["image_channels"]).to(device)

generator = AdaINResnetGenerator(input_nc=CONFIG["image_channels"]).to(device)


# ** Regularisation

adaptive_discriminator_augmentation = AdaptiveDiscriminatorAugmentation(
    xflip=1,
    rotate90=1,
    xint=1,
    scale=1,
    rotate=1,
    aniso=1,
    xfrac=1,
    brightness=1,
    contrast=1,
    lumaflip=1,
    hue=1,
    saturation=1,
).to(device)

ada_p = ADAp(
    ada_e=CONFIG["ada_E"],
    ada_adjustment_size=CONFIG["ada_adjustment_size"],
    batch_size=CONFIG["batch_size"],
    discriminator_overfitting_target=CONFIG["discriminator_overfitting_target"],
)

# ** Optimisers

discriminator_optimiser = torch.optim.Adam(
    discriminator.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["adam_betas"]
)

generator_optimiser = torch.optim.Adam(
    generator.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["adam_betas"]
)

# ** Losses

criterion = nn.MSELoss()

# ** Data

transform = transforms.Compose(
    [
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

shoemark_data = ShoeDataset(
    "~/Datasets/GAN Partitioned/Shoemarks", mode="train", transform=transform
)
shoemark_dataloader = torch.utils.data.DataLoader(
    shoemark_data,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

shoeprint_data = ShoeDataset(
    "~/Datasets/GAN Partitioned/Shoeprints", mode="train", transform=transform
)
shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

shoemark_iter = itertools.cycle(shoemark_dataloader)
shoeprint_iter = itertools.cycle(shoeprint_dataloader)


# ** Logging
logger = Logger(CONFIG["training_steps"])

# * Training Loop

# ** Discriminator


def discriminator_step(step: int):
    """Take a step with the discriminator and return loss."""
    discriminator_optimiser.zero_grad()

    log_disc_loss = torch.zeros((), device=device)
    log_real_accuracy = torch.zeros((), device=device)
    log_fake_accuracy = torch.zeros((), device=device)
    for _ in range(CONFIG["gradient_accumulate_steps"]):
        shoeprint_images = next(shoeprint_iter).to(device)
        generated_shoemarks = generator(shoeprint_images)
        augmented_fake_shoemarks = adaptive_discriminator_augmentation(
            generated_shoemarks.detach()
        )
        fake_scores = discriminator(augmented_fake_shoemarks)

        real_shoemarks = next(shoemark_iter).to(device)
        augmented_real_shoemarks = adaptive_discriminator_augmentation(real_shoemarks)
        real_scores = discriminator(augmented_real_shoemarks)

        real_loss = criterion(real_scores, torch.ones_like(real_scores))
        fake_loss = criterion(fake_scores, torch.zeros_like(fake_scores))

        disc_loss = (real_loss + fake_loss) / 2

        mean_real_score = real_scores.detach().mean()
        mean_fake_score = fake_scores.detach().mean()
        # r_t for adaptive discriminator normalisation
        log_real_accuracy += mean_real_score

        # Update ADA p value
        ada_p.update_p(mean_real_score)

        log_fake_accuracy += mean_fake_score

        log_disc_loss += disc_loss

        disc_loss.backward()

    discriminator_optimiser.step()

    log_disc_loss /= CONFIG["gradient_accumulate_steps"]
    log_real_accuracy /= CONFIG["gradient_accumulate_steps"]
    log_fake_accuracy /= CONFIG["gradient_accumulate_steps"]

    return log_disc_loss.detach().cpu().numpy(), (
        log_real_accuracy.cpu().numpy(),
        log_fake_accuracy.cpu().numpy(),
    )


# ** Generator


def generator_step(step: int):
    """Take a step with the generator and return loss."""
    generator_optimiser.zero_grad()

    log_gen_loss = torch.zeros((), device=device)
    for _ in range(CONFIG["gradient_accumulate_steps"]):
        shoeprint_images = next(shoeprint_iter).to(device)
        generated_shoemarks = generator(shoeprint_images)

        augmented_generated_images = adaptive_discriminator_augmentation(
            generated_shoemarks
        )
        fake_shoemark_scores = discriminator(augmented_generated_images)

        # Non-saturating loss
        gen_loss = criterion(
            fake_shoemark_scores, torch.ones_like(fake_shoemark_scores)
        )

        log_gen_loss += gen_loss

        gen_loss.backward()

    generator_optimiser.step()

    log_gen_loss /= CONFIG["gradient_accumulate_steps"]

    return log_gen_loss.detach().cpu().numpy()


# ** Main Loop


def main():
    """Training loop."""
    for step in tqdm(
        range(CONFIG["training_steps"]),
    ):
        logger.step()

        # set adaptive discriminator augmentation p
        adaptive_discriminator_augmentation.set_p(ada_p().item())
        logger.log_ada_p += ada_p().item()

        disc_losses = []
        disc_real_accuracies = []
        disc_fake_accuracies = []
        for _ in range(CONFIG["discriminator_steps"]):
            disc_loss, (real_accuracy, fake_accuracy) = discriminator_step(step)
            disc_losses.append(disc_loss)
            disc_real_accuracies.append(real_accuracy)
            disc_fake_accuracies.append(fake_accuracy)

        logger.log_disc_losses += np.mean(disc_losses).item()
        logger.log_disc_real_acc += np.mean(disc_real_accuracies).item()
        logger.log_disc_fake_acc += np.mean(disc_fake_accuracies).item()

        gen_losses = []
        for _ in range(CONFIG["generator_steps"]):
            gen_loss = generator_step(step)
            gen_losses.append(gen_loss)

        logger.log_gen_loss += np.mean(gen_losses).item()

        # Logging

        if (step + 1) % CONFIG["log_generated_interval"] == 0 or (step + 1) == CONFIG[
            "training_steps"
        ]:
            tqdm.write(logger.print())

        if (step + 1) % CONFIG["save_checkpoint_interval"] == 0 or (step + 1) == CONFIG[
            "training_steps"
        ]:
            # Generate images
            generator.eval()
            with torch.no_grad():
                if CONFIG["batch_size"] < 8:
                    shoeprint_images = [
                        next(shoeprint_iter).to(device)
                        for _ in range(math.ceil(CONFIG["batch_size"] / 8))
                    ]

                    shoeprint_images = torch.cat(shoeprint_images)
                else:
                    shoeprint_images = next(shoeprint_iter).to(device)

                shoeprint_images = shoeprint_images[:8]

                shoemark_images = []
                for column in range(8):
                    column_images = []
                    column_images.append(shoeprint_images[column][None, ...])

                    for _ in range(3):
                        row_image = generator(shoeprint_images[column][None, ...])
                        column_images.append(row_image)

                    shoemark_images.append(column_images)

            generator.train()

            save_grid(step + 1, shoemark_images)


if __name__ == "__main__":
    main()
