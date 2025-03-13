"""Orchestrate training of model."""

import itertools
import math
from typing import cast

import numpy as np
import torch
import torch.utils.data
from ada import AdaptiveDiscriminatorAugmentation
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
from xogan.data import Edges2ShoesDataset, ShoeDataset
from xogan.loss import (
    ADAp,
    GradientPenalty,
    PathLengthPenalty,
    SiameseTripletLoss,
    discriminator_loss,
    generator_loss,
)
from xogan.siamese import GlobalContextSiamese
from xogan.stylegan2 import Discriminator, Generator, MappingNetwork
from xogan.utils import Logger, save_grid

# * Hyperparameters

CONFIG = {
    "gradient_penalty_coeficcient": 10.0,
    "path_length_penalty_coeficcient": 0.99,
    "siamese_generator_coefficient": 0.2,
    "batch_size": 16,
    "d_latent": 64,
    "siamese_embedding_dimensions": 256,
    "image_size": (256, 128),
    "image_channels": 1,
    "mapping_network_layers": 4,
    "learning_rate": 1e-3,
    "mapping_network_learning_rate": 1e-5,  # 100x less
    "gradient_accumulate_steps": 1,
    "discriminator_steps": 1,
    "generator_steps": 1,
    "siamese_steps": 1,
    "generator_bottleneck_resolution": (16, 8),
    "generator_bottleneck_blocks": 6,
    "discriminator_overfitting_target": 0.6,
    "ada_E": 256,  # Number of images over which to take the mean discriminator overfitting
    "ada_adjustment_size": 5.12e-4,  # Adjustment amount per image, multiplied by ada_E
    "adam_betas": (0.0, 0.99),
    "style_mixing_prob": 0.9,
    "triplet_loss_margin": 0.8,
    "training_steps": 1_000_000,
    "lazy_gradient_penalty_interval": 4,
    "lazy_path_penalty_interval": 32,
    "lazy_path_penalty_after": 5_000_000,
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

log_resolution = int(math.log2(min(CONFIG["image_size"])))

discriminator = Discriminator(
    log_resolution=log_resolution, in_channels=CONFIG["image_channels"]
).to(device)

generator = Generator(
    log_resolution=log_resolution,
    bottleneck_resolution=CONFIG["generator_bottleneck_resolution"],
    d_latent=CONFIG["d_latent"],
    n_bottleneck_blocks=CONFIG["generator_bottleneck_blocks"],
    img_channels=CONFIG["image_channels"],
).to(device)
n_gen_style_blocks = generator.n_style_blocks

mapping_network = MappingNetwork(
    features=CONFIG["d_latent"],
    n_layers=CONFIG["mapping_network_layers"],
    style_mixing_prob=CONFIG["style_mixing_prob"],
).to(device)

shoemark_siamese = GlobalContextSiamese(
    in_channels=CONFIG["image_channels"], emb_dim=CONFIG["siamese_embedding_dimensions"]
).to(device)
shoeprint_siamese = GlobalContextSiamese(
    in_channels=CONFIG["image_channels"], emb_dim=CONFIG["siamese_embedding_dimensions"]
).to(device)

# ** Regularisation

path_length_penalty = PathLengthPenalty(
    beta=CONFIG["path_length_penalty_coeficcient"]
).to(device)

gradient_penalty = GradientPenalty().to(device)

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

# ** Siamese

triplet_loss = SiameseTripletLoss(margin=CONFIG["triplet_loss_margin"])

# ** Optimisers

discriminator_optimiser = torch.optim.Adam(
    discriminator.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["adam_betas"]
)

generator_optimiser = torch.optim.Adam(
    generator.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["adam_betas"]
)

mapping_network_optimiser = torch.optim.Adam(
    mapping_network.parameters(),
    lr=CONFIG["mapping_network_learning_rate"],
    betas=CONFIG["adam_betas"],
)

# Use AdamW as siamese net uses batch norm
shoemark_siamese_optimiser = torch.optim.AdamW(
    shoemark_siamese.parameters(),
    lr=CONFIG["mapping_network_learning_rate"],
    betas=CONFIG["adam_betas"],
)

shoeprint_siamese_optimiser = torch.optim.AdamW(
    shoeprint_siamese.parameters(),
    lr=CONFIG["mapping_network_learning_rate"],
    betas=CONFIG["adam_betas"],
)

# ** Data

transform = transforms.Compose(
    [
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

shoemark_data = ShoeDataset(
    "~/Datasets/Santa Partitioned/Shoemarks", mode="train", transform=transform
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
    "~/Datasets/Santa Partitioned/Shoeprints", mode="train", transform=transform
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
        w = mapping_network.get_w(CONFIG["batch_size"], n_gen_style_blocks, device)

        shoeprint_images = next(shoeprint_iter).to(device)
        generated_shoemarks = generator.generate_images(
            CONFIG["batch_size"], shoeprint_images, w, device
        )

        augmented_fake_shoemarks = adaptive_discriminator_augmentation(
            generated_shoemarks.detach()
        )
        fake_output = discriminator(augmented_fake_shoemarks)

        real_shoemarks = next(shoemark_iter).to(device)
        augmented_real_shoemarks = adaptive_discriminator_augmentation(real_shoemarks)

        if (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0:
            augmented_real_shoemarks.requires_grad_()

        real_output = discriminator(augmented_real_shoemarks)

        # r_t for adaptive discriminator normalisation
        sign_real = torch.sign(real_output.detach())
        log_real_accuracy += torch.mean(sign_real)

        # Update ADA p value
        ada_p.update_p(sign_real)

        sign_fake = torch.sign(fake_output.detach())
        log_fake_accuracy += torch.mean(sign_fake)

        disc_loss = discriminator_loss(real_output, fake_output)

        if (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0:
            gp = gradient_penalty(augmented_real_shoemarks, real_output)

            disc_loss = (
                disc_loss
                + 0.5
                * CONFIG["gradient_penalty_coeficcient"]
                * gp
                * CONFIG["lazy_gradient_penalty_interval"]
            )

        log_disc_loss += disc_loss

        disc_loss.backward()

    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

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
    mapping_network_optimiser.zero_grad()

    log_gen_loss = torch.zeros((), device=device)
    for _ in range(CONFIG["gradient_accumulate_steps"]):
        w = mapping_network.get_w(CONFIG["batch_size"], n_gen_style_blocks, device)

        shoeprint_images = next(shoeprint_iter).to(device)
        generated_shoemarks = generator.generate_images(
            CONFIG["batch_size"], shoeprint_images, w, device
        )

        augmented_generated_images = adaptive_discriminator_augmentation(
            generated_shoemarks
        )
        fake_shoemarks_score = discriminator(augmented_generated_images)

        gen_loss = generator_loss(fake_shoemarks_score)

        # TODO add ADA
        anchor = shoeprint_images
        positive = generated_shoemarks
        negative = next(shoemark_iter).to(device)

        anchor_emb = shoeprint_siamese(anchor)
        positive_emb = shoemark_siamese(positive)
        negative_emb = shoemark_siamese(negative)

        siamese_loss, _ = triplet_loss(anchor_emb, positive_emb, negative_emb)

        gen_loss += siamese_loss * CONFIG["siamese_generator_coefficient"]

        if (
            step > CONFIG["lazy_path_penalty_after"]
            and (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0
        ):
            plp = path_length_penalty(w, generated_shoemarks)

            if not torch.isnan(plp):
                gen_loss = gen_loss + plp

        log_gen_loss += gen_loss

        gen_loss.backward()

    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(mapping_network.parameters(), max_norm=1.0)

    generator_optimiser.step()
    mapping_network_optimiser.step()

    log_gen_loss /= CONFIG["gradient_accumulate_steps"]

    return log_gen_loss.detach().cpu().numpy()


# ** Siamese


def siamese_step():
    """Take a step with the siamese network and return loss."""
    shoemark_siamese_optimiser.zero_grad()
    shoeprint_siamese_optimiser.zero_grad()

    log_siamese_loss = torch.zeros((), device=device)
    log_siamese_positive_accuracy = torch.zeros((), device=device)
    log_siamese_negative_accuracy = torch.zeros((), device=device)

    for _ in range(CONFIG["gradient_accumulate_steps"]):
        w = mapping_network.get_w(CONFIG["batch_size"], n_gen_style_blocks, device)

        shoeprint_images = next(shoeprint_iter).to(device)
        generated_shoemarks = generator.generate_images(
            CONFIG["batch_size"], shoeprint_images, w, device
        )

        anchor = shoeprint_images
        positive = generated_shoemarks.detach()
        negative = next(shoemark_iter).to(device)

        # TODO add ADA
        anchor_emb = shoeprint_siamese(anchor)
        positive_emb = shoemark_siamese(positive)
        negative_emb = shoemark_siamese(negative)

        siamese_loss, (norm_anchor_emb, norm_positive_emb, norm_negative_emb) = (
            triplet_loss(anchor_emb, positive_emb, negative_emb)
        )

        log_siamese_loss += siamese_loss

        positive_similarities = F.cosine_similarity(norm_anchor_emb, norm_positive_emb)
        negative_similarities = F.cosine_similarity(norm_anchor_emb, norm_negative_emb)

        log_siamese_positive_accuracy += torch.mean(
            torch.sign(positive_similarities.detach())
        )
        log_siamese_negative_accuracy += torch.mean(
            torch.sign(negative_similarities.detach())
        )

        siamese_loss.backward()

    torch.nn.utils.clip_grad_norm_(shoemark_siamese.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(shoeprint_siamese.parameters(), max_norm=1.0)

    shoeprint_siamese_optimiser.step()
    shoemark_siamese_optimiser.step()

    log_siamese_loss /= CONFIG["gradient_accumulate_steps"]
    log_siamese_positive_accuracy /= CONFIG["gradient_accumulate_steps"]
    log_siamese_negative_accuracy /= CONFIG["gradient_accumulate_steps"]

    return log_siamese_loss.detach().cpu().numpy(), (
        log_siamese_positive_accuracy.cpu().numpy(),
        log_siamese_negative_accuracy.cpu().numpy(),
    )


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

        siamese_losses = []
        siamese_positive_accuracies = []
        siamese_negative_accuracies = []
        for _ in range(CONFIG["siamese_steps"]):
            siamese_loss, (siamese_positive_acc, siamese_negative_acc) = siamese_step()
            siamese_losses.append(siamese_loss)
            siamese_positive_accuracies.append(siamese_positive_acc)
            siamese_negative_accuracies.append(siamese_negative_acc)

        logger.log_siamese_loss += np.mean(siamese_losses).item()
        logger.log_siamese_positive_accuracy += np.mean(
            siamese_positive_accuracies
        ).item()
        logger.log_siamese_negative_accuracy += np.mean(
            siamese_negative_accuracies
        ).item()

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
            mapping_network.eval()
            with torch.no_grad():
                w = [
                    mapping_network.get_w(
                        1, n_gen_style_blocks, device, mix_styles=False
                    )
                    for _ in range(3)
                ]

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
                    column_images.append(shoeprint_images[column])

                    for row in range(3):
                        row_image = generator.generate_images(
                            1, shoeprint_images[column], w[row], device
                        )
                        column_images.append(row_image)

                    shoemark_images.append(torch.cat(column_images))

                images = torch.cat(shoemark_images)

            generator.train()
            mapping_network.train()

            save_grid(step + 1, images)


if __name__ == "__main__":
    main()
