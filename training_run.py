"""Orchestrate training of model."""

import itertools
import math
import random
from typing import cast

import numpy as np
import torch
import torch.utils.data
from ada import AdaptiveDiscriminatorAugmentation
from src.data import ShoeDataset
from src.loss import ADAp, kl_loss_func, path_loss_func, style_cycle_loss_func
from src.models import Discriminator, Generator, MappingNetwork, StyleExtractor
from src.utils import ImageBuffer, Logger, save_grid
from torch import nn
from torchvision import transforms
from tqdm import tqdm

# * Hyperparameters

CONFIG = {
    "style_cycle_loss_lambda": 5,
    "identity_loss_lambda": 5,
    "reconstruction_loss_lambda": 5,
    "kl_loss_lambda": 0.01,
    "path_loss_lambda": 0.1,
    "path_loss_layers": [
        0,
        1,
        2,
        3,
        5,
        8,
    ],  # Layers of the decoder to calculate path length on
    "path_loss_granularity": (
        0.1,
        0.2,
    ),  # Min and max for sampling h to approximate jacobian
    "batch_size": 4,
    "image_buffer_size": 100,
    "w_dim": 6,
    "image_size": (512, 256),
    "image_channels": 1,
    "mapping_network_layers": 2,
    "learning_rate": 2e-3,
    "mapping_network_learning_rate": 2e-5,  # 100x less
    "gradient_accumulate_steps": 1,  # FIXME this doesn't work, I presume because tensors have to be kept in order to perform backprop
    "discriminator_steps": 1,
    "generator_steps": 1,
    "discriminator_overfitting_target": 0.6,
    "ada_E": 256,  # Number of images over which to take the mean discriminator overfitting
    "ada_adjustment_size": 5.12e-4,  # Adjustment amount per image, multiplied by ada_E
    "adam_betas": (0.5, 0.99),
    "style_mixing_prob": 0.9,
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

discriminator = Discriminator(input_nc=CONFIG["image_channels"]).to(device)

generator = Generator(input_nc=CONFIG["image_channels"], w_dim=CONFIG["w_dim"]).to(
    device
)

mapping_network = MappingNetwork(
    features=CONFIG["w_dim"],
    n_layers=CONFIG["mapping_network_layers"],
    style_mixing_prob=CONFIG["style_mixing_prob"],
).to(device)

style_extractor = StyleExtractor(
    input_nc=CONFIG["image_channels"], w_dim=CONFIG["w_dim"]
).to(device)

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

mapping_network_optimiser = torch.optim.Adam(
    mapping_network.parameters(),
    lr=CONFIG["mapping_network_learning_rate"],
    betas=CONFIG["adam_betas"],
)

style_extractor_optimiser = torch.optim.Adam(
    style_extractor.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["adam_betas"]
)

# ** Losses

gan_loss_func = nn.MSELoss()
l1_loss_func = torch.nn.L1Loss()

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

image_buffer = ImageBuffer(CONFIG["image_buffer_size"])

# ** Logging

logger = Logger(CONFIG["training_steps"])

# * Training Loop

# ** Discriminator


def discriminator_step():
    """Take a step with the discriminator and return loss."""
    discriminator_optimiser.zero_grad()

    log_disc_loss = torch.zeros((), device=device)
    log_real_accuracy = torch.zeros((), device=device)
    log_fake_accuracy = torch.zeros((), device=device)

    # Scale from [0,1] to [-1,1] and then take sign as indication of judgement
    discriminator_confidence = lambda scores: torch.sign(scores * 2 - 1).mean()

    for _ in range(CONFIG["gradient_accumulate_steps"]):
        shoeprint_images = next(shoeprint_iter).to(device)
        w = mapping_network.get_w(
            batch_size=CONFIG["batch_size"],
            n_gen_blocks=generator.n_style_blocks,
            device=device,
            domain_variable=1,
        )
        generated_shoemarks = generator(shoeprint_images, w)
        buffered_shoemarks, _ = image_buffer(generated_shoemarks, w[-1])
        augmented_fake_shoemarks = adaptive_discriminator_augmentation(
            buffered_shoemarks
        )
        fake_scores = discriminator(augmented_fake_shoemarks)

        real_shoemarks = next(shoemark_iter).to(device)
        augmented_real_shoemarks = adaptive_discriminator_augmentation(real_shoemarks)
        real_scores = discriminator(augmented_real_shoemarks)

        real_loss = gan_loss_func(real_scores, torch.ones_like(real_scores))
        fake_loss = gan_loss_func(fake_scores, torch.zeros_like(fake_scores))

        disc_loss = (real_loss + fake_loss) / 2

        sign_real = discriminator_confidence(real_scores.detach())
        sign_fake = discriminator_confidence(fake_scores.detach()) * -1
        # r_t for adaptive discriminator normalisation
        log_real_accuracy += sign_real
        log_fake_accuracy += sign_fake
        # Update ADA p value
        ada_p.update_p(sign_real)

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


def generator_step():
    """Take a step with the generator and return loss."""
    generator_optimiser.zero_grad()
    mapping_network_optimiser.zero_grad()
    style_extractor_optimiser.zero_grad()

    log_total_gen_loss = torch.zeros((), device=device)
    log_gan_loss = torch.zeros((), device=device)
    log_rec_loss = torch.zeros((), device=device)
    log_idt_loss = torch.zeros((), device=device)
    log_kl_loss = torch.zeros((), device=device)
    log_path_loss = torch.zeros((), device=device)
    log_style_loss = torch.zeros((), device=device)

    for _ in range(CONFIG["gradient_accumulate_steps"]):
        shoeprint_images = next(shoeprint_iter).to(device)
        real_shoemark_images = next(shoemark_iter).to(device)

        # KL loss
        # TODO in the Santa implementation they add Gaussian noise to latents after kl loss
        # Combine for single forward pass
        combined_images = torch.cat([shoeprint_images, real_shoemark_images], dim=0)
        combined_latents = generator.encode(combined_images)
        kl_loss = kl_loss_func(combined_latents)
        log_kl_loss += kl_loss

        # Encoded latent variables
        shoeprint_latent, shoemark_latent = combined_latents.chunk(2, dim=0)

        # Reconstruction loss
        reconstruct_w = mapping_network.get_w(
            batch_size=CONFIG["batch_size"],
            n_gen_blocks=generator.n_style_blocks,
            device=device,
            domain_variable=0,
        )
        reconstruct_w = cast(torch.Tensor, reconstruct_w)
        reconstructed_shoeprints = generator.decode(shoeprint_latent, reconstruct_w)
        reconstruction_loss = l1_loss_func(reconstructed_shoeprints, shoeprint_images)
        log_rec_loss += reconstruction_loss

        # Identity loss
        real_shoemark_w = style_extractor(real_shoemark_images)
        reconstructed_shoemarks = generator.decode(
            shoemark_latent,
            real_shoemark_w.expand(generator.n_style_blocks, *real_shoemark_w.shape),
        )
        identity_loss = l1_loss_func(reconstructed_shoemarks, real_shoemark_images)
        log_idt_loss += identity_loss

        # GAN loss
        translation_w = mapping_network.get_w(
            batch_size=CONFIG["batch_size"],
            n_gen_blocks=generator.n_style_blocks,
            device=device,
            domain_variable=1,
        )
        translation_w = cast(torch.Tensor, translation_w)
        generated_shoemarks = generator.decode(shoeprint_latent, translation_w)
        augmented_generated_images = adaptive_discriminator_augmentation(
            generated_shoemarks
        )
        fake_shoemark_scores = discriminator(augmented_generated_images)
        gan_loss = gan_loss_func(
            fake_shoemark_scores, torch.ones_like(fake_shoemark_scores)
        )
        log_gan_loss += gan_loss

        # Style cycle loss
        style_loss_shoemarks = generated_shoemarks
        style_loss_w = translation_w[-1]
        reconstructed_w = style_extractor(style_loss_shoemarks)
        style_loss = style_cycle_loss_func(style_loss_w, reconstructed_w)
        log_style_loss += style_loss

        # Path loss
        # Calculate random \theta for each image from uniform distribution between 0 and 1
        theta = torch.rand(CONFIG["batch_size"]).to(device)
        # H used in the central finite difference calculation
        cent_fin_diff_h = (
            torch.ones_like(theta)
            .to(device)
            .uniform_(
                CONFIG["path_loss_granularity"][0], CONFIG["path_loss_granularity"][1]
            )
        )
        d1 = (theta + cent_fin_diff_h / 2).clamp(0, 1)
        d2 = (theta - cent_fin_diff_h / 2).clamp(0, 1)
        w1, w2 = mapping_network.get_w(
            batch_size=CONFIG["batch_size"],
            n_gen_blocks=generator.n_style_blocks,
            device=device,
            domain_variable=(d1, d2),
        )
        features1 = generator.extract(shoeprint_latent, w1, CONFIG["path_loss_layers"])
        features2 = generator.extract(shoeprint_latent, w2, CONFIG["path_loss_layers"])
        path_loss = path_loss_func(features1, features2, cent_fin_diff_h)
        log_path_loss += path_loss

        total_gen_loss = (
            gan_loss
            + CONFIG["identity_loss_lambda"] * identity_loss
            + CONFIG["reconstruction_loss_lambda"] * reconstruction_loss
            + CONFIG["kl_loss_lambda"] * kl_loss
            + CONFIG["path_loss_lambda"] * path_loss
            + CONFIG["style_cycle_loss_lambda"] * style_loss
        )

        log_total_gen_loss += total_gen_loss

        total_gen_loss.backward(retain_graph=True)

    generator_optimiser.step()
    mapping_network_optimiser.step()
    style_extractor_optimiser.step()

    log_total_gen_loss /= CONFIG["gradient_accumulate_steps"]
    log_gan_loss /= CONFIG["gradient_accumulate_steps"]
    log_rec_loss /= CONFIG["gradient_accumulate_steps"]
    log_idt_loss /= CONFIG["gradient_accumulate_steps"]
    log_kl_loss /= CONFIG["gradient_accumulate_steps"]
    log_path_loss /= CONFIG["gradient_accumulate_steps"]
    log_style_loss /= CONFIG["gradient_accumulate_steps"]

    return log_total_gen_loss.detach().cpu().numpy(), (
        log_gan_loss.detach().cpu().numpy(),
        log_rec_loss.detach().cpu().numpy(),
        log_idt_loss.detach().cpu().numpy(),
        log_kl_loss.detach().cpu().numpy(),
        log_path_loss.detach().cpu().numpy(),
        log_style_loss.detach().cpu().numpy(),
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

        # Train discriminator
        disc_losses = []
        disc_real_accuracies = []
        disc_fake_accuracies = []
        for _ in range(CONFIG["discriminator_steps"]):
            disc_loss, (real_accuracy, fake_accuracy) = discriminator_step()
            disc_losses.append(disc_loss)
            disc_real_accuracies.append(real_accuracy)
            disc_fake_accuracies.append(fake_accuracy)

        logger.log_total_disc_loss += np.mean(disc_losses).item()
        logger.log_disc_real_acc += np.mean(disc_real_accuracies).item()
        logger.log_disc_fake_acc += np.mean(disc_fake_accuracies).item()

        # Train generator
        # TODO this is getting a bit crazy, clean it up
        total_gen_losses = []
        gan_losses = []
        rec_losses = []
        idt_losses = []
        kl_losses = []
        path_losses = []
        style_losses = []
        for _ in range(CONFIG["generator_steps"]):
            (
                total_gen_loss,
                (gan_loss, rec_loss, idt_loss, kl_loss, path_loss, style_loss),
            ) = generator_step()
            total_gen_losses.append(total_gen_loss)
            gan_losses.append(gan_loss)
            rec_losses.append(rec_loss)
            idt_losses.append(idt_loss)
            kl_losses.append(kl_loss)
            path_losses.append(path_loss)
            style_losses.append(style_loss)

        logger.log_total_gen_loss += np.mean(total_gen_losses).item()
        logger.log_gan_loss += np.mean(gan_losses).item()
        logger.log_rec_loss += np.mean(rec_losses).item()
        logger.log_idt_loss += np.mean(idt_losses).item()
        logger.log_kl_loss += np.mean(kl_losses).item()
        logger.log_path_loss += np.mean(path_losses).item()
        logger.log_style_loss += np.mean(style_losses).item()

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
                        batch_size=1,
                        n_gen_blocks=generator.n_style_blocks,
                        device=device,
                        mix_styles=False,
                        domain_variable=1,
                    )
                    for _ in range(8)
                ]
                if CONFIG["batch_size"] < 8:
                    shoeprint_images = [
                        next(shoeprint_iter).to(device)
                        for _ in range(math.ceil(8 / CONFIG["batch_size"]))
                    ]
                    shoeprint_images = torch.cat(shoeprint_images)
                else:
                    shoeprint_images = next(shoeprint_iter).to(device)

                shoeprint_images = shoeprint_images[:8]

                shoemark_images = []
                for column in range(8):
                    column_images = []
                    column_images.append(shoeprint_images[column][None, ...])

                    for row in range(8):
                        row_image = generator(
                            shoeprint_images[column][None, ...], w[row]
                        )
                        column_images.append(row_image)

                    shoemark_images.append(column_images)

            generator.train()
            mapping_network.train()

            save_grid(step + 1, shoemark_images)


if __name__ == "__main__":
    main()
