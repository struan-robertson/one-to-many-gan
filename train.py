"""Orchestrate training of model."""

import gc
import itertools
import math
import os
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.utils.data
import torchvision
from ada import AdaptiveDiscriminatorAugmentation
from src.config import load_config
from src.data import ShoeDataset
from src.loss import ADAp, kl_loss_func, path_loss_func, style_cycle_loss_func
from src.models import Discriminator, Generator, MappingNetwork, StyleExtractor
from src.utils import ImageBuffer, Logger, evaluator, save_grid
from torch import nn
from torchvision import transforms
from tqdm import tqdm, trange

# * Config

config = load_config("config.toml")

# * Initialisation

# ** Random Seeding
torch.manual_seed(config["training"]["random_seed"])
np.random.default_rng(config["training"]["random_seed"])
random.seed(config["training"]["random_seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["training"]["random_seed"])
    if config["training"]["deterministic_cuda_kernels"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    """Seed DataLoader workers with random seed."""
    worker_seed = (
        config["training"]["random_seed"] + worker_id
    ) % 2**32  # Ensure we don't overflow 32 bit
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)


# Passed to dataloaders
dataloader_g = torch.Generator()
dataloader_g.manual_seed(config["training"]["random_seed"])

# ** PyTorch

# *** Device

device = (
    torch.cuda.current_device()
    if torch.cuda.is_available()
    else torch.cpu.current_device()
)

# *** Config

torch.set_float32_matmul_precision("high")


# ** Models

discriminator = Discriminator(input_nc=config["data"]["image_channels"]).to(device)

generator = Generator(
    input_nc=config["data"]["image_channels"],
    w_dim=config["architecture"]["w_dim"],
    image_size=config["data"]["image_size"],
    min_latent_resolution=config["architecture"]["min_latent_resolution"],
    n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
).to(device)

mapping_network = MappingNetwork(
    features=config["architecture"]["w_dim"],
    n_layers=config["architecture"]["mapping_network_layers"],
    style_mixing_prob=config["training"]["style_mixing_prob"],
).to(device)

style_extractor = StyleExtractor(
    input_nc=config["data"]["image_channels"], w_dim=config["architecture"]["w_dim"]
).to(device)

# ** Optimisers

discriminator_optimiser = torch.optim.Adam(
    discriminator.parameters(),
    lr=config["optimisation"]["learning_rate"],
    betas=config["optimisation"]["adam_betas"],
)

generator_optimiser = torch.optim.Adam(
    generator.parameters(),
    lr=config["optimisation"]["learning_rate"],
    betas=config["optimisation"]["adam_betas"],
)

mapping_network_optimiser = torch.optim.Adam(
    mapping_network.parameters(),
    lr=config["optimisation"]["mapping_network_learning_rate"],
    betas=config["optimisation"]["adam_betas"],
)

style_extractor_optimiser = torch.optim.Adam(
    style_extractor.parameters(),
    lr=config["optimisation"]["learning_rate"],
    betas=config["optimisation"]["adam_betas"],
)

# ** Losses

mse_loss_func = nn.MSELoss()
l1_loss_func = torch.nn.L1Loss()

# ** Data

transform = transforms.Compose(
    [
        transforms.Resize(config["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

shoemark_data = ShoeDataset(
    config["data"]["shoemark_data_dir"], mode="train", transform=transform
)
shoemark_dataloader = torch.utils.data.DataLoader(
    shoemark_data,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=dataloader_g,
)

shoeprint_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"], mode="train", transform=transform
)
shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=dataloader_g,
)

shoeprint_val_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=config["training"]["batch_size"] * 4,
    shuffle=False,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=dataloader_g,
)

shoemark_iter = itertools.cycle(shoemark_dataloader)
shoeprint_iter = itertools.cycle(shoeprint_dataloader)
shoeprint_val_iter = itertools.cycle(shoeprint_val_dataloader)

image_buffer = ImageBuffer(config["training"]["image_buffer_size"])

# ** Augmentation

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
    ada_e=config["ada"]["ada_overfitting_measurement_n_images"],
    ada_adjustment_size=config["ada"]["ada_adjustment_size"],
    batch_size=config["training"]["batch_size"],
    discriminator_overfitting_target=config["ada"]["discriminator_real_acc_target"],
)

# ** Logging

logger = Logger(config["training"]["training_steps"])

# * Training Loop

# Clean up return value code
detacher = lambda x: x.detach().cpu().item()

# ** Discriminator


def discriminator_step():
    """Take a step with the discriminator and return loss."""
    # Scale from [0,1] to [-1,1] and then take sign as indication of judgement
    discriminator_confidence = lambda scores: torch.sign(scores * 2 - 1).mean()

    discriminator_optimiser.zero_grad()

    # Generate fake shoemarks
    shoeprint_images = next(shoeprint_iter).to(device)
    w = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=1,
    )
    generated_shoemarks = generator(shoeprint_images, w)
    buffered_shoemarks = image_buffer(generated_shoemarks)
    augmented_fake_shoemarks = adaptive_discriminator_augmentation(buffered_shoemarks)

    # Get real shoemarks
    real_shoemarks = next(shoemark_iter).to(device)
    augmented_real_shoemarks = adaptive_discriminator_augmentation(real_shoemarks)

    # Calculate discriminator scores
    fake_scores = discriminator(augmented_fake_shoemarks)
    real_scores = discriminator(augmented_real_shoemarks)

    # Calculate losses
    real_loss = mse_loss_func(real_scores, torch.ones_like(real_scores))
    fake_loss = mse_loss_func(fake_scores, torch.zeros_like(fake_scores))
    disc_loss = (real_loss + fake_loss) / 2

    # Calculate discriminator confidence
    sign_real = discriminator_confidence(real_scores.detach())
    sign_fake = discriminator_confidence(fake_scores.detach()) * -1

    # Update ADA p value
    ada_p.update_p(sign_real)

    disc_loss.backward()
    discriminator_optimiser.step()

    return detacher(disc_loss), (
        detacher(sign_real),  # Discriminator confidences
        detacher(sign_fake),
    )


# ** Generator


def generator_step():
    """Take a step with the generator and return loss."""
    generator_optimiser.zero_grad()
    mapping_network_optimiser.zero_grad()
    style_extractor_optimiser.zero_grad()

    real_shoeprint_images = next(shoeprint_iter).to(device)
    real_shoemark_images = next(shoemark_iter).to(device)

    # KL loss
    # Combine for single forward pass
    combined_images = torch.cat([real_shoeprint_images, real_shoemark_images], dim=0)
    combined_latents = generator.encode(combined_images)
    kl_loss = kl_loss_func(combined_latents)

    # Encoded latent variables
    # Not specified in paper, but in implementation Xie et al. add noise to latents.
    if config["architecture"]["add_latent_noise"]:
        combined_latents = combined_latents + torch.randn_like(combined_latents)
    shoeprint_latent, shoemark_latent = combined_latents.chunk(2, dim=0)

    # Reconstruction loss
    reconstruct_w = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=0,
    )
    reconstruct_w = cast(torch.Tensor, reconstruct_w)
    reconstructed_shoeprints = generator.decode(shoeprint_latent, reconstruct_w)
    reconstruction_loss = l1_loss_func(reconstructed_shoeprints, real_shoeprint_images)

    # Identity loss
    real_shoemark_w = style_extractor(real_shoemark_images)
    reconstructed_shoemarks = generator.decode(
        shoemark_latent,
        real_shoemark_w.expand(generator.n_style_blocks, *real_shoemark_w.shape),
    )
    identity_loss = l1_loss_func(reconstructed_shoemarks, real_shoemark_images)

    # GAN loss
    translation_w = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
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
    gan_loss = mse_loss_func(
        fake_shoemark_scores, torch.ones_like(fake_shoemark_scores)
    )

    # Style cycle loss
    style_loss_shoemarks = generated_shoemarks
    style_loss_w = translation_w[-1]
    reconstructed_w = style_extractor(style_loss_shoemarks)
    style_loss = style_cycle_loss_func(style_loss_w, reconstructed_w)

    # Path loss
    # Calculate random \theta for each image from uniform distribution between 0 and 1
    theta = torch.rand(config["training"]["batch_size"]).to(device)
    # H used in the central finite difference calculation
    cent_fin_diff_h = (
        torch.ones_like(theta)
        .to(device)
        .uniform_(
            config["optimisation"]["path_loss_jacobian_granularity"][0],
            config["optimisation"]["path_loss_jacobian_granularity"][1],
        )
    )
    d1 = (theta + cent_fin_diff_h / 2).clamp(0, 1)
    d2 = (theta - cent_fin_diff_h / 2).clamp(0, 1)
    w1, w2 = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=(d1, d2),
    )
    features1 = generator.extract(shoeprint_latent, w1)
    features2 = generator.extract(shoeprint_latent, w2)
    path_loss = path_loss_func(features1, features2, cent_fin_diff_h)

    total_gen_loss = (
        gan_loss
        + config["optimisation"]["identity_loss_lambda"] * identity_loss
        + config["optimisation"]["reconstruction_loss_lambda"] * reconstruction_loss
        + config["optimisation"]["kl_loss_lambda"] * kl_loss
        + config["optimisation"]["path_loss_lambda"] * path_loss
        + config["optimisation"]["style_cycle_loss_lambda"] * style_loss
    )

    total_gen_loss.backward()
    generator_optimiser.step()
    mapping_network_optimiser.step()
    style_extractor_optimiser.step()

    return detacher(total_gen_loss), (
        detacher(gan_loss),
        detacher(reconstruction_loss),
        detacher(identity_loss),
        detacher(kl_loss),
        detacher(path_loss),
        detacher(style_loss),
    )


# ** Main Loop


def main():
    """Training loop."""
    for step in trange(config["training"]["training_steps"], dynamic_ncols=True):
        # set adaptive discriminator augmentation p
        adaptive_discriminator_augmentation.set_p(ada_p())
        logger.log_ada_ps.append(ada_p())

        # Train discriminator
        disc_loss, (real_accuracy, fake_accuracy) = discriminator_step()
        logger.log_total_disc_losses.append(disc_loss)
        logger.log_disc_real_accs.append(real_accuracy)
        logger.log_disc_fake_accs.append(fake_accuracy)

        # Train generator
        (
            total_gen_loss,
            (gan_loss, rec_loss, idt_loss, kl_loss, path_loss, style_loss),
        ) = generator_step()
        logger.log_total_gen_losses.append(total_gen_loss)
        logger.log_gan_losses.append(gan_loss)
        logger.log_rec_losses.append(rec_loss)
        logger.log_idt_losses.append(idt_loss)
        logger.log_kl_losses.append(kl_loss)
        logger.log_path_losses.append(path_loss)
        logger.log_style_losses.append(style_loss)

        if (step + 1) % config["evaluation"]["log_interval"] == 0 or (
            step + 1
        ) == config["training"]["training_steps"]:
            log = logger.print(step + 1)
            tqdm.write(log)
            with Path("./checkpoints/log").open("a") as file:
                file.write(log + "\n")

        if (step + 1) % config["evaluation"]["checkpoint_interval"] == 0 or (
            step + 1
        ) == config["training"]["training_steps"]:
            generator.eval()
            mapping_network.eval()
            style_extractor.eval()

            Path("checkpoints/images").mkdir(parents=True, exist_ok=True)
            Path("checkpoints/models").mkdir(parents=True, exist_ok=True)
            Path("checkpoints/val").mkdir(parents=True, exist_ok=True)

            # Free memory
            generator_optimiser.zero_grad(set_to_none=True)
            mapping_network_optimiser.zero_grad(set_to_none=True)
            style_extractor_optimiser.zero_grad(set_to_none=True)
            discriminator_optimiser.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Bit of a long winded function but uses enough global state so keeping in this file
                # Declaring it as a function here introduces scope to clear up tensors when done
                def image_checkpoints(step: int):
                    w = mapping_network.get_w(
                        batch_size=8,
                        n_gen_blocks=generator.n_style_blocks,
                        device=device,
                        mix_styles=False,
                        domain_variable=1,
                    )
                    w = cast(torch.Tensor, w)

                    # Collect enough shoeprints and shoemarks
                    if config["training"]["batch_size"] < 8:
                        real_shoeprint_images = [
                            next(shoeprint_iter).to(device)
                            for _ in range(
                                math.ceil(8 / config["training"]["batch_size"])
                            )
                        ]
                        real_shoemark_images = [
                            next(shoemark_iter).to(device)
                            for _ in range(
                                math.ceil(8 / config["training"]["batch_size"])
                            )
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
                            *generator.decode(
                                shoeprint_latents[column].expand(8, -1, -1, -1), w
                            )
                        ]
                        translation_grid_images.append(column_images)

                    save_grid(
                        translation_grid_images,
                        f"./checkpoints/images/translation_{step + 1}.png",
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
                        real_shoemark_w.expand(
                            generator.n_style_blocks, *real_shoemark_w.shape
                        ),
                    )

                    translated_shoemarks = generator.decode(
                        shoeprint_latents,
                        real_shoemark_w.expand(
                            generator.n_style_blocks, *real_shoemark_w.shape
                        ),
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
                        f"./checkpoints/images/decoding_{step + 1}.png",
                        (5, 8),
                    )

                def validation_step(step):
                    i = 0
                    for _ in trange(
                        math.ceil(
                            config["evaluation"]["n_evaluation_images"]
                            / (config["training"]["batch_size"] * 4)
                        ),
                        desc="Generating shoemarks",
                        leave=False,
                    ):
                        shoeprints = next(shoeprint_val_iter).to(device)
                        w = mapping_network.get_w(
                            batch_size=config["training"]["batch_size"] * 4,
                            n_gen_blocks=generator.n_style_blocks,
                            device=device,
                            mix_styles=False,
                            domain_variable=1,
                        )
                        w = cast(torch.Tensor, w)

                        val_shoemarks = generator(shoeprints, w)

                        for shoemark in val_shoemarks:
                            torchvision.utils.save_image(
                                shoemark, f"checkpoints/val/{i}.png"
                            )
                            i += 1

                    fid, kid = evaluator(
                        "checkpoints/val/",
                        "/home/struan/Datasets/GAN Partitioned Half/Shoemarks/train",
                        device,
                    )
                    log = f"Step {step + 1} val | fid: {fid}, kid: {kid}"
                    tqdm.write(log)
                    with Path("./checkpoints/log").open("a") as file:
                        file.write(log + "\n")

                image_checkpoints(step)
                validation_step(step)

                torch.save(
                    {
                        "generator_state_dict": generator.state_dict(),
                        "generator_optim_state_dict": generator_optimiser.state_dict(),
                        "discriminator_state_dict": discriminator_optimiser.state_dict(),
                        "discriminator_optim_state_dict": discriminator_optimiser.state_dict(),
                        "mapping_network_state_dict": mapping_network.state_dict(),
                        "mapping_network_optim_state_dict": mapping_network_optimiser.state_dict(),
                        "style_extractor_state_dict": style_extractor.state_dict(),
                        "style_extractor_optim_state_dict": style_extractor_optimiser.state_dict(),
                        "ada_p": ada_p(),
                        "image_buffer_images": image_buffer.images,
                        "image_buffer_size": image_buffer.buffer_size,
                    },
                    f"./checkpoints/models/{step + 1}.tar",
                )

            generator.train()
            mapping_network.train()
            style_extractor.train()


if __name__ == "__main__":
    main()
