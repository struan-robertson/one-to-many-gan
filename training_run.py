"""Orchestrate training of model."""

import itertools
import math
from typing import cast

import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm
from xogan.data import Edges2ShoesDataset, ShoeDataset
from xogan.loss import (
    GradientPenalty,
    PathLengthPenalty,
    discriminator_loss,
    generator_loss,
)
from xogan.stylegan2 import Discriminator, Generator, MappingNetwork
from xogan.utils import save_grid

# * Hyperparameters

CONFIG = {
    "gradient_penalty_coeficcient": 10.0,
    "path_length_penalty_coeficcient": 0.99,
    "batch_size": 32,
    "d_latent": 32,
    "image_size": (128, 64),
    "image_channels": 1,
    "mapping_network_layers": 2,
    "learning_rate": 1e-3,
    "mapping_network_learning_rate": 1e-5,  # 100x less
    "gradient_accumulate_steps": 1,
    "discriminator_steps": 1,
    "generator_steps": 1,
    "adam_betas": (0.0, 0.99),
    "style_mixing_prob": 0.9,
    "training_steps": 300_000,
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
    d_latent=CONFIG["d_latent"],
    img_channels=CONFIG["image_channels"],
).to(device)
n_gen_blocks = generator.n_blocks

mapping_network = MappingNetwork(
    features=CONFIG["d_latent"],
    n_layers=CONFIG["mapping_network_layers"],
    style_mixing_prob=CONFIG["style_mixing_prob"],
).to(device)


# ** Regularisation

path_length_penalty = PathLengthPenalty(
    beta=CONFIG["path_length_penalty_coeficcient"]
).to(device)

gradient_penalty = GradientPenalty().to(device)

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


# ** Data

transform = transforms.Compose(
    [
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

shoemark_data = ShoeDataset(
    "~/Datasets/Santa Partitioned/Shoemarks", mode="train", transform=transform
)
# shoemark_data = Edges2ShoesDataset(
#     "~/Datasets/edges2shoes/", mode="train", transform=transform, type_="shoe"
# )
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
# shoeprint_data = Edges2ShoesDataset(
#     "~/Datasets/edges2shoes/", mode="train", transform=transform, type_="edge"
# )
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


# * Training Loop

# ** Discriminator


def discriminator_step(step: int):
    """Take a step with the discriminator and return loss."""
    discriminator_optimiser.zero_grad()

    log_disc_loss = 0
    real_accuracy = torch.zeros(())
    fake_accuracy = torch.zeros(())
    for _ in range(CONFIG["gradient_accumulate_steps"]):
        w = mapping_network.get_w(CONFIG["batch_size"], n_gen_blocks, device)

        shoeprint_images = next(shoeprint_iter).to(device)
        generated_shoemarks = generator.generate_images(
            CONFIG["batch_size"], shoeprint_images, w, device
        )

        fake_output = discriminator(generated_shoemarks.detach())
        real_shoemarks = next(shoemark_iter).to(device)

        if (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0:
            real_shoemarks.requires_grad_()

        real_output = discriminator(real_shoemarks)

        # r_t for adaptive discriminator normalisation
        sign_real = torch.sign(real_output.detach())
        real_accuracy = torch.mean(sign_real)

        sign_fake = torch.sign(fake_output.detach())
        fake_accuracy = torch.mean(sign_fake)

        disc_loss = discriminator_loss(real_output, fake_output)

        if (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0:
            gp = gradient_penalty(real_shoemarks, real_output)

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

    log_disc_loss = cast(torch.Tensor, log_disc_loss)
    return log_disc_loss.detach().cpu().numpy(), (
        real_accuracy.cpu().numpy(),
        fake_accuracy.cpu().numpy(),
    )


# ** Generator


def generator_step(step: int):
    """Take a step with the generator and return loss."""
    generator_optimiser.zero_grad()
    mapping_network_optimiser.zero_grad()

    log_gen_loss = 0
    for _ in range(CONFIG["gradient_accumulate_steps"]):
        w = mapping_network.get_w(CONFIG["batch_size"], n_gen_blocks, device)

        shoeprint_images = next(shoeprint_iter).to(device)
        generated_images = generator.generate_images(
            CONFIG["batch_size"], shoeprint_images, w, device
        )

        fake_shoemarks = discriminator(generated_images)

        gen_loss = generator_loss(fake_shoemarks)

        if (
            step > CONFIG["lazy_path_penalty_after"]
            and (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0
        ):
            plp = path_length_penalty(w, generated_images)

            if not torch.isnan(plp):
                gen_loss = gen_loss + plp

        log_gen_loss += gen_loss

        gen_loss.backward()

    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(mapping_network.parameters(), max_norm=1.0)

    generator_optimiser.step()
    mapping_network_optimiser.step()

    log_gen_loss = cast(torch.Tensor, log_gen_loss)
    return log_gen_loss.detach().cpu().numpy()


# ** Main Loop


def main():
    """Training loop."""
    for step in tqdm(
        range(CONFIG["training_steps"]),
    ):
        disc_losses = []
        real_accuracies = []
        fake_accuracies = []
        for _ in range(CONFIG["discriminator_steps"]):
            disc_loss, (real_accuracy, fake_accuracy) = discriminator_step(step)
            disc_losses.append(disc_loss)
            real_accuracies.append(real_accuracy)
            fake_accuracies.append(fake_accuracy)

        mean_disc_loss = np.mean(disc_losses)
        mean_real_accuracy = np.mean(real_accuracies)
        mean_fake_accuracy = np.mean(fake_accuracies)

        gen_losses = []
        for _ in range(CONFIG["generator_steps"]):
            gen_loss = generator_step(step)
            gen_losses.append(gen_loss)

        mean_gen_loss = np.mean(gen_losses)

        # Logging

        if (step + 1) % CONFIG["log_generated_interval"] == 0 or (step + 1) == CONFIG[
            "training_steps"
        ]:
            tqdm.write(
                f"Step: {step + 1}/{CONFIG['training_steps']}, Generator loss: {mean_gen_loss},"
                f"Discriminator loss: {mean_disc_loss},"
                f" Discriminator real/fake sign: {mean_real_accuracy}/{mean_fake_accuracy}"
            )

        if (step + 1) % CONFIG["save_checkpoint_interval"] == 0 or (step + 1) == CONFIG[
            "training_steps"
        ]:
            # Generate images
            generator.eval()
            mapping_network.eval()
            with torch.no_grad():
                w = mapping_network.get_w(32, n_gen_blocks, device)
                shoeprint_images = next(shoeprint_iter).to(device)
                images = generator.generate_images(32, shoeprint_images, w, device)
            generator.train()
            mapping_network.train()

            save_grid(step + 1, images)


if __name__ == "__main__":
    main()
