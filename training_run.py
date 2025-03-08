"""Orchestrate training of model."""

import functools
import math
import operator
import sys
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from xogan.loss import (
    GradientPenalty,
    PathLengthPenalty,
    discriminator_loss,
    generator_loss,
)
from xogan.stylegan2 import Discriminator, Generator, MappingNetwork

# * Hyperparameters

CONFIG = {
    "gradient_penalty_coeficcient": 10.0,
    "path_length_penalty_coeficcient": 0.99,
    "batch_size": 32,
    "d_latent": 512,
    "image_size": 32,
    "mapping_network_layers": 8,
    "learning_rate": 1e-3,
    "mapping_network_learning_rate": 1e-5,
    "gradient_accumulate_steps": 1,
    "adam_betas": (0.0, 0.99),
    "style_mixing_prob": 0.9,
    "training_steps": 150_000,
    "lazy_gradient_penalty_interval": 4,
    "lazy_path_penalty_interval": 32,
    "lazy_path_penalty_after": 5_000,
    "log_generated_interval": 500,
    "save_checkpoint_interval": 1_000,
}

# * Initialisation

# ** Device

device = (
    torch.cuda.current_device()
    if torch.cuda.is_available()
    else torch.cpu.current_device()
)

# ** Models

log_resolution = int(math.log2(CONFIG["image_size"]))

discriminator = Discriminator(log_resolution=log_resolution).to(device)

generator = Generator(log_resolution=log_resolution, d_latent=CONFIG["d_latent"]).to(
    device
)
n_gen_blocks = generator.n_blocks

mapping_network = MappingNetwork(
    features=CONFIG["d_latent"], n_layers=CONFIG["mapping_network_layers"]
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
        transforms.CenterCrop(CONFIG["image_size"]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ],
)

data = torchvision.datasets.CelebA(root="./data", transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)


def get_w():
    """Sample z randomly and get w from mapping network.

    Style mixing is also applied randomly."""
    if torch.rand(()).item() < CONFIG["style_mixing_prob"]:
        cross_over_point = int(torch.rand(()).item() * n_gen_blocks)

        z1 = torch.randn(CONFIG["batch_size"], CONFIG["d_latent"]).to(device)
        z2 = torch.randn(CONFIG["batch_size"], CONFIG["d_latent"]).to(device)

        w1 = mapping_network(z1)
        w2 = mapping_network(z2)

        w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
        w2 = w2[None, :, :].expand(n_gen_blocks - cross_over_point, -1, -1)
        return torch.cat((w1, w2), dim=0)

    z = torch.randn(CONFIG["batch_size"], CONFIG["d_latent"]).to(device)
    w = mapping_network(z)

    return w[None, :, :].expand(n_gen_blocks, -1, -1)


def get_noise():
    """Generate noise for each generator block."""
    noise = []
    resolution = 4  # Shape of learned constant

    for i in range(n_gen_blocks):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(
                CONFIG["batch_size"], 1, resolution, resolution, device=device
            )

        n2 = torch.randn(CONFIG["batch_size"], 1, resolution, resolution, device=device)

        noise.append((n1, n2))

        resolution *= 2

    return noise


def generate_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate images using the generator."""
    w = get_w()
    noise = get_noise()
    images = generator(w, noise)

    return images, w


def save_grid(step: int):
    """Save a grid of generated images to a file."""
    generator.eval()

    with torch.no_grad():
        images, _ = generate_images()

    # Put colour channel as final dim
    images = images.permute(0, 2, 3, 1)

    # Scale to between 0 and 1
    images = (images - images.min()) / (images.max() - images.min())

    # Convert to numpy and move to cpu
    images = images.cpu().numpy()

    plt.ioff()

    _, axes = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))

    for ax, image in zip(
        functools.reduce(operator.iadd, axes.tolist(), []),
        images,
        strict=True,
    ):
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()

    savepath: Path = Path("./checkpoints")
    savepath.mkdir(exist_ok=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(savepath / f"{step}.png", dpi=300, bbox_inches="tight")

    plt.close()
    plt.ion()

    generator.train()


def main():
    """Training loop."""
    data_iter = iter(dataloader)

    for step in tqdm(
        range(CONFIG["training_steps"]),
    ):
        # Discriminator
        discriminator_optimiser.zero_grad()

        log_disc_loss = 0
        for _ in range(CONFIG["gradient_accumulate_steps"]):
            generated_images, _ = generate_images()

            fake_output = discriminator(generated_images.detach())
            real_images, _ = next(data_iter)
            real_images = real_images.to(device)

            if (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0:
                real_images.requires_grad_()

            real_output = discriminator(real_images)

            real_loss, fake_loss = discriminator_loss(real_output, fake_output)
            disc_loss = real_loss + fake_loss

            if (step + 1) % CONFIG["lazy_gradient_penalty_interval"] == 0:
                gp = gradient_penalty(real_images, real_output)

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

        # Generator
        generator_optimiser.zero_grad()
        mapping_network_optimiser.zero_grad()

        log_gen_loss = 0
        for _ in range(CONFIG["gradient_accumulate_steps"]):
            generated_images, w = generate_images()

            fake_output = discriminator(generated_images)

            gen_loss = generator_loss(fake_output)

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

        # Logging

        if (step + 1) % CONFIG["log_generated_interval"] == 0:
            tqdm.write(
                f"Step: {step + 1}/{CONFIG['training_steps']}, Generator loss: {log_gen_loss},"
                f"Discriminator loss: {log_disc_loss}"
            )

        if (step + 1) % CONFIG["save_checkpoint_interval"] == 0:
            save_grid(step + 1)


if __name__ == "__main__":
    main()
