"""Orchestrate training of model."""

import itertools
import math

import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from tqdm import tqdm
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
    "d_latent": 512,
    "image_size": 128,
    "image_channels": 1,
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
    "save_checkpoint_interval": 2_000,
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

log_resolution = int(math.log2(CONFIG["image_size"]))

discriminator = Discriminator(
    log_resolution=log_resolution, in_channels=CONFIG["image_channels"]
).to(device)

generator = Generator(
    log_resolution=log_resolution,
    d_latent=CONFIG["d_latent"],
    out_channels=CONFIG["image_channels"],
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

base_transforms = [
    transforms.Resize(CONFIG["image_size"]),
    transforms.CenterCrop(CONFIG["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

if CONFIG["image_channels"] == 1:
    base_transforms.append(transforms.Grayscale())

transform = transforms.Compose(
    base_transforms,
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

# * Training Loop


def main():
    """Training loop."""
    data_iter = itertools.cycle(dataloader)

    for step in tqdm(
        range(CONFIG["training_steps"]),
    ):
        # Discriminator
        discriminator_optimiser.zero_grad()

        log_disc_loss = 0
        for _ in range(CONFIG["gradient_accumulate_steps"]):
            w = mapping_network.get_w(CONFIG["batch_size"], n_gen_blocks, device)
            generated_images = generator.generate_images(
                CONFIG["batch_size"], w, device
            )

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
            w = mapping_network.get_w(CONFIG["batch_size"], n_gen_blocks, device)
            generated_images = generator.generate_images(
                CONFIG["batch_size"], w, device
            )

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

        if (step + 1) % CONFIG["log_generated_interval"] == 0 or (step + 1) == CONFIG[
            "training_steps"
        ]:
            tqdm.write(
                f"Step: {step + 1}/{CONFIG['training_steps']}, Generator loss: {log_gen_loss},"
                f"Discriminator loss: {log_disc_loss}"
            )

        if (step + 1) % CONFIG["save_checkpoint_interval"] == 0 or (step + 1) == CONFIG[
            "training_steps"
        ]:
            # Generate images
            generator.eval()
            mapping_network.eval()
            with torch.no_grad():
                w = mapping_network.get_w(32, n_gen_blocks, device)
                images = generator.generate_images(32, w, device)
            generator.train()
            mapping_network.train()

            save_grid(step + 1, images)


if __name__ == "__main__":
    main()
