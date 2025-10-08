"""Initialise state and run training loop."""

import gc
import math
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from one_to_many_gan.core.evaluation import (
    Logger,
    create_image_checkpoint,
    create_model_checkpoint,
    validate_kid_fid,
    write_logfile,
)
from one_to_many_gan.core.training import ImageBuffer, discriminator_step, generator_step
from one_to_many_gan.data.config import load_config
from one_to_many_gan.data.datasets import CyclingDataLoader, ShoeDataset, dataset_transform
from one_to_many_gan.model.builder import Discriminator, Generator, MappingNetwork, StyleExtractor

config = (
    load_config("config.toml")
    if len(sys.argv) < 2 or sys.argv[1] == ""
    else load_config(sys.argv[1])
)

total_steps = math.ceil(
    config["training"]["batch_agnostic_steps"] / config["training"]["batch_size"]
)

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

# ** PyTorch

device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True

# ** Models

discriminator = Discriminator(input_nc=config["data"]["image_channels"]).to(device)

generator = Generator(
    input_nc=config["data"]["image_channels"],
    s_dim=config["architecture"]["s_dim"],
    image_size=config["data"]["image_size"],
    min_latent_resolution=config["architecture"]["min_latent_resolution"],
    n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
).to(device)

mapping_network = MappingNetwork(
    features=config["architecture"]["s_dim"],
    n_layers=config["architecture"]["mapping_network_layers"],
    style_mixing_prob=config["training"]["style_mixing_prob"],
    n_gen_blocks=generator.n_style_blocks,
).to(device)

style_extractor = StyleExtractor(
    input_nc=config["data"]["image_channels"], s_dim=config["architecture"]["s_dim"]
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

# ** Data

val_data_mode = "train" if config["evaluation"]["use_training_data"] else "val"

# A bit wasteful as we hold the same images in memory twice but the dataset is small
shoemark_data = ShoeDataset(
    config["data"]["shoemark_data_dir"],
    mode="train",
    transform=dataset_transform(
        config["data"]["image_size"],
        *config["data"]["shoemark_norm"],
        random_image_flip=config["training"]["random_image_flip"],
    ),
)
shoemark_val_data = ShoeDataset(
    config["data"]["shoemark_data_dir"],
    mode=val_data_mode,
    transform=dataset_transform(
        config["data"]["image_size"],
        *config["data"]["shoemark_norm"],
        random_image_flip=False,
    ),
)

shoemark_dataloader = torch.utils.data.DataLoader(
    shoemark_data,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=0,  # Data is held in memory
    drop_last=True,
    pin_memory=True,
)

shoeprint_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"],
    mode="train",
    transform=dataset_transform(
        config["data"]["image_size"],
        *config["data"]["shoeprint_norm"],
        random_image_flip=config["training"]["random_image_flip"],
    ),
)
shoeprint_val_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"],
    mode=val_data_mode,
    transform=dataset_transform(
        config["data"]["image_size"],
        *config["data"]["shoeprint_norm"],
        random_image_flip=False,
    ),
)

shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True,
)
shoeprint_val_dataloader = torch.utils.data.DataLoader(
    shoeprint_val_data,
    batch_size=config["inference"]["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True,
)

# Required as there are different numbers of shoemarks and shoeprints
shoemark_iter = CyclingDataLoader(shoemark_dataloader)
shoeprint_iter = CyclingDataLoader(shoeprint_dataloader)
shoeprint_val_iter = CyclingDataLoader(shoeprint_val_dataloader)

image_buffer = ImageBuffer(config["training"]["image_buffer_size"])

# ** Logging

logger = Logger(total_steps)

# * Training Loop


def _training_loop():
    # Use the same images in the image checkpoints
    shoeprint_checkpoint_images = shoeprint_val_data.random_sample(8)
    shoemark_checkpoint_images = shoemark_val_data.random_sample(8)

    for step in trange(1, total_steps, dynamic_ncols=True):
        shoeprints = next(shoeprint_iter).to(device)
        shoemarks = next(shoemark_iter).to(device)

        # Train discriminator
        disc_loss, (real_accuracy, fake_accuracy) = discriminator_step(
            config,
            device,
            discriminator,
            generator,
            mapping_network,
            discriminator_optimiser,
            shoeprints,
            shoemarks,
            image_buffer,
        )
        logger.log_total_disc_losses.append(disc_loss)
        logger.log_disc_real_accs.append(real_accuracy)
        logger.log_disc_fake_accs.append(fake_accuracy)

        # Train generator
        (
            total_gen_loss,
            (gan_loss, rec_loss, idt_loss, kl_loss, path_loss, style_loss),
        ) = generator_step(
            config,
            device,
            generator,
            discriminator,
            mapping_network,
            style_extractor,
            generator_optimiser,
            mapping_network_optimiser,
            style_extractor_optimiser,
            shoeprints,
            shoemarks,
        )
        logger.log_total_gen_losses.append(total_gen_loss)
        logger.log_gan_losses.append(gan_loss)
        logger.log_rec_losses.append(rec_loss)
        logger.log_idt_losses.append(idt_loss)
        logger.log_kl_losses.append(kl_loss)
        logger.log_path_losses.append(path_loss)
        logger.log_style_losses.append(style_loss)

        # TODO investigate why final evaluation is not running
        if step % config["evaluation"]["log_interval"] == 0 or step == total_steps:
            log_str = logger.print(step)
            write_logfile(config, log_str)

        if step % config["evaluation"]["checkpoint_interval"] == 0 or step == total_steps:
            generator.eval()
            mapping_network.eval()
            style_extractor.eval()

            # Free memory
            generator_optimiser.zero_grad(set_to_none=True)
            mapping_network_optimiser.zero_grad(set_to_none=True)
            style_extractor_optimiser.zero_grad(set_to_none=True)
            discriminator_optimiser.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                create_image_checkpoint(
                    step,
                    config,
                    device,
                    shoeprint_checkpoint_images,
                    shoemark_checkpoint_images,
                    mapping_network,
                    generator,
                    style_extractor,
                )

                fid_score, kid_score = validate_kid_fid(
                    config, device, shoeprint_val_iter, mapping_network, generator
                )
                write_logfile(config, f"Step {step} | fid: {fid_score}, kid: {kid_score}")

                create_model_checkpoint(
                    step,
                    config,
                    generator,
                    discriminator,
                    mapping_network,
                    style_extractor,
                    generator_optimiser,
                    discriminator_optimiser,
                    mapping_network_optimiser,
                    style_extractor_optimiser,
                    image_buffer,
                )

            generator.train()
            mapping_network.train()
            style_extractor.train()


if __name__ == "__main__":
    _training_loop()
