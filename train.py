"""Initialise state and run training loop."""

import gc
import itertools
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data
from ada import AdaptiveDiscriminatorAugmentation
from src.core.evaluation import (
    Logger,
    image_checkpoint,
    model_checkpoint,
    val_checkpoint,
)
from src.core.training import ImageBuffer, discriminator_step, generator_step
from src.data.config import load_config
from src.data.datasets import ShoeDataset
from src.model.builder import Discriminator, Generator, MappingNetwork, StyleExtractor
from src.model.loss import ADAp
from torchvision import transforms
from tqdm import tqdm, trange


def main(config_path: str):
    """Orchestrate training."""
    config = load_config(config_path)

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

    device = torch.device(
        f"cuda:{config['training']['gpu_number']}"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True

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
        batch_size=config["evaluation"]["inference_batch_size"],
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

    """Training loop."""
    for step in trange(config["training"]["training_steps"], dynamic_ncols=True):
        # set adaptive discriminator augmentation p
        adaptive_discriminator_augmentation.set_p(ada_p())
        logger.log_ada_ps.append(ada_p())

        # Train discriminator
        disc_loss, (real_accuracy, fake_accuracy) = discriminator_step(
            config,
            device,
            discriminator,
            generator,
            mapping_network,
            discriminator_optimiser,
            shoeprint_iter,
            shoemark_iter,
            image_buffer,
            adaptive_discriminator_augmentation,
            ada_p,
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
            shoeprint_iter,
            shoemark_iter,
            adaptive_discriminator_augmentation,
        )
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

            log_dir = (
                config["training"]["checkpoint_directory"]
                / config["training"]["training_run"]
            )

            log_dir.mkdir(parents=True, exist_ok=True)

            with (log_dir / "log").open("a") as file:
                file.write(log + "\n")

        if (step + 1) % config["evaluation"]["checkpoint_interval"] == 0 or (
            step + 1
        ) == config["training"]["training_steps"]:
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
                image_checkpoint(
                    step,
                    config,
                    device,
                    shoeprint_iter,
                    shoemark_iter,
                    mapping_network,
                    generator,
                    style_extractor,
                )

                val_checkpoint(
                    step, config, device, shoeprint_val_iter, mapping_network, generator
                )

                model_checkpoint(
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
                    ada_p,
                    image_buffer,
                )

            generator.train()
            mapping_network.train()
            style_extractor.train()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "":
        main("config.toml")
    else:
        main(sys.argv[1])
