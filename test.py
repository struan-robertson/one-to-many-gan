"""Utilities for running tests on trained models."""

# * Imports

import itertools
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.core.evaluation import validate_single
from src.data.config import load_config
from src.data.datasets import ShoeDataset, dataset_transform
from src.model.builder import Generator, MappingNetwork, StyleExtractor


def main(config_path: str):
    """Orchestrate evaluation."""
    # * Config

    config = load_config(config_path)

    # Initialisation
    device = torch.device(
        f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
    )

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    def load_checkpoint(path: Path):
        """Load model state from checkpoint dictionary."""
        checkpoint = torch.load(path)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])
        mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])

    # Models
    generator = (
        Generator(
            input_nc=config["data"]["image_channels"],
            w_dim=config["architecture"]["w_dim"],
            image_size=config["data"]["image_size"],
            min_latent_resolution=config["architecture"]["min_latent_resolution"],
            n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
        )
        .to(device)
        .eval()
    )

    mapping_network = (
        MappingNetwork(
            features=config["architecture"]["w_dim"],
            n_layers=config["architecture"]["mapping_network_layers"],
            style_mixing_prob=config["training"]["style_mixing_prob"],
        )
        .to(device)
        .eval()
    )

    # * Data

    transform = dataset_transform(config["data"]["image_size"])

    shoeprint_val_data = ShoeDataset(
        config["data"]["shoeprint_data_dir"], mode="is_val", transform=transform
    )

    shoeprint_val_dataloader = torch.utils.data.DataLoader(
        shoeprint_val_data,
        batch_size=100,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    shoeprints = next(itertools.cycle(shoeprint_val_dataloader))

    # * Evaluation

    training_runs = [entry for entry in Path("evaluations/").iterdir() if entry.is_dir()]

    work = len(training_runs) * 15 * 100

    with tqdm(total=work) as pbar:
        for run in training_runs:
            checkpoints = [file for file in run.glob("*.tar") if file.is_file()]
            checkpoints = sorted(checkpoints, key=lambda p: int(p.stem))

            for checkpoint in checkpoints:
                load_checkpoint(checkpoint)

                pbar.set_description(f"{run.name}/{checkpoint.stem}")

                inception_scores = []
                for shoeprint in shoeprints:
                    inception_score = validate_single(
                        config, device, shoeprint, mapping_network, generator
                    )
                    inception_scores.append(inception_score)

                    pbar.update(1)

                mean_score = np.mean(inception_scores)

                with (run / "scores.txt").open("a") as f:
                    f.write(f"{checkpoint.stem}: {mean_score}")
