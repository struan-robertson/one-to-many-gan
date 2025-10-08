"""Evalueate CIS score for a directory of trained models."""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from one_to_many_gan.core.evaluation import validate_cis
from one_to_many_gan.data.config import load_config
from one_to_many_gan.data.datasets import ShoeDataset, dataset_transform
from one_to_many_gan.model.builder import Generator, MappingNetwork

config = (
    load_config("config.toml")
    if len(sys.argv) < 2 or sys.argv[1] == ""
    else load_config(sys.argv[1])
)


# * Initialisation

device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)


# * Models
generator = (
    Generator(
        input_nc=config["data"]["image_channels"],
        s_dim=config["architecture"]["s_dim"],
        image_size=config["data"]["image_size"],
        min_latent_resolution=config["architecture"]["min_latent_resolution"],
        n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
    )
    .to(device)
    .eval()
)

mapping_network = (
    MappingNetwork(
        features=config["architecture"]["s_dim"],
        n_layers=config["architecture"]["mapping_network_layers"],
        style_mixing_prob=0,
        n_gen_blocks=generator.n_style_blocks,
    )
    .to(device)
    .eval()
)


def load_checkpoint(path: Path):
    """Load model state from checkpoint dictionary."""
    checkpoint = torch.load(path, map_location=device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])


# * Data

shoeprint_transform = dataset_transform(
    config["data"]["image_size"], *config["data"]["shoeprint_norm"], random_image_flip=False
)

shoeprint_val_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"], mode="val", transform=shoeprint_transform
)

shoeprint_val_dataloader = torch.utils.data.DataLoader(
    shoeprint_val_data,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)


# * Evaluation


def _test_cis(saved_models_path: Path):
    training_runs = [entry for entry in saved_models_path.iterdir() if entry.is_dir()]

    work = sum(1 for p in saved_models_path.rglob("*.tar") if p.is_file()) * 100

    shoeprints = next(iter(shoeprint_val_dataloader))

    with tqdm(total=work) as pbar:
        for run in training_runs:
            checkpoints = [file for file in run.glob("*.tar") if file.is_file()]
            checkpoints = sorted(checkpoints, key=lambda p: int(p.stem))

            for checkpoint in checkpoints:
                load_checkpoint(checkpoint)

                pbar.set_description(f"{run.name}/{checkpoint.stem}")

                inception_scores = []
                for shoeprint in shoeprints:
                    inception_score = validate_cis(
                        config, device, shoeprint, mapping_network, generator
                    )
                    inception_scores.append(inception_score)

                    pbar.update()

                mean_score = np.mean(inception_scores)

                with (run / "scores.txt").open("a") as f:
                    f.write(f"{checkpoint.stem}: {mean_score}\n")


if __name__ == "__main__":
    _test_cis(Path(sys.argv[2]))
