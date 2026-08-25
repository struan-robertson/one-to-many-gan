"""Evaluate CIS score for a directory of trained models."""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torchvision.models.inception import inception_v3
from tqdm import tqdm

from one_to_many_gan.core.evaluation import validate_cis, validate_kid_fid
from one_to_many_gan.data.config import parse_config
from one_to_many_gan.data.datasets import CyclingDataLoader, ShoeDataset, dataset_transform
from one_to_many_gan.model.builder import Generator, MappingNetwork

config = parse_config()


# * Initialisation

device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

torch.backends.fp32_precision = "tf32"
torch.backends.cuda.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"


# * Models
generator = Generator(
    input_nc=config["data"]["image_channels"],
    s_dim=config["architecture"]["s_dim"],
    image_size=config["data"]["image_size"],
    min_latent_resolution=config["architecture"]["min_latent_resolution"],
    n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
)

mapping_network = MappingNetwork(
    features=config["architecture"]["s_dim"],
    n_layers=config["architecture"]["mapping_network_layers"],
    style_mixing_prob=0,
    n_gen_blocks=generator.n_style_blocks,
)

generator = generator.to(device).eval()
mapping_network = mapping_network.to(device).eval()
inception_model = inception_v3(weights="DEFAULT").to(device)
inception_model.eval()


def load_checkpoint(path: Path):
    """Load model state from checkpoint dictionary."""
    checkpoint = torch.load(path, map_location=device)

    def strip_prefix(state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    generator.load_state_dict(strip_prefix(checkpoint["generator_state_dict"]))
    mapping_network.load_state_dict(strip_prefix(checkpoint["mapping_network_state_dict"]))


# * Data

shoeprint_transform = dataset_transform(
    config["data"]["image_size"],
    *config["data"]["shoeprint_norm"],
    random_image_flip=False,
)

mode = "train"

shoeprint_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"],
    mode=mode,
    transform=shoeprint_transform,
    channels=config["data"]["image_channels"],
)

shoeprint_dataloader = torch.utils.data.DataLoader(
    shoeprint_data,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)

# * Evaluation


def _test_kid_fid(saved_models_path: Path):
    training_runs = [entry for entry in saved_models_path.iterdir() if entry.is_dir()]
    work = sum(1 for p in saved_models_path.rglob("*.tar") if p.is_file())

    with tqdm(total=work, dynamic_ncols=True) as pbar:
        for run in training_runs:
            checkpoints = [file for file in run.rglob("*.tar") if file.is_file()]
            checkpoints = sorted(checkpoints, key=lambda p: int(p.stem))

            for checkpoint in checkpoints:
                shoeprint_iter = CyclingDataLoader(shoeprint_dataloader)
                load_checkpoint(checkpoint)
                pbar.set_description(f"{run.name}/{checkpoint.stem}")

                # Same per-checkpoint seeding as the in-training validation and
                # rescore_best.py, so all three score identically at batch 64
                torch.manual_seed(config["evaluation"].get("eval_seed", 0))
                with torch.no_grad():
                    fid_score, kid_score = validate_kid_fid(
                        config,
                        device,
                        shoeprint_iter,
                        mapping_network,
                        generator,
                        saved_models_path / "val",
                        config["data"]["shoemark_data_dir"] / "train",
                    )

                    with (run / "kid_fid_scores.txt").open("a") as f:
                        f.write(f"Step {checkpoint.stem} | kid: {kid_score} fid: {fid_score}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: test_kid_fid.py <config.toml> <checkpoints-dir>  "
                 "(sweeps every run directory under <checkpoints-dir>)")
    _test_kid_fid(Path(sys.argv[2]))
