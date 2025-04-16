"""Define typed config options."""

from pathlib import Path
from typing import TypedDict

import tomllib


class _Training(TypedDict):
    batch_size: int
    random_seed: int
    training_steps: int
    image_buffer_size: int
    style_mixing_prob: float
    deterministic_cuda_kernels: bool


class _Optimisation(TypedDict):
    style_cycle_loss_lambda: float
    identity_loss_lambda: float
    reconstruction_loss_lambda: float
    kl_loss_lambda: float
    path_loss_lambda: float
    path_loss_jacobian_granularity: tuple[float, float]
    learning_rate: float
    mapping_network_learning_rate: float
    adam_betas: tuple[float, float]


class _Ada(TypedDict):
    discriminator_real_acc_target: float
    ada_overfitting_measurement_n_images: int
    ada_adjustment_size: float


class _Evaluation(TypedDict):
    log_interval: int
    checkpoint_interval: int
    n_evaluation_images: int


class _Architecture(TypedDict):
    w_dim: int
    add_latent_noise: bool
    min_latent_resolution: int
    n_resnet_blocks: int
    mapping_network_layers: int


class _Data(TypedDict):
    image_size: tuple[int, int]
    image_channels: int
    shoeprint_data_dir: Path
    shoemark_data_dir: Path


class Config(TypedDict):
    """Config options used for training and running the model."""

    training: _Training
    optimisation: _Optimisation
    ada: _Ada
    evaluation: _Evaluation
    architecture: _Architecture
    data: _Data


def load_config(path: Path | str) -> Config:
    """Load a TOML file of hyperparameters into a dictionary."""
    path = Path(path)

    with path.open("rb") as f:
        config: Config = tomllib.load(f)  # type: ignore[assignment]

    return config
