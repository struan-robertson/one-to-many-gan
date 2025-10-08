"""Define typed config options."""

import tomllib
from pathlib import Path
from typing import TypedDict


class _Training(TypedDict):
    batch_size: int
    random_seed: int
    batch_agnostic_steps: int
    image_buffer_size: int
    style_mixing_prob: float
    random_image_flip: bool
    deterministic_cuda_kernels: bool
    gpu_number: int
    checkpoint_directory: Path
    training_run: str


class _Inference(TypedDict):
    checkpoint: Path
    batch_size: int


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


class _Evaluation(TypedDict):
    log_interval: int
    checkpoint_interval: int
    n_evaluation_images: int
    cond_is_n_evaluation_images: int
    use_training_data: bool


class _Architecture(TypedDict):
    s_dim: int
    add_latent_noise: bool
    min_latent_resolution: int
    n_resnet_blocks: int
    mapping_network_layers: int


class _Data(TypedDict):
    image_size: tuple[int, int]
    image_channels: int
    shoeprint_data_dir: Path
    shoemark_data_dir: Path
    shoemark_norm: tuple[float, float]
    shoeprint_norm: tuple[float, float]


class Config(TypedDict):
    """Config options used for training and running the model."""

    training: _Training
    inference: _Inference
    optimisation: _Optimisation
    evaluation: _Evaluation
    architecture: _Architecture
    data: _Data


def load_config(path: Path | str) -> Config:
    """Load a TOML file of hyperparameters into a dictionary."""
    path = Path(path)

    with path.open("rb") as f:
        config: Config = tomllib.load(f)  # type: ignore[assignment]

    # Initialise Path objects
    config["training"]["checkpoint_directory"] = Path(config["training"]["checkpoint_directory"])
    config["data"]["shoeprint_data_dir"] = Path(config["data"]["shoeprint_data_dir"])
    config["data"]["shoemark_data_dir"] = Path(config["data"]["shoemark_data_dir"])
    config["inference"]["checkpoint"] = Path(config["inference"]["checkpoint"])

    return config
