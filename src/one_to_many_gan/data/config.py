"""Typed configuration for the one-to-many GAN.

Contract: defaults < config.toml < CLI overrides. Every key is declared in the
dataclasses below; an unknown key in the TOML or on the command line is a hard
error. CLI overrides use the dotted path of the field, values parsed as TOML
literals so booleans and lists work:

    python src/train.py config.toml --training.training_run santa_kl_1 \
        --optimisation.true_kl_loss true --architecture.add_latent_noise true

The loaders return a plain nested dict, so options are read as
config["training"]["batch_size"] throughout the project.
"""

import argparse
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints


@dataclass
class Training:
    """Batch, seeding, and run identity."""

    batch_size: int = 4
    random_seed: int = 421
    # Equivalent to the number of steps if batch_size=1
    batch_agnostic_steps: int = 300_000
    image_buffer_size: int = 100
    # Probability to use two different style vectors in the decoder
    style_mixing_prob: float = 0.9
    random_image_flip: bool = True
    # Exact results between different GPUs, but 4x slower
    deterministic_cuda_kernels: bool = False
    gpu_number: int = 0
    checkpoint_directory: Path = Path("checkpoints")
    training_run: str | None = None


@dataclass
class Inference:
    """The checkpoint generation and evaluation scripts load."""

    checkpoint: Path | None = None
    batch_size: int = 32


@dataclass
class Optimisation:
    """Loss weights and optimiser settings."""

    style_cycle_loss_lambda: float = 5.0
    identity_loss_lambda: float = 5.0
    reconstruction_loss_lambda: float = 5.0
    kl_loss_lambda: float = 0.01
    # Substitution study: use the SANTA (Xie et al.) KL formulation — mean
    # squared latents, requires add_latent_noise — instead of moment matching
    true_kl_loss: bool = False
    path_loss_lambda: float = 0.1
    # Min and max for sampling h to approximate the Jacobian for path length
    path_loss_jacobian_granularity: tuple[float, float] = (0.1, 0.2)
    learning_rate: float = 2e-3
    mapping_network_learning_rate: float = 2e-5  # 100x less
    adam_betas: tuple[float, float] = (0.5, 0.99)


@dataclass
class Evaluation:
    """Logging cadence and the seeded FID/KID/CIS measurements."""

    log_interval: int = 500
    checkpoint_interval: int = 5
    n_evaluation_images: int = 10_000
    # Number of images generated for the conditional inception score
    cond_is_n_evaluation_images: int = 100
    # Counter-intuitive but recommended by the literature
    use_training_data: bool = False
    eval_seed: int = 0
    validate_during_training: bool = True


@dataclass
class Architecture:
    """Generator capacity and the style pathway."""

    s_dim: int = 6  # Dimensionality of the style vector
    add_latent_noise: bool = False  # Add noise to latent feature maps
    min_latent_resolution: int = 64
    n_resnet_blocks: int = 7
    mapping_network_layers: int = 2


@dataclass
class Data:
    """Image geometry and the impression directories."""

    image_size: tuple[int, int] = (512, 256)
    image_channels: int = 1
    shoeprint_data_dir: Path | None = None
    shoemark_data_dir: Path | None = None
    shoemark_norm: tuple[float, float] = (0.5, 0.5)
    shoeprint_norm: tuple[float, float] = (0.5, 0.5)


@dataclass
class Config:
    """Every option used for training and running the model."""

    training: Training = field(default_factory=Training)
    inference: Inference = field(default_factory=Inference)
    optimisation: Optimisation = field(default_factory=Optimisation)
    evaluation: Evaluation = field(default_factory=Evaluation)
    architecture: Architecture = field(default_factory=Architecture)
    data: Data = field(default_factory=Data)


_REQUIRED = (
    "training.training_run",
    "data.shoeprint_data_dir",
    "data.shoemark_data_dir",
)


def _coerce(annotation, value, path):
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        if value == "" or value is None:
            return None
        annotation = next(a for a in get_args(annotation) if a is not type(None))
        origin = get_origin(annotation)
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            msg = f"{path}: expected a path string, got {value!r}"
            raise TypeError(msg)
        return Path(value).expanduser()
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            msg = f"{path}: expected a list, got {value!r}"
            raise TypeError(msg)
        return tuple(value)
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if annotation is int and isinstance(value, bool):
        msg = f"{path}: expected int, got bool"
        raise TypeError(msg)
    if annotation in (bool, int, float, str) and not isinstance(value, annotation):
        msg = f"{path}: expected {annotation.__name__}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _apply(obj, mapping, path=""):
    hints = get_type_hints(type(obj))
    valid = {f.name for f in fields(obj)}
    for key, value in mapping.items():
        if key not in valid:
            msg = f"unknown config key: {path}{key}"
            raise KeyError(msg)
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                msg = f"{path}{key}: expected a table"
                raise TypeError(msg)
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, _coerce(hints[key], value, f"{path}{key}"))


def _get(config, dotted):
    obj = config
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _validate(config: Config):
    missing = [dotted for dotted in _REQUIRED if _get(config, dotted) is None]
    if missing:
        msg = "missing required config keys: " + ", ".join(missing)
        raise ValueError(msg)


def load_config(path: Path | str) -> dict:
    """Load a TOML file over the schema defaults; unknown keys are errors."""
    config = Config()
    with Path(path).open("rb") as f:
        _apply(config, tomllib.load(f))
    _validate(config)
    return asdict(config)


def _leaf_paths(cls, prefix=""):
    hints = get_type_hints(cls)
    for f in fields(cls):
        if is_dataclass(hints[f.name]):
            yield from _leaf_paths(hints[f.name], f"{prefix}{f.name}.")
        else:
            yield f"{prefix}{f.name}"


def parse_config(argv=None, default_config="config.toml") -> dict:
    """Config path plus dotted-path CLI overrides (--training.batch_size 8)."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("config", nargs="?", default=default_config,
                        help="config TOML (default: %(default)s)")
    for dotted in _leaf_paths(Config):
        parser.add_argument(f"--{dotted}", dest=dotted, metavar="VALUE")
    args = parser.parse_args(argv)

    config = Config()
    with Path(args.config).open("rb") as f:
        _apply(config, tomllib.load(f))
    for dotted, raw in vars(args).items():
        if dotted == "config" or raw is None:
            continue
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # bare strings need no quoting on the command line
        *parents, leaf = dotted.split(".")
        _apply(_get(config, ".".join(parents)) if parents else config, {leaf: value},
               f"{'.'.join(parents)}." if parents else "")
    _validate(config)
    return asdict(config)
