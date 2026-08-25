"""Generate n image checkpoints using random images from the dataset."""

import sys
from pathlib import Path
from typing import cast

import torch
import torchvision

from one_to_many_gan.data.config import parse_config
from one_to_many_gan.data.datasets import ShoeDataset, dataset_transform
from one_to_many_gan.model.builder import Generator, MappingNetwork, StyleExtractor

config = parse_config()

device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

# * Load models

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

checkpoint = torch.load(config["inference"]["checkpoint"], map_location=device)


def strip_prefix(state_dict):
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


generator.load_state_dict(strip_prefix(checkpoint["generator_state_dict"]))
mapping_network.load_state_dict(strip_prefix(checkpoint["mapping_network_state_dict"]))
style_extractor.load_state_dict(strip_prefix(checkpoint["style_extractor_state_dict"]))

generator = cast(Generator, generator.eval())
mapping_network = cast(MappingNetwork, mapping_network.eval())
style_extractor = cast(StyleExtractor, style_extractor.eval())

for param in generator.parameters():
    param.requires_grad = False
for param in mapping_network.parameters():
    param.requires_grad = False
for param in style_extractor.parameters():
    param.requires_grad = False

# * Load datasets

shoeprint_data = ShoeDataset(
    config["data"]["shoeprint_data_dir"],
    mode=None,
    transform=dataset_transform(
        config["data"]["image_size"],
        *config["data"]["shoeprint_norm"],
        random_image_flip=False,
    ),
    channels=config["data"]["image_channels"],
)

shoemark_data = ShoeDataset(
    config["data"]["shoemark_data_dir"],
    mode=None,
    transform=dataset_transform(
        config["data"]["image_size"],
        *config["data"]["shoemark_norm"],
        random_image_flip=False,
    ),
    channels=config["data"]["image_channels"],
)

# * Generation

ROWS = 8
TRANSLATION_COLS = 9
STYLE_COLS = 5


def generate_checkpoints(n: int, output_dir: Path | str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(n):
            shoeprints = shoeprint_data.random_sample(ROWS).to(device)
            shoemarks = shoemark_data.random_sample(ROWS).to(device)

            # Style vectors for translation grid columns
            s = mapping_network.get_single_s(
                batch_size=TRANSLATION_COLS - 1,
                device=device,
                mix_styles=False,
                domain_variable=1,
            )

            shoeprint_latents = generator.encode(shoeprints)
            shoemark_latents = generator.encode(shoemarks)

            # Reconstructed shoeprints (s=0 → shoeprint domain)
            s0 = torch.zeros(
                (generator.n_style_blocks, TRANSLATION_COLS - 1, config["architecture"]["s_dim"]),
                device=device,
            )
            reconstructed_shoeprints = generator.decode(shoeprint_latents, s0)

            real_shoemark_s = style_extractor(shoemarks)
            reconstructed_shoemarks = generator.decode(
                shoemark_latents,
                real_shoemark_s.expand(generator.n_style_blocks, *real_shoemark_s.shape),
            )
            translated_shoemarks = generator.decode(
                shoeprint_latents,
                real_shoemark_s.expand(generator.n_style_blocks, *real_shoemark_s.shape),
            )

            # ---- Translation checkpoint
            translation_grid = torch.stack(
                [
                    torch.stack(
                        [
                            real_shoeprint,
                            *generator.decode(latent.expand(TRANSLATION_COLS - 1, -1, -1, -1), s),
                        ]
                    )
                    for real_shoeprint, latent in zip(shoeprints, shoeprint_latents, strict=True)
                ]
            )

            torchvision.utils.save_image(
                translation_grid.view(
                    ROWS * TRANSLATION_COLS,
                    config["data"]["image_channels"],
                    *config["data"]["image_size"],
                ),
                output_dir / f"translation_{i}.png",
                nrow=TRANSLATION_COLS,
                padding=2,
                normalize=True,
            )

            # ---- Style checkpoint
            style_grid = torch.stack(
                [
                    torch.stack(
                        [
                            shoeprints[row],
                            reconstructed_shoeprints[row],
                            translated_shoemarks[row],
                            shoemarks[row],
                            reconstructed_shoemarks[row],
                        ]
                    )
                    for row in range(ROWS)
                ]
            )

            torchvision.utils.save_image(
                style_grid.view(
                    ROWS * STYLE_COLS,
                    config["data"]["image_channels"],
                    *config["data"]["image_size"],
                ),
                output_dir / f"style_{i}.png",
                nrow=STYLE_COLS,
                padding=2,
                normalize=True,
            )

            print(f"Checkpoint {i + 1}/{n} saved.")


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_dir = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else config["training"]["checkpoint_directory"]
        / config["training"]["training_run"]
        / "random_image_checkpoints"
    )
    generate_checkpoints(n, output_dir)
