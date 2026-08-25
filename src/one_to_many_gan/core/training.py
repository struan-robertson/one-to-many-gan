"""Classes and methods used for training models."""

from typing import cast

import torch

from one_to_many_gan.data.config import Config
from one_to_many_gan.model.builder import Discriminator, Generator, MappingNetwork, StyleExtractor
from one_to_many_gan.model.loss import (
    kl_loss_func,
    path_loss_func,
    style_cycle_loss_func,
    true_kl_loss_func,
)

# Clean up return value code
_detacher = lambda x: x.detach().cpu().item()


# * Image Buffer


# Adapted from CycleGAN
class ImageBuffer:
    """An image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    buffer_size: int
    num_imgs: int
    images: list[torch.Tensor]

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        if self.buffer_size < 1:
            raise ValueError

        self.num_imgs = 0
        self.images = []

    def __call__(self, images: torch.Tensor):
        return_images = []

        for image in images:
            image_unsqueezed = torch.unsqueeze(image.detach(), 0)
            # Fill buffer if it is not full
            if self.num_imgs < self.buffer_size:
                self.num_imgs += 1
                self.images.append(image_unsqueezed)
                return_images.append(image_unsqueezed)
            elif torch.rand(()).gt(0.5):
                random_id = cast(int, torch.randint(self.buffer_size, ()).item())
                # Clone tensors as they may be used many times
                cloned_image = self.images[random_id].clone()
                self.images[random_id] = image_unsqueezed
                return_images.append(cloned_image)
            else:
                return_images.append(image_unsqueezed)

        return torch.cat(return_images, 0)


# * Discriminator


def discriminator_step(
    config: Config,
    device: torch.device,
    discriminator: Discriminator,
    generator: Generator,
    mapping_network: MappingNetwork,
    discriminator_optimiser: torch.optim.Optimizer,
    shoeprints: torch.Tensor,
    real_shoemarks: torch.Tensor,
    image_buffer: ImageBuffer,
):
    """Take a step with the discriminator and return loss."""
    # Scale from [0,1] to [-1,1] and then take sign as indication of judgement
    discriminator_confidence = lambda scores: torch.sign(scores * 2 - 1).mean()

    discriminator_optimiser.zero_grad()

    # Generate fake shoemarks
    s = mapping_network.get_single_s(
        batch_size=config["training"]["batch_size"],
        device=device,
        domain_variable=1,
    )
    generated_shoemarks = generator(shoeprints, s)
    buffered_shoemarks = image_buffer(generated_shoemarks)

    # Calculate discriminator scores
    fake_scores = discriminator(buffered_shoemarks)
    real_scores = discriminator(real_shoemarks)

    # Calculate losses
    real_loss = torch.nn.functional.mse_loss(real_scores, torch.ones_like(real_scores))
    fake_loss = torch.nn.functional.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    disc_loss = (real_loss + fake_loss) / 2

    # Calculate discriminator confidence
    sign_real = discriminator_confidence(real_scores.detach())
    sign_fake = discriminator_confidence(fake_scores.detach()) * -1

    disc_loss.backward()
    discriminator_optimiser.step()

    return _detacher(disc_loss), (
        _detacher(sign_real),  # Discriminator confidences
        _detacher(sign_fake),
    )


# * Generator


def generator_step(
    config: Config,
    device: torch.device,
    generator: Generator,
    discriminator: Discriminator,
    mapping_network: MappingNetwork,
    style_extractor: StyleExtractor,
    generator_optimiser: torch.optim.Optimizer,
    mapping_network_optimiser: torch.optim.Optimizer,
    style_extractor_optimiser: torch.optim.Optimizer,
    real_shoeprints: torch.Tensor,
    real_shoemarks: torch.Tensor,
):
    """Take a step with the generator and return loss."""
    generator_optimiser.zero_grad()
    mapping_network_optimiser.zero_grad()
    style_extractor_optimiser.zero_grad()

    # KL loss
    # Combine for single forward pass
    combined_images = torch.cat([real_shoeprints, real_shoemarks], dim=0)
    combined_latents = generator.encode(combined_images)
    # true_kl_loss substitutes the SANTA formulation of Xie et al. (mean
    # squared pre-noise latents; pair with add_latent_noise) for the default
    # moment matching. .get keeps configs written before the flag working.
    if config["optimisation"].get("true_kl_loss", False):
        kl_loss = true_kl_loss_func(combined_latents)
    else:
        kl_loss = kl_loss_func(combined_latents)

    # Encoded latent variables
    # Not specified in paper, but in implementation Xie et al. add noise to latents.
    if config["architecture"]["add_latent_noise"]:
        combined_latents = combined_latents + torch.randn_like(combined_latents)
    shoeprint_latent, shoemark_latent = combined_latents.chunk(2, dim=0)

    # Reconstruction loss
    reconstruct_s = mapping_network.get_single_s(
        batch_size=config["training"]["batch_size"],
        device=device,
        domain_variable=0,
    )
    reconstructed_shoeprints = generator.decode(shoeprint_latent, reconstruct_s)
    reconstruction_loss = torch.nn.functional.l1_loss(reconstructed_shoeprints, real_shoeprints)

    # Identity loss
    real_shoemark_s = style_extractor(real_shoemarks)
    reconstructed_shoemarks = generator.decode(
        shoemark_latent,
        real_shoemark_s.expand(generator.n_style_blocks, *real_shoemark_s.shape),
    )
    identity_loss = torch.nn.functional.l1_loss(reconstructed_shoemarks, real_shoemarks)

    # GAN loss
    translation_s = mapping_network.get_single_s(
        batch_size=config["training"]["batch_size"],
        device=device,
        domain_variable=1,
    )
    generated_shoemarks = generator.decode(shoeprint_latent, translation_s)
    fake_shoemark_scores = discriminator(generated_shoemarks)
    gan_loss = torch.nn.functional.mse_loss(
        fake_shoemark_scores, torch.ones_like(fake_shoemark_scores)
    )

    # Style cycle loss
    style_loss_shoemarks = generated_shoemarks
    style_loss_s = translation_s[-1]
    reconstructed_s = style_extractor(style_loss_shoemarks)
    style_loss = style_cycle_loss_func(style_loss_s, reconstructed_s)

    # Path loss
    # Calculate random \theta for each image from uniform distribution between 0 and 1
    theta = torch.rand(config["training"]["batch_size"]).to(device)
    # H used in the central finite difference calculation
    cent_fin_diff_h = (
        torch.ones_like(theta)
        .to(device)
        .uniform_(
            config["optimisation"]["path_loss_jacobian_granularity"][0],
            config["optimisation"]["path_loss_jacobian_granularity"][1],
        )
    )
    d1 = (theta + cent_fin_diff_h / 2).clamp(0, 1)
    d2 = (theta - cent_fin_diff_h / 2).clamp(0, 1)
    s1, s2 = mapping_network.get_two_s(
        batch_size=config["training"]["batch_size"],
        device=device,
        domain_variables=(d1, d2),
    )
    features1 = generator.extract(shoeprint_latent, s1)
    features2 = generator.extract(shoeprint_latent, s2)
    path_loss = path_loss_func(features1, features2, cent_fin_diff_h)

    total_gen_loss = (
        gan_loss
        + config["optimisation"]["identity_loss_lambda"] * identity_loss
        + config["optimisation"]["reconstruction_loss_lambda"] * reconstruction_loss
        + config["optimisation"]["kl_loss_lambda"] * kl_loss
        + config["optimisation"]["path_loss_lambda"] * path_loss
        + config["optimisation"]["style_cycle_loss_lambda"] * style_loss
    )

    total_gen_loss.backward()
    generator_optimiser.step()
    mapping_network_optimiser.step()
    style_extractor_optimiser.step()

    return _detacher(total_gen_loss), (
        _detacher(gan_loss),
        _detacher(reconstruction_loss),
        _detacher(identity_loss),
        _detacher(kl_loss),
        _detacher(path_loss),
        _detacher(style_loss),
    )
