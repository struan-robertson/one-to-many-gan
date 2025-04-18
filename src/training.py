"""Classes and methods used for training models."""

import random
from collections.abc import Iterator
from typing import cast

import torch
from ada import AdaptiveDiscriminatorAugmentation

from .config import Config
from .loss import ADAp, kl_loss_func, path_loss_func, style_cycle_loss_func
from .models import Discriminator, Generator, MappingNetwork, StyleExtractor

# Clean up return value code
_detacher = lambda x: x.detach().cpu().item()


# * Discriminator

# ** Image Buffer


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
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(
                        0, self.buffer_size - 1
                    )  # randint is inclusive
                    # Clone tensors as they may be used many times
                    cloned_image = self.images[random_id].clone()
                    self.images[random_id] = image_unsqueezed
                    return_images.append(cloned_image)
                else:
                    return_images.append(image_unsqueezed)

        return torch.cat(return_images, 0)


# ** D Training Step


def discriminator_step(
    config: Config,
    device: torch.device,
    discriminator: Discriminator,
    generator: Generator,
    mapping_network: MappingNetwork,
    discriminator_optimiser: torch.optim.Optimizer,
    shoeprint_iter: Iterator[torch.Tensor],
    shoemark_iter: Iterator[torch.Tensor],
    image_buffer: ImageBuffer,
    ada: AdaptiveDiscriminatorAugmentation,
    ada_p: ADAp,
):
    """Take a step with the discriminator and return loss."""
    # Scale from [0,1] to [-1,1] and then take sign as indication of judgement
    discriminator_confidence = lambda scores: torch.sign(scores * 2 - 1).mean()

    discriminator_optimiser.zero_grad()

    # Generate fake shoemarks
    shoeprint_images = next(shoeprint_iter).to(device)
    w = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=1,
    )
    generated_shoemarks = generator(shoeprint_images, w)
    buffered_shoemarks = image_buffer(generated_shoemarks)
    augmented_fake_shoemarks = ada(buffered_shoemarks)

    # Get real shoemarks
    real_shoemarks = next(shoemark_iter).to(device)
    augmented_real_shoemarks = ada(real_shoemarks)

    # Calculate discriminator scores
    fake_scores = discriminator(augmented_fake_shoemarks)
    real_scores = discriminator(augmented_real_shoemarks)

    # Calculate losses
    real_loss = torch.nn.functional.mse_loss(real_scores, torch.ones_like(real_scores))
    fake_loss = torch.nn.functional.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    disc_loss = (real_loss + fake_loss) / 2

    # Calculate discriminator confidence
    sign_real = discriminator_confidence(real_scores.detach())
    sign_fake = discriminator_confidence(fake_scores.detach()) * -1

    # Update ADA p value
    ada_p.update_p(sign_real)

    disc_loss.backward()
    discriminator_optimiser.step()

    return _detacher(disc_loss), (
        _detacher(sign_real),  # Discriminator confidences
        _detacher(sign_fake),
    )


# * Generator

# ** Generator Train Step


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
    shoeprint_iter: Iterator[torch.Tensor],
    shoemark_iter: Iterator[torch.Tensor],
    ada: AdaptiveDiscriminatorAugmentation,
):
    """Take a step with the generator and return loss."""
    generator_optimiser.zero_grad()
    mapping_network_optimiser.zero_grad()
    style_extractor_optimiser.zero_grad()

    real_shoeprint_images = next(shoeprint_iter).to(device)
    real_shoemark_images = next(shoemark_iter).to(device)

    # KL loss
    # Combine for single forward pass
    combined_images = torch.cat([real_shoeprint_images, real_shoemark_images], dim=0)
    combined_latents = generator.encode(combined_images)
    kl_loss = kl_loss_func(combined_latents)

    # Encoded latent variables
    # Not specified in paper, but in implementation Xie et al. add noise to latents.
    if config["architecture"]["add_latent_noise"]:
        combined_latents = combined_latents + torch.randn_like(combined_latents)
    shoeprint_latent, shoemark_latent = combined_latents.chunk(2, dim=0)

    # Reconstruction loss
    reconstruct_w = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=0,
    )
    reconstruct_w = cast(torch.Tensor, reconstruct_w)
    reconstructed_shoeprints = generator.decode(shoeprint_latent, reconstruct_w)
    reconstruction_loss = torch.nn.functional.l1_loss(
        reconstructed_shoeprints, real_shoeprint_images
    )

    # Identity loss
    real_shoemark_w = style_extractor(real_shoemark_images)
    reconstructed_shoemarks = generator.decode(
        shoemark_latent,
        real_shoemark_w.expand(generator.n_style_blocks, *real_shoemark_w.shape),
    )
    identity_loss = torch.nn.functional.l1_loss(
        reconstructed_shoemarks, real_shoemark_images
    )

    # GAN loss
    translation_w = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=1,
    )
    translation_w = cast(torch.Tensor, translation_w)
    generated_shoemarks = generator.decode(shoeprint_latent, translation_w)
    augmented_generated_images = ada(generated_shoemarks)
    fake_shoemark_scores = discriminator(augmented_generated_images)
    gan_loss = torch.nn.functional.mse_loss(
        fake_shoemark_scores, torch.ones_like(fake_shoemark_scores)
    )

    # Style cycle loss
    style_loss_shoemarks = generated_shoemarks
    style_loss_w = translation_w[-1]
    reconstructed_w = style_extractor(style_loss_shoemarks)
    style_loss = style_cycle_loss_func(style_loss_w, reconstructed_w)

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
    w1, w2 = mapping_network.get_w(
        batch_size=config["training"]["batch_size"],
        n_gen_blocks=generator.n_style_blocks,
        device=device,
        domain_variable=(d1, d2),
    )
    features1 = generator.extract(shoeprint_latent, w1)
    features2 = generator.extract(shoeprint_latent, w2)
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
