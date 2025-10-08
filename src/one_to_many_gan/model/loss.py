"""Losses and penalties."""

import torch
from torch.nn import functional as F

# * Loss Functions

# ** Style Cycle Loss


def style_cycle_loss_func(
    original_s: torch.Tensor,
    reconstructed_s: torch.Tensor,
    *,
    normalise=True,
    cos_l2_ratio: float = 0.2,
):
    """Calculate cycle consistency loss for style vector s."""
    if normalise:
        norm_original_s = F.normalize(original_s, dim=-1)
        norm_reconstructed_s = F.normalize(reconstructed_s, dim=-1)
        cos_loss = 1 - F.cosine_similarity(norm_original_s, norm_reconstructed_s, dim=-1).mean()
    else:
        cos_loss = 1 - F.cosine_similarity(original_s, reconstructed_s, dim=-1).mean()

    l2_loss = F.mse_loss(original_s, reconstructed_s)
    return cos_loss + cos_l2_ratio * l2_loss


# ** KL Loss


def kl_loss_func(
    combined_latents: torch.Tensor,
):
    """Calculate the KL divergence between the latent vectors and a normal distribution."""
    mean = combined_latents.mean()
    var = combined_latents.var(correction=0)  # Don't use Bassel's correction

    # Loss terms for mean=0 and var=1
    loss_mean = mean**2
    loss_var = (var - 1) ** 2
    return loss_mean + loss_var


# ** Path Length Loss


def path_loss_func(
    features1: list[torch.Tensor],
    features2: list[torch.Tensor],
    cent_fin_diff_h: torch.Tensor,
) -> torch.Tensor:
    """Calculate path length loss."""
    path_loss = torch.zeros((), device=features1[0].device)
    for feature1, feature2 in zip(features1, features2, strict=True):
        jacobian = (feature1 - feature2) / cent_fin_diff_h[:, None, None, None]
        energy = (jacobian**2).mean()
        path_loss += energy
    path_loss /= len(features1)

    return path_loss
