"""Losses and penalties."""

import torch
from torch import nn
from torch.nn import functional as F

from .utils import compile_

# * Adaptive Discriminator Augmentation


# Can't compile this
class ADAp:
    """Adaptive discriminator augmentation state."""

    def __init__(
        self,
        ada_e: int,
        ada_adjustment_size: float,
        batch_size: int,
        discriminator_overfitting_target: float,
    ):
        # Number of batches required to reach images to calculate mean overfitting
        self.n_batches = ada_e // batch_size
        # Amount to adjust ADA each time
        self.ada_adjustment = ada_adjustment_size * ada_e

        self.overfitting_target = discriminator_overfitting_target

        self.p = torch.zeros(())
        self.curr_batch = 0
        self.mean_real_scores = []

    def update_p(self, mean_score: torch.Tensor):
        if self.curr_batch == self.n_batches:
            self.mean_real_scores.append(mean_score)

            mean_sign = torch.mean(torch.stack(self.mean_real_scores))

            if mean_sign < self.overfitting_target:
                self.p -= self.ada_adjustment
            elif mean_sign > self.overfitting_target:
                self.p += self.ada_adjustment

            self.curr_batch = 0
            self.mean_real_scores = []

            self.p = nn.functional.relu(self.p, inplace=True)

        self.curr_batch += 1
        self.mean_real_scores.append(mean_score)

    def __call__(self) -> float:
        return self.p.item()


# * Loss Functions

# ** Style Cycle Loss


@compile_
def style_cycle_loss_func(
    original_w: torch.Tensor,
    reconstructed_w: torch.Tensor,
    *,
    normalise=True,
    cos_l2_ratio: float = 0.2,
):
    """Calculate cycle consistency loss for style vector w."""
    if normalise:
        original_w = F.normalize(original_w, dim=-1)
        reconstructed_w = F.normalize(reconstructed_w, dim=-1)

    # Calculate cycle loss
    cos_loss = 1 - F.cosine_similarity(original_w, reconstructed_w, dim=-1).mean()
    l2_loss = F.mse_loss(original_w, reconstructed_w)
    return cos_loss + cos_l2_ratio * l2_loss


# ** KL Loss


# Would be more appropriate to call domain alignment loss
@compile_
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


@compile_
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
