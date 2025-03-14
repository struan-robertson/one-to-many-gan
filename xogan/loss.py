"""Losses and penalties."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import compile_

# * Penalties


# torch.autograd.grad not supported in full graph compilation
@torch.compile
class GradientPenalty(nn.Module):
    """Ensure discriminator is 1-Lipshitz for Wasserstein-style losses."""

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        batch_size = x.shape[0]

        gradients, *_ = torch.autograd.grad(
            outputs=d, inputs=x, grad_outputs=d.new_ones(d.shape), create_graph=True
        )

        # Reshape and calculate L_2 norm
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)

        return torch.mean(norm**2)


# torch.autograd.grad not supported in full graph compilation
@torch.compile
class PathLengthPenalty(nn.Module):
    """Encourages a fixed-size step in w to result in a fixed-magnitude change in the image."""

    def __init__(self, beta: float):
        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        device = x.device

        image_size = x.shape[2] * x.shape[3]

        y = torch.randn(x.shape, device=device)

        # Normalise output
        output = (x * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=device),
            create_graph=True,
        )

        norm = (gradients**2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta**self.steps)

            loss = torch.mean((norm - a) ** 2)

        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()

        # alpha is subtracted before adding
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)

        self.steps.add_(1.0)

        return loss


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

    def __call__(self):
        return self.p


# * Loss Functions


@compile_
class SiameseTripletLoss(nn.Module):
    """Triplet margin loss with L2-normalisation."""

    def __init__(self, margin=0.8):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ):
        # Normalise
        norm_anchor = F.normalize(anchor, p=2, dim=1)
        norm_positive = F.normalize(positive, p=2, dim=1)
        norm_negative = F.normalize(negative, p=2, dim=1)

        return self.loss(anchor, positive, negative), (
            norm_anchor,
            norm_positive,
            norm_negative,
        )
