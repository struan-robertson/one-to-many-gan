"""Losses and penalties."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import compile_


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


@compile_
def discriminator_loss(f_real: torch.Tensor, f_fake: torch.Tensor):
    """Calculate discriminator loss on real and fake batches.

    This is kept separate to allow logging both losses."""
    # Use ReLUs to clip the loss to keep f in [-1, +1] range
    return F.relu(-f_real).mean() + F.relu(f_fake).mean()


@compile_
def generator_loss(f_fake: torch.Tensor):
    """Calculate generator loss for generated fake batch."""
    return -f_fake.mean()
    # return F.relu(-f_fake).mean()
