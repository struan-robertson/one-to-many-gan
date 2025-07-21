"""Siamese model implementation."""

from typing import Literal

import torch
import torchvision
from torch import nn


class SharedSiamese(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.model = torchvision.models.efficientnet_v2_s(weights=None)

        # Replace final FC layer to get embeddings
        fc = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 500), nn.Linear(500, embedding_size)
        )
        self.model.classifier = fc

    def forward(self, x):
        return self.model(x)
