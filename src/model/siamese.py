"""Siamese model implementation."""

from typing import Literal

import torch
import torchvision
from torch import nn


class SharedSiamese(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=None)

        # Modify first conv for 512x256 grayscale (1-channel) input
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace final FC layer to get embeddings
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)

    def forward(self, x):
        return self.resnet(x)
