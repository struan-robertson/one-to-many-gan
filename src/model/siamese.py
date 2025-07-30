"""Siamese model implementation."""

from typing import Literal

import torch
import torchvision
from torch import nn


class SharedSiamese(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        # Create ResNet-50 model without pretrained weights
        self.model = torchvision.models.resnet18(weights=None)

        # Replace final FC layer with embedding layers
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)

        self.model.apply(self.init_weights)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
