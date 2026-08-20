"""
Dynamic CNN model used during architecture search.
"""

import torch.nn as nn


class SearchCNN(nn.Module):
    def __init__(
        self,
        filters=32,
        kernel_size=3,
        pooling="max",
        activation="relu",
    ):
        super().__init__()

        padding = kernel_size // 2

        if activation == "relu":
            activation_layer = nn.ReLU()
        else:
            activation_layer = nn.GELU()

        if pooling == "max":
            pooling_layer = nn.MaxPool2d(2)
        else:
            pooling_layer = nn.AvgPool2d(2)

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size, padding=padding),
            nn.BatchNorm2d(filters),
            activation_layer,
            pooling_layer,

            nn.Conv2d(filters, filters * 2, kernel_size, padding=padding),
            nn.BatchNorm2d(filters * 2),
            activation_layer,
            pooling_layer,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters * 2, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
