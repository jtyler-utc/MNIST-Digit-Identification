"""Base MNIST CNN model for classification."""

import torch
import torch.nn as nn


class MNISTCNN(nn.Module):
    """Standard CNN for MNIST classification.
    
    Architecture:
        - Conv2d(1, 32, kernel_size=3) + BatchNorm + ReLU + MaxPool
        - Conv2d(32, 64, kernel_size=3) + BatchNorm + ReLU + MaxPool
        - Flatten + Linear(64*7*7, 128) + ReLU + Dropout + Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x