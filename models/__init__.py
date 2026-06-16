"""PyTorch model definitions for MNIST classification."""

from .base_model import MNISTCNN
from .interference_model import MNISTInterferenceTolerantCNN

__all__ = ["MNISTCNN", "MNISTInterferenceTolerantCNN"]