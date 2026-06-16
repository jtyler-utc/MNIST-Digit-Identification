"""Interference functions for applying noise and blur to image batches."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def apply_random_interference(
    images: torch.Tensor,
    noise_levels: np.ndarray,
    blur_levels: np.ndarray
) -> torch.Tensor:
    """Apply random blur and noise to a batch of images.

    Args:
        images: Tensor of shape (B, 1, 28, 28) in range [0, 1]
        noise_levels: Array of continuous values [0, 1] for each sample
        blur_levels: Array of continuous values [0, 1] for each sample

    Returns:
        Distorted images as tensor in range [0, 1]
    """
    batch_size = images.size(0)
    # Convert to numpy for processing: (B, 1, 28, 28) -> (B, 28, 28)
    images_np = images.cpu().numpy() * 255.0  # Scale to [0, 255]
    distorted_batch = []

    for i in range(batch_size):
        img = images_np[i, 0].copy()  # (28, 28)

        # Apply Gaussian blur
        if blur_levels[i] > 0:
            sigma = blur_levels[i] * 3.0  # Scale to sigma [0, 3]
            img = gaussian_filter(img, sigma=sigma)

        # Apply Gaussian noise
        if noise_levels[i] > 0:
            noise_std = noise_levels[i] * 64.0  # Scale to std [0, 64]
            noise = np.random.normal(0, noise_std, img.shape)
            img = img + noise

        # Clip and convert
        img = np.clip(img, 0, 255).astype(np.uint8)
        distorted_batch.append(img)

    # Convert back to tensor: (B, 28, 28) -> (B, 1, 28, 28)
    result = torch.tensor(np.array(distorted_batch), dtype=torch.float32).unsqueeze(1) / 255.0
    return result