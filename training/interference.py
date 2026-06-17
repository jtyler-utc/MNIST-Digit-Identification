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


def compute_distortion_level(noise_level: float, blur_level: float) -> float:
    """Compute combined distortion level in [0, 1].

    Args:
        noise_level: Continuous value [0, 1] for noise strength
        blur_level: Continuous value [0, 1] for blur strength

    Returns:
        Combined distortion level in [0, 1]
    """
    return 0.5 * noise_level + 0.5 * blur_level


def generate_batch_interference(
    batch_size: int,
    min_noise: float = 0.0,
    max_noise: float = 1.0,
    min_blur: float = 0.0,
    max_blur: float = 1.0
) -> tuple:
    """Generate random interference levels for a batch.

    Args:
        batch_size: Number of samples in the batch
        min_noise: Minimum noise level [0, 1]
        max_noise: Maximum noise level [0, 1]
        min_blur: Minimum blur level [0, 1]
        max_blur: Maximum blur level [0, 1]

    Returns:
        Tuple of (noise_levels, blur_levels, distortion_levels) as numpy arrays
    """
    noise_levels = np.random.uniform(min_noise, max_noise, batch_size)
    blur_levels = np.random.uniform(min_blur, max_blur, batch_size)
    distortion_levels = 0.5 * noise_levels + 0.5 * blur_levels
    return noise_levels, blur_levels, distortion_levels


def apply_interference_with_level(
    images: torch.Tensor,
    noise_levels: np.ndarray,
    blur_levels: np.ndarray
) -> tuple:
    """Apply interference and compute distortion levels for a batch.

    Args:
        images: Tensor of shape (B, 1, 28, 28) in range [0, 1]
        noise_levels: Array of continuous values [0, 1] for each sample
        blur_levels: Array of continuous values [0, 1] for each sample

    Returns:
        Tuple of (distorted_images, distortion_levels) where:
        - distorted_images: Tensor of shape (B, 1, 28, 28)
        - distortion_levels: numpy array of shape (B,) in [0, 1]
    """
    distorted = apply_random_interference(images, noise_levels, blur_levels)
    distortion_levels = 0.5 * noise_levels + 0.5 * blur_levels
    return distorted, distortion_levels
