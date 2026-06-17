"""Joint Autoencoder/Classifier (JAC) models for MNIST image recovery.

Each JAC model consists of:
1. Encoder: Architecture-specific feature extraction from distorted input
2. Decoder: Architecture-specific reconstruction of clean 28×28 image
3. Classifier Head: Standard CNN classifier (shared across all architectures)

The pipeline: Distorted Input → Encoder → Latent Code → Decoder → Reconstruction
              Reconstruction → Classifier Head → Class Logits
"""

from typing import Tuple

import math
import torch
import torch.nn as nn


# =============================================================================
# Shared Classifier Head (identical across all JAC architectures)
# =============================================================================

class JACClassifierHead(nn.Module):
    """Standard CNN classifier head shared by all JAC architectures.

    Takes a flattened 64×7×7 feature vector and outputs 10 class logits.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# =============================================================================
# Residual Block (ResNet-style)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connections for deep feature learning."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# =============================================================================
# JAC-MLP (Fully Connected Autoencoder)
# =============================================================================

class JACMLP(nn.Module):
    """MLP-based autoencoder with shared classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        latent_dim = 49

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

        self.encoder_for_classifier = nn.Sequential(
            nn.Linear(49, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
        )

        self.classifier_head = JACClassifierHead(num_classes)

        # Distortion discrimination head: learns to predict distortion level in [0, 1]
        self.distortion_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        classifier_features = self.encoder_for_classifier(latent)
        class_logits = self.classifier_head(classifier_features.flatten(1))
        distortion_score = self.distortion_discriminator(latent)
        return class_logits, reconstruction, distortion_score


# =============================================================================
# JAC-LSTM (Sequential Autoencoder)
# =============================================================================

class JACLSTM(nn.Module):
    """LSTM-based autoencoder with shared classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.hidden_dim = 128
        self.num_layers = 2

        self.conv_pre_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.lstm_encoder = nn.LSTM(
            input_size=64, hidden_size=self.hidden_dim,
            num_layers=self.num_layers, batch_first=True
        )
        self.lstm_input_proj = nn.Linear(448, 64)
        # Decoder: project each timestep's LSTM output (128) -> 64
        self.decoder_linear = nn.Linear(self.hidden_dim, 64)

        # Decoder: upsample from 7x7 to 28x28 using transposed convolutions
        # Input: (B, 64, 7, 7) - conv features directly
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (B, 64, 7, 7)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # (B, 64, 14, 14)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (B, 32, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # (B, 1, 28, 28)
            nn.Sigmoid(),
        )

        self.classifier_head = JACClassifierHead(num_classes)

        # Distortion discrimination head: learns to predict distortion level in [0, 1]
        # Uses the 128-dim LSTM hidden state as input
        self.distortion_discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        conv_features = self.conv_pre_encoder(x)  # (B, 64, 7, 7)

        # Reshape conv features for LSTM: (B, 64, 7, 7) -> process 7 columns as sequence
        # Permute to (B, 7, 64, 7), then flatten last 2 dims -> (B, 7, 448)
        seq_features = conv_features.permute(0, 2, 1, 3)  # (B, 7, 64, 7)
        seq_features = seq_features.reshape(batch_size, 7, -1)  # (B, 7, 448)
        seq_features = self.lstm_input_proj(seq_features)  # (B, 7, 64)
        lstm_out, _ = self.lstm_encoder(seq_features)  # (B, 7, 128)

        # Reconstruction: use conv features directly (they have 7x7 spatial structure)
        # Add an extra conv layer to learn better feature transformation
        reconstruction = self.decoder_conv(conv_features)

        # Classifier uses conv features (preserves spatial info)
        class_logits = self.classifier_head(conv_features.flatten(1))

        # Distortion score from the final LSTM hidden state (128-dim latent)
        hidden_state = lstm_out[:, -1, :]
        distortion_score = self.distortion_discriminator(hidden_state)

        return class_logits, reconstruction, distortion_score


# =============================================================================
# JAC-CNN (Convolutional Autoencoder with Skip Connections)
# =============================================================================

class JACCNN(nn.Module):
    """Convolutional autoencoder with skip connections and shared classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_skip1 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.decoder_skip2 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.classifier_head = JACClassifierHead(num_classes)

        # Distortion discrimination head: learns to predict distortion level in [0, 1]
        # Uses adaptive average pooling to get 64 features regardless of spatial size
        # (bottleneck outputs 64 channels)
        self.distortion_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool to (B, 64, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc1 = self.encoder[:4](x)
        enc2 = self.encoder[4:8](enc1)
        enc3 = self.encoder[8:](enc2)
        bottleneck = self.bottleneck(enc3)

        dec1 = torch.cat([bottleneck, enc2], dim=1)
        dec1 = self.decoder_skip1(dec1)
        dec1_up = nn.functional.interpolate(dec1, size=(14, 14), mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec1_up, enc1], dim=1)
        dec2 = self.decoder_skip2(dec2)
        reconstruction = self.decoder(dec2)
        class_logits = self.classifier_head(bottleneck.flatten(1))

        # Distortion score from bottleneck features
        distortion_score = self.distortion_discriminator(bottleneck)

        return class_logits, reconstruction, distortion_score


# =============================================================================
# JAC-ResNet (Residual Autoencoder)
# =============================================================================

class JACResNet(nn.Module):
    """ResNet-based autoencoder with shared classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.init_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder_block1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=2),
        )

        self.encoder_block2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
        )
        self.bottleneck_project = nn.Conv2d(128, 64, kernel_size=1)

        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
        )

        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )
        self.classifier_head = JACClassifierHead(num_classes)

        # Distortion discrimination head: learns to predict distortion level in [0, 1]
        # Uses adaptive average pooling to get 64 features regardless of spatial size
        self.distortion_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool to (B, 64, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.init_encoder(x)
        features = self.encoder_block1(features)
        features = self.encoder_block2(features)
        bottleneck_128 = self.bottleneck(features)
        bottleneck = self.bottleneck_project(bottleneck_128)
        decoded = self.decoder_block1(bottleneck_128)
        # decoder_output gives [-1, 1], scale to [0, 1]
        reconstruction = (self.decoder_output(decoded) + 1) / 2
        class_logits = self.classifier_head(bottleneck.flatten(1))

        # Distortion score from bottleneck features (64-dim, via adaptive pooling)
        distortion_score = self.distortion_discriminator(bottleneck)

        return class_logits, reconstruction, distortion_score


# =============================================================================
# JAC-Transformer (Vision Transformer Autoencoder)
# =============================================================================

class JACTransformer(nn.Module):
    """Vision Transformer-based autoencoder with shared classifier head."""

    def __init__(self, num_classes: int = 10, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=4, stride=4),
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.classifier_head = JACClassifierHead(num_classes)

        # Distortion discrimination head: learns to predict distortion level in [0, 1]
        # Uses the 64-dim latent from latent_proj
        self.distortion_discriminator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        patches = self.patch_embedding(x)
        patches = patches.flatten(2).permute(0, 2, 1)
        patches = self.pos_encoding(patches)
        encoded = self.transformer_encoder(patches)

        latent = self.latent_proj(encoded)  # (B, 16, 64)
        latent_reshaped = latent.permute(0, 2, 1).view(batch_size, 64, 7, 7)
        decoded = encoded.permute(0, 2, 1).view(batch_size, self.d_model, 7, 7)
        reconstruction = self.decoder_conv(decoded)
        class_logits = self.classifier_head(latent_reshaped.flatten(1))

        # Distortion score: average over spatial tokens, then project
        # latent is (B, 16, 64), mean gives (B, 64)
        latent_mean = latent.mean(dim=1)  # (B, 64)
        distortion_score = self.distortion_discriminator(latent_mean)

        return class_logits, reconstruction, distortion_score


# =============================================================================
# Positional Encoding (for Transformer)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# =============================================================================
# Model Registry & Factory
# =============================================================================

JAC_MODELS = {
    'mlp': ('JAC-MLP (Fully Connected Autoencoder)', JACMLP),
    'lstm': ('JAC-LSTM (Sequential Autoencoder)', JACLSTM),
    'cnn': ('JAC-CNN (Convolutional Autoencoder)', JACCNN),
    'resnet': ('JAC-ResNet (Residual Autoencoder)', JACResNet),
    'transformer': ('JAC-Transformer (Vision Transformer Autoencoder)', JACTransformer),
}


def get_jac_model(model_name: str, num_classes: int = 10) -> nn.Module:
    if model_name not in JAC_MODELS:
        available = ', '.join(JAC_MODELS.keys())
        raise ValueError(f"Unknown JAC model: '{model_name}'. Available: {available}")
    model_class = JAC_MODELS[model_name][1]
    return model_class(num_classes=num_classes)


def get_jac_model_info(model_name: str) -> dict:
    if model_name not in JAC_MODELS:
        return {'display_name': model_name, 'description': 'Unknown architecture'}

    display_name, model_class = JAC_MODELS[model_name]
    descriptions = {
        'mlp': 'Fully connected autoencoder.',
        'lstm': 'LSTM-based sequential autoencoder.',
        'cnn': 'Deep convolutional autoencoder with skip connections.',
        'resnet': 'Residual network autoencoder.',
        'transformer': 'Vision Transformer autoencoder.',
    }

    try:
        test_model = model_class(num_classes=10)
        total_params = sum(p.numel() for p in test_model.parameters())
    except Exception:
        total_params = 0

    return {
        'display_name': display_name,
        'description': descriptions.get(model_name, 'Custom architecture.'),
        'total_params': total_params,
    }