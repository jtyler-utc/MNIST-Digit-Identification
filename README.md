# MNIST CNN Trainer & Classifier

A PyQt6 desktop application for training Convolutional Neural Networks (CNNs) on the MNIST digits dataset and classifying user-drawn digits in real-time. Supports **5 JAC (Joint Autoencoder/Classifier) architectures** with reconstruction capabilities and noise-robust training.

## Overview

This application provides an interactive GUI with three tabs:

1. **Training** - Select from 6 model architectures (Standard CNN + 5 JAC models), configure hyperparameters, monitor training progress, and view live charts
2. **Classification** - Draw digits on a canvas, apply distortions (blur/noise), and get real-time predictions with confidence levels and reconstruction output
3. **Terminal** - View all training logs, status messages, and system output

## Prerequisites

- **Python 3.10+** (must be installed and added to PATH)
- **Windows 10/11** (tested on Windows)

## Installation

### 1. Install Python

If Python is not installed:

```bash
# Using Windows Package Manager (winget)
winget install Python.Python.3.11

# Or download from https://www.python.org/downloads/
# IMPORTANT: Check "Add Python to PATH" during installation
```

Verify installation:
```bash
python --version
# Should output: Python 3.10.x or higher
```

### 2. Install Dependencies Using Conda (Recommended for GPU)

```bash
# Create conda environment with Python 3.12
conda create -n mnist-cnn python=3.12 -y
conda activate mnist-cnn

# Install dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip install pyqt6 matplotlib numpy
```

**Note on GPU support:** CUDA-enabled PyTorch wheels may not yet be available for all Python versions. Check `torch.cuda.is_available()` after installation. If False, you're running on CPU.

### Alternative: Using System Python

```bash
pip install torch torchvision pyqt6 matplotlib numpy
```

### 3. Install Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch deep learning framework |
| `torchvision` | Datasets and transforms for image classification |
| `pyqt6` | GUI framework |
| `matplotlib` | Chart visualization |
| `numpy` | Numerical computing |

## Usage

### Run the Application

**With conda environment:**
```bash
conda activate mnist-cnn
python mnist_cnn_gui.py
```

**With system Python:**
```bash
python mnist_cnn_gui.py
```

### Tab 1: Training

#### Model Selection

Choose from **6 model architectures** in the unified dropdown:

| Model | Description |
|-------|-------------|
| **Standard** | Traditional CNN classifier (baseline) |
| **JAC-MLP** | Fully Connected Autoencoder + classifier |
| **JAC-LSTM** | LSTM-based Sequential Autoencoder + classifier |
| **JAC-CNN** | Convolutional Autoencoder with Skip Connections + classifier |
| **JAC-ResNet** | Residual Network Autoencoder + classifier |
| **JAC-Transformer** | Vision Transformer Autoencoder + classifier |

JAC models perform **dual tasks**: digit classification AND image reconstruction (denoising). They are trained with combined Cross-Entropy + MSE loss.

#### Hyperparameters

- **Model** - Select architecture from dropdown (1-6)
- **Max Epochs** (1-500, default: 10) - Number of complete passes through the dataset
- **Learn Rate** (0.0001-1.0, default: 0.001) - Step size for optimizer
- **Batch Size** (2-1024, powers of 2, default: 256) - Number of samples per gradient update
- **Verbose Freq** (1-10000, default: 100) - How often (in iterations) to log metrics and update charts

#### Controls

- **Start Training** - Begin training with current hyperparameters (blue button)
- **Stop Training** - Interrupt training early (red button, toggles during training)
- **Progress Bar** - Shows current iteration progress (goes gold on completion)

#### Live Metrics

- **Iteration** - Current training iteration number
- **Train Loss** / **Train Acc** - Training loss and accuracy
- **Val Loss** / **Val Acc** - Validation loss and accuracy
- **Training Charts** - Real-time line charts for loss and accuracy curves (dark theme)
- **Reconstruction Chart** - For JAC models: shows original, distorted, and reconstructed images side-by-side

#### Model Saving

Trained models are saved to architecture-specific subfolders under `saved_models/`:
- `saved_models/standard/best_mnist_model.pth` - Standard CNN
- `saved_models/standard_mlp/best_mnist_model_mlp.pth` - JAC-MLP
- `saved_models/standard_lstm/best_mnist_model_lstm.pth` - JAC-LSTM
- `saved_models/standard_cnn/best_mnist_model_cnn.pth` - JAC-CNN
- `saved_models/standard_resnet/best_mnist_model_resnet.pth` - JAC-ResNet
- `saved_models/standard_transformer/best_mnist_model_transformer.pth` - JAC-Transformer

Noise-robust variants are saved to `saved_models/noise_robust_{arch}/` folders.

### Tab 2: Classification

#### Drawing Canvas

- Draw a digit (0-9) using mouse input on the white canvas
- The canvas is 280×280 pixels, scaled from the 28×28 MNIST input size
- Use the **Clear Canvas** button to reset
- Predictions update at ~60fps as you draw

#### Model Loading

- **Model Dropdown** - Shows available models with checkmark (✓) for trained models, greyed out for unavailable
- **Load Model** - Loads the selected saved model for classification
- **Model Status** - Displays accuracy and model type (Standard or +Decoder for JAC/noise-robust)

#### Distortion Controls

Test model robustness with real-time distortions:

- **Gaussian Blur** (0-10) - Apply Gaussian blur to the drawn digit
- **Noise** (0-10) - Add random noise to the drawn digit

#### Real-Time Prediction

- Predictions update at ~60fps as you draw
- Shows predicted digit and confidence percentage
- **Bar Chart** - Displays confidence for all 10 digits (dark theme)
  - Predicted digit highlighted in **red**
  - Other digits shown in **blue**
- **Image Previews** - Shows distorted input at bottom; for JAC/noise-robust models, also shows reconstructed (denoised) output side-by-side

#### Requirements

- A trained model must be loaded (complete training first)
- Empty canvas will not trigger predictions

### Tab 3: Terminal

- Displays all training logs, status messages, errors, and classification events
- Monospace font (Courier New, 10pt)
- Auto-scrolls to latest output
- Captures both stdout and stderr

## Model Architectures

### Standard CNN (Baseline)

```
Input: (1, 28, 28) grayscale image
       │
Conv2d(1→32, kernel=3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2×2)
       │
Conv2d(32→64, kernel=3, pad=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2×2)
       │
Flatten → (3136)
       │
Linear(3136→128) → ReLU → Dropout(0.5)
       │
Linear(128→10)
       │
Output: 10 class logits
```

**Total Parameters:** ~535,000

### JAC-MLP (Fully Connected Autoencoder)

```
Encoder:
  Flatten(784) → Linear(784→256) → ReLU → Dropout(0.3)
  → Linear(256→128) → ReLU → Linear(128→49) [latent code]

Decoder:
  Linear(49→128) → ReLU → Linear(128→256) → ReLU
  → Linear(256→784) → Sigmoid → Unflatten(1, 28, 28)

Classifier Head:
  Latent(49) → Linear(49→3136) → Unflatten → Linear(3136→10)
```

**Total Parameters:** ~15,000

### JAC-LSTM (Sequential Autoencoder)

```
Pre-Encoder: Conv2d → MaxPool → Conv2d → MaxPool (reduces to 7×7×64)
Encoder: Project rows → 2-layer LSTM → hidden state (128 dim)
Decoder: 2-layer LSTM → reshape → Conv2d upsample ×2
Classifier Head: Conv features → Linear(3136→10)
```

**Total Parameters:** ~1,200,000

### JAC-CNN (Convolutional Autoencoder with Skip Connections)

```
Encoder:
  Conv2d(1→32, p=1) → BN → ReLU → MaxPool(2)
  → Conv2d(32→64, p=1) → BN → ReLU → MaxPool(2)
  → Conv2d(64→128, p=1) → BN → ReLU

Bottleneck: Conv2d(128→64, p=1) → BN → ReLU

Decoder (with skip connections):
  Conv(64+64→64) → Upsample(14×14)
  → Cat with encoder1 → Conv(64+32→32)
  → ConvTranspose(32→32, stride=2) → Conv(32→1) → Sigmoid

Classifier Head: Bottleneck features → Linear(3136→10)
```

**Total Parameters:** ~850,000

### JAC-ResNet (Residual Autoencoder)

```
Encoder:
  Conv2d(1→64, p=1) → BN → ReLU
  → 3× ResidualBlock(64, stride=2) → 3× ResidualBlock(128, stride=2)

Bottleneck: 2× ResidualBlock(128) → Conv2d(128→64, 1×1)

Decoder:
  ConvTranspose(128→64, stride=2) → 2× ResidualBlock(64)
  → ConvTranspose(64→1, stride=2) → ResidualBlock(1) → Sigmoid

Classifier Head: Bottleneck(64×7×7) → Linear(3136→10)
```

**Total Parameters:** ~2,500,000

### JAC-Transformer (Vision Transformer Autoencoder)

```
Patch Embedding: Conv2d(1→128, kernel=4, stride=4) → [8×8 patches]
Position Encoding: Sinusoidal PE (max 64 tokens)

Encoder: 4-layer TransformerEncoder (d_model=128, nhead=4, FFN=256)

Latent: Linear(128→64) → reshape to [64, 7, 7]

Decoder: Conv2d(128→64) → BN → ReLU → Upsample ×2
  → Conv2d(64→32) → BN → ReLU → Upsample ×2
  → Conv2d(32→1) → Sigmoid

Classifier Head: Latent(64×7×7) → Linear(3136→10)
```

**Total Parameters:** ~3,000,000

## Training Configuration

### Standard Training

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Data Augmentation | RandomHorizontalFlip, RandomRotation(10°) |
| Normalization | mean=0.1307, std=0.3081 |

### JAC / Noise-Robust Training

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss + MSE Reconstruction Loss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Data Augmentation | RandomHorizontalFlip, RandomRotation(10°) + Gaussian Blur + Gaussian Noise |
| Normalization | mean=0.1307, std=0.3081 |

## File Structure

```
RET Example 2026/
├── mnist_cnn_gui.py          # Main application file
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── gui/
│   ├── __init__.py
│   ├── canvas.py             # Drawing canvas with distortion support
│   ├── charts.py             # Matplotlib chart widgets (dark theme)
│   ├── main_window.py        # Main application window
│   └── terminal.py           # Terminal widget for logs
├── models/
│   ├── __init__.py
│   ├── base_model.py         # Standard CNN (MNISTCNN)
│   ├── interference_model.py # Noise-robust CNN
│   └── jac_model.py          # JAC models (MLP, LSTM, CNN, ResNet, Transformer)
├── training/
│   ├── __init__.py
│   ├── interference.py       # Interference/noise generation
│   ├── standard_trainer.py   # Standard training thread
│   └── noise_robust_trainer.py  # JAC/noise-robust training thread
└── saved_models/             # Saved model checkpoints
    ├── standard/
    │   └── best_mnist_model.pth
    ├── standard_mlp/
    │   └── best_mnist_model_mlp.pth
    ├── standard_lstm/
    │   └── best_mnist_model_lstm.pth
    ├── standard_cnn/
    │   └── best_mnist_model_cnn.pth
    ├── standard_resnet/
    │   └── best_mnist_model_resnet.pth
    ├── standard_transformer/
    │   └── best_mnist_model_transformer.pth
    ├── noise_robust_legacy/
    │   └── best_mnist_noise_robust.pth
    └── noise_robust_{arch}/  # Noise-robust JAC variants
        └── best_mnist_noise_robust_{arch}.pth
```

## Troubleshooting

### "Python not found" error
- Python is not installed or not in PATH
- Run: `winget install Python.Python.3.11`
- Or download from python.org (check "Add to PATH")

### "Module not found" errors
- Install missing dependencies: `pip install torch torchvision pyqt6 matplotlib numpy`

### GUI doesn't appear / crashes on startup
- Ensure PyQt6 is installed: `pip install pyqt6`
- Try different PyQt6 style: modify `app.setStyle("Fusion")` in `main()`

### Training is very slow
- Use a smaller batch size or fewer epochs for testing
- GPU acceleration is automatic if CUDA-enabled PyTorch is installed
- Check device with: `torch.cuda.is_available()`
- JAC-Transformer and JAC-ResNet are larger models and train slower

### Drawing canvas doesn't respond
- Ensure you're using left-click and dragging
- Try clearing the canvas and redrawing
- Check terminal tab for error messages

### Classification always shows same digit
- Make sure training completed successfully (check Terminal tab)
- Draw a clearer, centered digit
- Try increasing training epochs
- Load a different trained model

### No models available in dropdown
- Train a model first using the Training tab
- Models appear with checkmark (✓) once trained

### Reconstruction output not showing
- Ensure you've loaded a JAC or noise-robust model
- Standard models only produce classification output

## Technical Details

### Threading
- Training runs in a background `QThread` to keep the GUI responsive
- PyQt signals/slots communicate progress from training thread to UI
- Classification runs at ~60fps (16ms interval) on the main thread
- Chart updates are throttled to ~10fps to reduce rendering load

### Image Preprocessing
1. Drawing captured at 280×280 pixels
2. Scaled to 28×28 with aspect ratio preserved
3. Centered in a black (zero-padded) 28×28 canvas
4. Normalized to tensor: `(pixel - 0.1307) / 0.3081`
5. Added batch dimension: shape `(1, 1, 28, 28)`

### Data Augmentation (Training Only)
- Random horizontal flip (50% probability)
- Random rotation up to ±10 degrees
- For JAC/noise-robust training: Gaussian blur and Gaussian noise applied to simulate distorted inputs

### Distortion Application (Classification)
- **Gaussian Blur**: Kernel size scales with slider value (0-10)
- **Gaussian Noise**: Standard deviation scales with slider value (0-10)
- Distortions are applied in real-time to the drawn digit before classification
- For JAC models: both distorted input and reconstructed output are displayed

### UI Theme
- Dark theme for all charts (background: #1a1a2e, chart: #16213e)
- Accent colors: Blue (#00d4ff), Red (#ff6b6b), Green (#51cf66), Yellow (#ffd43b)
- Fusion style for Qt widgets for consistent cross-platform appearance