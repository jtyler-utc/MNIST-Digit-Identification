"""Noise-robust training thread for interference-tolerant model training.

Supports multiple encoder/decoder architectures via the JAC (Joint Autoencoder/Classifier)
framework. The architecture is selected via the 'architecture' config parameter.
"""

import os
import time
import traceback

# Model save directory configuration
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "saved_models")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from PyQt6.QtCore import QThread, pyqtSignal

from models.jac_model import get_jac_model, JAC_MODELS
from training.interference import apply_random_interference


class NoiseRobustTrainingThread(QThread):
    """Background thread for interference-tolerant model training.

    Supports multiple encoder/decoder architectures via the JAC framework.
    The architecture is selected via config['architecture'] (default: 'cnn').
    
    Available architectures:
        - 'standard': JAC-Standard (CNN Autoencoder)
        - 'mlp': JAC-MLP (Fully Connected Autoencoder)
        - 'lstm': JAC-LSTM (Sequential Autoencoder)
        - 'cnn': JAC-CNN (Convolutional Autoencoder)
        - 'resnet': JAC-ResNet (Residual Autoencoder)
        - 'transformer': JAC-Transformer (Vision Transformer Autoencoder)
        - 'legacy': Legacy MNISTInterferenceTolerantCNN (backward compatible)
    """

    log_signal = pyqtSignal(str)
    epoch_signal = pyqtSignal(int, float, float, float, float)
    sample_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # original, distorted, reconstructed
    finish_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop_requested = False
        self.architecture = config.get('architecture', 'cnn')

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_signal.emit(f"Using device: {device}")
            self.log_signal.emit("Training mode: Interference Tolerance (Encoder → Decoder → Classifier Pipeline)")

            # Data loading (no transforms needed - interference is applied in training loop)
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.log_signal.emit("Downloading/loading MNIST dataset...")
            train_transform_basic = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform_basic)
            test_dataset = MNIST(root='./data', train=False, download=True, transform=test_transform)

            train_loader = DataLoader(
                train_dataset, batch_size=self.config['batch_size'],
                shuffle=True, num_workers=2, pin_memory=device.type == 'cuda',
                persistent_workers=True if torch.cuda.is_available() else False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.config['batch_size'],
                shuffle=False, num_workers=2, pin_memory=device.type == 'cuda',
                persistent_workers=True if torch.cuda.is_available() else False
            )

            # Model setup — select architecture
            arch_display = JAC_MODELS.get(self.architecture, ('Unknown', None))[0] if self.architecture in JAC_MODELS else 'Legacy CNN'
            if self.architecture == 'legacy':
                # Use the original legacy model for backward compatibility
                self.log_signal.emit(f"Initializing legacy MNISTInterferenceTolerantCNN model...")
                model = MNISTInterferenceTolerantCNN(num_classes=10)
            else:
                self.log_signal.emit(f"Initializing JAC model with architecture: {arch_display}")
                model = get_jac_model(self.architecture, num_classes=10)
            model = model.to(device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log_signal.emit(f"Total parameters: {total_params:,}")
            self.log_signal.emit(f"Trainable parameters: {trainable_params:,}")

            # Loss functions
            ce_criterion = nn.CrossEntropyLoss()
            mse_criterion = nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # Prefetch all test data to GPU
            self.log_signal.emit("Prefetching test data to GPU device...")
            prefetch_start = time.time()

            train_data_list, train_labels_list = [], []
            for data, target in train_loader:
                train_data_list.append(data.to(device, non_blocking=True))
                train_labels_list.append(target.to(device, non_blocking=True))

            test_data_list, test_labels_list = [], []
            for data, target in test_loader:
                test_data_list.append(data.to(device, non_blocking=True))
                test_labels_list.append(target.to(device, non_blocking=True))

            prefetch_time = time.time() - prefetch_start
            self.log_signal.emit(f"Data prefetch complete: {len(train_data_list)} train batches, {len(test_data_list)} val batches ({prefetch_time:.2f}s)")

            num_epochs = self.config['epochs']
            verbose_freq = self.config.get('verbose_freq', 100)
            best_val_acc = 0.0
            train_losses, train_accs, val_losses, val_accs = [], [], [], []
            start_time = time.time()
            global_iteration = 0
            prev_val_loss = 0.0
            prev_val_acc = 0.0

            # Track a sample image for visualization
            # Select one random image at start and apply fixed distortion
            sample_idx = min(4, len(test_data_list[0]))  # Pick a sample from first test batch
            sample_original_tensor = test_data_list[0][sample_idx].to(device)  # (1, 28, 28) on GPU
            
            # Apply fixed distortion once at the start
            sample_distorted_tensor = apply_random_interference(
                sample_original_tensor.unsqueeze(0),  # Add batch dim
                np.array([0.5]),  # Fixed noise level
                np.array([0.5])   # Fixed blur level
            ).squeeze(0).to(device)  # Remove batch dim and move to device
            
            sample_original = sample_original_tensor.cpu().numpy().squeeze()  # (28, 28)
            sample_distorted = sample_distorted_tensor.cpu().numpy().squeeze()  # (28, 28) - fixed throughout training

            self.log_signal.emit("=" * 60)
            self.log_signal.emit(f"Starting interference tolerance training for {num_epochs} epochs...")
            self.log_signal.emit(f"Batch size: {self.config['batch_size']}, Learning rate: {self.config['learning_rate']}")
            self.log_signal.emit(f"Verbose frequency: every {verbose_freq} iterations")
            self.log_signal.emit("=" * 60)

            # --- Initial validation at iteration 0 (before any training) ---
            self.log_signal.emit("Running initial validation at iteration 0...")
            model.eval()
            val_ce_loss = 0.0
            val_mse_loss = 0.0
            val_total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx in range(len(test_data_list)):
                    data = test_data_list[batch_idx]
                    target = test_labels_list[batch_idx]

                    # Apply moderate interference for validation
                    bsz_val = data.size(0)
                    val_noise = np.full(bsz_val, 0.5)
                    val_blur = np.full(bsz_val, 0.5)
                    distorted_val = apply_random_interference(data, val_noise, val_blur).to(device)

                    class_logits, reconstruction = model(distorted_val)

                    ce_loss = ce_criterion(class_logits, target)
                    mse_loss = mse_criterion(reconstruction, data)
                    total_loss = 0.5 * ce_loss + 0.5 * mse_loss

                    val_ce_loss += ce_loss.item()
                    val_mse_loss += mse_loss.item()
                    val_total_loss += total_loss.item()
                    _, predicted = class_logits.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            val_total_loss = val_total_loss / len(test_loader)
            val_acc = 100. * correct / total
            prev_val_loss = val_total_loss
            prev_val_acc = val_acc

            # Emit initial validation data at iteration 0
            self.epoch_signal.emit(0, 0.0, 0.0, val_total_loss, val_acc)
            self.log_signal.emit(f"  Iteration 0 | Initial Val Loss: {val_total_loss:.4f} Val Acc: {val_acc:.2f}%")
            
            # Emit initial sample reconstruction (before any training)
            with torch.no_grad():
                _, initial_recon = model(sample_distorted_tensor.unsqueeze(0))
                initial_recon_np = initial_recon.squeeze().detach().cpu().numpy().squeeze()
            self.sample_signal.emit(sample_original, sample_distorted, initial_recon_np)

            train_losses.append(0.0)  # Placeholder for iteration 0
            train_accs.append(0.0)
            val_losses.append(val_total_loss)
            val_accs.append(val_acc)

            for epoch in range(1, num_epochs + 1):
                if self._stop_requested:
                    self.log_signal.emit("Training stopped by user.")
                    break

                epoch_start = time.time()

                # --- Train ---
                model.train()
                running_ce_loss = 0.0
                running_mse_loss = 0.0
                running_total_loss = 0.0
                correct = 0
                total = 0

                for batch_idx in range(len(train_data_list)):
                    if self._stop_requested:
                        break

                    originals = train_data_list[batch_idx]  # (B, 1, 28, 28) in [0, 1]
                    target = train_labels_list[batch_idx]
                    bsz = originals.size(0)

                    # Generate random interference levels for this batch
                    current_noise_levels = np.random.uniform(0, 1, bsz)
                    current_blur_levels = np.random.uniform(0, 1, bsz)

                    # Apply interference to create distorted inputs
                    distorted = apply_random_interference(originals, current_noise_levels, current_blur_levels)
                    distorted = distorted.to(device)

                    optimizer.zero_grad()
                    class_logits, reconstruction = model(distorted)

                    ce_loss = ce_criterion(class_logits, target)
                    mse_loss = mse_criterion(reconstruction, originals)
                    total_loss = 0.5 * ce_loss + 0.5 * mse_loss

                    total_loss.backward()
                    optimizer.step()

                    running_ce_loss += ce_loss.item()
                    running_mse_loss += mse_loss.item()
                    running_total_loss += total_loss.item()
                    _, predicted = class_logits.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    global_iteration += 1

                    # Log and emit GUI updates at verbose intervals
                    if global_iteration % verbose_freq == 0:
                        iter_total_loss = running_total_loss / (batch_idx + 1)
                        iter_ce_loss = running_ce_loss / (batch_idx + 1)
                        iter_mse_loss = running_mse_loss / (batch_idx + 1)
                        iter_acc = 100. * correct / total
                        log_msg = (f"  Iter {global_iteration} | Total: {iter_total_loss:.4f} | "
                                   f"CE: {iter_ce_loss:.4f} | MSE: {iter_mse_loss:.4f} | Acc: {iter_acc:.2f}%")
                        self.log_signal.emit(log_msg)
                        self.epoch_signal.emit(global_iteration, iter_total_loss, iter_acc, prev_val_loss, prev_val_acc)

                        # Emit sample reconstruction for visualization at verbose intervals
                        # Original and distorted images are fixed - only reconstruction changes
                        with torch.no_grad():
                            _, recon_sample = model(sample_distorted_tensor.unsqueeze(0))
                            recon_np = recon_sample.squeeze().detach().cpu().numpy().squeeze()
                        self.sample_signal.emit(sample_original, sample_distorted, recon_np)

                train_total_loss = running_total_loss / len(train_loader)
                train_ce_loss = running_ce_loss / len(train_loader)
                train_mse_loss = running_mse_loss / len(train_loader)
                train_acc = 100. * correct / total

                # --- Validate ---
                model.eval()
                val_ce_loss = 0.0
                val_mse_loss = 0.0
                val_total_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_idx in range(len(test_data_list)):
                        data = test_data_list[batch_idx]
                        target = test_labels_list[batch_idx]

                        # Apply moderate interference for validation
                        bsz_val = data.size(0)
                        val_noise = np.full(bsz_val, 0.5)
                        val_blur = np.full(bsz_val, 0.5)
                        distorted_val = apply_random_interference(data, val_noise, val_blur).to(device)

                        class_logits, reconstruction = model(distorted_val)

                        ce_loss = ce_criterion(class_logits, target)
                        mse_loss = mse_criterion(reconstruction, data)
                        total_loss = 0.5 * ce_loss + 0.5 * mse_loss

                        val_ce_loss += ce_loss.item()
                        val_mse_loss += mse_loss.item()
                        val_total_loss += total_loss.item()
                        _, predicted = class_logits.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()

                val_total_loss = val_total_loss / len(test_loader)
                val_acc = 100. * correct / total
                elapsed = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]['lr']

                train_losses.append(train_total_loss)
                train_accs.append(train_acc)
                val_losses.append(val_total_loss)
                val_accs.append(val_acc)

                if global_iteration % verbose_freq == 0:
                    self.epoch_signal.emit(global_iteration, train_total_loss, train_acc, val_total_loss, val_acc)

                    # Emit sample reconstruction for visualization (synced with epoch_signal updates)
                    # Original and distorted are fixed - only reconstruction changes as decoder improves
                    with torch.no_grad():
                        _, recon_sample = model(sample_distorted_tensor.unsqueeze(0))
                        recon_np = recon_sample.squeeze().detach().cpu().numpy().squeeze()
                    self.sample_signal.emit(sample_original, sample_distorted, recon_np)

                    log_msg = (f"Epoch {epoch}/{num_epochs} | "
                               f"Train Loss: {train_total_loss:.4f} (CE: {train_ce_loss:.4f}, MSE: {train_mse_loss:.4f}) Acc: {train_acc:.2f}% | "
                               f"Val Loss: {val_total_loss:.4f} Acc: {val_acc:.2f}% | "
                               f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
                    self.log_signal.emit(log_msg)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                scheduler.step(val_total_loss)
                prev_val_loss = val_total_loss
                prev_val_acc = val_acc

            total_time = time.time() - start_time
            self.log_signal.emit("=" * 60)
            self.log_signal.emit(f"Training complete! Best Val Acc: {best_val_acc:.2f}% | Total time: {total_time:.1f}s")
            self.log_signal.emit("=" * 60)

            # Save model with architecture name in architecture-based subfolder
            architecture = self.architecture  # 'mlp', 'lstm', 'cnn', 'resnet', 'transformer', or 'legacy'
            
            # Create architecture-based subfolder
            if architecture == 'legacy':
                # Legacy model uses old naming for backward compatibility
                arch_folder = "legacy_noise_robust"
                model_filename = "best_mnist_noise_robust.pth"
            else:
                # JAC model uses architecture-specific naming
                arch_folder = f"noise_robust_{architecture}"
                model_filename = f"best_mnist_noise_robust_{architecture}.pth"
            
            # Create the architecture subfolder if it doesn't exist
            arch_dir = os.path.join(MODEL_SAVE_DIR, arch_folder)
            os.makedirs(arch_dir, exist_ok=True)
            
            model_path = os.path.join(arch_dir, model_filename)
            
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'architecture': architecture,
                'is_noise_robust': True,
                'model_type': 'jac' if architecture != 'legacy' else 'noise_robust',
            }, model_path)
            self.log_signal.emit(f"Model saved to: {model_path}")

            self.finish_signal.emit({
                'model': model,
                'device': device,
                'best_val_acc': best_val_acc,
                'history': {
                    'train_losses': train_losses, 'train_accs': train_accs,
                    'val_losses': val_losses, 'val_accs': val_accs,
                },
                'is_noise_robust': True
            })

        except Exception as e:
            self.error_signal.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")