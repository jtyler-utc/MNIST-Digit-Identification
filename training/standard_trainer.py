"""Standard training thread for MNIST CNN training."""

import os
import time
import traceback
import sys

# Model save directory configuration
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "saved_models")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from PyQt6.QtCore import QThread, pyqtSignal

from models.base_model import MNISTCNN


class TrainingThread(QThread):
    """Background thread for standard model training."""

    log_signal = pyqtSignal(str)
    epoch_signal = pyqtSignal(int, float, float, float, float)
    finish_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_signal.emit(f"Using device: {device}")

            # Data loading
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.log_signal.emit("Downloading/loading MNIST dataset...")
            train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform)
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

            # Model setup
            self.log_signal.emit("Initializing CNN model...")
            model = MNISTCNN(num_classes=10)
            model = model.to(device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log_signal.emit(f"Total parameters: {total_params:,}")
            self.log_signal.emit(f"Trainable parameters: {trainable_params:,}")

            # Loss, optimizer, scheduler
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # Prefetch all data to GPU device before training starts
            self.log_signal.emit("Prefetching all data to GPU device...")
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

            self.log_signal.emit("=" * 60)
            self.log_signal.emit(f"Starting training for {num_epochs} epochs...")
            self.log_signal.emit(f"Batch size: {self.config['batch_size']}, Learning rate: {self.config['learning_rate']}")
            self.log_signal.emit(f"Verbose frequency: every {verbose_freq} iterations")
            self.log_signal.emit("=" * 60)

            # --- Initial validation at iteration 0 (before any training) ---
            self.log_signal.emit("Running initial validation at iteration 0...")
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx in range(len(test_data_list)):
                    data = test_data_list[batch_idx]
                    target = test_labels_list[batch_idx]
                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            val_loss = val_loss / len(test_loader)
            val_acc = 100. * correct / total
            prev_val_loss = val_loss
            prev_val_acc = val_acc

            # Emit initial validation data at iteration 0
            self.epoch_signal.emit(0, 0.0, 0.0, val_loss, val_acc)
            self.log_signal.emit(f"  Iteration 0 | Initial Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

            train_losses.append(0.0)  # Placeholder for iteration 0
            train_accs.append(0.0)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            for epoch in range(1, num_epochs + 1):
                if self._stop_requested:
                    self.log_signal.emit("Training stopped by user.")
                    break

                epoch_start = time.time()

                # --- Train ---
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for batch_idx in range(len(train_data_list)):
                    if self._stop_requested:
                        break
                    data = train_data_list[batch_idx]
                    target = train_labels_list[batch_idx]

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    global_iteration += 1

                    # Log and emit GUI updates at verbose intervals
                    if global_iteration % verbose_freq == 0:
                        iter_loss = running_loss / (batch_idx + 1)
                        iter_acc = 100. * correct / total
                        self.log_signal.emit(f"  Iteration {global_iteration} | Loss: {iter_loss:.4f} Acc: {iter_acc:.2f}%")
                        self.epoch_signal.emit(global_iteration, iter_loss, iter_acc, prev_val_loss, prev_val_acc)

                train_loss = running_loss / len(train_loader)
                train_acc = 100. * correct / total

                # --- Validate ---
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_idx in range(len(test_data_list)):
                        data = test_data_list[batch_idx]
                        target = test_labels_list[batch_idx]
                        output = model(data)
                        loss = criterion(output, target)

                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()

                val_loss = val_loss / len(test_loader)
                val_acc = 100. * correct / total
                elapsed = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]['lr']

                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                if global_iteration % verbose_freq == 0:
                    self.epoch_signal.emit(global_iteration, train_loss, train_acc, val_loss, val_acc)

                if global_iteration % verbose_freq == 0:
                    log_msg = (f"Epoch {epoch}/{num_epochs} | "
                               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                               f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
                    self.log_signal.emit(log_msg)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                scheduler.step(val_loss)

                prev_val_loss = val_loss
                prev_val_acc = val_acc

            total_time = time.time() - start_time
            self.log_signal.emit("=" * 60)
            self.log_signal.emit(f"Training complete! Best Val Acc: {best_val_acc:.2f}% | Total time: {total_time:.1f}s")
            self.log_signal.emit("=" * 60)

            # Save model with architecture name in architecture-based subfolder
            architecture = self.config.get('architecture', 'standard')
            is_noise_robust = self.config.get('trainer', 'standard') == 'noise_robust'
            
            # Create architecture-based subfolder
            if is_noise_robust:
                # Noise-robust JAC model subfolder
                arch_folder = f"noise_robust_{architecture}"
                model_filename = f"best_mnist_noise_robust_{architecture}.pth"
            elif architecture == 'standard':
                # Standard model subfolder
                arch_folder = "standard"
                model_filename = "best_mnist_model.pth"
            else:
                # Standard variant of JAC models
                arch_folder = f"standard_{architecture}"
                model_filename = f"best_mnist_model_{architecture}.pth"
            
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
                'is_noise_robust': is_noise_robust,
                'model_type': 'jac' if not is_noise_robust and architecture != 'standard' else 'standard',
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
                'is_noise_robust': False
            })

        except Exception as e:
            self.error_signal.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")