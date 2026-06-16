"""Main application window for MNIST CNN Trainer & Classifier."""

import os
import sys
import time
import traceback
from typing import Optional

import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLabel, QSpinBox, QComboBox,
    QDoubleSpinBox, QPushButton, QProgressBar, QTextEdit,
    QGroupBox, QCheckBox, QFileDialog, QMessageBox, QSizePolicy,
    QSlider
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QListView


class PowerOfTwoSpinBox(QSpinBox):
    """QSpinBox that displays and handles powers of two."""

    def __init__(self, base_exponent_range: tuple = (1, 10), parent=None):
        super().__init__(parent)
        self._min_exp, self._max_exp = base_exponent_range
        self.setRange(self._min_exp, self._max_exp)

    def textFromValue(self, value: int) -> str:
        return str(2 ** value)

    def valueToText(self, value: int) -> str:
        return str(2 ** value)
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

from models.base_model import MNISTCNN
from models.jac_model import JAC_MODELS, get_jac_model_info, JACMLP, JACLSTM, JACCNN, JACResNet, JACTransformer
from training.standard_trainer import TrainingThread
from training.noise_robust_trainer import NoiseRobustTrainingThread
from gui.canvas import DrawingCanvas
from gui.charts import ConfidenceChart, ReconstructionChart, TrainingChart
from gui.terminal import TerminalWidget, TerminalStream

# JAC architecture keys (for checking if a model is JAC-based)
JAC_KEYS = {'mlp', 'lstm', 'cnn', 'resnet', 'transformer'}

# Model class mapping for loading
MODEL_CLASS_MAP = {
    'standard': MNISTCNN,
    'mlp': JACMLP,
    'lstm': JACLSTM,
    'cnn': JACCNN,
    'resnet': JACResNet,
    'transformer': JACTransformer,
}

# Display names for JAC models
JAC_DISPLAY_NAMES = {
    'mlp': 'JAC-MLP',
    'lstm': 'JAC-LSTM',
    'cnn': 'JAC-CNN',
    'resnet': 'JAC-ResNet',
    'transformer': 'JAC-Transformer',
}


class MNISTApp(QMainWindow):
    """Main application window for MNIST training and classification."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST CNN Trainer & Classifier")
        self.setMinimumSize(900, 700)

        self.training_thread: Optional[TrainingThread] = None
        self.trained_model = None
        self.model_device = None
        self.classification_timer: Optional[QTimer] = None
        self.is_noise_robust_model = False
        self.current_model_architecture: str = "standard"  # Track architecture type

        self._build_ui()
        self._connect_signals()
        # Initialize model combo after all widgets are created
        self._update_model_combo()

    def _is_jac_model(self, model_name: str) -> bool:
        """Check if a model name corresponds to a JAC architecture."""
        return model_name in JAC_KEYS

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Tab 1: Training ---
        self.training_tab = self._build_training_tab()
        self.tabs.addTab(self.training_tab, "Training")

        # --- Tab 2: Classification ---
        self.classification_tab = self._build_classification_tab()
        self.tabs.addTab(self.classification_tab, "Classification")

        # --- Tab 3: Terminal ---
        self.terminal = TerminalWidget()
        self.tabs.addTab(self.terminal, "Terminal")

        # Redirect stdout to terminal
        self.terminal_stream = TerminalStream(self.terminal)
        sys.stdout = self.terminal_stream
        sys.stderr = self.terminal_stream

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # === Training Options group: Single Model dropdown + Hyperparameters ===
        options_group = QGroupBox("Training Options")
        options_layout = QHBoxLayout()

        # Single unified model selection dropdown
        model_col = QWidget()
        model_inner = QVBoxLayout(model_col)
        model_inner.setContentsMargins(5, 5, 5, 5)
        model_inner.setSpacing(2)
        self.model_select_combo = QComboBox()
        self._populate_model_select_combo()
        model_lbl = QLabel("Model:")
        model_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_lbl.setStyleSheet("font-weight: bold;")
        model_inner.addWidget(model_lbl)
        model_inner.addWidget(self.model_select_combo)
        options_layout.addWidget(model_col)

        # Model info label (shows parameter count for JAC models)
        self.model_info_label = QLabel("")
        self.model_info_label.setFont(QFont("Arial", 7))
        self.model_info_label.setStyleSheet("color: gray;")
        self.model_info_label.setWordWrap(True)
        options_layout.addWidget(self.model_info_label)
        options_layout.addSpacing(20)

        # Hyperparameters (each vertical: label on top, value below)
        # Max Epochs
        epochs_col = QWidget()
        epochs_inner = QVBoxLayout(epochs_col)
        epochs_inner.setContentsMargins(5, 5, 5, 5)
        epochs_inner.setSpacing(2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(10)
        lbl = QLabel("Max Epochs:")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        epochs_inner.addWidget(lbl)
        epochs_inner.addWidget(self.epochs_spin)
        options_layout.addWidget(epochs_col)

        # Learn Rate
        lr_col = QWidget()
        lr_inner = QVBoxLayout(lr_col)
        lr_inner.setContentsMargins(5, 5, 5, 5)
        lr_inner.setSpacing(2)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(0.0001)
        lr_lbl = QLabel("Learn Rate:")
        lr_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lr_inner.addWidget(lr_lbl)
        lr_inner.addWidget(self.learning_rate_spin)
        options_layout.addWidget(lr_col)

        # Batch Size (powers of two, default 256)
        batch_col = QWidget()
        batch_inner = QVBoxLayout(batch_col)
        batch_inner.setContentsMargins(5, 5, 5, 5)
        batch_inner.setSpacing(2)
        self.batch_size_spin = PowerOfTwoSpinBox((1, 10))  # 2^1 to 2^10, displays 2 to 1024
        self.batch_size_spin.setValue(8)  # 2^8 = 256
        batch_lbl = QLabel("Batch Size:")
        batch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        batch_inner.addWidget(batch_lbl)
        batch_inner.addWidget(self.batch_size_spin)
        options_layout.addWidget(batch_col)

        # Verbose Freq
        verbose_col = QWidget()
        verbose_inner = QVBoxLayout(verbose_col)
        verbose_inner.setContentsMargins(5, 5, 5, 5)
        verbose_inner.setSpacing(2)
        self.verbose_freq_spin = QSpinBox()
        self.verbose_freq_spin.setRange(1, 10000)
        self.verbose_freq_spin.setValue(100)
        self.verbose_freq_spin.setSingleStep(50)
        verbose_lbl = QLabel("Verbose Freq:")
        verbose_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        verbose_inner.addWidget(verbose_lbl)
        verbose_inner.addWidget(self.verbose_freq_spin)
        options_layout.addWidget(verbose_col)

        options_layout.addStretch()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # === Single row of training metrics ===
        metrics_row = QHBoxLayout()
        metrics_row.addWidget(QLabel("Iteration:"))
        self.epoch_label = QLabel("0")
        self.train_loss_label = QLabel("Train Loss: -")
        self.train_acc_label = QLabel("Train Acc: -")
        self.val_loss_label = QLabel("Val Loss: -")
        self.val_acc_label = QLabel("Val Acc: -")
        metrics_row.addWidget(self.epoch_label)
        metrics_row.addWidget(self.train_loss_label)
        metrics_row.addWidget(self.train_acc_label)
        metrics_row.addWidget(self.val_loss_label)
        metrics_row.addWidget(self.val_acc_label)
        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        # Sample reconstruction chart (for JAC training only)
        self.reconstruction_chart = ReconstructionChart(width=8, height=2, dpi=100)
        self.reconstruction_chart.setVisible(False)  # Hidden by default
        layout.addWidget(self.reconstruction_chart)

        # Chart - use stretch=1 to make it fill all remaining vertical space
        self.training_chart = TrainingChart()
        layout.addWidget(self.training_chart, stretch=1)

        # Progress bar (above buttons at bottom)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # percentage-based
        self.progress_bar.setValue(0)
        # Black background before training starts
        self.progress_bar.setStyleSheet("QProgressBar { background-color: black; color: white; }")
        layout.addWidget(self.progress_bar)

        # Start/Stop button at the very bottom (centered, single button that toggles)
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setMinimumWidth(180)
        # Blue style for start button
        self.start_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #4CA1CF; "
            "color: white; "
            "font-weight: bold; "
            "font-size: 14px; "
            "border-radius: 5px; "
            "padding: 10px 20px; "
            "} "
            "QPushButton:hover { "
            "background-color: #3D8EB8; "
            "} "
            "QPushButton:disabled { "
            "background-color: #A0A0A0; "
            "}"
        )
        self.stop_btn = None  # No separate stop button
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        return tab

    def _populate_model_select_combo(self):
        """Populate the unified model selection dropdown."""
        self.model_select_combo.clear()
        # Standard first
        self.model_select_combo.addItem("Standard", "standard")
        # JAC architectures
        for key, (display_name, _) in JAC_MODELS.items():
            self.model_select_combo.addItem(display_name, key)
        # Default to Standard
        self.model_select_combo.setCurrentIndex(0)

    def _should_show_reconstruction_chart(self) -> bool:
        """Check if reconstruction chart should be visible.

        Only show when a JAC model is selected AND training is running.
        """
        selected = self.model_select_combo.currentData()
        return (self._is_jac_model(selected) and
                self.training_thread is not None and
                self.training_thread.isRunning())

    def _build_classification_tab(self) -> QWidget:
        tab = QWidget()
        # Main layout: vertical with top section (canvas + chart) and bottom image panel
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # === Top section: canvas (left) + confidence chart (right) ===
        top_layout = QHBoxLayout()

        # Left: Drawing canvas + controls
        left_layout = QVBoxLayout()
        self.canvas = DrawingCanvas()
        left_layout.addWidget(self.canvas)

        # Model loading controls
        model_control_layout = QHBoxLayout()

        # Model selection dropdown
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        model_control_layout.addWidget(QLabel("Model:"))
        model_control_layout.addWidget(self.model_combo)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self._load_model)
        model_control_layout.addWidget(self.load_model_btn)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setFont(QFont("Arial", 8))
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_control_layout.addWidget(self.model_status_label)
        left_layout.addLayout(model_control_layout)

        # Model name label (shows which model is being used for classification)
        self.model_name_label = QLabel("Classification Model: None")
        self.model_name_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_name_label.setStyleSheet("color: gray; padding: 4px; background-color: #f0f0f0; border-radius: 3px;")
        left_layout.addWidget(self.model_name_label)

        clear_btn = QPushButton("Clear Canvas")
        left_layout.addWidget(clear_btn)

        # Status
        self.prediction_label = QLabel("No model loaded - load a model first")
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setStyleSheet("color: red;")
        left_layout.addWidget(self.prediction_label)

        # Distortion controls group
        distortion_group = QGroupBox("Distortions")
        distortion_layout = QVBoxLayout()

        # Gaussian blur slider
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Gaussian Blur:"))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(0, 10)
        self.blur_slider.setValue(0)
        self.blur_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blur_slider.setTickInterval(1)
        self.blur_label = QLabel("0")
        blur_layout.addWidget(self.blur_slider)
        blur_layout.addWidget(self.blur_label)
        distortion_layout.addLayout(blur_layout)

        # Noise slider
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise:"))
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 10)
        self.noise_slider.setValue(0)
        self.noise_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.noise_slider.setTickInterval(1)
        self.noise_label = QLabel("0")
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_label)
        distortion_layout.addLayout(noise_layout)

        distortion_group.setLayout(distortion_layout)
        left_layout.addWidget(distortion_group)

        left_layout.addStretch()
        left_group = QWidget()
        left_group.setLayout(left_layout)

        # Right: Confidence chart
        right_layout = QVBoxLayout()
        self.confidence_chart = ConfidenceChart(width=6, height=5, dpi=100)
        right_layout.addWidget(self.confidence_chart)
        right_layout.addStretch()
        right_group = QWidget()
        right_group.setLayout(right_layout)

        top_layout.addWidget(left_group, 1)
        top_layout.addWidget(right_group, 1)

        layout.addLayout(top_layout, stretch=2)

        # === Bottom section: Image display panel ===
        # This panel shows either:
        # - Standard model: distorted image centered at bottom
        # - JAC model: distorted and reconstructed side-by-side at bottom
        self.bottom_image_panel = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_image_panel)
        self.bottom_layout.setContentsMargins(5, 5, 5, 5)
        self.bottom_layout.setSpacing(10)

        # Distorted image preview (always visible when model loaded)
        self.distorted_image_container = QVBoxLayout()
        self.distorted_image_label = QLabel()
        self.distorted_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distorted_image_label.setStyleSheet(
            "background-color: white; border: 2px solid #cccccc; border-radius: 5px;"
        )
        self.distorted_image_label.setMinimumSize(150, 150)
        self.distorted_image_label.setMaximumSize(400, 400)
        self.distorted_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.distorted_image_container.addWidget(self.distorted_image_label)
        self.distorted_image_label.setVisible(False)  # Hidden until model loaded
        self.bottom_layout.addLayout(self.distorted_image_container)

        # Reconstruction image preview (only for JAC models)
        self.reconstruction_image_container = QVBoxLayout()
        self.reconstruction_image_label = QLabel()
        self.reconstruction_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reconstruction_image_label.setStyleSheet(
            "background-color: white; border: 2px solid #cccccc; border-radius: 5px;"
        )
        self.reconstruction_image_label.setMinimumSize(150, 150)
        self.reconstruction_image_label.setMaximumSize(400, 400)
        self.reconstruction_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.reconstruction_image_container.addWidget(self.reconstruction_image_label)
        self.reconstruction_image_label.setVisible(False)  # Hidden until JAC model loaded
        self.bottom_layout.addLayout(self.reconstruction_image_container)

        layout.addWidget(self.bottom_image_panel, stretch=1)

        # Connect clear button
        clear_btn.clicked.connect(self.canvas.clear)

        # Connect distortion sliders
        self.blur_slider.valueChanged.connect(self._on_blur_changed)
        self.noise_slider.valueChanged.connect(self._on_noise_changed)

        self.prediction_label.setText("Draw a digit to classify")

        # Real-time classification timer (60fps = ~16ms per frame)
        self.classification_timer = QTimer(self)
        self.classification_timer.timeout.connect(self._classify)
        self.classification_timer.start(16)  # classify every ~16ms (60fps)
        self._last_chart_update = 0  # Track last chart update time for throttling

        return tab

    def _connect_signals(self):
        self.start_btn.clicked.connect(self._start_training)
        # Update model combo when selection changes
        self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)
        # Update model info when model selection changes
        self.model_select_combo.currentIndexChanged.connect(self._on_model_select_changed)

    def _on_model_select_changed(self, index: int):
        """Update model info label when selection changes."""
        if index < 0 or index >= self.model_select_combo.count():
            return
        key = self.model_select_combo.itemData(index)
        if key and self._is_jac_model(key):
            info = get_jac_model_info(key)
            params = info.get('total_params', 0)
            self.model_info_label.setText(f"~{params:,} params")
        else:
            self.model_info_label.setText("Standard CNN classifier")

    def _update_model_combo(self):
        """Update the model selection dropdown with available models.
        
        Scans for all possible model files in architecture-based subfolders:
        saved_models/standard/best_mnist_model.pth
        saved_models/noise_robust_legacy/best_mnist_noise_robust.pth
        saved_models/standard_{arch}/best_mnist_model_{arch}.pth
        saved_models/noise_robust_{arch}/best_mnist_noise_robust_{arch}.pth
        
        Available models show with checkmark (✓) and are enabled/clickable
        Unavailable models are greyed out (disabled)
        """
        # Use QStandardItemModel for per-item enable/disable
        self.model_combo.clear()
        model = QStandardItemModel()
        
        # Model save directory
        model_save_dir = os.path.join(os.getcwd(), "saved_models")
        
        # Check standard model in saved_models/standard/
        standard_path = os.path.join(model_save_dir, "standard", "best_mnist_model.pth")
        has_standard = os.path.exists(standard_path)

        # Add standard model item with path info
        item = QStandardItem("✓ Standard Model" if has_standard else "Standard Model (not available)")
        item.setData("standard", Qt.ItemDataRole.UserRole)
        item.setData(standard_path, Qt.ItemDataRole.UserRole + 1)  # Store actual file path
        item.setEnabled(has_standard)
        if has_standard:
            item.setForeground(Qt.GlobalColor.darkGreen)
        else:
            item.setForeground(Qt.GlobalColor.gray)
        model.appendRow(item)

        # JAC models: show all architectures, grey out unavailable
        jac_archs = ['mlp', 'lstm', 'cnn', 'resnet', 'transformer']
        for arch in jac_archs:
            standard_jac_path = os.path.join(model_save_dir, f"standard_{arch}", f"best_mnist_model_{arch}.pth")
            noise_robust_jac_path = os.path.join(model_save_dir, f"noise_robust_{arch}", f"best_mnist_noise_robust_{arch}.pth")
            
            has_standard_jac = os.path.exists(standard_jac_path)
            has_noise_jac = os.path.exists(noise_robust_jac_path)
            
            jac_name = JAC_DISPLAY_NAMES.get(arch, arch.upper())
            
            if has_standard_jac or has_noise_jac:
                variants = []
                if has_standard_jac:
                    variants.append("Standard")
                if has_noise_jac:
                    variants.append("Noise-Robust")
                variant_text = "+".join(variants)
                display_text = f"✓ {jac_name} ({variant_text})"
                item = QStandardItem(display_text)
                item.setData(arch, Qt.ItemDataRole.UserRole)
                # Store noise-robust path as primary (it's preferred), fallback to standard path
                primary_path = noise_robust_jac_path if has_noise_jac else standard_jac_path
                item.setData(primary_path, Qt.ItemDataRole.UserRole + 1)
                item.setEnabled(True)
                item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                display_text = f"{jac_name} (not available)"
                item = QStandardItem(display_text)
                item.setData(arch, Qt.ItemDataRole.UserRole)
                # Store a placeholder path for unavailable models
                item.setData(noise_robust_jac_path, Qt.ItemDataRole.UserRole + 1)
                item.setEnabled(False)
                item.setForeground(Qt.GlobalColor.gray)
            model.appendRow(item)

        # Set the model on the combo box
        self.model_combo.setModel(model)
        self.model_combo.setView(QListView())

        # Set default selection: prefer standard model if available
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.isEnabled():
                self.model_combo.setCurrentIndex(i)
                break
        
        # Disable combo if no models available
        any_available = any(model.item(i).isEnabled() for i in range(model.rowCount()))
        self.model_combo.setEnabled(any_available)
        
        if not any_available:
            self.model_combo.addItem("No models available - train first")

    def _on_model_selection_changed(self, text: str):
        """Handle model selection changes."""
        pass  # Just visual feedback, actual load happens on button click

    def _get_selected_model_key(self) -> str:
        """Get the selected model key from the training dropdown."""
        return self.model_select_combo.currentData() or "standard"

    def _load_model(self):
        """Load a saved MNIST model from disk based on dropdown selection."""
        # Check if any model is available
        if not self.model_combo.isEnabled():
            QMessageBox.warning(self, "No Models Available", 
                              "No trained models are available. Please train a model first.")
            return
        
        # Get the selected index
        index = self.model_combo.currentIndex()
        if index < 0 or index >= self.model_combo.count():
            QMessageBox.warning(self, "No Model Selected", "Please select a valid model from the dropdown.")
            return
        
        # Check if the selected item is enabled (has a checkmark)
        selected_text = self.model_combo.itemText(index)
        if not selected_text.startswith("✓"):
            QMessageBox.warning(self, "Model Not Available", 
                              "The selected model file is not available. Please train a model first.")
            return
        
        model_key = self.model_combo.itemData(index)
        if model_key is None:
            QMessageBox.warning(self, "No Model Selected", "Please select a valid model from the dropdown.")
            return

        # Get the stored file path for this model item
        model_path = self.model_combo.itemData(index, Qt.ItemDataRole.UserRole + 1)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._on_log(f"Using device: {device}")
        if torch.cuda.is_available():
            self._on_log(f"GPU: {torch.cuda.get_device_name(0)}")

        # Determine model path and type based on selection
        is_noise_robust = False
        architecture = "standard"
        display_name = "Unknown"

        model_save_dir = os.path.join(os.getcwd(), "saved_models")

        if model_key == "standard":
            # Use stored path or fallback
            if model_path and os.path.exists(model_path):
                self._on_log(f"Loading standard model from: {model_path}")
            else:
                # Fallback for backwards compatibility
                model_path = os.path.join(model_save_dir, "standard", "best_mnist_model.pth")
                self._on_log(f"Loading standard model (best_mnist_model.pth)...")
            is_noise_robust = False
            architecture = "standard"
            display_name = "Standard Model"
            
        else:
            # JAC model: check which variant exists in architecture subfolders
            jac_name = JAC_DISPLAY_NAMES.get(model_key, model_key.upper())
            architecture = model_key
            
            # Try noise-robust variant first in saved_models/noise_robust_{arch}/
            noise_jac_path = os.path.join(model_save_dir, f"noise_robust_{model_key}", f"best_mnist_noise_robust_{model_key}.pth")
            if os.path.exists(noise_jac_path):
                model_path = noise_jac_path
                is_noise_robust = True
                display_name = f"{jac_name} (Noise-Robust)"
                self._on_log(f"Loading {display_name} from: {model_path}")
            else:
                # Fall back to standard variant in saved_models/standard_{arch}/
                standard_jac_path = os.path.join(model_save_dir, f"standard_{model_key}", f"best_mnist_model_{model_key}.pth")
                if os.path.exists(standard_jac_path):
                    model_path = standard_jac_path
                    is_noise_robust = False
                    display_name = f"{jac_name} (Standard)"
                    self._on_log(f"Loading {display_name} from: {model_path}")
                else:
                    QMessageBox.warning(self, "Model Not Found", 
                                      f"Model file for {jac_name} not found.")
                    return

        if model_path is None or not os.path.exists(model_path):
            QMessageBox.warning(
                self, "Model Not Found",
                f"Selected model file not found at:\n{model_path}\n\nPlease train the appropriate model first."
            )
            return

        try:
            self._on_log(f"Attempting to load model from: {model_path}")
            self._on_log("Loading checkpoint...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self._on_log(f"Checkpoint keys: {list(checkpoint.keys())}")
            self._on_log(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A')}")

            # Detect model type from file or checkpoint
            is_noise_robust = "noise_robust" in model_path or checkpoint.get('is_noise_robust', False)
            self.is_noise_robust_model = is_noise_robust
            self.current_model_architecture = architecture

            # Get model class
            model_class = MODEL_CLASS_MAP.get(architecture, MNISTCNN)
            
            self._on_log(f"Loading model class: {model_class.__name__}")
            self.trained_model = model_class(num_classes=10)

            self.trained_model.load_state_dict(checkpoint['model_state_dict'])
            self.trained_model = self.trained_model.to(device).eval()
            self.model_device = device

            self._on_log("Model state dict loaded successfully")
            self._on_log(f"Model moved to device: {self.model_device}")

            # Test the model with a dummy input
            self._on_log("Running model test with dummy input...")
            with torch.no_grad():
                dummy = torch.randn(1, 1, 28, 28).to(device)
                if self.is_noise_robust_model or self._is_jac_model(architecture):
                    test_output, test_recon = self.trained_model(dummy)
                    self._on_log(f"Test output shape: {test_output.shape}, Reconstruction shape: {test_recon.shape}")
                else:
                    test_output = self.trained_model(dummy)
                    self._on_log(f"Test output shape: {test_output.shape}")

            # Update UI based on model type
            if self.is_noise_robust_model or self._is_jac_model(architecture):
                self.load_model_btn.setText(f"{display_name} \u2713")
                self.model_status_label.setText(f"Acc: {checkpoint['val_acc']:.1f}% | +Decoder")
                self.model_status_label.setStyleSheet("color: darkblue; font-weight: bold;")
                # Show both image previews for noise-robust/JAC model
                self.distorted_image_label.setVisible(True)
                self.reconstruction_image_label.setVisible(True)
                self._on_log(f"{display_name} loaded - reconstruction output enabled")
            else:
                self.load_model_btn.setText(f"{display_name} \u2713")
                self.model_status_label.setText(f"Acc: {checkpoint['val_acc']:.1f}%")
                self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
                # Show only distorted image for standard model
                self.distorted_image_label.setVisible(True)
                self.reconstruction_image_label.setVisible(False)

            # Update model name label
            self.model_name_label.setText(f"Classification Model: {display_name}")
            if self.is_noise_robust_model or self._is_jac_model(architecture):
                self.model_name_label.setStyleSheet("color: darkblue; padding: 4px; background-color: #e0f0ff; border-radius: 3px;")
            else:
                self.model_name_label.setStyleSheet("color: green; padding: 4px; background-color: #e0ffe0; border-radius: 3px;")

            self.prediction_label.setText("Model ready - draw a digit!")
            self.prediction_label.setStyleSheet("color: green;")

            # Update bottom layout for the model type
            self._update_bottom_layout_for_model_type()

            self._on_log(f"Model loaded successfully! Validation Acc: {checkpoint['val_acc']:.2f}%")
            QMessageBox.information(
                self, "Model Loaded",
                f"Model loaded successfully!\nModel: {display_name}\nValidation Accuracy: {checkpoint['val_acc']:.2f}%\nDevice: {device}" +
                (f"\n\nThis is a noise-robust model with reconstruction capability." if self.is_noise_robust_model else "")
            )

        except Exception as e:
            import traceback
            error_msg = f"Failed to load model: {str(e)}\n{traceback.format_exc()}"
            self._on_log(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")

    def _get_config(self) -> dict:
        model_key = self._get_selected_model_key()
        config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': 2 ** self.batch_size_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'verbose_freq': self.verbose_freq_spin.value(),
        }
        if self._is_jac_model(model_key):
            config['trainer'] = 'noise_robust'
            config['architecture'] = model_key
        else:
            config['trainer'] = 'standard'
        return config

    def _start_training(self):
        try:
            self._on_log("DEBUG: _start_training called")
            if self.training_thread and self.training_thread.isRunning():
                # Training is running, stop it
                self._on_log("DEBUG: Stopping training...")
                self.training_thread.stop()
                self._on_log("Stop requested...")
                return

            self._on_log("DEBUG: Calling _get_config...")
            config = self._get_config()
            model_key = self._get_selected_model_key()
            is_jac = self._is_jac_model(model_key)
            self._on_log(f"DEBUG: Config = {config}, Model key = {model_key}, Is JAC = {is_jac}")

            # Show/hide reconstruction chart based on JAC selection
            if is_jac:
                self.reconstruction_chart.setVisible(True)
                self.reconstruction_chart.draw_empty()
            else:
                self.reconstruction_chart.setVisible(False)

            if is_jac:
                self._on_log("DEBUG: Creating NoiseRobustTrainingThread...")
                self.training_thread = NoiseRobustTrainingThread(config)
                self.training_thread.log_signal.connect(self._on_log)
                self.training_thread.epoch_signal.connect(self._on_epoch_progress)
                self.training_thread.sample_signal.connect(self._on_sample_reconstruction)
                self.training_thread.finish_signal.connect(self._on_training_finished)
                self.training_thread.error_signal.connect(self._on_training_error)
            else:
                self._on_log("DEBUG: Creating TrainingThread...")
                self.training_thread = TrainingThread(config)
                self.training_thread.log_signal.connect(self._on_log)
                self.training_thread.epoch_signal.connect(self._on_epoch_progress)
                self.training_thread.finish_signal.connect(self._on_training_finished)
                self.training_thread.error_signal.connect(self._on_training_error)

            self._on_log("DEBUG: Disabling controls...")
            # Switch button to red "Stop Training" style
            self.start_btn.setText("Stop Training")
            self.start_btn.setStyleSheet(
                "QPushButton { "
                "background-color: #E74C3C; "
                "color: white; "
                "font-weight: bold; "
                "font-size: 14px; "
                "border-radius: 5px; "
                "padding: 10px 20px; "
                "} "
                "QPushButton:hover { "
                "background-color: #C0392B; "
                "} "
                "QPushButton:disabled { "
                "background-color: #A0A0A0; "
                "}"
            )
            self.epochs_spin.setEnabled(False)
            self.batch_size_spin.setEnabled(False)
            self.learning_rate_spin.setEnabled(False)
            self.verbose_freq_spin.setEnabled(False)
            # Reset to black before training starts
            self.progress_bar.setValue(0)
            self.progress_bar.setStyleSheet("QProgressBar { background-color: black; color: white; }")
            # Calculate total iterations for percentage-based progress
            # MNIST has 60,000 training images
            num_train_samples = 60000
            batch_size = config['batch_size']
            num_batches = (num_train_samples + batch_size - 1) // batch_size
            total_iterations = config['epochs'] * num_batches
            self._total_training_iterations = total_iterations
            self.progress_bar.setRange(0, 100)

            self._on_log("DEBUG: Resetting chart...")
            self.training_chart.reset()
            self._on_log("DEBUG: Starting thread...")
            self.training_thread.start()
            self._on_log(f"Training thread started (model: {model_key}).")
        except Exception as e:
            import traceback
            error_msg = f"Exception in _start_training:\n{str(e)}\n{traceback.format_exc()}"
            self._on_log(error_msg)

    # _stop_training is now integrated into _start_training as a toggle

    def _on_log(self, message: str):
        self.terminal.write(message)

    def _on_training_error(self, message: str):
        self._on_log(message)
        self._reset_training_button()
        self.epochs_spin.setEnabled(True)
        self.batch_size_spin.setEnabled(True)
        self.learning_rate_spin.setEnabled(True)
        self.verbose_freq_spin.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: black; color: white; }")
        self.reconstruction_chart.setVisible(False)
        self._on_log("Training failed due to an error.")
        self._update_model_combo()

    def _on_epoch_progress(self, iteration: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        config = self._get_config()
        is_jac = config.get('trainer') == 'noise_robust'

        # Update single-row metrics display
        self.epoch_label.setText(f"Iteration: {iteration}")
        
        if is_jac:
            self.train_loss_label.setText(f"Train Loss: {train_loss:.4f} (CE+MSE)")
            self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
        else:
            self.train_loss_label.setText(f"Train Loss: {train_loss:.4f}")
            self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")

        self.train_acc_label.setText(f"Train Acc: {train_acc:.2f}%")
        self.val_acc_label.setText(f"Val Acc: {val_acc:.2f}%")
        self.training_chart.add_data(iteration, train_loss, train_acc, val_loss, val_acc)

        # Update progress bar percentage based on iteration progress
        total_iterations = getattr(self, '_total_training_iterations', 10000)
        if total_iterations > 0:
            progress_pct = int((iteration / total_iterations) * 100)
            progress_pct = min(progress_pct, 99)  # Don't reach 100 until training finishes
            self.progress_bar.setValue(progress_pct)

    def _on_sample_reconstruction(self, original: np.ndarray, distorted: np.ndarray, reconstructed: np.ndarray):
        """Update the reconstruction chart with sample image data."""
        # Use QTimer to defer update to next event loop iteration (proper Qt cross-thread pattern)
        QTimer.singleShot(0, lambda o=original, d=distorted, r=reconstructed: self._update_reconstruction_chart(o, d, r))

    def _update_reconstruction_chart(self, original: np.ndarray, distorted: np.ndarray, reconstructed: np.ndarray):
        """Actually update the reconstruction chart."""
        self.reconstruction_chart.update_images(original, distorted, reconstructed)

    def _reset_training_button(self):
        """Reset the training button to blue 'Start Training' style."""
        self.start_btn.setText("Start Training")
        self.start_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #4CA1CF; "
            "color: white; "
            "font-weight: bold; "
            "font-size: 14px; "
            "border-radius: 5px; "
            "padding: 10px 20px; "
            "} "
            "QPushButton:hover { "
            "background-color: #3D8EB8; "
            "} "
            "QPushButton:disabled { "
            "background-color: #A0A0A0; "
            "}"
        )

    def _on_training_finished(self, result: dict):
        self.trained_model = result['model'].eval()
        self.model_device = result['device']
        self.is_noise_robust_model = result.get('is_noise_robust', False)

        self._reset_training_button()
        self.epochs_spin.setEnabled(True)
        self.batch_size_spin.setEnabled(True)
        self.learning_rate_spin.setEnabled(True)
        self.verbose_freq_spin.setEnabled(True)
        self.reconstruction_chart.setVisible(False)

        # Set progress bar to 100% and go gold (defer to next event loop for proper rendering)
        QTimer.singleShot(0, self._set_progress_bar_gold)

        training_type = "Noise-robust" if self.is_noise_robust_model else "Standard"
        self._on_log(f"Training finished ({training_type} mode). Best Val Accuracy: {result['best_val_acc']:.2f}%")
        self._on_log(f"Model device: {self.model_device}")

        # Get architecture from training config
        model_key = self._get_selected_model_key()
        if self._is_jac_model(model_key):
            self.current_model_architecture = model_key
            jac_name = JAC_DISPLAY_NAMES.get(model_key, model_key.upper())
            model_display = f"{jac_name} {'(Noise-Robust)' if self.is_noise_robust_model else ''}"
        else:
            self.current_model_architecture = "standard"
            model_display = "Standard Model" if not self.is_noise_robust_model else "Noise-Robust Model"

        # Update classification tab status
        if self.is_noise_robust_model:
            self.load_model_btn.setText("Noise-Robust Model \u2713")
            self.model_status_label.setText(f"From Training: {result['best_val_acc']:.1f}% | +Decoder")
            self.model_status_label.setStyleSheet("color: darkblue; font-weight: bold;")
            # Show both image previews for noise-robust model
            self.distorted_image_label.setVisible(True)
            self.reconstruction_image_label.setVisible(True)
            self._on_log("Noise-robust model from training - reconstruction output enabled")
        else:
            self.load_model_btn.setText("Model Ready \u2713")
            self.model_status_label.setText(f"From Training: {result['best_val_acc']:.1f}%")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
            # Show only distorted image for standard model
            self.distorted_image_label.setVisible(True)
            self.reconstruction_image_label.setVisible(False)

        # Update model name label
        self.model_name_label.setText(f"Classification Model: {model_display}")
        if self.is_noise_robust_model:
            self.model_name_label.setStyleSheet("color: darkblue; padding: 4px; background-color: #e0f0ff; border-radius: 3px;")
        else:
            self.model_name_label.setStyleSheet("color: green; padding: 4px; background-color: #e0ffe0; border-radius: 3px;")

        self.prediction_label.setText("Model ready - draw a digit!")
        self.prediction_label.setStyleSheet("color: green;")

        # Update bottom layout for the model type
        self._update_bottom_layout_for_model_type()

        # Store history for chart
        verbose_freq = self.verbose_freq_spin.value()
        self.training_chart.set_history(result['history'], verbose_freq=verbose_freq)
        self.training_chart.draw_chart()

        # Refresh model combo to show the newly trained model
        self._update_model_combo()

    def _set_progress_bar_gold(self):
        """Apply gold style to progress bar."""
        # Force apply the gold style with centered text
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet(
            "QProgressBar { "
            "background-color: gold; "
            "color: black; "
            "font-weight: bold; "
            "border-radius: 5px; "
            "border: 1px solid #cccccc; "
            "text-align: center; "
            "} QProgressBar::chunk { background-color: gold; border-radius: 3px; }"
        )
        self.progress_bar.setValue(100)
        self.progress_bar.repaint()
        self.repaint()
        self.update()

    def _on_blur_changed(self, value: int):
        """Handle blur slider value changes."""
        self.blur_label.setText(str(value))
        self.canvas.set_blur_level(value)

    def _on_noise_changed(self, value: int):
        """Handle noise slider value changes."""
        self.noise_label.setText(str(value))
        self.canvas.set_noise_level(value)

    def _update_bottom_layout_for_model_type(self):
        """Update the bottom layout to match the current model type.

        - Standard model: distorted image centered at bottom
        - JAC/noise-robust model: distorted and reconstructed side-by-side at bottom
        """
        # Clear existing stretch/spacer items
        while self.bottom_layout.count() > 2:
            item = self.bottom_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout():
                # Remove spacer widgets from layouts
                pass

        if self.is_noise_robust_model or self._is_jac_model(self.current_model_architecture):
            # Side-by-side layout: distorted | reconstructed
            # Both images get equal stretch
            self.distorted_image_container.addWidget(
                QLabel("<b>Distorted Input</b>"), stretch=0
            )
            self.reconstruction_image_container.addWidget(
                QLabel("<b>Reconstructed Output</b>"), stretch=0
            )
            # Set stretch factors for side-by-side
            self.bottom_layout.setStretch(0, 1)
            self.bottom_layout.setStretch(1, 1)
        else:
            # Standard model: single distorted image centered
            # Add spacer before to center it
            if self.bottom_layout.count() == 1:
                spacer = QWidget()
                spacer.setMaximumWidth(0)
                spacer.setMaximumHeight(0)
                self.bottom_layout.insertWidget(0, spacer)
            self.bottom_layout.setStretch(0, 1)
            if self.bottom_layout.count() > 1:
                self.bottom_layout.setStretch(1, 2)

        self.bottom_image_panel.layout().setAlignment(
            self.distorted_image_container, Qt.AlignmentFlag.AlignCenter
        )
        self.bottom_image_panel.layout().setAlignment(
            self.reconstruction_image_container, Qt.AlignmentFlag.AlignCenter
        )

    def _update_distortion_preview(self):
        """Update the distorted image label with the current distorted image."""
        if not self.distorted_image_label.isVisible():
            return

        distorted_array = self.canvas.get_distorted_array()
        # Convert to QImage for display
        rgb_array = np.stack([distorted_array] * 3, axis=2)
        height, width, channel = rgb_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit the available space while maintaining aspect ratio
        available_size = self.distorted_image_label.size()
        if available_size.width() > 50 and available_size.height() > 50:
            scaled_pixmap = pixmap.scaled(
                available_size.width() - 10, available_size.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            # Fallback: scale to a reasonable default
            scaled_pixmap = pixmap.scaled(
                200, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        self.distorted_image_label.setPixmap(scaled_pixmap)

    def _update_reconstruction_preview(self, reconstructed_array: np.ndarray):
        """Update the reconstruction image label with the reconstructed image."""
        if not self.reconstruction_image_label.isVisible():
            return

        rgb_array = np.stack([reconstructed_array] * 3, axis=2)
        height, width, channel = rgb_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit the available space while maintaining aspect ratio
        available_size = self.reconstruction_image_label.size()
        if available_size.width() > 50 and available_size.height() > 50:
            scaled_pixmap = pixmap.scaled(
                available_size.width() - 10, available_size.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            # Fallback: scale to a reasonable default
            scaled_pixmap = pixmap.scaled(
                200, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        self.reconstruction_image_label.setPixmap(scaled_pixmap)

    def _classify(self):
        import time as _time

        if self.trained_model is None:
            return

        if self.model_device is None:
            return

        # Check if canvas has any drawing
        try:
            img_data = self.canvas.drawing_image
            w = img_data.width()
            h = img_data.height()
            num_pixels = w * h
            bits_ptr = img_data.bits()
            arr = np.frombuffer(bits_ptr.asarray(num_pixels), dtype=np.uint8)
            if np.all(arr == 0):
                return  # Empty canvas (black background with no white strokes)
        except Exception:
            return

        with torch.no_grad():
            try:
                image_tensor = self.canvas.get_normalized_image()
                image_tensor = image_tensor.to(self.model_device)

                if self.is_noise_robust_model or self._is_jac_model(self.current_model_architecture):
                    # JAC/noise-robust model: get both classification and reconstruction
                    output, reconstruction = self.trained_model(image_tensor)
                    recon_np = reconstruction.squeeze().cpu().numpy()

                    # Update reconstruction preview
                    recon_display = (recon_np * 255).astype(np.uint8)
                    self._update_reconstruction_preview(recon_display)
                else:
                    # Standard model
                    output = self.trained_model(image_tensor)

                probabilities = torch.softmax(output, dim=1).squeeze()
                confidences = probabilities.cpu().numpy().tolist()

                predicted_digit = int(torch.argmax(probabilities).item())
                confidence = confidences[predicted_digit]

                # Update prediction label
                self.prediction_label.setText(
                    f"Predicted: {predicted_digit} ({confidence:.1%})"
                )

                # Throttle chart updates to ~10fps
                current_time = _time.time()
                if current_time - self._last_chart_update >= 0.1:
                    self.confidence_chart.update_chart(confidences)
                    self._last_chart_update = current_time

                    # Update distortion preview at 10fps
                    self._update_distortion_preview()

                    # Log to terminal (throttled)
                    self._on_log(f"Classified: digit {predicted_digit} with {confidence:.1%} confidence")
            except Exception as e:
                import traceback
                error_msg = f"Classification error: {str(e)}\n{traceback.format_exc()}"
                self._on_log(error_msg)