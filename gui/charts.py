"""Matplotlib-based chart widgets for visualization."""

import numpy as np
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Global dark theme settings
DARK_BG = '#1a1a2e'
DARK_CHARTBG = '#16213e'
WHITE_TEXT = '#e0e0e0'
WHITE_SEMI = (0.85, 0.85, 0.85, 0.3)
GRID_COLOR = (0.85, 0.85, 0.85, 0.15)
ACCENT_BLUE = '#00d4ff'
ACCENT_RED = '#ff6b6b'
ACCENT_GREEN = '#51cf66'
ACCENT_YELLOW = '#ffd43b'


class ConfidenceChart(FigureCanvas):
    """Matplotlib bar chart for digit confidence levels."""

    def __init__(self, width=5, height=4, dpi=100):
        self.dpi = dpi
        self.base_width = width
        self.base_height = height
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.fig.set_facecolor(DARK_BG)
        self.ax.set_facecolor(DARK_CHARTBG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_chart([0.1] * 10)

    def resizeEvent(self, event):
        """Redraw the chart when the widget is resized."""
        super().resizeEvent(event)
        size = event.size()
        if size.width() > 0 and size.height() > 0:
            self.fig.set_size_inches(size.width() / self.dpi, size.height() / self.dpi)
            self.fig.canvas.draw_idle()

    def update_chart(self, confidences: list):
        digits = range(10)
        max_idx = np.argmax(confidences)
        colors = [ACCENT_BLUE if i != max_idx else ACCENT_RED for i in digits]
        self.ax.clear()
        self.ax.set_facecolor(DARK_CHARTBG)
        self.fig.set_facecolor(DARK_BG)
        bars = self.ax.bar([str(d) for d in digits], confidences, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
        self.ax.set_ylabel('Confidence', color=WHITE_TEXT, fontsize=11)
        self.ax.set_title('Digit Classification Confidence', color=WHITE_TEXT, fontsize=12, fontweight='bold')
        self.ax.set_ylim(0, 1.05)
        self.ax.set_xlabel('Digit', color=WHITE_TEXT, fontsize=11)
        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_color(WHITE_TEXT)
        for spine in self.ax.spines.values():
            spine.set_color(WHITE_SEMI)
        self.ax.grid(False)
        for bar, val in zip(bars, confidences):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.1%}', ha='center', va='bottom', fontsize=9, color=WHITE_TEXT, fontweight='bold')
        self.fig.canvas.draw()


class ReconstructionChart(FigureCanvas):
    """Matplotlib chart showing original, distorted, and reconstructed images."""

    def __init__(self, width=5, height=2, dpi=100):
        self.dpi = dpi
        self.fig, self.axes = plt.subplots(1, 3, figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(DARK_BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._setup_axes()
        self.draw_empty()

    def _setup_axes(self):
        for ax in self.axes:
            ax.set_facecolor(DARK_CHARTBG)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    def draw_empty(self):
        for i, ax in enumerate(self.axes):
            ax.set_title(['Original', 'Distorted', 'Reconstructed'][i], color=WHITE_TEXT, fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes,
                    color=WHITE_TEXT, alpha=0.5)
        self.fig.canvas.draw_idle()

    def update_images(self, original: np.ndarray, distorted: np.ndarray, reconstructed: np.ndarray):
        """Update the chart with new image data.

        Args:
            original: (28, 28) clean original image
            distorted: (28, 28) distorted input image
            reconstructed: (28, 28) model's reconstruction
        """
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(DARK_CHARTBG)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        self.axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
        self.axes[0].set_title('Original (Clean)', color=WHITE_TEXT, fontsize=10, fontweight='bold')

        self.axes[1].imshow(distorted, cmap='gray', vmin=0, vmax=1)
        self.axes[1].set_title('Distorted Input', color=WHITE_TEXT, fontsize=10, fontweight='bold')

        self.axes[2].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
        self.axes[2].set_title('Reconstructed Output', color=WHITE_TEXT, fontsize=10, fontweight='bold')

        self.fig.canvas.draw()


class InterferencePlot(FigureCanvas):
    """Matplotlib plot showing JAC's interference/quality estimation."""

    def __init__(self, width=3, height=3, dpi=100):
        self.dpi = dpi
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.fig.set_facecolor(DARK_BG)
        self.ax.set_facecolor(DARK_CHARTBG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.draw_empty()

    def resizeEvent(self, event):
        """Redraw the chart when the widget is resized."""
        super().resizeEvent(event)
        size = event.size()
        if size.width() > 0 and size.height() > 0:
            self.fig.set_size_inches(size.width() / self.dpi, size.height() / self.dpi)
            self.fig.canvas.draw_idle()

    def draw_empty(self):
        self.ax.clear()
        self.ax.set_facecolor(DARK_CHARTBG)
        self.fig.set_facecolor(DARK_BG)
        self.ax.set_title('Interference Estimation', color=WHITE_TEXT, fontsize=12, fontweight='bold')
        self.ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=self.ax.transAxes,
                     color=WHITE_TEXT, alpha=0.5)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.fig.canvas.draw()

    def update_interference(self, interference_score: float):
        """Update the plot with interference estimation.

        Args:
            interference_score: Float in [0, 1] where 0 = clean, 1 = highly distorted.
        """
        self.ax.clear()
        self.ax.set_facecolor(DARK_CHARTBG)
        self.fig.set_facecolor(DARK_BG)

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, interference_score))

        # Color gradient: green (clean) -> yellow (medium) -> red (distorted)
        if score < 0.5:
            bar_color = ACCENT_GREEN  # Green for low interference
            label = 'Clean'
        elif score < 0.7:
            # Yellow-green
            bar_color = '#aaff00'
            label = 'Low Interference'
        elif score < 0.85:
            bar_color = ACCENT_YELLOW  # Yellow for medium interference
            label = 'Medium Interference'
        else:
            bar_color = ACCENT_RED  # Red for high interference
            label = 'High Interference'

        # Draw horizontal bar
        bar_height = 0.15
        bar_y = 0.45
        bar_x = 0.1
        bar_width = 0.8

        # Background track
        self.ax.barh(bar_y, bar_width, height=bar_height, left=bar_x,
                     color=(0.3, 0.3, 0.3, 0.3), edgecolor='white', linewidth=1, alpha=0.5)

        # Filled portion
        fill_width = bar_width * score
        if fill_width > 0:
            self.ax.barh(bar_y, fill_width, height=bar_height, left=bar_x,
                         color=bar_color, edgecolor='white', linewidth=1, alpha=0.8)

        # Score marker
        marker_x = bar_x + fill_width
        self.ax.plot(marker_x, bar_y, 'o', color='white', markersize=10, zorder=5)

        # Labels
        self.ax.set_title('Interference Estimation', color=WHITE_TEXT, fontsize=12, fontweight='bold')

        # Score text
        score_text = f'{score:.2f}'
        self.ax.text(0.5, 0.85, f'Score: {score_text}', ha='center', va='top',
                     transform=self.ax.transAxes, color=WHITE_TEXT, fontsize=11, fontweight='bold')

        # Level text
        self.ax.text(0.5, 0.72, label, ha='center', va='top',
                     transform=self.ax.transAxes, color=bar_color, fontsize=10, fontweight='bold')

        # Tick labels
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        self.ax.set_xticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'], color=WHITE_TEXT, fontsize=8)
        self.ax.set_xlabel('Interference Level', color=WHITE_TEXT, fontsize=9)
        self.ax.set_yticks([])
        for label in self.ax.get_xticklabels():
            label.set_color(WHITE_TEXT)
        for spine in self.ax.spines.values():
            spine.set_color(WHITE_SEMI)

        self.fig.canvas.draw()


class TrainingChart(FigureCanvas):
    """Matplotlib chart showing training progress."""

    def __init__(self, width=7, height=4, dpi=100):
        self.dpi = dpi
        self.base_width = width
        self.base_height = height
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(width, height), dpi=dpi,
            gridspec_kw={'height_ratios': [1, 1]},
            constrained_layout=True
        )
        self.fig.set_facecolor(DARK_BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.epoch_count = 0
        self._iterations = []
        self._train_losses = []
        self._train_accs = []
        self._val_losses = []
        self._val_accs = []
        self._setup_axes()
        self.draw_chart()

    def resizeEvent(self, event):
        """Redraw the chart when the widget is resized."""
        super().resizeEvent(event)
        size = event.size()
        if size.width() > 0 and size.height() > 0:
            self.fig.set_size_inches(size.width() / self.dpi, size.height() / self.dpi)
            self.fig.canvas.draw_idle()

    def _setup_axes(self):
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(DARK_CHARTBG)
            ax.title.set_color(WHITE_TEXT)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color(WHITE_TEXT)
            for spine in ax.spines.values():
                spine.set_color(WHITE_SEMI)

        self.ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Iteration', color=WHITE_TEXT)
        self.ax1.set_ylabel('Loss', color=WHITE_TEXT)
        self.ax1.grid(True, alpha=0.15, color=GRID_COLOR)

        self.ax2.set_title('Training Accuracy', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Iteration', color=WHITE_TEXT)
        self.ax2.set_ylabel('Accuracy (%)', color=WHITE_TEXT)
        self.ax2.grid(True, alpha=0.15, color=GRID_COLOR)

    def add_data(self, iteration, train_loss, train_acc, val_loss, val_acc):
        """Add a new data point to the training chart.

        Args:
            iteration: The iteration number (x-axis value)
            train_loss: Training loss value
            train_acc: Training accuracy value
            val_loss: Validation loss value
            val_acc: Validation accuracy value
        """
        self._iterations.append(iteration)
        self._train_losses.append(train_loss)
        self._train_accs.append(train_acc)
        self._val_losses.append(val_loss)
        self._val_accs.append(val_acc)
        self.draw_chart()

    def set_history(self, history: dict, verbose_freq: int = 100):
        """Set training history from completed training.

        Args:
            history: Dictionary containing train_losses, train_accs, val_losses, val_accs
            verbose_freq: The verbose frequency used during training (to compute iteration numbers)
        """
        self._train_losses = history['train_losses']
        self._train_accs = history['train_accs']
        self._val_losses = history['val_losses']
        self._val_accs = history['val_accs']
        self.epoch_count = len(self._train_losses)
        self._iterations = list(range(0, self.epoch_count * verbose_freq, verbose_freq))
        self.draw_chart()

    def draw_chart(self):
        for ax in [self.ax1, self.ax2]:
            ax.clear()
            ax.set_facecolor(DARK_CHARTBG)

        self._setup_axes()

        if self._train_losses:
            iterations = self._iterations if self._iterations else list(range(1, len(self._train_losses) + 1))
            self.ax1.plot(iterations, self._train_losses, color=ACCENT_BLUE, linewidth=1.5, label='Train Loss')
            self.ax1.plot(iterations, self._val_losses, color=ACCENT_RED, linewidth=1.5, label='Val Loss')
            self.ax1.legend(facecolor=DARK_CHARTBG, edgecolor=WHITE_SEMI,
                           labelcolor=[WHITE_TEXT, WHITE_TEXT])
            self.ax1.grid(True, alpha=0.15, color=GRID_COLOR)

            self.ax2.plot(iterations, self._train_accs, color=ACCENT_GREEN, linewidth=1.5, label='Train Acc')
            self.ax2.plot(iterations, self._val_accs, color=ACCENT_YELLOW, linewidth=1.5, label='Val Acc')
            self.ax2.legend(facecolor=DARK_CHARTBG, edgecolor=WHITE_SEMI,
                           labelcolor=[WHITE_TEXT, WHITE_TEXT])
            self.ax2.set_ylim(0, 100)  # Fixed 0%-100% scale
            self.ax2.grid(True, alpha=0.15, color=GRID_COLOR)

        self.fig.canvas.draw()

    def reset(self):
        self.epoch_count = 0
        self._iterations = []
        self._train_losses = []
        self._train_accs = []
        self._val_losses = []
        self._val_accs = []
        self.draw_chart()