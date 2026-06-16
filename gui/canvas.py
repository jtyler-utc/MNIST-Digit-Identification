"""Drawing canvas widget for user digit input."""

import numpy as np
import torch
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPixmap
from PyQt6.QtCore import Qt, QPointF, QRectF
from scipy.ndimage import gaussian_filter


class DrawingCanvas(QWidget):
    """Custom widget for drawing digits."""

    def __init__(self, size: int = 280, parent=None):
        super().__init__(parent)
        self.size = size
        self.image_size = 28
        # In PyQt6, Qt.GlobalColor is an int enum, not a QColor
        self.background_color = QColor(0, 0, 0)  # Black background
        self.foreground_color = QColor(255, 255, 255)  # White strokes
        self.pen_width = 18
        self.drawing = False
        self.last_point = None
        self.drawing_image = QImage(self.size, self.size, QImage.Format.Format_Grayscale8)
        self.drawing_image.fill(self.background_color.rgb())
        self.setMinimumSize(self.size + 10, self.size + 10)
        self.setMaximumSize(self.size + 10, self.size + 10)

        # Distortion parameters
        self.blur_level = 0
        self.noise_level = 0

    def clear(self):
        self.drawing_image.fill(self.background_color.rgb())
        self.last_point = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        source = QRectF(0, 0, self.size, self.size)
        target = QRectF(5, 5, self.size, self.size)
        painter.drawImage(target, self.drawing_image, source)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position()

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() & Qt.MouseButton.LeftButton:
            painter = QPainter(self.drawing_image)
            pen = QPen(self.foreground_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(QPointF(self.last_point), event.position())
            self.last_point = event.position()
            painter.end()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.last_point = None

    def set_blur_level(self, level: int):
        """Set the Gaussian blur level (0-10)."""
        self.blur_level = level

    def set_noise_level(self, level: int):
        """Set the Gaussian noise level (0-10)."""
        self.noise_level = level

    def _apply_distortions(self, array: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur and noise to a 2D numpy array (0-255 uint8).

        Returns a float32 array in range [0, 255].
        """
        # Convert to float for processing
        result = array.astype(np.float32)

        # Apply Gaussian blur if blur_level > 0
        if self.blur_level > 0:
            sigma = self.blur_level / 3.0
            result = gaussian_filter(result, sigma=sigma)

        # Apply Gaussian noise if noise_level > 0
        if self.noise_level > 0:
            noise_std = self.noise_level / 25.0 * 255.0
            noise = np.random.normal(0, noise_std, result.shape)
            result = result + noise

        # Clip to valid range and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def get_distorted_array(self) -> np.ndarray:
        """Get the 28x28 distorted array for preview display.

        Returns a 28x28 uint8 array (inverted: white digits on black background,
        matching MNIST format) after applying distortions.
        """
        # Scale to 28x28 keeping aspect ratio
        scaled = self.drawing_image.scaled(
            self.image_size, self.image_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        # Create a 28x28 white canvas
        canvas = QImage(self.image_size, self.image_size, QImage.Format.Format_Grayscale8)
        canvas.fill(255)  # Start with white background
        x_offset = (self.image_size - scaled.width()) // 2
        y_offset = (self.image_size - scaled.height()) // 2
        painter = QPainter(canvas)
        painter.drawImage(x_offset, y_offset, scaled)
        painter.end()

        # Convert to numpy array
        num_pixels = canvas.width() * canvas.height()
        buffer = canvas.bits().asarray(num_pixels)
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(self.image_size, self.image_size)
        # Canvas now stores white (255) on black (0), matching MNIST format directly

        # Apply distortions
        array = self._apply_distortions(array)
        return array

    def get_normalized_image(self) -> torch.Tensor:
        """Return a normalized tensor suitable for the CNN (1, 28, 28).

        MNIST expects white digits on black background. The canvas draws
        black strokes on white background, so we need to invert the colors
        before passing to the model. Distortions (blur + noise) are applied
        after inversion.
        """
        # Scale to 28x28 keeping aspect ratio
        scaled = self.drawing_image.scaled(
            self.image_size, self.image_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        # Create a 28x28 white canvas (MNIST has white digits on black, we invert: black background, white digits)
        canvas = QImage(self.image_size, self.image_size, QImage.Format.Format_Grayscale8)
        canvas.fill(255)  # Start with white background
        x_offset = (self.image_size - scaled.width()) // 2
        y_offset = (self.image_size - scaled.height()) // 2
        painter = QPainter(canvas)
        painter.drawImage(x_offset, y_offset, scaled)
        painter.end()

        # Convert to tensor
        num_pixels = canvas.width() * canvas.height()
        buffer = canvas.bits().asarray(num_pixels)
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(self.image_size, self.image_size)
        # Canvas now stores white digits (255) on black (0), matching MNIST format directly

        # Apply distortions (blur + noise)
        array = self._apply_distortions(array)

        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        # Normalize with MNIST stats
        tensor = (tensor - 0.1307) / 0.3081
        return tensor