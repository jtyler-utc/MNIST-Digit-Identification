"""MNIST CNN Training Application with PyQt6 GUI.

A three-tab application for training a CNN on MNIST and classifying user-drawn digits.
Supports interference tolerance training with autoencoder reconstruction.
"""

import sys
from PyQt6.QtWidgets import QApplication

from gui.main_window import MNISTApp

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MNISTApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()