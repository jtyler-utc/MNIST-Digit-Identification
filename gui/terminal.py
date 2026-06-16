"""Terminal widget and stream handler for capturing console output."""

import sys
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtGui import QFont


class TerminalWidget(QTextEdit):
    """A terminal-like text display widget."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Courier New", 10))
        self.setPlaceholderText("Terminal output will appear here...")

    def write(self, text: str):
        self.append(text)
        # Auto-scroll
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def clear_terminal(self):
        self.clear()


class TerminalStream:
    """Custom stream to redirect print/output to terminal widget."""

    def __init__(self, terminal: TerminalWidget, original=sys.stdout):
        self.terminal = terminal
        self.original = original

    def write(self, text):
        self.terminal.write(text)
        if self.original:
            self.original.write(text)

    def flush(self):
        if self.original:
            self.original.flush()