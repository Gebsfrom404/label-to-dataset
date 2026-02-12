"""Batch reorder tags dialog (adapted from taggui)."""
import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout)


class BatchReorderDialog(QDialog):
    def __init__(self, tag_frequencies: dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Batch Reorder Tags')
        self.setMinimumWidth(400)
        self._tag_frequencies = tag_frequencies
        self.operation: str | None = None
        self.move_to_front_tags: list[str] = []
        self.keep_first: bool = False

        layout = QVBoxLayout(self)

        # Option: do not reorder first tag
        self.keep_first_cb = QCheckBox('Do not reorder first tag')
        self.keep_first_cb.setToolTip(
            'Preserve the first tag in each image when reordering')
        layout.addWidget(self.keep_first_cb)

        layout.addSpacing(8)

        # Quick actions
        layout.addWidget(QLabel('Quick actions:'))

        btn_row1 = QHBoxLayout()
        self.sort_alpha_btn = QPushButton('Sort Alphabetically')
        self.sort_freq_btn = QPushButton('Sort by Frequency')
        btn_row1.addWidget(self.sort_alpha_btn)
        btn_row1.addWidget(self.sort_freq_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.reverse_btn = QPushButton('Reverse Order')
        self.shuffle_btn = QPushButton('Shuffle Randomly')
        btn_row2.addWidget(self.reverse_btn)
        btn_row2.addWidget(self.shuffle_btn)
        layout.addLayout(btn_row2)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Move tags to front
        layout.addWidget(QLabel('Move tags to front (comma-separated):'))
        move_layout = QHBoxLayout()
        self.move_input = QLineEdit()
        self.move_input.setPlaceholderText('tag1, tag2, tag3')
        move_layout.addWidget(self.move_input)
        self.move_btn = QPushButton('Move to Front')
        self.move_btn.setEnabled(False)
        move_layout.addWidget(self.move_btn)
        layout.addLayout(move_layout)

        layout.addSpacing(8)

        # Close
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Signals
        self.sort_alpha_btn.clicked.connect(lambda: self._do('sort_alpha'))
        self.sort_freq_btn.clicked.connect(lambda: self._do('sort_freq'))
        self.reverse_btn.clicked.connect(lambda: self._do('reverse'))
        self.shuffle_btn.clicked.connect(lambda: self._do('shuffle'))
        self.move_btn.clicked.connect(lambda: self._do('move_to_front'))
        self.move_input.textChanged.connect(
            lambda t: self.move_btn.setEnabled(bool(t.strip())))

    def _do(self, operation: str):
        self.operation = operation
        self.keep_first = self.keep_first_cb.isChecked()
        if operation == 'move_to_front':
            raw = self.move_input.text()
            self.move_to_front_tags = [
                t.strip() for t in raw.split(',') if t.strip()]
        self.accept()

    def reorder_tags(self, tags: list[str]) -> list[str]:
        """Apply the selected operation to a tag list."""
        if not tags or self.operation is None:
            return tags

        first = None
        work = list(tags)
        if self.keep_first and len(work) > 1:
            first = work.pop(0)

        if self.operation == 'sort_alpha':
            work.sort(key=str.lower)
        elif self.operation == 'sort_freq':
            work.sort(key=lambda t: self._tag_frequencies.get(t, 0),
                      reverse=True)
        elif self.operation == 'reverse':
            work.reverse()
        elif self.operation == 'shuffle':
            random.shuffle(work)
        elif self.operation == 'move_to_front':
            front = []
            rest = []
            front_set = set(self.move_to_front_tags)
            # Maintain the order specified by user
            for t in self.move_to_front_tags:
                if t in [w for w in work]:
                    front.append(t)
            rest = [t for t in work if t not in front_set]
            work = front + rest

        if first is not None:
            work.insert(0, first)
        return work
