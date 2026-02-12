from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListView, QPushButton,
                               QVBoxLayout, QWidget, QFileDialog)

from ltd.data.image_list_model import ImageListModel


class ImageListWidget(QWidget):
    """Reusable widget: thumbnail list + image counter + load button."""
    current_changed = Signal(int)  # emits the new row index
    load_directory_requested = Signal(str)  # emits the directory path

    def __init__(self, model: ImageListModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton('Load Folder...')
        layout.addWidget(self.load_button)

        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.setViewMode(QListView.ViewMode.ListMode)
        from PySide6.QtCore import QSize
        self.list_view.setIconSize(QSize(160, 90))
        self.list_view.setSpacing(2)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        layout.addWidget(self.list_view, stretch=1)

        counter_layout = QHBoxLayout()
        self.counter_label = QLabel('0 / 0')
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        counter_layout.addWidget(self.counter_label)
        layout.addLayout(counter_layout)

    def _connect_signals(self):
        self.list_view.selectionModel().currentChanged.connect(
            self._on_current_changed)
        self.model.modelReset.connect(self._on_model_reset)
        self.load_button.clicked.connect(self._on_load_clicked)

    def _on_model_reset(self):
        # Reconnect selection model after reset (Qt invalidates it)
        self.list_view.selectionModel().currentChanged.connect(
            self._on_current_changed)
        self._update_counter()

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        if current.isValid():
            self.current_changed.emit(current.row())
            self._update_counter()

    def _update_counter(self):
        current = self.list_view.currentIndex()
        row = current.row() + 1 if current.isValid() else 0
        total = self.model.rowCount()
        self.counter_label.setText(f'{row} / {total}')

    def _on_load_clicked(self):
        from ltd.settings import get_settings, DEFAULT_SETTINGS
        settings = get_settings()
        last_dir = settings.value('last_label_directory',
                                  DEFAULT_SETTINGS['last_label_directory'],
                                  type=str)
        directory = QFileDialog.getExistingDirectory(
            self, 'Select Image Directory', last_dir)
        if directory:
            settings.setValue('last_label_directory', directory)
            self.load_directory_requested.emit(directory)

    def select_index(self, index: int):
        if 0 <= index < self.model.rowCount():
            model_index = self.model.index(index)
            self.list_view.setCurrentIndex(model_index)
            self.list_view.scrollTo(model_index)

    def current_row(self) -> int:
        current = self.list_view.currentIndex()
        return current.row() if current.isValid() else -1

    def go_to_previous(self):
        row = self.current_row()
        if row > 0:
            self.select_index(row - 1)

    def go_to_next(self):
        row = self.current_row()
        if row < self.model.rowCount() - 1:
            self.select_index(row + 1)
