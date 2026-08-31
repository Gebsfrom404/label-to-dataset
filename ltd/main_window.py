from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from ltd.app import apply_theme
from ltd.settings import get_settings
from ltd.tabs.label_tab import LabelTab
from ltd.tabs.modify_tab import ModifyTab
from ltd.tabs.caption_tab import CaptionTab
from ltd.tabs.train_tab import TrainTab
from ltd.tabs.gen_images_tab import GenImagesTab
from ltd.tabs.duplicates_tab import DuplicatesTab
from ltd.tabs.extras_tab import ExtrasTab
from ltd.utils.file_utils import cleanup_all_temp
from ltd.widgets.toolbar_widget import ToolbarWidget


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.settings = get_settings()
        self.setWindowTitle('Label-to-Dataset')

        # Toolbar
        self.toolbar = ToolbarWidget(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.label_tab = LabelTab(self)
        self.modify_tab = ModifyTab(self)
        self.caption_tab = CaptionTab(self)
        self.train_tab = TrainTab(self)
        self.gen_images_tab = GenImagesTab(self)
        self.duplicates_tab = DuplicatesTab(self)
        self.extras_tab = ExtrasTab(self)

        self.tab_widget.addTab(self.label_tab, 'Label')
        self.tab_widget.addTab(self.modify_tab, 'Modify')
        self.tab_widget.addTab(self.caption_tab, 'Caption')
        self.tab_widget.addTab(self.train_tab, 'Train YOLO')
        self.tab_widget.addTab(self.gen_images_tab, 'Manage Gen Images')
        self.tab_widget.addTab(self.duplicates_tab, 'Manage Duplicates')
        self.tab_widget.addTab(self.extras_tab, 'Extras')

        # Connect toolbar signals
        self.toolbar.theme_changed.connect(self._on_theme_changed)
        self.toolbar.font_size_changed.connect(self._on_font_size_changed)

        # Connect inter-tab signals
        self._connect_tab_signals()

        # Restore window state
        self._restore()

    def _connect_tab_signals(self):
        # Label → Modify
        self.label_tab.copy_to_modify_requested.connect(
            self._on_copy_to_modify)
        # Modify → Caption
        self.modify_tab.copy_to_caption_requested.connect(
            self._on_copy_to_caption)
        # Caption → Modify
        self.caption_tab.open_in_modify_requested.connect(
            self._on_open_in_modify)
        # Label → Train (dataset path)
        self.label_tab.dataset_saved.connect(self._on_dataset_saved)
        self.label_tab.copy_to_train_requested.connect(
            self._on_copy_to_train)

    def _on_copy_to_modify(self, items, colors):
        self.modify_tab.load_from_label_tab(items, colors)
        self.tab_widget.setCurrentWidget(self.modify_tab)

    def _on_copy_to_caption(self, folder_path: str):
        self.caption_tab._load_directory(folder_path)
        self.tab_widget.setCurrentWidget(self.caption_tab)

    def _on_open_in_modify(self, items):
        self.modify_tab.load_from_label_tab(items)
        self.tab_widget.setCurrentWidget(self.modify_tab)

    def _on_dataset_saved(self, path: str):
        self.train_tab.set_dataset_path(path)

    def _on_copy_to_train(self, path: str, model_type: str):
        self.train_tab.set_dataset_path(path)
        self.train_tab.set_model_type(model_type)
        self.tab_widget.setCurrentWidget(self.train_tab)

    def _on_theme_changed(self, theme: str):
        apply_theme(self.app, theme)

    def _on_font_size_changed(self, size: int):
        font = self.app.font()
        font.setPointSize(size)
        self.app.setFont(font)

    def _restore(self):
        if self.settings.contains('geometry'):
            self.restoreGeometry(self.settings.value('geometry', type=bytes))
        else:
            self.resize(1400, 900)
            # Center on screen
            screen = self.screen().geometry()
            self.move((screen.width() - self.width()) // 2,
                      (screen.height() - self.height()) // 2)
        if self.settings.contains('window_state'):
            self.restoreState(self.settings.value('window_state', type=bytes))
        if self.settings.contains('active_tab'):
            self.tab_widget.setCurrentIndex(self.settings.value('active_tab', type=int))

    def closeEvent(self, event: QCloseEvent):
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('window_state', self.saveState())
        self.settings.setValue('active_tab', self.tab_widget.currentIndex())
        self.duplicates_tab.shutdown()
        cleanup_all_temp()
        super().closeEvent(event)
