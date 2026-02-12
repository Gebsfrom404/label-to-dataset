"""Modify tab: image modification with before/after comparison."""
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import IMAGE_EXTENSIONS, ImageListModel
from ltd.data.label_data import DEFAULT_COLORS
from ltd.utils.file_utils import get_temp_dir
from ltd.utils.image_utils import load_pixmap
from ltd.widgets.comparison_slider import ComparisonSlider
from ltd.widgets.image_list_widget import ImageListWidget
from ltd.widgets.module_selector import ModuleSelector
from ltd.workers.modification_worker import ModificationWorker

from modules.base import BaseModificationModule
from modules import discover_modules


class ModifyTab(QWidget):
    copy_to_caption_requested = Signal(list)  # list[ImageItem]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self._current_image_index = -1
        self._worker: ModificationWorker | None = None
        self._class_colors: list[str] = list(DEFAULT_COLORS)

        # Discover modification modules
        self._modification_modules = discover_modules(
            BaseModificationModule, 'modules/modifications')

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image list
        self.image_list = ImageListWidget(self.model)
        splitter.addWidget(self.image_list)

        # Center: Comparison slider
        self.comparison = ComparisonSlider()
        splitter.addWidget(self.comparison)

        # Right: Modifications + Output
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Modification settings
        mod_group = QGroupBox('Modification')
        mod_layout = QVBoxLayout(mod_group)

        if self._modification_modules:
            self.module_selector = ModuleSelector(
                'Modification module:', self._modification_modules,
                settings_key='modification')
            mod_layout.addWidget(self.module_selector)
        else:
            mod_layout.addWidget(QLabel('No modification modules found'))
            self.module_selector = None

        orig_btn_layout = QHBoxLayout()
        self.run_current_original_btn = QPushButton('Run Current (Original)')
        self.run_all_original_btn = QPushButton('Run All (Original)')
        orig_btn_layout.addWidget(self.run_current_original_btn)
        orig_btn_layout.addWidget(self.run_all_original_btn)
        mod_layout.addLayout(orig_btn_layout)

        cur_btn_layout = QHBoxLayout()
        self.run_current_modified_btn = QPushButton('Run Current (Modified)')
        self.run_all_modified_btn = QPushButton('Run All (Modified)')
        cur_btn_layout.addWidget(self.run_current_modified_btn)
        cur_btn_layout.addWidget(self.run_all_modified_btn)
        mod_layout.addLayout(cur_btn_layout)

        self.mod_progress = QProgressBar()
        self.mod_progress.setVisible(False)
        mod_layout.addWidget(self.mod_progress)

        self.mod_status = QLabel('')
        mod_layout.addWidget(self.mod_status)

        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setVisible(False)
        mod_layout.addWidget(self.cancel_btn)

        right_layout.addWidget(mod_group)

        # Output
        output_group = QGroupBox('Output')
        output_layout = QVBoxLayout(output_group)

        self.save_modified_btn = QPushButton('Save Modified Images...')
        output_layout.addWidget(self.save_modified_btn)

        self.copy_caption_btn = QPushButton('Copy to Caption')
        output_layout.addWidget(self.copy_caption_btn)

        right_layout.addWidget(output_group)
        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 600, 300])
        layout.addWidget(splitter)

    def _connect_signals(self):
        self.image_list.current_changed.connect(self._on_image_changed)
        self.image_list.load_directory_requested.connect(self._load_directory)

        self.run_current_original_btn.clicked.connect(
            lambda: self._run_modification(use_current=False, single=True))
        self.run_all_original_btn.clicked.connect(
            lambda: self._run_modification(use_current=False, single=False))
        self.run_current_modified_btn.clicked.connect(
            lambda: self._run_modification(use_current=True, single=True))
        self.run_all_modified_btn.clicked.connect(
            lambda: self._run_modification(use_current=True, single=False))
        self.cancel_btn.clicked.connect(self._cancel_modification)

        self.save_modified_btn.clicked.connect(self._save_modified)
        self.copy_caption_btn.clicked.connect(self._copy_to_caption)

    def load_from_label_tab(self, items: list[ImageItem],
                           colors: list[str] | None = None):
        """Load images with masks from the label tab."""
        if colors:
            self._class_colors = colors
        self.model.load_items(items)
        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _load_directory(self, path: str):
        """Load a directory, detecting -masklabel pairs."""
        directory = Path(path)
        self.model.load_directory(directory)

        # Detect mask pairs: imagename-masklabel.png
        for image in self.model.images:
            mask_path = image.path.parent / f'{image.name}-masklabel.png'
            if mask_path.exists():
                image.mask_path = mask_path

        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _on_image_changed(self, index: int):
        self._current_image_index = index
        image = self.model.get_image(index)
        if image is None:
            return

        # Before: original with mask overlay
        before_pixmap = self._create_before_pixmap(image)
        if before_pixmap:
            self.comparison.set_before(before_pixmap)

        # After: modified image (if exists) or original
        after_pixmap = None
        if image.modified_path and image.modified_path.exists():
            after_pixmap = load_pixmap(image.modified_path)
        if after_pixmap is None:
            after_pixmap = load_pixmap(image.path)
        if after_pixmap:
            self.comparison.set_after(after_pixmap)

    def _create_before_pixmap(self, image: ImageItem) -> QPixmap | None:
        """Create before image with label overlays (matching Label tab look)."""
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import (QBrush, QColor, QPainter, QPen, QPolygonF)
        pixmap = load_pixmap(image.path)
        if pixmap is None:
            return None

        if not image.labels:
            return pixmap

        result = pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = pixmap.width()
        h = pixmap.height()
        colors = self._class_colors

        for label in image.labels:
            base = QColor(colors[label.class_id % len(colors)])
            fill = QColor(base)
            fill.setAlpha(60)
            pen = QPen(base, 2)

            if label.has_polygon:
                points = [QPointF(x * w, y * h) for x, y in label.polygon]
                painter.setPen(pen)
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(QPolygonF(points))
            elif label.has_bbox:
                cx, cy, bw, bh = label.bbox
                rx = (cx - bw / 2) * w
                ry = (cy - bh / 2) * h
                rw = bw * w
                rh = bh * h
                painter.setPen(pen)
                painter.setBrush(QBrush(fill))
                painter.drawRect(QRectF(rx, ry, rw, rh))

        painter.end()
        return result

    # --- Modification ---

    def _set_run_buttons_enabled(self, enabled: bool):
        self.run_current_original_btn.setEnabled(enabled)
        self.run_all_original_btn.setEnabled(enabled)
        self.run_current_modified_btn.setEnabled(enabled)
        self.run_all_modified_btn.setEnabled(enabled)

    def _run_modification(self, use_current: bool = False,
                          single: bool = False):
        if self.module_selector is None:
            return
        module = self.module_selector.current_module()
        if module is None:
            return

        if single:
            image = self.model.get_image(self._current_image_index)
            if image is None or image.mask_path is None:
                QMessageBox.information(
                    self, 'Info', 'Current image has no mask.')
                return
            images_to_run = [image]
            self._single_mode = True
        else:
            images_to_run = [img for img in self.model.images
                             if img.mask_path is not None]
            if not images_to_run:
                QMessageBox.information(
                    self, 'Info', 'No images with masks.')
                return
            self._single_mode = False

        try:
            module.prepare()
        except Exception as e:
            QMessageBox.critical(self, 'Module Error', str(e))
            return

        self._worker = ModificationWorker(
            module, images_to_run, use_current=use_current)
        self._worker.progress.connect(self._on_mod_progress)
        self._worker.status.connect(self._on_mod_status)
        self._worker.modification_complete.connect(self._on_mod_result)
        self._worker.finished_work.connect(self._on_mod_finished)
        self._worker.error.connect(self._on_mod_error)

        self.mod_progress.setVisible(not single)
        self.cancel_btn.setVisible(not single)
        self._set_run_buttons_enabled(False)
        self._worker.start()

    def _cancel_modification(self):
        if self._worker:
            self._worker.cancel()

    def _on_mod_progress(self, current, total):
        self.mod_progress.setMaximum(total)
        self.mod_progress.setValue(current)

    def _on_mod_status(self, text):
        self.mod_status.setText(text)

    def _on_mod_result(self, index, output_path):
        if getattr(self, '_single_mode', False):
            # Single mode: the worker got a list of 1 image
            image = self.model.get_image(self._current_image_index)
            if image:
                image.modified_path = Path(output_path)
                self._on_image_changed(self._current_image_index)
        else:
            images_with_masks = [img for img in self.model.images
                                 if img.mask_path is not None]
            if 0 <= index < len(images_with_masks):
                image = images_with_masks[index]
                image.modified_path = Path(output_path)
                all_idx = self.model.images.index(image)
                if all_idx == self._current_image_index:
                    self._on_image_changed(self._current_image_index)

    def _on_mod_finished(self):
        self.mod_progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._set_run_buttons_enabled(True)
        self._worker = None

    def _on_mod_error(self, msg):
        QMessageBox.critical(self, 'Modification Error', msg)

    # --- Output ---

    def _save_modified(self):
        folder = QFileDialog.getExistingDirectory(self, 'Save Modified Images To')
        if not folder:
            return
        out = Path(folder)
        count = 0
        for image in self.model.images:
            source = image.modified_path if image.modified_path else image.path
            shutil.copy2(source, out / image.filename)
            count += 1
        self.mod_status.setText(f'Saved {count} images to {folder}')

    def _copy_to_caption(self):
        temp_dir = get_temp_dir('caption')
        items = []
        for image in self.model.images:
            source = image.modified_path if image.modified_path else image.path
            dest = temp_dir / image.filename
            shutil.copy2(source, dest)
            item = ImageItem(path=dest, width=image.width,
                             height=image.height)
            items.append(item)

        if items:
            self.copy_to_caption_requested.emit(items)
        else:
            QMessageBox.information(self, 'Info', 'No images to copy.')
