"""Modify tab: image modification with before/after comparison."""
import shutil
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QSlider, QSpinBox,
                               QSplitter, QVBoxLayout, QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import IMAGE_EXTENSIONS, ImageListModel
from ltd.data.label_data import DEFAULT_COLORS
from ltd.utils.file_utils import get_temp_dir, get_temp_dir_no_clear
from ltd.utils.image_utils import load_pixmap_preview
from ltd.widgets.comparison_slider import ComparisonSlider
from ltd.widgets.image_list_widget import ImageListWidget
from ltd.widgets.loading_dialog import loading_dialog
from ltd.widgets.module_selector import ModuleSelector
from ltd.workers.modification_worker import ModificationWorker

from modules.base import BaseModificationModule
from modules import discover_modules


class ModifyTab(QWidget):
    copy_to_caption_requested = Signal(list)  # list[ImageItem]

    _PIXMAP_CACHE_MAX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self._current_image_index = -1
        self._worker: ModificationWorker | None = None
        self._class_colors: list[str] = list(DEFAULT_COLORS)
        self._pixmap_cache: OrderedDict[str, object] = OrderedDict()

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

        # Right: Modifications + Tools + Output
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

        # Tools group
        tools_group = QGroupBox('Tools')
        tools_layout = QVBoxLayout(tools_group)

        # Crop row
        crop_row = QHBoxLayout()
        self.crop_toggle = QPushButton('Crop')
        self.crop_toggle.setCheckable(True)
        self.apply_crop_btn = QPushButton('Apply Crop')
        self.apply_crop_btn.setEnabled(False)
        crop_row.addWidget(self.crop_toggle)
        crop_row.addWidget(self.apply_crop_btn)
        tools_layout.addLayout(crop_row)

        # Split row
        split_row = QHBoxLayout()
        self.split_toggle = QPushButton('Split')
        self.split_toggle.setCheckable(True)
        self.split_h_radio = QRadioButton('H')
        self.split_v_radio = QRadioButton('V')
        self.split_v_radio.setChecked(True)
        self.split_orientation_group = QButtonGroup(self)
        self.split_orientation_group.addButton(self.split_h_radio)
        self.split_orientation_group.addButton(self.split_v_radio)
        split_row.addWidget(self.split_toggle)
        split_row.addWidget(self.split_h_radio)
        split_row.addWidget(self.split_v_radio)
        tools_layout.addLayout(split_row)

        # Split position slider + spinbox
        split_pos_row = QHBoxLayout()
        split_pos_row.addWidget(QLabel('Pos:'))
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 1000)
        self.split_slider.setValue(500)
        self.split_spinbox = QSpinBox()
        self.split_spinbox.setRange(0, 100)
        self.split_spinbox.setValue(50)
        self.split_spinbox.setSuffix('%')
        split_pos_row.addWidget(self.split_slider)
        split_pos_row.addWidget(self.split_spinbox)
        tools_layout.addLayout(split_pos_row)

        self.apply_split_btn = QPushButton('Apply Split')
        self.apply_split_btn.setEnabled(False)
        tools_layout.addWidget(self.apply_split_btn)

        # Restore original row
        restore_row = QHBoxLayout()
        self.restore_current_btn = QPushButton('Restore Current')
        self.restore_all_btn = QPushButton('Restore All')
        restore_row.addWidget(self.restore_current_btn)
        restore_row.addWidget(self.restore_all_btn)
        tools_layout.addLayout(restore_row)

        right_layout.addWidget(tools_group)

        # Output
        output_group = QGroupBox('Output')
        output_layout = QVBoxLayout(output_group)

        self.save_modified_btn = QPushButton('Save Modified Images...')
        output_layout.addWidget(self.save_modified_btn)

        self.save_in_place_btn = QPushButton('Save Modified In Place')
        output_layout.addWidget(self.save_in_place_btn)

        self.copy_caption_btn = QPushButton('Copy to Caption')
        output_layout.addWidget(self.copy_caption_btn)

        right_layout.addWidget(output_group)
        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 600, 300])
        layout.addWidget(splitter)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key.Key_A, Qt.Key.Key_PageUp):
            self.image_list.go_to_previous()
        elif key in (Qt.Key.Key_D, Qt.Key.Key_PageDown):
            self.image_list.go_to_next()
        else:
            super().keyPressEvent(event)

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
        self.save_in_place_btn.clicked.connect(self._save_modified_in_place)
        self.copy_caption_btn.clicked.connect(self._copy_to_caption)

        self.restore_current_btn.clicked.connect(self._restore_current)
        self.restore_all_btn.clicked.connect(self._restore_all)

        # Tools
        self.crop_toggle.toggled.connect(self._on_crop_toggled)
        self.apply_crop_btn.clicked.connect(self._apply_crop)

        self.split_toggle.toggled.connect(self._on_split_toggled)
        self.split_orientation_group.buttonToggled.connect(
            self._on_split_orientation_changed)
        self.split_slider.valueChanged.connect(self._on_split_slider_changed)
        self.split_spinbox.valueChanged.connect(self._on_split_spinbox_changed)
        self.apply_split_btn.clicked.connect(self._apply_split)

        self.comparison.split_pos_changed.connect(self._on_split_pos_from_view)

    def load_from_label_tab(self, items: list[ImageItem],
                           colors: list[str] | None = None):
        """Load images with masks from the label tab."""
        self._pixmap_cache.clear()
        if colors:
            self._class_colors = colors
        self.model.load_items(items)
        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _load_directory(self, path: str):
        """Load a directory, detecting -masklabel pairs."""
        directory = Path(path)

        with loading_dialog('Loading images...', self):
            self._pixmap_cache.clear()
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

        # Snapshot max_dim once so both pixmaps use the same scale
        max_dim = self._preview_max_dim()

        # Before: original with mask overlay
        before_pixmap = self._create_before_pixmap(image, max_dim)

        # After: modified image (if exists) or original
        after_pixmap = None
        if image.modified_path and image.modified_path.exists():
            after_pixmap = self._load_cached(image.modified_path, max_dim)
        if after_pixmap is None:
            after_pixmap = self._load_cached(image.path, max_dim)

        if before_pixmap and after_pixmap:
            self.comparison.set_images(before_pixmap, after_pixmap)
        elif before_pixmap:
            self.comparison.set_before(before_pixmap)
        elif after_pixmap:
            self.comparison.set_after(after_pixmap)

        QTimer.singleShot(0, lambda idx=index: self._preload_adjacent(idx))

    def _preview_max_dim(self) -> int:
        vp = self.comparison.viewport().size()
        return max(vp.width(), vp.height(), 800) * 2

    def _load_cached(self, path, max_dim: int | None = None):
        key = str(path)
        if key in self._pixmap_cache:
            self._pixmap_cache.move_to_end(key)
            return self._pixmap_cache[key]
        if max_dim is None:
            max_dim = self._preview_max_dim()
        pixmap = load_pixmap_preview(path, max_dim)
        self._pixmap_cache[key] = pixmap
        while len(self._pixmap_cache) > self._PIXMAP_CACHE_MAX:
            self._pixmap_cache.popitem(last=False)
        return pixmap

    def _preload_adjacent(self, index: int):
        if index != self._current_image_index:
            return
        for offset in (-1, 1):
            image = self.model.get_image(index + offset)
            if image and str(image.path) not in self._pixmap_cache:
                self._load_cached(image.path)

    def _create_before_pixmap(self, image: ImageItem,
                              max_dim: int | None = None) -> QPixmap | None:
        """Create before image with label overlays (matching Label tab look)."""
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import (QBrush, QColor, QPainter, QPen, QPolygonF)
        pixmap = self._load_cached(image.path, max_dim)
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
        # Invalidate cache for this output so we load the fresh result
        self._pixmap_cache.pop(output_path, None)

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

    # --- Tools: Crop ---

    def _on_crop_toggled(self, checked: bool):
        if checked and self.split_toggle.isChecked():
            self.split_toggle.setChecked(False)
        self.apply_crop_btn.setEnabled(checked)
        self.comparison.set_crop_mode(checked)

    def _apply_crop(self):
        from PIL import Image
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return

        x, y, w, h = self.comparison.get_crop_rect()
        if w <= 0 or h <= 0:
            QMessageBox.information(self, 'Info', 'Invalid crop area.')
            return

        # Need to scale crop rect from preview coords to actual image coords
        source_path = image.modified_path if image.modified_path else image.path
        pil_img = Image.open(str(source_path))
        actual_w, actual_h = pil_img.size

        # The preview pixmap may be scaled - get the scale factor
        max_dim = self._preview_max_dim()
        preview_scale = min(max_dim / actual_w, max_dim / actual_h, 1.0)
        # Scale crop coords back to actual image coords
        ax = int(x / preview_scale)
        ay = int(y / preview_scale)
        aw = int(w / preview_scale)
        ah = int(h / preview_scale)
        # Clamp
        ax = max(0, min(ax, actual_w))
        ay = max(0, min(ay, actual_h))
        aw = min(aw, actual_w - ax)
        ah = min(ah, actual_h - ay)

        if aw <= 0 or ah <= 0:
            QMessageBox.information(self, 'Info', 'Crop area too small.')
            return

        temp_dir = get_temp_dir_no_clear('crop')
        cropped = pil_img.crop((ax, ay, ax + aw, ay + ah))
        out_path = temp_dir / f'{image.name}_cropped{image.suffix}'
        cropped.save(str(out_path))
        pil_img.close()

        # Crop mask if exists
        if image.mask_path and image.mask_path.exists():
            mask_img = Image.open(str(image.mask_path))
            # Scale mask to match actual image if needed
            if mask_img.size != (actual_w, actual_h):
                mask_img = mask_img.resize((actual_w, actual_h), Image.NEAREST)
            mask_cropped = mask_img.crop((ax, ay, ax + aw, ay + ah))
            mask_out = temp_dir / f'{image.name}_cropped-masklabel.png'
            mask_cropped.save(str(mask_out))
            image.mask_path = mask_out
            mask_img.close()

        # Update image
        image.modified_path = out_path
        image.width = aw
        image.height = ah
        self._pixmap_cache.pop(str(source_path), None)
        self._pixmap_cache.pop(str(out_path), None)
        self.model.invalidate_thumbnail(self._current_image_index)

        # Disable crop mode
        self.crop_toggle.setChecked(False)
        self._on_image_changed(self._current_image_index)
        self.mod_status.setText(f'Cropped to {aw}x{ah}')

    # --- Tools: Split ---

    def _on_split_toggled(self, checked: bool):
        if checked and self.crop_toggle.isChecked():
            self.crop_toggle.setChecked(False)
        self.apply_split_btn.setEnabled(checked)
        orientation = 'H' if self.split_h_radio.isChecked() else 'V'
        self.comparison.set_split_mode(checked, orientation)

    def _on_split_orientation_changed(self):
        if self.split_toggle.isChecked():
            orientation = 'H' if self.split_h_radio.isChecked() else 'V'
            self.comparison.set_split_mode(True, orientation)

    def _on_split_slider_changed(self, value: int):
        self.split_spinbox.blockSignals(True)
        self.split_spinbox.setValue(value // 10)
        self.split_spinbox.blockSignals(False)
        if self.split_toggle.isChecked():
            self.comparison.set_split_pos(value / 1000.0)

    def _on_split_spinbox_changed(self, value: int):
        self.split_slider.blockSignals(True)
        self.split_slider.setValue(value * 10)
        self.split_slider.blockSignals(False)
        if self.split_toggle.isChecked():
            self.comparison.set_split_pos(value / 100.0)

    def _on_split_pos_from_view(self, pos: float):
        """Sync UI controls when split line is dragged in the view."""
        self.split_slider.blockSignals(True)
        self.split_spinbox.blockSignals(True)
        self.split_slider.setValue(int(pos * 1000))
        self.split_spinbox.setValue(int(pos * 100))
        self.split_slider.blockSignals(False)
        self.split_spinbox.blockSignals(False)

    def _apply_split(self):
        from PIL import Image
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return

        split_pos = self.comparison.get_split_pos()
        orientation = 'H' if self.split_h_radio.isChecked() else 'V'

        source_path = image.modified_path if image.modified_path else image.path
        pil_img = Image.open(str(source_path))
        actual_w, actual_h = pil_img.size

        temp_dir = get_temp_dir_no_clear('split')
        stem = image.name
        ext = image.suffix

        if orientation == 'V':
            split_x = int(split_pos * actual_w)
            split_x = max(1, min(split_x, actual_w - 1))
            part1 = pil_img.crop((0, 0, split_x, actual_h))
            part2 = pil_img.crop((split_x, 0, actual_w, actual_h))
            sizes = [(split_x, actual_h), (actual_w - split_x, actual_h)]
        else:
            split_y = int(split_pos * actual_h)
            split_y = max(1, min(split_y, actual_h - 1))
            part1 = pil_img.crop((0, 0, actual_w, split_y))
            part2 = pil_img.crop((0, split_y, actual_w, actual_h))
            sizes = [(actual_w, split_y), (actual_w, actual_h - split_y)]

        path1 = temp_dir / f'{stem}_part1{ext}'
        path2 = temp_dir / f'{stem}_part2{ext}'
        part1.save(str(path1))
        part2.save(str(path2))
        pil_img.close()

        # Split mask if exists
        mask_path1 = None
        mask_path2 = None
        if image.mask_path and image.mask_path.exists():
            mask_img = Image.open(str(image.mask_path))
            if mask_img.size != (actual_w, actual_h):
                mask_img = mask_img.resize((actual_w, actual_h), Image.NEAREST)
            if orientation == 'V':
                m1 = mask_img.crop((0, 0, split_x, actual_h))
                m2 = mask_img.crop((split_x, 0, actual_w, actual_h))
            else:
                m1 = mask_img.crop((0, 0, actual_w, split_y))
                m2 = mask_img.crop((0, split_y, actual_w, actual_h))
            mask_path1 = temp_dir / f'{stem}_part1-masklabel.png'
            mask_path2 = temp_dir / f'{stem}_part2-masklabel.png'
            m1.save(str(mask_path1))
            m2.save(str(mask_path2))
            mask_img.close()

        # Create new ImageItems
        item1 = ImageItem(path=path1, width=sizes[0][0], height=sizes[0][1],
                          mask_path=mask_path1)
        item2 = ImageItem(path=path2, width=sizes[1][0], height=sizes[1][1],
                          mask_path=mask_path2)

        # Insert after current
        insert_pos = self._current_image_index + 1
        self.model.insert_items(insert_pos, [item1, item2])

        # Disable split mode and select first new part
        self.split_toggle.setChecked(False)
        self.image_list.select_index(insert_pos)
        self.mod_status.setText(
            f'Split into {sizes[0][0]}x{sizes[0][1]} + '
            f'{sizes[1][0]}x{sizes[1][1]}')

    # --- Restore ---

    def _restore_current(self):
        image = self.model.get_image(self._current_image_index)
        if image is None or not image.modified_path:
            QMessageBox.information(self, 'Info', 'Current image has no modifications.')
            return
        self._pixmap_cache.pop(str(image.modified_path), None)
        self._pixmap_cache.pop(str(image.path), None)
        image.modified_path = None
        self.model.invalidate_thumbnail(self._current_image_index)
        self._on_image_changed(self._current_image_index)
        self.mod_status.setText('Restored original image')

    def _restore_all(self):
        modified = [img for img in self.model.images if img.modified_path]
        if not modified:
            QMessageBox.information(self, 'Info', 'No modified images to restore.')
            return
        reply = QMessageBox.question(
            self, 'Restore All',
            f'Discard modifications for {len(modified)} image(s)?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        for image in modified:
            self._pixmap_cache.pop(str(image.modified_path), None)
            self._pixmap_cache.pop(str(image.path), None)
            image.modified_path = None
            idx = self.model.images.index(image)
            self.model.invalidate_thumbnail(idx)
        if self._current_image_index >= 0:
            self._on_image_changed(self._current_image_index)
        self.mod_status.setText(f'Restored {len(modified)} original images')

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

    def _save_modified_in_place(self):
        """Overwrite original files with their modified versions."""
        modified = [img for img in self.model.images if img.modified_path]
        if not modified:
            QMessageBox.information(self, 'Info', 'No modified images to save.')
            return

        reply = QMessageBox.warning(
            self, 'Save Modified In Place',
            f'This will overwrite {len(modified)} original image(s) with '
            f'their modified versions. This cannot be undone.\n\n'
            f'Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        count = 0
        for image in modified:
            try:
                shutil.copy2(str(image.modified_path), str(image.path))
                # Invalidate caches
                self._pixmap_cache.pop(str(image.path), None)
                self._pixmap_cache.pop(str(image.modified_path), None)
                image.modified_path = None
                idx = self.model.images.index(image)
                self.model.invalidate_thumbnail(idx)
                count += 1
            except Exception as e:
                QMessageBox.warning(
                    self, 'Error',
                    f'Failed to save {image.filename}: {e}')

        # Refresh current view
        if self._current_image_index >= 0:
            self._on_image_changed(self._current_image_index)
        self.mod_status.setText(f'Saved {count} modified images in place')

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
