"""Label tab: image labeling with canvas tools, detection, and output."""
import copy
import random
import shutil
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                               QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox, QSplitter,
                               QToolButton, QVBoxLayout, QWidget,
                               QButtonGroup, QLineEdit)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel
from ltd.data.label_data import (DEFAULT_COLORS, DetectionType, Label,
                                  LabelClass)
from ltd.settings import get_settings
from ltd.utils.file_utils import get_temp_dir, get_temp_dir_no_clear
from ltd.utils.image_utils import load_pixmap_preview
from ltd.utils.mask_utils import (bbox_to_mask, create_empty_mask,
                                   label_to_mask, mask_from_qimage,
                                   mask_to_polygons, masks_overlap,
                                   merge_masks, polygon_to_mask, save_mask)
from ltd.utils.yolo_format import read_yolo_labels, write_yolo_labels
from ltd.widgets.canvas_widget import CanvasWidget, DrawMode, Tool
from ltd.widgets.label_image_list import LabelImageList
from ltd.widgets.loading_dialog import loading_dialog
from ltd.widgets.module_selector import ModuleSelector
from ltd.workers.detection_worker import DetectionWorker

from modules.base import BaseDetectionModule
from modules import discover_modules


class LabelTab(QWidget):
    copy_to_modify_requested = Signal(list, list)  # (list[ImageItem], colors)
    copy_to_train_requested = Signal(str, str)  # (dataset path, model_type)
    dataset_saved = Signal(str)  # dataset directory path

    _PIXMAP_CACHE_MAX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self.classes: list[LabelClass] = []
        self._current_image_index = -1
        self._worker: DetectionWorker | None = None
        self._labels_dir: Path | None = None
        self._pixmap_cache: OrderedDict[str, object] = OrderedDict()

        # Undo: per-image stack of label snapshots
        # key = image index, value = list of label list snapshots
        self._undo_stacks: dict[int, list[list[Label]]] = {}
        self._max_undo = 50

        # Discover detection modules
        self._detection_modules = discover_modules(
            BaseDetectionModule, 'modules/detection')

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()
        self._restore_classes()
        self._install_space_filter()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left: Image list ---
        self.image_list = LabelImageList(self.model)
        splitter.addWidget(self.image_list)

        # --- Center: Canvas ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # Tool buttons
        tool_bar = QHBoxLayout()
        self.tool_buttons = {}
        tools = [
            ('Hand (M)', Tool.HAND), ('Pointer (P)', Tool.POINTER),
            ('BBox (R)', Tool.BBOX), ('Polygon (V)', Tool.POLYGON),
            ('Brush (B)', Tool.MARKER),
        ]
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for text, tool in tools:
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setProperty('tool', tool)
            self.tool_group.addButton(btn)
            self.tool_buttons[tool] = btn
            tool_bar.addWidget(btn)
        self.tool_buttons[Tool.HAND].setChecked(True)

        tool_bar.addWidget(self._make_separator())

        # Draw mode buttons (New / Combine / Erase)
        self.mode_buttons: dict[DrawMode, QToolButton] = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        modes = [
            ('New (1)', DrawMode.NEW),
            ('Combine (2)', DrawMode.COMBINE),
            ('Erase (3)', DrawMode.ERASE),
        ]
        for text, mode in modes:
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setProperty('draw_mode', mode)
            self.mode_group.addButton(btn)
            self.mode_buttons[mode] = btn
            tool_bar.addWidget(btn)
        self.mode_buttons[DrawMode.NEW].setChecked(True)

        tool_bar.addWidget(self._make_separator())

        # Brush size
        tool_bar.addWidget(QLabel('Brush:'))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(20)
        tool_bar.addWidget(self.brush_spin)

        tool_bar.addStretch()
        center_layout.addLayout(tool_bar)

        self.canvas = CanvasWidget()
        center_layout.addWidget(self.canvas, stretch=1)
        splitter.addWidget(center_widget)

        # --- Right: Classes + Labels + Detection + Output ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Classes group
        classes_group = QGroupBox('Classes')
        classes_layout = QVBoxLayout(classes_group)

        add_class_layout = QHBoxLayout()
        self.class_name_input = QLineEdit()
        self.class_name_input.setPlaceholderText('Class name')
        add_class_layout.addWidget(self.class_name_input)
        self.detection_type_combo = QComboBox()
        self.detection_type_combo.addItems(['BBox', 'Polygon'])
        add_class_layout.addWidget(self.detection_type_combo)
        self.add_class_btn = QPushButton('+')
        self.add_class_btn.setFixedWidth(30)
        add_class_layout.addWidget(self.add_class_btn)
        classes_layout.addLayout(add_class_layout)

        self.classes_list = QListWidget()
        self.classes_list.setMaximumHeight(120)
        classes_layout.addWidget(self.classes_list)

        self.remove_class_btn = QPushButton('Remove Class')
        classes_layout.addWidget(self.remove_class_btn)
        right_layout.addWidget(classes_group)

        # Labels list for current image
        labels_group = QGroupBox('Labels (current image)')
        labels_layout = QVBoxLayout(labels_group)
        self.labels_list = QListWidget()
        labels_layout.addWidget(self.labels_list)

        labels_btn_layout = QHBoxLayout()
        self.delete_label_btn = QPushButton('Delete Label')
        labels_btn_layout.addWidget(self.delete_label_btn)
        self.clear_labels_btn = QPushButton('Clear All')
        labels_btn_layout.addWidget(self.clear_labels_btn)
        labels_layout.addLayout(labels_btn_layout)
        right_layout.addWidget(labels_group)

        # Detection group
        detection_group = QGroupBox('Auto-Detection')
        detection_layout = QVBoxLayout(detection_group)

        if self._detection_modules:
            self.module_selector = ModuleSelector(
                'Detection module:', self._detection_modules,
                settings_key='detection')
            detection_layout.addWidget(self.module_selector)
        else:
            detection_layout.addWidget(QLabel('No detection modules found'))
            self.module_selector = None

        detect_btn_layout = QHBoxLayout()
        self.auto_label_current_btn = QPushButton('Auto-Label Current')
        detect_btn_layout.addWidget(self.auto_label_current_btn)
        self.auto_label_btn = QPushButton('Auto-Label All')
        detect_btn_layout.addWidget(self.auto_label_btn)
        self.auto_label_unlabeled_btn = QPushButton('Auto-Label Unlabeled')
        detect_btn_layout.addWidget(self.auto_label_unlabeled_btn)
        detection_layout.addLayout(detect_btn_layout)

        self.detection_progress = QProgressBar()
        self.detection_progress.setVisible(False)
        detection_layout.addWidget(self.detection_progress)

        self.detection_status = QLabel('')
        detection_layout.addWidget(self.detection_status)

        self.cancel_detection_btn = QPushButton('Cancel')
        self.cancel_detection_btn.setVisible(False)
        detection_layout.addWidget(self.cancel_detection_btn)

        right_layout.addWidget(detection_group)

        # Output group
        output_group = QGroupBox('Output')
        output_layout = QVBoxLayout(output_group)

        self.include_unlabeled_cb = QCheckBox('Include unlabeled images')
        self.include_unlabeled_cb.setChecked(
            get_settings().value('label_tab/include_unlabeled', False, type=bool))
        self.include_unlabeled_cb.toggled.connect(
            lambda v: get_settings().setValue('label_tab/include_unlabeled', v))
        output_layout.addWidget(self.include_unlabeled_cb)

        self.save_masks_btn = QPushButton('Save Masks...')
        output_layout.addWidget(self.save_masks_btn)

        self.copy_modify_btn = QPushButton('Proceed to Modify')
        output_layout.addWidget(self.copy_modify_btn)

        self.save_dataset_btn = QPushButton('Save as YOLO Dataset...')
        output_layout.addWidget(self.save_dataset_btn)

        self.copy_train_btn = QPushButton('Proceed to Train')
        output_layout.addWidget(self.copy_train_btn)

        right_layout.addWidget(output_group)
        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 600, 300])
        layout.addWidget(splitter)

    @staticmethod
    def _make_separator():
        sep = QLabel('|')
        sep.setStyleSheet('color: gray; margin: 0 2px;')
        return sep

    def _setup_shortcuts(self):
        shortcuts = {
            'M': Tool.HAND, 'P': Tool.POINTER, 'R': Tool.BBOX,
            'V': Tool.POLYGON, 'B': Tool.MARKER,
        }
        for key, tool in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda t=tool: self._set_tool(t))

        # Draw mode shortcuts: 1=New, 2=Combine, 3=Erase
        mode_shortcuts = {
            '1': DrawMode.NEW, '2': DrawMode.COMBINE, '3': DrawMode.ERASE,
        }
        for key, mode in mode_shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda m=mode: self._set_draw_mode(m))

        QShortcut(QKeySequence('A'), self).activated.connect(
            self.image_list.go_to_previous)
        QShortcut(QKeySequence('D'), self).activated.connect(
            self.image_list.go_to_next)
        QShortcut(QKeySequence('PgUp'), self).activated.connect(
            self.image_list.go_to_previous)
        QShortcut(QKeySequence('PgDown'), self).activated.connect(
            self.image_list.go_to_next)
        QShortcut(QKeySequence('W'), self).activated.connect(
            self._next_class)
        QShortcut(QKeySequence('S'), self).activated.connect(
            self._prev_class)
        QShortcut(QKeySequence('Delete'), self).activated.connect(
            self._delete_selected_label)
        QShortcut(QKeySequence('Ctrl+Z'), self).activated.connect(
            self._undo)

    def _install_space_filter(self):
        """Install event filter on all child widgets to capture spacebar."""
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
                if event.type() == QEvent.Type.KeyPress:
                    self.canvas.keyPressEvent(event)
                else:
                    self.canvas.keyReleaseEvent(event)
                return True
        return super().eventFilter(obj, event)

    def _connect_signals(self):
        self.image_list.current_changed.connect(self._on_image_changed)
        self.image_list.load_directory_requested.connect(self._load_directory)

        self.tool_group.buttonClicked.connect(self._on_tool_button)
        self.mode_group.buttonClicked.connect(self._on_mode_button)
        self.brush_spin.valueChanged.connect(
            lambda v: setattr(self.canvas, 'brush_size', v))
        # Sync brush_spin when canvas changes brush size (scroll wheel)
        self.canvas.brush_size_changed.connect(self.brush_spin.setValue)

        self.add_class_btn.clicked.connect(self._add_class)
        self.class_name_input.returnPressed.connect(self._add_class)
        self.remove_class_btn.clicked.connect(self._remove_class)
        self.classes_list.currentRowChanged.connect(self._on_class_selected)

        self.canvas.label_created.connect(self._on_label_created)
        self.canvas.label_selected.connect(self._on_label_selected)
        self.canvas.label_modified.connect(self._on_label_modified)
        self.canvas.mask_updated.connect(self._on_mask_updated)

        self.delete_label_btn.clicked.connect(self._delete_selected_label)
        self.clear_labels_btn.clicked.connect(self._clear_all_labels)
        self.labels_list.currentRowChanged.connect(self._on_label_list_selected)

        self.auto_label_current_btn.clicked.connect(self._run_auto_detection_current)
        self.auto_label_btn.clicked.connect(self._run_auto_detection)
        self.auto_label_unlabeled_btn.clicked.connect(self._run_auto_detection_unlabeled)
        self.cancel_detection_btn.clicked.connect(self._cancel_detection)

        self.save_masks_btn.clicked.connect(self._save_masks)
        self.copy_modify_btn.clicked.connect(self._copy_to_modify)
        self.save_dataset_btn.clicked.connect(self._save_dataset)
        self.copy_train_btn.clicked.connect(self._copy_to_train)

    # --- Class persistence ---

    def _save_classes(self):
        settings = get_settings()
        data = []
        for c in self.classes:
            data.append(f'{c.name}|{c.color}|{c.detection_type.value}')
        settings.setValue('label_classes', data)

    def _restore_classes(self):
        settings = get_settings()
        data = settings.value('label_classes', [], type=list)
        for i, entry in enumerate(data):
            parts = entry.split('|')
            if len(parts) == 3:
                name, color, det_str = parts
                det_type = (DetectionType.POLYGON if det_str == 'polygon'
                            else DetectionType.BBOX)
                lc = LabelClass(name=name, color=color,
                                detection_type=det_type, class_id=i)
                self.classes.append(lc)
                item = QListWidgetItem(
                    f'{i}: {name} ({det_type.value})')
                item.setForeground(QColor(color))
                self.classes_list.addItem(item)
        if self.classes:
            self.classes_list.setCurrentRow(0)
        self.image_list.update_class_names(self.classes)

    def _load_directory(self, path: str):
        import hashlib
        directory = Path(path)

        with loading_dialog('Loading images...', self) as dlg:
            self._pixmap_cache.clear()
            self.model.load_directory(directory)

            # Stable per-directory temp labels dir (survives app restarts)
            dir_hash = hashlib.md5(
                str(directory.resolve()).encode()).hexdigest()[:12]
            self._labels_dir = get_temp_dir_no_clear(f'labels_{dir_hash}')

            dlg.set_message('Loading labels...')
            QApplication.processEvents()
            for image in self.model.images:
                # Prefer temp labels (previous session), fall back to in-place
                temp_txt = self._label_path_for(image)
                inplace_txt = image.path.with_suffix('.txt')
                if temp_txt.exists():
                    image.labels = read_yolo_labels(temp_txt)
                elif inplace_txt.exists():
                    image.labels = read_yolo_labels(inplace_txt)

            self._undo_stacks.clear()

        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _on_image_changed(self, index: int):
        self._current_image_index = index
        image = self.model.get_image(index)
        if image is None:
            return

        pixmap = self._load_cached(image.path)
        if pixmap:
            self.canvas.load_image(pixmap)
            colors = self._get_colors()
            self.canvas.display_labels(image.labels, colors)
            self._refresh_labels_list()
            self.canvas.setFocus()

        QTimer.singleShot(0, lambda idx=index: self._preload_adjacent(idx))

    def _preview_max_dim(self) -> int:
        vp = self.canvas.viewport().size()
        return max(vp.width(), vp.height(), 800) * 2

    def _load_cached(self, path):
        key = str(path)
        if key in self._pixmap_cache:
            self._pixmap_cache.move_to_end(key)
            return self._pixmap_cache[key]
        pixmap = load_pixmap_preview(path, self._preview_max_dim())
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

    def _get_colors(self) -> list[str]:
        return [c.color for c in self.classes] if self.classes else list(DEFAULT_COLORS)

    def _label_path_for(self, image) -> Path:
        """Return the label .txt path inside the temp labels dir."""
        return self._labels_dir / f'{image.name}.txt'

    def _save_current_labels(self):
        if self._current_image_index < 0:
            return
        image = self.model.get_image(self._current_image_index)
        if image is None or self._labels_dir is None:
            return
        write_yolo_labels(self._label_path_for(image), image.labels)
        self.image_list.reapply_filter()

    def _push_undo(self):
        """Snapshot current image's labels before a mutation."""
        idx = self._current_image_index
        image = self.model.get_image(idx)
        if image is None:
            return
        stack = self._undo_stacks.setdefault(idx, [])
        stack.append(copy.deepcopy(image.labels))
        if len(stack) > self._max_undo:
            stack.pop(0)

    def _undo(self):
        """Restore the previous label snapshot for the current image."""
        idx = self._current_image_index
        stack = self._undo_stacks.get(idx)
        if not stack:
            return
        image = self.model.get_image(idx)
        if image is None:
            return
        image.labels = stack.pop()
        colors = self._get_colors()
        self.canvas.display_labels(image.labels, colors)
        self._refresh_labels_list()
        self._save_current_labels()

    def _refresh_labels_list(self):
        self.labels_list.clear()
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        colors = self._get_colors()
        for i, label in enumerate(image.labels):
            class_name = self._class_name(label.class_id)
            if label.has_polygon:
                kind = f'polygon ({len(label.polygon)} pts)'
            elif label.has_bbox:
                cx, cy, w, h = label.bbox
                kind = f'bbox ({w:.2f}x{h:.2f})'
            else:
                kind = 'empty'
            text = f'{i}: [{label.class_id}] {class_name} - {kind}'
            item = QListWidgetItem(text)
            color = colors[label.class_id % len(colors)]
            item.setForeground(QColor(color))
            self.labels_list.addItem(item)

    def _class_name(self, class_id: int) -> str:
        for c in self.classes:
            if c.class_id == class_id:
                return c.name
        return f'class_{class_id}'

    def _set_tool(self, tool: Tool):
        self.canvas.current_tool = tool
        btn = self.tool_buttons.get(tool)
        if btn:
            btn.setChecked(True)

    def _on_tool_button(self, btn):
        tool = btn.property('tool')
        if tool:
            self.canvas.current_tool = tool

    def _set_draw_mode(self, mode: DrawMode):
        self.canvas.draw_mode = mode
        btn = self.mode_buttons.get(mode)
        if btn:
            btn.setChecked(True)

    def _on_mode_button(self, btn):
        mode = btn.property('draw_mode')
        if mode:
            self.canvas.draw_mode = mode

    # --- Class management ---

    def _add_class(self):
        name = self.class_name_input.text().strip()
        if not name:
            return
        det_type = (DetectionType.POLYGON
                    if self.detection_type_combo.currentIndex() == 1
                    else DetectionType.BBOX)
        class_id = len(self.classes)
        color = DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]
        lc = LabelClass(name=name, color=color, detection_type=det_type,
                        class_id=class_id)
        self.classes.append(lc)
        item = QListWidgetItem(f'{class_id}: {name} ({det_type.value})')
        item.setForeground(QColor(color))
        self.classes_list.addItem(item)
        self.class_name_input.clear()
        self.classes_list.setCurrentRow(class_id)
        self._save_classes()
        self.image_list.update_class_names(self.classes)

    def _remove_class(self):
        row = self.classes_list.currentRow()
        if row < 0 or row >= len(self.classes):
            return
        removed_id = self.classes[row].class_id
        self.classes.pop(row)
        self.classes_list.takeItem(row)

        self.classes_list.clear()
        for i, c in enumerate(self.classes):
            c.class_id = i
            item = QListWidgetItem(
                f'{i}: {c.name} ({c.detection_type.value})')
            item.setForeground(QColor(c.color))
            self.classes_list.addItem(item)

        for image in self.model.images:
            updated = []
            for label in image.labels:
                if label.class_id == removed_id:
                    continue
                if label.class_id > removed_id:
                    label.class_id -= 1
                updated.append(label)
            image.labels = updated
            write_yolo_labels(self._label_path_for(image), image.labels)

        self._refresh_labels_list()
        colors = self._get_colors()
        image = self.model.get_image(self._current_image_index)
        if image:
            self.canvas.display_labels(image.labels, colors)

        self._save_classes()
        self.image_list.update_class_names(self.classes)

    def _on_class_selected(self, row: int):
        if 0 <= row < len(self.classes):
            self.canvas.set_class_id(row)

    def _next_class(self):
        row = self.classes_list.currentRow()
        if row < len(self.classes) - 1:
            self.classes_list.setCurrentRow(row + 1)

    def _prev_class(self):
        row = self.classes_list.currentRow()
        if row > 0:
            self.classes_list.setCurrentRow(row - 1)

    # --- Label operations ---

    def _on_label_created(self, label: Label):
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        self._push_undo()
        mode = self.canvas.draw_mode
        if mode == DrawMode.NEW:
            image.labels.append(label)
        elif mode in (DrawMode.COMBINE, DrawMode.ERASE):
            self._apply_label_with_mode(image, label, mode)
        colors = self._get_colors()
        self.canvas.display_labels(image.labels, colors)
        self._refresh_labels_list()
        self._save_current_labels()

    def _on_label_selected(self, index: int):
        if 0 <= index < self.labels_list.count():
            if self.labels_list.currentRow() == index:
                # Same row — setCurrentRow won't fire currentRowChanged,
                # so apply highlight directly.
                self.canvas.highlight_label(index)
            else:
                self.labels_list.setCurrentRow(index)
        else:
            self.canvas.highlight_label(-1)

    def _on_label_modified(self, index: int, label):
        """Handle label modification from canvas drag."""
        image = self.model.get_image(self._current_image_index)
        if image is None or index < 0 or index >= len(image.labels):
            return
        self._push_undo()
        image.labels[index] = label
        self._refresh_labels_list()
        self._save_current_labels()
        # Re-select the same label in the list
        if 0 <= index < self.labels_list.count():
            self.labels_list.setCurrentRow(index)

    def _on_label_list_selected(self, row: int):
        self.canvas.highlight_label(row)

    def _on_mask_updated(self, draw_mode=None):
        """Convert drawn mask to polygon label, then apply using draw mode."""
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        mask_qimage = self.canvas.get_mask_image()
        if mask_qimage is None:
            return

        drawn_mask = mask_from_qimage(mask_qimage)

        # Check if anything was actually drawn
        if drawn_mask.max() < 128:
            self.canvas.clear_mask()
            return

        h, w = drawn_mask.shape
        current_class = self.canvas._current_class_id

        # Convert drawn mask to polygon label
        new_polygons = mask_to_polygons(drawn_mask)
        if not new_polygons:
            self.canvas.clear_mask()
            return

        # Build a label from the drawn mask
        poly_pts = new_polygons[0]
        # If multiple disconnected regions, merge them all
        all_pts = []
        for pts in new_polygons:
            all_pts.extend(pts)

        if draw_mode is None:
            draw_mode = self.canvas.draw_mode

        self._push_undo()

        if draw_mode == DrawMode.NEW:
            # Each polygon region becomes its own label
            for poly_pts in new_polygons:
                xs = [p[0] for p in poly_pts]
                ys = [p[1] for p in poly_pts]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                label = Label(class_id=current_class,
                              bbox=(cx, cy, bw, bh), polygon=poly_pts)
                image.labels.append(label)
        else:
            # Combine or Erase: work with overlapping same-class labels
            self._apply_brush_with_mode(image, drawn_mask, current_class,
                                        w, h, draw_mode)

        # Clear mask buffer, update display
        self.canvas.clear_mask()
        colors = self._get_colors()
        self.canvas.display_labels(image.labels, colors)
        self._refresh_labels_list()
        self._save_current_labels()

    def _apply_brush_with_mode(self, image, drawn_mask, current_class,
                                w, h, mode: DrawMode):
        """Apply brush drawn_mask using Combine or Erase mode."""
        import cv2

        # Find existing same-class labels that overlap with the drawn mask
        overlapping_indices = []
        for i, label in enumerate(image.labels):
            if label.class_id != current_class:
                continue
            lmask = label_to_mask(label, w, h)
            if lmask is not None and masks_overlap(drawn_mask, lmask):
                overlapping_indices.append(i)

        if mode == DrawMode.ERASE:
            if not overlapping_indices:
                return
            overlap_masks = []
            for i in overlapping_indices:
                lmask = label_to_mask(image.labels[i], w, h)
                if lmask is not None:
                    overlap_masks.append(lmask)
            if not overlap_masks:
                return
            combined = merge_masks(overlap_masks)
            result = cv2.bitwise_and(combined, cv2.bitwise_not(drawn_mask))
        else:
            # Combine mode
            combined = drawn_mask.copy()
            for i in overlapping_indices:
                lmask = label_to_mask(image.labels[i], w, h)
                if lmask is not None:
                    combined = cv2.bitwise_or(combined, lmask)
            result = combined

        # Convert result mask to polygon(s)
        new_polygons = mask_to_polygons(result)

        # Remove overlapping labels (reverse order to preserve indices)
        for i in sorted(overlapping_indices, reverse=True):
            image.labels.pop(i)

        # Add new polygon labels
        for poly_pts in new_polygons:
            xs = [p[0] for p in poly_pts]
            ys = [p[1] for p in poly_pts]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            label = Label(class_id=current_class,
                          bbox=(cx, cy, bw, bh), polygon=poly_pts)
            image.labels.append(label)

    def _apply_label_with_mode(self, image, label: Label, mode: DrawMode):
        """Apply a polygon label using Combine or Erase mode (for polygon tool)."""
        if not label.has_polygon:
            # BBox-only labels can't combine/erase, just add
            image.labels.append(label)
            return

        h, w = self.canvas._image_height, self.canvas._image_width
        if h == 0 or w == 0:
            return

        drawn_mask = polygon_to_mask(label.polygon, w, h)
        if drawn_mask is None:
            return

        current_class = label.class_id
        self._apply_brush_with_mode(image, drawn_mask, current_class,
                                     w, h, mode)

    def _delete_selected_label(self):
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        idx = self.labels_list.currentRow()
        if idx < 0:
            idx = self.canvas._selected_label_index
        if 0 <= idx < len(image.labels):
            self._push_undo()
            image.labels.pop(idx)
            colors = self._get_colors()
            self.canvas.display_labels(image.labels, colors)
            self._refresh_labels_list()
            self._save_current_labels()

    def _clear_all_labels(self):
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        self._push_undo()
        image.labels.clear()
        colors = self._get_colors()
        self.canvas.display_labels(image.labels, colors)
        self._refresh_labels_list()
        self._save_current_labels()

    # --- Auto detection ---

    def _get_class_map(self, module):
        """Build class name → our class ID mapping for a detection module."""
        module_classes = module.get_class_names()
        class_map = {}
        if module_classes:
            for i, name in enumerate(module_classes):
                for lc in self.classes:
                    if lc.name.lower() == name.lower():
                        class_map[name] = lc.class_id
                        break
        return class_map

    def _run_auto_detection_current(self):
        """Run detection on the current image only, appending results."""
        if not self.classes:
            QMessageBox.warning(self, 'Warning',
                                'Add at least one class before auto-labeling.')
            return
        if self.module_selector is None:
            return
        module = self.module_selector.current_module()
        if module is None:
            return
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return

        self.detection_status.setText(f'Detecting: {image.filename}...')
        self.auto_label_current_btn.setEnabled(False)
        self.auto_label_btn.setEnabled(False)
        self.auto_label_unlabeled_btn.setEnabled(False)

        try:
            results = module.run(image.path)
            class_map = self._get_class_map(module)
            new_labels = []
            for det in results:
                class_id = det['class_id']
                if class_map:
                    class_name = det.get('class_name', '')
                    if class_name in class_map:
                        class_id = class_map[class_name]
                label = Label(class_id=class_id,
                              bbox=det.get('bbox'),
                              polygon=det.get('polygon'))
                new_labels.append(label)

            # Append, not replace
            image.labels.extend(new_labels)
            self._save_current_labels()
            colors = self._get_colors()
            self.canvas.display_labels(image.labels, colors)
            self._refresh_labels_list()
            self.detection_status.setText(
                f'Added {len(new_labels)} labels to {image.filename}')
        except Exception as e:
            QMessageBox.critical(self, 'Detection Error', str(e))
        finally:
            self.auto_label_current_btn.setEnabled(True)
            self.auto_label_btn.setEnabled(True)
            self.auto_label_unlabeled_btn.setEnabled(True)

    def _run_auto_detection(self):
        if not self.classes:
            QMessageBox.warning(self, 'Warning',
                                'Add at least one class before auto-labeling.')
            return
        if self.module_selector is None:
            return
        module = self.module_selector.current_module()
        if module is None:
            return

        class_map = self._get_class_map(module)

        try:
            module.prepare()
        except Exception as e:
            QMessageBox.critical(self, 'Module Error', str(e))
            return

        self._worker = DetectionWorker(module, self.model.images,
                                        class_map=class_map)
        self._worker.progress.connect(self._on_detection_progress)
        self._worker.status.connect(self._on_detection_status)
        self._worker.detection_complete.connect(self._on_detection_result)
        self._worker.finished_work.connect(self._on_detection_finished)
        self._worker.error.connect(self._on_detection_error)

        self.detection_progress.setVisible(True)
        self.cancel_detection_btn.setVisible(True)
        self.auto_label_btn.setEnabled(False)
        self.auto_label_current_btn.setEnabled(False)
        self.auto_label_unlabeled_btn.setEnabled(False)
        self._worker.start()

    def _run_auto_detection_unlabeled(self):
        if not self.classes:
            QMessageBox.warning(self, 'Warning',
                                'Add at least one class before auto-labeling.')
            return
        if self.module_selector is None:
            return
        module = self.module_selector.current_module()
        if module is None:
            return

        unlabeled = [img for img in self.model.images if not img.labels]
        if not unlabeled:
            QMessageBox.information(self, 'Info', 'All images already have labels.')
            return

        class_map = self._get_class_map(module)

        try:
            module.prepare()
        except Exception as e:
            QMessageBox.critical(self, 'Module Error', str(e))
            return

        self._worker = DetectionWorker(module, unlabeled,
                                        class_map=class_map)
        self._worker.progress.connect(self._on_detection_progress)
        self._worker.status.connect(self._on_detection_status)
        self._worker.detection_complete.connect(self._on_detection_result)
        self._worker.finished_work.connect(self._on_detection_finished)
        self._worker.error.connect(self._on_detection_error)

        self.detection_progress.setVisible(True)
        self.cancel_detection_btn.setVisible(True)
        self.auto_label_btn.setEnabled(False)
        self.auto_label_current_btn.setEnabled(False)
        self.auto_label_unlabeled_btn.setEnabled(False)
        self._worker.start()

    def _cancel_detection(self):
        if self._worker:
            self._worker.cancel()

    def _on_detection_progress(self, current, total):
        self.detection_progress.setMaximum(total)
        self.detection_progress.setValue(current)

    def _on_detection_status(self, text):
        self.detection_status.setText(text)

    def _on_detection_result(self, worker_index, labels):
        """Append detected labels to existing ones."""
        if not self._worker:
            return
        image = self._worker.images[worker_index]
        # Find the real model index for undo stack and display check
        try:
            model_index = self.model.images.index(image)
        except ValueError:
            return
        # Push undo snapshot (uses model index since detection may
        # target a different image than the currently displayed one)
        stack = self._undo_stacks.setdefault(model_index, [])
        stack.append(copy.deepcopy(image.labels))
        if len(stack) > self._max_undo:
            stack.pop(0)
        image.labels.extend(labels)
        write_yolo_labels(self._label_path_for(image), image.labels)
        if model_index == self._current_image_index:
            colors = self._get_colors()
            self.canvas.display_labels(image.labels, colors)
            self._refresh_labels_list()

    def _on_detection_finished(self):
        self.detection_progress.setVisible(False)
        self.cancel_detection_btn.setVisible(False)
        self.auto_label_btn.setEnabled(True)
        self.auto_label_current_btn.setEnabled(True)
        self.auto_label_unlabeled_btn.setEnabled(True)
        self._worker = None
        self.image_list.reapply_filter()
        from ltd.utils.sound import play_completion_sound
        play_completion_sound()

    def _on_detection_error(self, msg):
        QMessageBox.critical(self, 'Detection Error', msg)

    # --- Output actions ---

    def _save_masks(self):
        folder = QFileDialog.getExistingDirectory(self, 'Save Masks To')
        if not folder:
            return
        out = Path(folder)
        include_unlabeled = self.include_unlabeled_cb.isChecked()
        for image in self.model.images:
            if not include_unlabeled and not image.labels:
                continue
            shutil.copy2(image.path, out / image.filename)
            masks = []
            for label in image.labels:
                if label.has_polygon:
                    masks.append(polygon_to_mask(label.polygon,
                                                 image.width, image.height))
                elif label.has_bbox:
                    masks.append(bbox_to_mask(label.bbox,
                                              image.width, image.height))
            if masks:
                merged = merge_masks(masks)
            else:
                merged = create_empty_mask(image.height, image.width)
            mask_name = f'{image.name}-mask.png'
            save_mask(merged, out / mask_name)

        self.detection_status.setText(f'Masks saved to {folder}')

    def _copy_to_modify(self):
        self._save_current_labels()
        items = []
        temp_dir = get_temp_dir('modify')
        include_unlabeled = self.include_unlabeled_cb.isChecked()
        for image in self.model.images:
            if not include_unlabeled and not image.labels:
                continue
            dest_img = temp_dir / image.filename
            shutil.copy2(image.path, dest_img)
            mask_path = None
            masks = []
            for label in image.labels:
                if label.has_polygon:
                    masks.append(polygon_to_mask(label.polygon,
                                                 image.width, image.height))
                elif label.has_bbox:
                    masks.append(bbox_to_mask(label.bbox,
                                              image.width, image.height))
            if masks:
                merged = merge_masks(masks)
                mask_path = temp_dir / f'{image.name}-masklabel.png'
                save_mask(merged, mask_path)
            item = ImageItem(path=dest_img, width=image.width,
                             height=image.height, mask_path=mask_path,
                             labels=list(image.labels),
                             original_path=image.path)
            items.append(item)

        if items:
            self.copy_to_modify_requested.emit(items, self._get_colors())
        else:
            QMessageBox.information(self, 'Info', 'No images to copy.')

    def _get_dataset_format(self) -> str:
        """Return 'segmentation' if any class is Polygon, else 'detection'."""
        for c in self.classes:
            if c.detection_type == DetectionType.POLYGON:
                return 'segmentation'
        return 'detection'

    def _save_dataset_to(self, out: Path):
        """Save YOLO dataset to the given directory."""
        self._save_current_labels()
        images_dir = out / 'images'
        labels_dir = out / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        classes_txt = out / 'classes.txt'
        with open(classes_txt, 'w') as f:
            for c in self.classes:
                f.write(c.name + '\n')

        fmt = self._get_dataset_format()
        include_unlabeled = self.include_unlabeled_cb.isChecked()
        for image in self.model.images:
            if not include_unlabeled and not image.labels:
                continue
            shutil.copy2(image.path, images_dir / image.filename)
            txt_path = labels_dir / f'{image.name}.txt'
            write_yolo_labels(txt_path, image.labels, force_format=fmt)

    def _save_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, 'Save YOLO Dataset To')
        if not folder:
            return
        self._save_dataset_to(Path(folder))
        self.dataset_saved.emit(folder)
        self.detection_status.setText(f'Dataset saved to {folder}')

    def _copy_to_train(self):
        if not self.model.images:
            QMessageBox.information(self, 'Info', 'No images loaded.')
            return
        if not self.classes:
            QMessageBox.warning(self, 'Warning',
                                'Add at least one class before copying.')
            return
        self._save_current_labels()
        temp_dir = get_temp_dir('train_dataset')
        fmt = self._get_dataset_format()
        class_names = [c.name for c in self.classes]

        # Filter and shuffle, then split 80/20
        include_unlabeled = self.include_unlabeled_cb.isChecked()
        images = [img for img in self.model.images
                  if include_unlabeled or img.labels]
        if not images:
            QMessageBox.information(self, 'Info', 'No labeled images to copy.')
            return
        random.shuffle(images)
        split_idx = max(1, int(len(images) * 0.8))
        train_images = images[:split_idx]
        val_images = images[split_idx:] if split_idx < len(images) else []

        # If only 1 image, put it in both train and valid
        if not val_images:
            val_images = train_images[:1]

        for subset_name, subset in [('train', train_images),
                                    ('valid', val_images)]:
            img_dir = temp_dir / subset_name / 'images'
            lbl_dir = temp_dir / subset_name / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for image in subset:
                shutil.copy2(image.path, img_dir / image.filename)
                txt_path = lbl_dir / f'{image.name}.txt'
                write_yolo_labels(txt_path, image.labels, force_format=fmt)

        # Write data.yaml
        import yaml
        data_config = {
            'path': str(temp_dir.resolve()),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(class_names),
            'names': class_names,
        }
        with open(temp_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        model_type = 'segment' if fmt == 'segmentation' else 'detect'
        self.copy_to_train_requested.emit(str(temp_dir), model_type)
        n_train = len(train_images)
        n_val = len(val_images)
        self.detection_status.setText(
            f'Dataset copied to Train tab ({n_train} train, {n_val} valid)')
