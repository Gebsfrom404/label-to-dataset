"""Modify tab: image modification with before/after comparison."""
import shutil
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox,
                               QSplitter, QToolButton, QVBoxLayout, QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel
from ltd.data.label_data import DEFAULT_COLORS
from ltd.utils.file_utils import get_temp_dir_no_clear
from ltd.utils.image_utils import load_pixmap_preview
from ltd.utils.mask_utils import mask_from_qimage
from ltd.widgets.canvas_widget import CanvasWidget, DrawMode, Tool
from ltd.widgets.image_list_widget import ImageListWidget
from ltd.widgets.elided_label import ElidedLabel
from ltd.widgets.loading_dialog import loading_dialog
from ltd.widgets.module_selector import ModuleSelector
from ltd.workers.modification_worker import ModificationWorker

from modules.base import BaseModificationModule
from modules import discover_modules

# Internal tool names for crop/split (not in canvas Tool enum)
_TOOL_CROP = 'crop'
_TOOL_SPLIT_V = 'split_v'
_TOOL_SPLIT_H = 'split_h'


class ModifyTab(QWidget):
    copy_to_caption_requested = Signal(str)  # folder path

    _PIXMAP_CACHE_MAX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self._current_image_index = -1
        self._worker: ModificationWorker | None = None
        self._class_colors: list[str] = list(DEFAULT_COLORS)
        self._pixmap_cache: OrderedDict[str, object] = OrderedDict()
        self._active_tool = None  # Track which tool is active (Tool enum or str)
        # Mask edit history per image: keyed by image index.
        # Each value is (history_list, history_pos).
        # History entry dict keys:
        #   name: str, mask: QImage, pixmap: QPixmap,
        #   modified_path: Path|None, mask_path: Path|None,
        #   width: int, height: int
        self._image_histories: dict[int, tuple[list, int]] = {}
        self._mask_history: list[dict] = []
        self._history_pos: int = -1
        self._HISTORY_MAX = 50

        # Discover modification modules
        self._modification_modules = discover_modules(
            BaseModificationModule, 'modules/modifications')

        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._install_space_filter()

    # --- UI Setup ---

    @staticmethod
    def _make_separator():
        sep = QLabel('|')
        sep.setStyleSheet('color: gray; margin: 0 2px;')
        return sep

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image list
        self.image_list = ImageListWidget(self.model)
        self._setup_image_list_context_menu()
        splitter.addWidget(self.image_list)

        # Center: toolbar + canvas
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(2)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 2, 2, 2)
        toolbar.setSpacing(2)

        # All tools in one exclusive group: Hand, BBox, Polygon, Brush,
        # Crop, SplitV, SplitH
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict = {}  # key: Tool enum or str

        canvas_tools = [
            ('Hand (M)', Tool.HAND),
            ('BBox (R)', Tool.BBOX),
            ('Polygon (V)', Tool.POLYGON),
            ('Brush (B)', Tool.MARKER),
        ]
        for label, tool in canvas_tools:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setProperty('tool_id', tool)
            self.tool_group.addButton(btn)
            toolbar.addWidget(btn)
            self.tool_buttons[tool] = btn
        self.tool_buttons[Tool.HAND].setChecked(True)

        toolbar.addWidget(self._make_separator())

        # Crop and Split as tools in same group
        overlay_tools = [
            ('Crop (C)', _TOOL_CROP),
            ('Split V', _TOOL_SPLIT_V),
            ('Split H', _TOOL_SPLIT_H),
        ]
        for label, tool_id in overlay_tools:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setProperty('tool_id', tool_id)
            self.tool_group.addButton(btn)
            toolbar.addWidget(btn)
            self.tool_buttons[tool_id] = btn

        toolbar.addWidget(self._make_separator())

        # Draw mode buttons: Draw / Erase
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[DrawMode, QToolButton] = {}

        draw_btn = QToolButton()
        draw_btn.setText('Draw (1)')
        draw_btn.setCheckable(True)
        draw_btn.setChecked(True)
        draw_btn.setProperty('draw_mode', DrawMode.NEW)
        self.mode_group.addButton(draw_btn)
        toolbar.addWidget(draw_btn)
        self.mode_buttons[DrawMode.NEW] = draw_btn

        erase_btn = QToolButton()
        erase_btn.setText('Erase (2)')
        erase_btn.setCheckable(True)
        erase_btn.setProperty('draw_mode', DrawMode.ERASE)
        self.mode_group.addButton(erase_btn)
        toolbar.addWidget(erase_btn)
        self.mode_buttons[DrawMode.ERASE] = erase_btn

        toolbar.addWidget(self._make_separator())

        # Brush size
        toolbar.addWidget(QLabel('Brush:'))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(20)
        toolbar.addWidget(self.brush_spin)

        toolbar.addWidget(self._make_separator())

        # Invert Mask
        self.invert_mask_btn = QPushButton('Invert Mask')
        toolbar.addWidget(self.invert_mask_btn)

        toolbar.addStretch()
        center_layout.addLayout(toolbar)

        # --- Canvas ---
        self.canvas = CanvasWidget()
        center_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(center_widget)

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

        run_btn_layout = QHBoxLayout()
        self.run_current_btn = QPushButton('Run Current')
        self.run_all_btn = QPushButton('Run All')
        run_btn_layout.addWidget(self.run_current_btn)
        run_btn_layout.addWidget(self.run_all_btn)
        mod_layout.addLayout(run_btn_layout)

        self.mod_progress = QProgressBar()
        self.mod_progress.setVisible(False)
        mod_layout.addWidget(self.mod_progress)

        self.mod_status = ElidedLabel('')
        mod_layout.addWidget(self.mod_status)

        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setVisible(False)
        mod_layout.addWidget(self.cancel_btn)

        right_layout.addWidget(mod_group)

        # Tools group (restore only)
        tools_group = QGroupBox('Tools')
        tools_layout = QVBoxLayout(tools_group)
        restore_row = QHBoxLayout()
        self.restore_current_btn = QPushButton('Restore Current')
        self.restore_all_btn = QPushButton('Restore All')
        restore_row.addWidget(self.restore_current_btn)
        restore_row.addWidget(self.restore_all_btn)
        tools_layout.addLayout(restore_row)
        right_layout.addWidget(tools_group)

        # Mask edit history
        history_group = QGroupBox('Mask History')
        history_layout = QVBoxLayout(history_group)

        history_btn_row = QHBoxLayout()
        self.history_start_btn = QPushButton('Jump to Start')
        self.history_end_btn = QPushButton('Jump to End')
        history_btn_row.addWidget(self.history_start_btn)
        history_btn_row.addWidget(self.history_end_btn)
        history_layout.addLayout(history_btn_row)

        self.history_list = QListWidget()
        self.history_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        history_layout.addWidget(self.history_list, stretch=1)

        right_layout.addWidget(history_group, stretch=1)

        # Output
        output_group = QGroupBox('Output')
        output_layout = QVBoxLayout(output_group)

        self.save_modified_btn = QPushButton('Save Modified Images...')
        output_layout.addWidget(self.save_modified_btn)

        self.save_in_place_btn = QPushButton('Save Modified In Place')
        output_layout.addWidget(self.save_in_place_btn)

        self.copy_caption_btn = QPushButton('Open in Caption')
        output_layout.addWidget(self.copy_caption_btn)

        right_layout.addWidget(output_group)
        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 600, 300])
        layout.addWidget(splitter)

    # --- Image list context menu ---

    def _setup_image_list_context_menu(self):
        self.image_list.list_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.list_view.customContextMenuRequested.connect(
            self._show_image_context_menu)

    def _show_image_context_menu(self, pos):
        index = self.image_list.list_view.indexAt(pos)
        if not index.isValid():
            return
        menu = QMenu(self)
        delete_action = menu.addAction('Delete Image\tCtrl+Del')
        action = menu.exec(self.image_list.list_view.mapToGlobal(pos))
        if action == delete_action:
            self._delete_current_image()

    def _delete_current_image(self):
        row = self.image_list.current_row()
        if row < 0:
            return
        image = self.model.get_image(row)
        if image is None:
            return

        reply = QMessageBox.question(
            self, 'Delete Image',
            f'Permanently delete "{image.filename}" from disk?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete files from disk
        try:
            if image.path and image.path.exists():
                image.path.unlink()
            if image.modified_path and image.modified_path.exists():
                image.modified_path.unlink()
            if image.mask_path and image.mask_path.exists():
                image.mask_path.unlink()
            # Delete associated label file if exists
            label_path = image.path.with_suffix('.txt')
            if label_path.exists():
                label_path.unlink()
        except OSError:
            pass

        # Clean up history for this image
        self._image_histories.pop(row, None)

        self.model.remove_rows([row])
        if self.model.rowCount() > 0:
            new_row = min(row, self.model.rowCount() - 1)
            self.image_list.select_index(new_row)
        else:
            self._mask_history.clear()
            self._history_pos = -1
            self._sync_history_list()
            self.canvas.clear_canvas()

    # --- Shortcuts ---

    def _setup_shortcuts(self):
        tool_shortcuts = {
            'M': Tool.HAND, 'R': Tool.BBOX,
            'V': Tool.POLYGON, 'B': Tool.MARKER,
            'C': _TOOL_CROP,
        }
        for key, tool in tool_shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda t=tool: self._set_tool(t))

        # Draw(1) / Erase(2)
        sc = QShortcut(QKeySequence('1'), self)
        sc.activated.connect(lambda: self._set_draw_mode(DrawMode.NEW))
        sc = QShortcut(QKeySequence('2'), self)
        sc.activated.connect(lambda: self._set_draw_mode(DrawMode.ERASE))

        # Enter = apply crop/split
        sc = QShortcut(QKeySequence('Return'), self)
        sc.activated.connect(self._apply_current_overlay)

        # Escape = cancel current action
        sc = QShortcut(QKeySequence('Escape'), self)
        sc.activated.connect(self._cancel_current_action)

        # Undo last mask edit
        sc = QShortcut(QKeySequence('Ctrl+Z'), self)
        sc.activated.connect(self._undo_mask_edit)

        # Delete image
        sc = QShortcut(QKeySequence('Ctrl+Delete'), self)
        sc.activated.connect(self._delete_current_image)

    def _install_space_filter(self):
        """Install event filter on all child widgets to capture spacebar."""
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            key = event.key()
            if key == Qt.Key.Key_Space and not event.isAutoRepeat():
                if event.type() == QEvent.Type.KeyPress:
                    self.canvas.keyPressEvent(event)
                else:
                    self.canvas.keyReleaseEvent(event)
                return True
            # Intercept navigation keys to prevent QListView page-jumping
            if key in (Qt.Key.Key_A, Qt.Key.Key_PageUp):
                if event.type() == QEvent.Type.KeyPress:
                    self.image_list.go_to_previous()
                return True
            if key in (Qt.Key.Key_D, Qt.Key.Key_PageDown):
                if event.type() == QEvent.Type.KeyPress:
                    self.image_list.go_to_next()
                return True
        return super().eventFilter(obj, event)

    # --- Tool/mode setters ---

    def _set_tool(self, tool_id):
        """Set active tool. tool_id is Tool enum or string for crop/split."""
        btn = self.tool_buttons.get(tool_id)
        if btn:
            btn.setChecked(True)
        # setChecked doesn't emit buttonClicked, so apply directly
        self._apply_tool(tool_id)

    def _set_draw_mode(self, mode: DrawMode):
        self.canvas.draw_mode = mode
        btn = self.mode_buttons.get(mode)
        if btn:
            btn.setChecked(True)

    def _on_tool_changed(self, btn):
        tool_id = btn.property('tool_id')
        if tool_id is not None:
            self._apply_tool(tool_id)

    def _apply_tool(self, tool_id):
        """Apply the given tool — disable previous overlay, enable new one."""
        if tool_id == self._active_tool:
            return

        # Disable previous overlay
        if self._active_tool == _TOOL_CROP:
            self.canvas.set_crop_mode(False)
        elif self._active_tool in (_TOOL_SPLIT_V, _TOOL_SPLIT_H):
            self.canvas.set_split_mode(False)

        self._active_tool = tool_id

        # Enable new tool
        if isinstance(tool_id, Tool):
            self.canvas.current_tool = tool_id
        elif tool_id == _TOOL_CROP:
            self.canvas._current_tool = Tool.HAND
            self.canvas._finish_polygon()
            self.canvas._update_brush_cursor_visibility()
            from PySide6.QtWidgets import QGraphicsView
            self.canvas.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.canvas.set_crop_mode(True)
        elif tool_id in (_TOOL_SPLIT_V, _TOOL_SPLIT_H):
            self.canvas._current_tool = Tool.HAND
            self.canvas._finish_polygon()
            self.canvas._update_brush_cursor_visibility()
            from PySide6.QtWidgets import QGraphicsView
            self.canvas.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)
            orientation = 'V' if tool_id == _TOOL_SPLIT_V else 'H'
            self.canvas.set_split_mode(True, orientation)

    def _on_mode_changed(self, btn):
        mode = btn.property('draw_mode')
        if mode:
            self.canvas.draw_mode = mode

    def _cancel_current_action(self):
        """Esc: cancel polygon, bbox drag, reset crop, or switch to Hand."""
        # Cancel in-progress polygon (cleans markers/lines, no label emitted
        # unless >= 3 points — but we clear points first to prevent that)
        if self.canvas._polygon_points:
            # Clear points so _finish_polygon won't emit a label
            self.canvas._polygon_points = self.canvas._polygon_points[:0]
            self.canvas._finish_polygon()
            return

        # Cancel in-progress bbox drag
        if self.canvas._drawing and self.canvas._current_tool == Tool.BBOX:
            if self.canvas._temp_rect:
                if self.canvas._temp_rect.scene() is not None:
                    self.canvas.scene_.removeItem(self.canvas._temp_rect)
                self.canvas._temp_rect = None
            self.canvas._drawing = False
            self.canvas._draw_start = None
            return

        # Reset crop to full image edges
        if self._active_tool == _TOOL_CROP:
            from PySide6.QtCore import QRectF
            self.canvas._crop_rect = QRectF(
                0, 0, self.canvas._image_width, self.canvas._image_height)
            self.canvas._update_crop_overlay()
            return

        # For split or any other state: switch to Hand
        self._set_tool(Tool.HAND)

    # --- Signals ---

    def _connect_signals(self):
        self.image_list.current_changed.connect(self._on_image_changed)
        self.image_list.load_directory_requested.connect(self._load_directory)

        # Toolbar
        self.tool_group.buttonClicked.connect(self._on_tool_changed)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        self.brush_spin.valueChanged.connect(self._on_brush_spin_changed)
        self.canvas.brush_size_changed.connect(self._on_brush_size_from_canvas)

        # Canvas signals
        self.canvas.mask_updated.connect(self._on_mask_updated)
        self.canvas.label_created.connect(self._on_label_created)
        self.canvas.split_pos_changed.connect(self._on_split_pos_from_canvas)

        # Invert mask
        self.invert_mask_btn.clicked.connect(self._invert_mask)

        # History
        self.history_start_btn.clicked.connect(self._history_jump_to_start)
        self.history_end_btn.clicked.connect(self._history_jump_to_end)
        self.history_list.currentRowChanged.connect(self._on_history_clicked)
        self.history_list.customContextMenuRequested.connect(
            self._show_history_context_menu)

        # Modification
        self.run_current_btn.clicked.connect(
            lambda: self._run_modification(single=True))
        self.run_all_btn.clicked.connect(
            lambda: self._run_modification(single=False))
        self.cancel_btn.clicked.connect(self._cancel_modification)

        self.save_modified_btn.clicked.connect(self._save_modified)
        self.save_in_place_btn.clicked.connect(self._save_modified_in_place)
        self.copy_caption_btn.clicked.connect(self._copy_to_caption)

        self.restore_current_btn.clicked.connect(self._restore_current)
        self.restore_all_btn.clicked.connect(self._restore_all)

    # --- Brush size sync ---

    def _on_brush_spin_changed(self, value: int):
        self.canvas.brush_size = value

    def _on_brush_size_from_canvas(self, size: int):
        self.brush_spin.blockSignals(True)
        self.brush_spin.setValue(size)
        self.brush_spin.blockSignals(False)

    # --- Image loading ---

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

            for image in self.model.images:
                mask_path = image.path.parent / f'{image.name}-masklabel.png'
                if mask_path.exists():
                    image.mask_path = mask_path

        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _preview_max_dim(self) -> int:
        vp = self.canvas.viewport().size()
        return max(vp.width(), vp.height(), 800) * 2

    def _reload_and_record(self, image: ImageItem, action_name: str):
        """Reload canvas from image state and record in history.

        Used by crop/modify to update the canvas without resetting history.
        """
        display_path = (image.modified_path
                        if image.modified_path else image.path)
        self._pixmap_cache.pop(str(display_path), None)
        max_dim = self._preview_max_dim()
        pixmap = load_pixmap_preview(display_path, max_dim)
        if pixmap is None or pixmap.isNull():
            return
        self.canvas.load_image(pixmap)

        if image.mask_path and image.mask_path.exists():
            mask_qimage = QImage(str(image.mask_path))
            if not mask_qimage.isNull():
                if (mask_qimage.width() != pixmap.width()
                        or mask_qimage.height() != pixmap.height()):
                    mask_qimage = mask_qimage.scaled(
                        pixmap.width(), pixmap.height(),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.FastTransformation)
                self.canvas.set_mask(mask_qimage)

        self.model.invalidate_thumbnail(self._current_image_index)
        self._history_record(action_name)

    def _on_image_changed(self, index: int):
        # Save current image's history before switching
        if self._current_image_index >= 0 and self._mask_history:
            self._image_histories[self._current_image_index] = (
                list(self._mask_history), self._history_pos)

        self._current_image_index = index
        image = self.model.get_image(index)
        if image is None:
            return

        # Disable crop/split on image change — switch to Hand
        if self._active_tool in (_TOOL_CROP, _TOOL_SPLIT_V, _TOOL_SPLIT_H):
            self._set_tool(Tool.HAND)

        # Load image on canvas (preview-sized for speed)
        max_dim = self._preview_max_dim()
        display_path = (image.modified_path
                        if image.modified_path else image.path)
        pixmap = self._load_cached(display_path, max_dim)
        if pixmap is None or pixmap.isNull():
            return
        self.canvas.load_image(pixmap)

        # Load mask if exists — scale to match preview pixmap
        if image.mask_path and image.mask_path.exists():
            mask_qimage = QImage(str(image.mask_path))
            if not mask_qimage.isNull():
                if (mask_qimage.width() != pixmap.width() or
                        mask_qimage.height() != pixmap.height()):
                    mask_qimage = mask_qimage.scaled(
                        pixmap.width(), pixmap.height(),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.FastTransformation)
                self.canvas.set_mask(mask_qimage)

        # Display labels on canvas
        if image.labels:
            self.canvas.display_labels(image.labels, self._class_colors)

        # Restore or initialize history for this image
        if index in self._image_histories:
            saved_history, saved_pos = self._image_histories[index]
            self._mask_history = list(saved_history)
            self._history_pos = saved_pos
            self._sync_history_list()
        else:
            self._history_init()

        QTimer.singleShot(0, lambda idx=index: self._preload_adjacent(idx))

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

    # --- Mask edit history ---

    def _make_history_entry(self, name: str) -> dict | None:
        """Capture current full state as a history entry."""
        mask_buf = self.canvas.get_mask_image()
        if mask_buf is None:
            return None
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return None
        # QPixmap uses implicit sharing so storing same ref is cheap
        return {
            'name': name,
            'mask': mask_buf.copy(),
            'pixmap': self.canvas._pixmap_item.pixmap() if self.canvas._pixmap_item else QPixmap(),
            'modified_path': image.modified_path,
            'mask_path': image.mask_path,
            'width': image.width,
            'height': image.height,
        }

    def _history_init(self):
        """Reset history with current state as 'Start'."""
        self._mask_history.clear()
        self._history_pos = -1
        entry = self._make_history_entry('Start')
        if entry:
            self._mask_history.append(entry)
            self._history_pos = 0
        self._sync_history_list()

    def _history_record(self, action_name: str):
        """Record current state as a new history entry after an edit."""
        entry = self._make_history_entry(action_name)
        if entry is None:
            return
        # If we're not at the end, truncate forward history
        if self._history_pos < len(self._mask_history) - 1:
            self._mask_history = self._mask_history[:self._history_pos + 1]
        self._mask_history.append(entry)
        if len(self._mask_history) > self._HISTORY_MAX:
            self._mask_history.pop(0)
        self._history_pos = len(self._mask_history) - 1
        self._sync_history_list()

    def _undo_mask_edit(self):
        """Go one step back in history (Ctrl+Z)."""
        if self._history_pos <= 0:
            return
        self._history_navigate(self._history_pos - 1)

    def _history_jump_to_start(self):
        """Jump to the initial mask state."""
        if self._mask_history:
            self._history_navigate(0)

    def _history_jump_to_end(self):
        """Jump to the most recent mask state."""
        if self._mask_history:
            self._history_navigate(len(self._mask_history) - 1)

    def _on_history_clicked(self, row: int):
        """Navigate to a specific history entry when clicked."""
        if 0 <= row < len(self._mask_history) and row != self._history_pos:
            self._history_navigate(row)

    def _history_navigate(self, pos: int):
        """Restore full state at the given history position."""
        if pos < 0 or pos >= len(self._mask_history):
            return
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        self._history_pos = pos
        entry = self._mask_history[pos]

        # Check if the image itself changed (different path or dimensions)
        need_image_reload = False
        if image.modified_path != entry['modified_path']:
            need_image_reload = True
        elif entry['pixmap'] and self.canvas._pixmap_item:
            cur_pm = self.canvas._pixmap_item.pixmap()
            ent_pm = entry['pixmap']
            if (cur_pm.width() != ent_pm.width()
                    or cur_pm.height() != ent_pm.height()):
                need_image_reload = True

        # Restore image metadata
        image.modified_path = entry['modified_path']
        image.mask_path = entry['mask_path']
        image.width = entry['width']
        image.height = entry['height']

        if need_image_reload:
            self.canvas.load_image(entry['pixmap'])
            # Invalidate caches and thumbnail
            self._pixmap_cache.clear()
            self.model.invalidate_thumbnail(self._current_image_index)

        # Restore mask
        mask_snapshot = entry['mask']
        if need_image_reload:
            if (mask_snapshot.width() != self.canvas._image_width
                    or mask_snapshot.height() != self.canvas._image_height):
                mask_snapshot = mask_snapshot.scaled(
                    self.canvas._image_width, self.canvas._image_height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation)
        self.canvas.set_mask(mask_snapshot.copy())
        self.canvas._update_mask_overlay_fast()
        self._save_mask_buffer(image)
        self._sync_history_list()

    def _show_history_context_menu(self, pos):
        """Context menu to delete a history entry and all after it."""
        item = self.history_list.itemAt(pos)
        if item is None:
            return
        row = self.history_list.row(item)
        if row <= 0:
            return  # Can't delete 'Start'
        menu = QMenu(self)
        delete_action = menu.addAction('Delete this and all after')
        action = menu.exec(self.history_list.mapToGlobal(pos))
        if action == delete_action:
            self._mask_history = self._mask_history[:row]
            if self._history_pos >= row:
                self._history_navigate(len(self._mask_history) - 1)
            else:
                self._sync_history_list()

    def _sync_history_list(self):
        """Sync the history QListWidget with internal state."""
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for i, entry in enumerate(self._mask_history):
            label = f'{i}. {entry["name"]}'
            item = QListWidgetItem(label)
            if i > self._history_pos:
                item.setForeground(Qt.GlobalColor.gray)
            self.history_list.addItem(item)
        if 0 <= self._history_pos < self.history_list.count():
            self.history_list.setCurrentRow(self._history_pos)
        self.history_list.blockSignals(False)

    # --- Canvas mask handling ---

    def _on_mask_updated(self, draw_mode):
        """Save canvas mask buffer directly to file and record in history."""
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        self._save_mask_buffer(image)
        self._history_record('Brush')

    def _on_label_created(self, label):
        """Handle bbox/polygon tool — paint the shape onto the mask buffer."""
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return

        mask_buf = self.canvas.get_mask_image()
        if mask_buf is None:
            return

        w = self.canvas._image_width
        h = self.canvas._image_height
        if w <= 0 or h <= 0:
            return

        erase = self.canvas.draw_mode == DrawMode.ERASE
        color = Qt.GlobalColor.black if erase else Qt.GlobalColor.white

        painter = QPainter(mask_buf)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        from PySide6.QtGui import QBrush, QPolygonF
        from PySide6.QtCore import QPointF, QRectF
        painter.setBrush(QBrush(color))

        if label.has_polygon:
            points = [QPointF(x * w, y * h) for x, y in label.polygon]
            painter.drawPolygon(QPolygonF(points))
        elif label.has_bbox:
            cx, cy, bw, bh = label.bbox
            rx = (cx - bw / 2) * w
            ry = (cy - bh / 2) * h
            rw = bw * w
            rh = bh * h
            painter.drawRect(QRectF(rx, ry, rw, rh))
        painter.end()

        # Update overlay and save
        self.canvas._update_mask_overlay_fast()
        self._save_mask_buffer(image)
        action = 'Polygon' if label.has_polygon else 'BBox'
        self._history_record(action)

    def _save_mask_buffer(self, image: ImageItem):
        """Save current canvas mask buffer to file."""
        mask_qimage = self.canvas.get_mask_image()
        if mask_qimage is None:
            return

        mask_arr = mask_from_qimage(mask_qimage)

        if mask_arr.max() < 128:
            image.mask_path = None
            return

        from PIL import Image as PILImage
        temp_dir = get_temp_dir_no_clear('masks')
        mask_out = temp_dir / f'{image.name}-masklabel.png'

        # Scale mask back to original image dimensions for saving
        source_path = (image.modified_path
                       if image.modified_path else image.path)
        orig_pixmap = QPixmap(str(source_path))
        orig_w, orig_h = orig_pixmap.width(), orig_pixmap.height()

        if (mask_arr.shape[1] != orig_w or mask_arr.shape[0] != orig_h):
            pil_mask = PILImage.fromarray(mask_arr)
            pil_mask = pil_mask.resize((orig_w, orig_h), PILImage.NEAREST)
            pil_mask.save(str(mask_out))
        else:
            PILImage.fromarray(mask_arr).save(str(mask_out))
        image.mask_path = mask_out

    def _on_split_pos_from_canvas(self, pos: float):
        pass

    # --- Apply crop/split ---

    def _apply_current_overlay(self):
        """Apply crop or split when Enter is pressed."""
        if self._active_tool == _TOOL_CROP:
            self._apply_crop()
        elif self._active_tool in (_TOOL_SPLIT_V, _TOOL_SPLIT_H):
            self._apply_split()

    def _apply_crop(self):
        from PIL import Image
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return

        x, y, w, h = self.canvas.get_crop_rect()
        if w <= 0 or h <= 0:
            QMessageBox.information(self, 'Info', 'Invalid crop area.')
            return

        source_path = image.modified_path if image.modified_path else image.path
        pil_img = Image.open(str(source_path))
        actual_w, actual_h = pil_img.size

        # Scale crop rect from preview coords to actual image coords
        canvas_w = self.canvas._image_width
        canvas_h = self.canvas._image_height
        if canvas_w > 0 and canvas_h > 0:
            sx = actual_w / canvas_w
            sy = actual_h / canvas_h
        else:
            sx = sy = 1.0
        ax = int(x * sx)
        ay = int(y * sy)
        aw = int(w * sx)
        ah = int(h * sy)
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

        if image.mask_path and image.mask_path.exists():
            mask_img = Image.open(str(image.mask_path))
            if mask_img.size != (actual_w, actual_h):
                mask_img = mask_img.resize((actual_w, actual_h), Image.NEAREST)
            mask_cropped = mask_img.crop((ax, ay, ax + aw, ay + ah))
            mask_out = temp_dir / f'{image.name}_cropped-masklabel.png'
            mask_cropped.save(str(mask_out))
            image.mask_path = mask_out
            mask_img.close()

        image.modified_path = out_path
        image.width = aw
        image.height = ah
        self._pixmap_cache.pop(str(source_path), None)

        self._set_tool(Tool.HAND)
        self._reload_and_record(image, 'Crop')
        self.mod_status.setText(f'Cropped to {aw}x{ah}')

    def _apply_split(self):
        from PIL import Image
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return

        split_pos = self.canvas.get_split_pos()
        orientation = 'H' if self._active_tool == _TOOL_SPLIT_H else 'V'

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

        item1 = ImageItem(path=path1, width=sizes[0][0], height=sizes[0][1],
                          mask_path=mask_path1)
        item2 = ImageItem(path=path2, width=sizes[1][0], height=sizes[1][1],
                          mask_path=mask_path2)

        insert_pos = self._current_image_index + 1
        self.model.insert_items(insert_pos, [item1, item2])

        self._set_tool(Tool.HAND)
        self.image_list.select_index(insert_pos)
        self.mod_status.setText(
            f'Split into {sizes[0][0]}x{sizes[0][1]} + '
            f'{sizes[1][0]}x{sizes[1][1]}')

    # --- Invert Mask ---

    def _invert_mask(self):
        image = self.model.get_image(self._current_image_index)
        if image is None:
            QMessageBox.information(self, 'Info', 'No image selected.')
            return
        if not image.mask_path or not image.mask_path.exists():
            QMessageBox.information(self, 'Info', 'No mask to invert.')
            return

        import numpy as np
        from PIL import Image as PILImage

        mask_img = PILImage.open(str(image.mask_path)).convert('L')
        mask_arr = np.array(mask_img)
        inverted_arr = 255 - mask_arr
        inverted_img = PILImage.fromarray(inverted_arr)
        inverted_img.save(str(image.mask_path))

        # Reload mask on canvas at preview scale
        inverted_qimage = QImage(str(image.mask_path))
        if not inverted_qimage.isNull():
            pw = self.canvas._image_width
            ph = self.canvas._image_height
            if (inverted_qimage.width() != pw or
                    inverted_qimage.height() != ph):
                inverted_qimage = inverted_qimage.scaled(
                    pw, ph,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation)
            self.canvas.set_mask(inverted_qimage)

        self._history_record('Invert')
        self.mod_status.setText('Mask inverted')

    # --- Modification ---

    def _set_run_buttons_enabled(self, enabled: bool):
        self.run_current_btn.setEnabled(enabled)
        self.run_all_btn.setEnabled(enabled)

    def _run_modification(self, single: bool = False):
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
            module, images_to_run, use_current=True)
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
        self._pixmap_cache.pop(output_path, None)

        if getattr(self, '_single_mode', False):
            image = self.model.get_image(self._current_image_index)
            if image:
                image.modified_path = Path(output_path)
                self._reload_and_record(image, 'Modify')
        else:
            images_with_masks = [img for img in self.model.images
                                 if img.mask_path is not None]
            if 0 <= index < len(images_with_masks):
                image = images_with_masks[index]
                image.modified_path = Path(output_path)
                all_idx = self.model.images.index(image)
                if all_idx == self._current_image_index:
                    self._reload_and_record(image, 'Modify')

    def _on_mod_finished(self):
        self.mod_progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._set_run_buttons_enabled(True)
        self._worker = None
        from ltd.utils.sound import play_completion_sound
        play_completion_sound()

    def _on_mod_error(self, msg):
        QMessageBox.critical(self, 'Modification Error', msg)

    # --- Restore ---

    def _restore_current(self):
        image = self.model.get_image(self._current_image_index)
        if image is None or not image.modified_path:
            QMessageBox.information(self, 'Info',
                                    'Current image has no modifications.')
            return
        self._pixmap_cache.pop(str(image.modified_path), None)
        self._pixmap_cache.pop(str(image.path), None)
        image.modified_path = None
        self._reload_and_record(image, 'Restore')
        self.mod_status.setText('Restored original image')

    def _restore_all(self):
        modified = [img for img in self.model.images if img.modified_path]
        if not modified:
            QMessageBox.information(self, 'Info',
                                    'No modified images to restore.')
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
            # Clear per-image histories for other images
            if idx != self._current_image_index:
                self._image_histories.pop(idx, None)
        cur_image = self.model.get_image(self._current_image_index)
        if cur_image:
            self._reload_and_record(cur_image, 'Restore')
        self.mod_status.setText(f'Restored {len(modified)} original images')

    # --- Output ---

    def _save_modified(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Save Modified Images To')
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
            QMessageBox.information(self, 'Info',
                                    'No modified images to save.')
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
                save_target = image.original_path if image.original_path else image.path
                shutil.copy2(str(image.modified_path), str(save_target))
                if save_target != image.path:
                    shutil.copy2(str(image.modified_path), str(image.path))
                self._pixmap_cache.pop(str(image.path), None)
                self._pixmap_cache.pop(str(image.modified_path), None)
                self._pixmap_cache.pop(str(save_target), None)
                image.modified_path = None
                idx = self.model.images.index(image)
                self.model.invalidate_thumbnail(idx)
                count += 1
            except Exception as e:
                QMessageBox.warning(
                    self, 'Error',
                    f'Failed to save {image.filename}: {e}')

        if self._current_image_index >= 0:
            self._on_image_changed(self._current_image_index)
        self.mod_status.setText(f'Saved {count} modified images in place')

    def _copy_to_caption(self):
        if not self.model.images:
            QMessageBox.information(self, 'Info', 'No images to open.')
            return

        # Check for unsaved modifications
        modified = [img for img in self.model.images if img.modified_path]
        if modified:
            reply = QMessageBox.question(
                self, 'Unsaved Modifications',
                f'{len(modified)} image(s) have unsaved modifications.\n'
                f'Save them in place before opening in Caption?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_modified_in_place()

        first = self.model.images[0]
        folder = first.original_path.parent if first.original_path else first.path.parent
        self.copy_to_caption_requested.emit(str(folder))
