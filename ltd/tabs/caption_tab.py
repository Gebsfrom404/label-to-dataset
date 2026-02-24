"""Caption tab: tag editing and auto-captioning (adapted from taggui)."""
import csv
import re
import shutil
from collections import OrderedDict
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox,
                               QDialog, QDoubleSpinBox,
                               QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QFileDialog, QMenu,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from ltd.comfyui.client import ComfyUIClient
from ltd.comfyui.workflow import (load_workflow, validate_caption_workflow,
                                  set_input_image)
from ltd.data.image_item import ImageItem
from ltd.data.tag_dictionary import TagDictionary
from ltd.settings import get_settings
from ltd.data.image_list_model import IMAGE_EXTENSIONS, ImageListModel
from ltd.dialogs.batch_reorder_dialog import BatchReorderDialog
from ltd.dialogs.find_replace_dialog import FindReplaceDialog
from ltd.utils.file_utils import get_temp_dir
from ltd.utils.image_utils import load_pixmap_preview
from ltd.widgets.caption_image_list import CaptionImageList
from ltd.widgets.loading_dialog import loading_dialog
from ltd.widgets.tag_completer_popup import TagCompleterPopup
from ltd.widgets.workflow_selector import WorkflowSelector
from ltd.workers.caption_worker import CaptionWorker


# ---------------------------------------------------------------------------
# Captioner backends (unchanged)
# ---------------------------------------------------------------------------

class WdTaggerCaptioner:
    """WD Tagger auto-captioning using ONNX Runtime."""

    MODEL_REPO = 'SmilingWolf/wd-eva02-large-tagger-v3'
    MODELS_BASE = Path('./models/caption')

    def __init__(self, min_probability: float = 0.35, max_tags: int = 50,
                 exclude_tags: list[str] | None = None):
        self.model_dir = self.MODELS_BASE / self.MODEL_REPO
        self.min_probability = min_probability
        self.max_tags = max_tags
        self.exclude_tags = set(exclude_tags or [])
        self._model = None
        self._tags = []
        self._rating_indices = []

    @property
    def name(self) -> str:
        return self.MODEL_REPO

    def _ensure_model(self):
        if self._model is not None:
            return

        import huggingface_hub
        from onnxruntime import InferenceSession

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.model_dir / 'model.onnx'
        tags_path = self.model_dir / 'selected_tags.csv'

        if not model_path.exists():
            model_path = Path(huggingface_hub.hf_hub_download(
                self.MODEL_REPO, filename='model.onnx',
                local_dir=str(self.model_dir)))

        if not tags_path.exists():
            tags_path = Path(huggingface_hub.hf_hub_download(
                self.MODEL_REPO, filename='selected_tags.csv',
                local_dir=str(self.model_dir)))

        self._model = InferenceSession(str(model_path),providers=['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

        self._tags = []
        self._rating_indices = []
        with open(tags_path, 'r') as f:
            reader = csv.DictReader(f)
            for idx, line in enumerate(reader):
                tag = line['name'].replace('_', ' ')
                self._tags.append(tag)
                if line['category'] == '9':
                    self._rating_indices.append(idx)

    def caption(self, image_path: Path) -> list[str]:
        self._ensure_model()
        from PIL import Image as PilImage

        img = PilImage.open(image_path).convert('RGBA')
        canvas = PilImage.new('RGBA', img.size, (255, 255, 255))
        canvas.alpha_composite(img)
        img = canvas.convert('RGB')

        max_dim = max(img.size)
        canvas = PilImage.new('RGB', (max_dim, max_dim), (255, 255, 255))
        h_pad = (max_dim - img.width) // 2
        v_pad = (max_dim - img.height) // 2
        canvas.paste(img, (h_pad, v_pad))

        _, input_dim, *_ = self._model.get_inputs()[0].shape
        if max_dim != input_dim:
            canvas = canvas.resize((input_dim, input_dim),
                                   resample=PilImage.Resampling.BICUBIC)

        arr = np.array(canvas, dtype=np.float32)[:, :, ::-1]
        arr = np.expand_dims(arr, axis=0)

        input_name = self._model.get_inputs()[0].name
        output_name = self._model.get_outputs()[0].name
        probs = self._model.run([output_name], {input_name: arr})[0][0]
        probs = probs.astype(np.float32)

        results = []
        for i, (tag, prob) in enumerate(zip(self._tags, probs)):
            if i in self._rating_indices:
                continue
            if tag in self.exclude_tags:
                continue
            if prob >= self.min_probability:
                results.append((tag, prob))

        results.sort(key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in results[:self.max_tags]]


class ComfyUICaptioner:
    """Captioning via ComfyUI workflow."""

    def __init__(self, workflow_json: str):
        self.workflow_json = workflow_json

    def caption(self, image_path: Path) -> list[str]:
        workflow = load_workflow(self.workflow_json)
        if workflow is None:
            raise ValueError('Invalid workflow JSON')

        valid, msg = validate_caption_workflow(workflow)
        if not valid:
            raise ValueError(msg)

        client = ComfyUIClient()
        if not client.health_check():
            raise ConnectionError('Cannot connect to ComfyUI')

        upload_result = client.upload_image(image_path)
        set_input_image(workflow, upload_result.get('name', image_path.name))

        output_dir = get_temp_dir('comfyui_caption_output')
        result = client.run_workflow(workflow, output_dir)

        if result['texts']:
            text = result['texts'][0]
            return [t.strip() for t in text.split(',') if t.strip()]
        return []


# ---------------------------------------------------------------------------
# Editable Tags List (supports inline edit, delete key, drag-drop)
# ---------------------------------------------------------------------------

class EditableTagsList(QListWidget):
    """Tags list with drag-drop reorder, inline editing, Delete key."""
    tags_changed = Signal()
    tags_delete_requested = Signal(list)  # list of tag strings
    navigate_image = Signal(int)  # -1 = prev, +1 = next

    _HIGHLIGHT_BRUSH = QBrush(QColor(255, 200, 50, 70))
    _CLEAR_BRUSH = QBrush()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed)
        self.model().rowsMoved.connect(lambda: self.tags_changed.emit())
        self.itemChanged.connect(lambda: self.tags_changed.emit())
        self._highlight_patterns: list[tuple] = []
        self._tag_dict: TagDictionary | None = None
        self._dark_theme = True

    def set_tag_dictionary(self, tag_dict: TagDictionary | None,
                           dark: bool = True):
        self._tag_dict = tag_dict
        self._dark_theme = dark

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            tags = [self.item(idx.row()).text()
                    for idx in self.selectedIndexes()]
            if tags:
                self.tags_delete_requested.emit(tags)
            return
        if event.key() in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            direction = 1 if event.key() == Qt.Key.Key_PageDown else -1
            self.navigate_image.emit(direction)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            current = self.currentRow()
            count = self.count()
            if event.key() == Qt.Key.Key_Up and current <= 0:
                self.navigate_image.emit(-1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down and current >= count - 1:
                self.navigate_image.emit(1)
                event.accept()
                return
        super().keyPressEvent(event)

    def get_tags(self) -> list[str]:
        return [self.item(i).text() for i in range(self.count())]

    def set_tags(self, tags: list[str]):
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        self.clear()
        for tag in tags:
            item = QListWidgetItem(tag)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._apply_tag_color(item, tag)
            self.addItem(item)
        self.blockSignals(False)
        self._apply_highlights()
        self.setUpdatesEnabled(True)

    def _apply_tag_color(self, item: QListWidgetItem, tag: str):
        """Apply category color to a tag item if dictionary is available."""
        if self._tag_dict and self._tag_dict.is_loaded():
            color = self._tag_dict.get_color(tag, dark=self._dark_theme)
            if color:
                item.setForeground(QBrush(color))

    # -- search highlight --

    def set_highlight_patterns(self, patterns: list[tuple]):
        """Set tag patterns to highlight. Each is ('tag', value) or ('text', value)."""
        self._highlight_patterns = patterns
        self._apply_highlights()

    def _apply_highlights(self):
        # Block signals: setBackground triggers itemChanged → tags_changed
        # which would cause _rebuild_all_tags per item (O(n*images) each).
        self.blockSignals(True)
        hl = self._HIGHLIGHT_BRUSH if self._highlight_patterns else None
        for i in range(self.count()):
            item = self.item(i)
            if hl and self._tag_matches(item.text()):
                item.setBackground(hl)
            else:
                item.setBackground(self._CLEAR_BRUSH)
        self.blockSignals(False)

    def _tag_matches(self, tag_text: str) -> bool:
        tag_lower = tag_text.strip().lower()
        for kind, value in self._highlight_patterns:
            v = value.lower()
            if kind == 'tag':
                if '*' in v or '?' in v:
                    if fnmatchcase(tag_lower, v):
                        return True
                else:
                    # Exact tag match (same as filter proxy)
                    if tag_lower == v:
                        return True
            elif kind == 'text':
                if v in tag_lower:
                    return True
        return False


# ---------------------------------------------------------------------------
# All Tags List (sort, filter, click actions, delete, rename)
# ---------------------------------------------------------------------------

class AllTagsList(QListWidget):
    """All-tags list with delete key, right-click context menu."""
    delete_tags_requested = Signal(list)  # list of tag names
    add_to_image_requested = Signal(str)  # single tag name
    filter_by_tag_requested = Signal(str)  # single tag name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _get_tag_at(self, item):
        """Extract tag name from display text like 'tag (5)'."""
        text = item.text()
        return text.rsplit(' (', 1)[0] if ' (' in text else text

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        tag = self._get_tag_at(item)

        menu = QMenu(self)
        act_add = menu.addAction(f'Add "{tag}" to image')
        act_filter = menu.addAction(f'Filter images by "{tag}"')
        menu.addSeparator()
        selected = self.selectedIndexes()
        if len(selected) > 1:
            tags = [self._get_tag_at(self.item(idx.row()))
                    for idx in selected]
            act_delete = menu.addAction(
                f'Delete {len(tags)} tags from all images')
        else:
            act_delete = menu.addAction(
                f'Delete "{tag}" from all images')

        action = menu.exec(self.mapToGlobal(pos))
        if action == act_add:
            self.add_to_image_requested.emit(tag)
        elif action == act_filter:
            self.filter_by_tag_requested.emit(tag)
        elif action == act_delete:
            if len(selected) > 1:
                self.delete_tags_requested.emit(tags)
            else:
                self.delete_tags_requested.emit([tag])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.selectAll()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            tags = []
            for idx in self.selectedIndexes():
                text = idx.data()
                tag = text.rsplit(' (', 1)[0] if ' (' in text else text
                tags.append(tag)
            if tags:
                self.delete_tags_requested.emit(tags)
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Tag Input (fixes completer eating the clear on Enter)
# ---------------------------------------------------------------------------

class TagInputField(QLineEdit):
    """Tag input that clears properly after Enter, with popup navigation."""
    tag_submitted = Signal()
    popup_navigate = Signal(int)   # -1 = up, +1 = down
    popup_confirm = Signal()
    popup_cancel = Signal()
    page_navigate = Signal(int)    # -1 = prev image, +1 = next image
    undo_requested = Signal()
    redo_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.popup_confirm.emit()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.popup_cancel.emit()
            return
        # Ctrl+Z / Ctrl+Y → app-level undo/redo (not QLineEdit undo)
        if (event.key() == Qt.Key.Key_Z
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.undo_requested.emit()
            return
        if (event.key() == Qt.Key.Key_Y
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.redo_requested.emit()
            return
        if event.key() == Qt.Key.Key_Down:
            self.popup_navigate.emit(1)
            return
        if event.key() == Qt.Key.Key_Up:
            self.popup_navigate.emit(-1)
            return
        if event.key() == Qt.Key.Key_PageDown:
            self.page_navigate.emit(1)
            return
        if event.key() == Qt.Key.Key_PageUp:
            self.page_navigate.emit(-1)
            return
        if event.key() == Qt.Key.Key_Tab:
            self.popup_confirm.emit()
            event.accept()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Image Viewer (auto-fit, Ctrl+scroll zoom, spacebar+drag pan)
# ---------------------------------------------------------------------------

class CaptionImageViewer(QGraphicsView):
    """Image viewer with auto-fit, Ctrl+scroll zoom, Space+drag pan."""
    navigate_image = Signal(int)  # -1 = prev, +1 = next

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap_item = None
        self._manually_zoomed = False

    def load_image(self, pixmap):
        self._scene.clear()
        self._pixmap_item = None
        if pixmap:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            # Reset scene rect to this pixmap's bounds so fitInView
            # doesn't use a stale rect from a previously larger image.
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
            self._manually_zoomed = False
            self._fit_image()
        else:
            self._scene.setSceneRect(0, 0, 0, 0)

    def _fit_image(self):
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)
            self._manually_zoomed = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._manually_zoomed and self._pixmap_item:
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            self._manually_zoomed = True
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return
        # Ctrl+0 to reset zoom to fit
        if (event.key() == Qt.Key.Key_0
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._fit_image()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_PageDown, Qt.Key.Key_Down):
            self.navigate_image.emit(1)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_PageUp, Qt.Key.Key_Up):
            self.navigate_image.emit(-1)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)


# ---------------------------------------------------------------------------
# Caption Tab
# ---------------------------------------------------------------------------

class CaptionTab(QWidget):
    _PIXMAP_CACHE_MAX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self._current_image_index = -1
        self._all_tags: dict[str, int] = {}  # tag -> count
        self._worker: CaptionWorker | None = None
        self._pixmap_cache: OrderedDict[str, object] = OrderedDict()

        # Undo/redo stacks (per image index)
        self._undo_stacks: dict[int, list[list[str]]] = {}
        self._redo_stacks: dict[int, list[list[str]]] = {}
        self._pre_tags_snapshot: list[str] | None = None
        self._max_undo = 50
        self._skip_snapshot = False

        # Tag dictionary for autocomplete + colors
        self._tag_dict = TagDictionary()
        csv_path = Path('tags.csv')
        if csv_path.exists():
            self._tag_dict.load_csv(csv_path)

        self._setup_ui()
        self._tag_popup = TagCompleterPopup(self._tag_dict, self)
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        self.tags_list.set_tag_dictionary(self._tag_dict, dark=dark)
        self._connect_signals()
        self._setup_shortcuts()
        self._restore_settings()
        self._connect_settings_persistence()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left: Image list (with filter, multi-select, context menu) ---
        self.image_list = CaptionImageList(self.model)
        splitter.addWidget(self.image_list)

        # --- Center: Image viewer (auto-fit, Ctrl+scroll zoom, Space+drag) -
        self.image_viewer = CaptionImageViewer()
        splitter.addWidget(self.image_viewer)

        # --- Right: Tabbed panel ---
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.right_tabs = QTabWidget()

        # ---- Tab: Image Tags ----
        image_tags_tab = QWidget()
        tag_layout = QVBoxLayout(image_tags_tab)

        input_layout = QHBoxLayout()
        self.tag_input = TagInputField()
        self.tag_input.setPlaceholderText('Add tag (Enter to add)...')
        input_layout.addWidget(self.tag_input)
        self.add_tag_btn = QPushButton('Add')
        input_layout.addWidget(self.add_tag_btn)
        tag_layout.addLayout(input_layout)

        self.tags_list = EditableTagsList()
        tag_layout.addWidget(self.tags_list)

        tag_btn_layout = QHBoxLayout()
        self.remove_tag_btn = QPushButton('Remove Selected')
        self.clear_tags_btn = QPushButton('Clear All')
        self.remove_dupes_btn = QPushButton('Remove Duplicates')
        tag_btn_layout.addWidget(self.remove_tag_btn)
        tag_btn_layout.addWidget(self.clear_tags_btn)
        tag_btn_layout.addWidget(self.remove_dupes_btn)
        tag_layout.addLayout(tag_btn_layout)

        self.token_count_label = QLabel('Tags: 0 | ~0 tokens')
        tag_layout.addWidget(self.token_count_label)

        self.right_tabs.addTab(image_tags_tab, 'Image Tags')

        # ---- Tab: All Tags ----
        all_tags_tab = QWidget()
        all_tags_layout = QVBoxLayout(all_tags_tab)

        self.all_tags_filter = QLineEdit()
        self.all_tags_filter.setPlaceholderText('Filter tags...')
        self.all_tags_filter.setClearButtonEnabled(True)
        all_tags_layout.addWidget(self.all_tags_filter)

        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel('Sort:'))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(['Frequency', 'Name', 'Length'])
        sort_layout.addWidget(self.sort_combo)
        self.sort_order_combo = QComboBox()
        self.sort_order_combo.addItems(['Descending', 'Ascending'])
        sort_layout.addWidget(self.sort_order_combo)
        all_tags_layout.addLayout(sort_layout)

        self.all_tags_list = AllTagsList()
        all_tags_layout.addWidget(self.all_tags_list)

        self.all_tags_count_label = QLabel('0 tags')
        all_tags_layout.addWidget(self.all_tags_count_label)

        self.right_tabs.addTab(all_tags_tab, 'All Tags')

        # ---- Tab: Auto-Caption ----
        caption_tab_w = QWidget()
        caption_layout = QVBoxLayout(caption_tab_w)

        self.captioner_combo = QComboBox()
        self.captioner_combo.addItems(
            [WdTaggerCaptioner.MODEL_REPO, 'ComfyUI Workflow'])
        caption_layout.addWidget(self.captioner_combo)

        # WD Tagger settings
        self.wd_settings = QWidget()
        wd_layout = QVBoxLayout(self.wd_settings)
        wd_layout.setContentsMargins(0, 0, 0, 0)

        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel('Min probability:'))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.01, 1.0)
        self.prob_spin.setValue(0.35)
        self.prob_spin.setSingleStep(0.05)
        prob_layout.addWidget(self.prob_spin)
        wd_layout.addLayout(prob_layout)

        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel('Max tags:'))
        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(1, 200)
        self.max_tags_spin.setValue(50)
        max_layout.addWidget(self.max_tags_spin)
        wd_layout.addLayout(max_layout)

        self.exclude_input = QLineEdit()
        self.exclude_input.setPlaceholderText(
            'Tags to exclude (comma-separated)...')
        wd_layout.addWidget(self.exclude_input)

        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel('Position:'))
        self.caption_position_combo = QComboBox()
        self.caption_position_combo.addItems([
            'Before existing tags',
            'After existing tags',
            'Overwrite all tags',
        ])
        self.caption_position_combo.setCurrentIndex(1)
        pos_layout.addWidget(self.caption_position_combo)
        wd_layout.addLayout(pos_layout)

        caption_layout.addWidget(self.wd_settings)

        # ComfyUI workflow
        self.comfy_settings = QWidget()
        comfy_layout_inner = QVBoxLayout(self.comfy_settings)
        comfy_layout_inner.setContentsMargins(0, 0, 0, 0)
        self._caption_workflow_selector = WorkflowSelector(
            settings_key='caption')
        comfy_layout_inner.addWidget(self._caption_workflow_selector)
        info = QLabel('Required nodes: LTD_Input_Image, LTD_Output_Text')
        info.setWordWrap(True)
        comfy_layout_inner.addWidget(info)
        caption_layout.addWidget(self.comfy_settings)
        self.comfy_settings.setVisible(False)

        cap_btn_layout = QHBoxLayout()
        self.caption_current_btn = QPushButton('Current')
        self.caption_selected_btn = QPushButton('Selected')
        self.caption_all_btn = QPushButton('All')
        self.caption_cancel_btn = QPushButton('Cancel')
        self.caption_cancel_btn.setVisible(False)
        cap_btn_layout.addWidget(QLabel('Caption:'))
        cap_btn_layout.addWidget(self.caption_current_btn)
        cap_btn_layout.addWidget(self.caption_selected_btn)
        cap_btn_layout.addWidget(self.caption_all_btn)
        cap_btn_layout.addWidget(self.caption_cancel_btn)
        caption_layout.addLayout(cap_btn_layout)

        self.caption_progress = QProgressBar()
        self.caption_progress.setVisible(False)
        caption_layout.addWidget(self.caption_progress)

        caption_layout.addStretch()
        self.right_tabs.addTab(caption_tab_w, 'Auto-Caption')

        # ---- Tab: Tools ----
        tools_tab = QWidget()
        tools_layout = QVBoxLayout(tools_tab)

        tools_row1 = QHBoxLayout()
        self.find_replace_btn = QPushButton('Find && Replace  (Ctrl+R)')
        self.batch_reorder_btn = QPushButton('Batch Reorder  (Ctrl+B)')
        tools_row1.addWidget(self.find_replace_btn)
        tools_row1.addWidget(self.batch_reorder_btn)
        tools_layout.addLayout(tools_row1)

        tools_row2 = QHBoxLayout()
        self.remove_empty_btn = QPushButton('Remove Empty  (Ctrl+E)')
        self.remove_dupes_all_btn = QPushButton('Deduplicate All  (Ctrl+D)')
        tools_row2.addWidget(self.remove_empty_btn)
        tools_row2.addWidget(self.remove_dupes_all_btn)
        tools_layout.addLayout(tools_row2)

        tools_row3 = QHBoxLayout()
        self.snapshot_save_btn = QPushButton('Create Captions Snapshot...')
        self.snapshot_restore_btn = QPushButton('Restore Captions from Snapshot...')
        tools_row3.addWidget(self.snapshot_save_btn)
        tools_row3.addWidget(self.snapshot_restore_btn)
        tools_layout.addLayout(tools_row3)

        tools_layout.addStretch()
        self.right_tabs.addTab(tools_tab, 'Tools')

        right_layout.addWidget(self.right_tabs)

        # Separator config
        sep_layout = QHBoxLayout()
        sep_layout.addWidget(QLabel('Separator:'))
        self.separator_input = QLineEdit(', ')
        self.separator_input.setFixedWidth(80)
        self.separator_input.setToolTip(
            'Tag separator for reading/writing files. '
            'Supports escape sequences: \\n (newline), \\t (tab)')
        sep_layout.addWidget(self.separator_input)
        self.reload_tags_btn = QPushButton('Reload Tags')
        self.reload_tags_btn.setToolTip(
            'Re-read all tag files using the current separator')
        sep_layout.addWidget(self.reload_tags_btn)
        sep_layout.addStretch()
        right_layout.addLayout(sep_layout)

        # Save/export (outside tabs)
        self.save_captions_btn = QPushButton('Save All Captions  (Ctrl+S)')
        right_layout.addWidget(self.save_captions_btn)

        self.export_captions_btn = QPushButton('Export Images + Captions...')
        right_layout.addWidget(self.export_captions_btn)

        self.caption_status = QLabel('')
        right_layout.addWidget(self.caption_status)

        splitter.addWidget(right)
        splitter.setSizes([200, 500, 300])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # Image list
        self.image_list.current_changed.connect(self._on_image_changed)
        self.image_list.load_directory_requested.connect(self._load_directory)
        self.image_list.tags_paste_requested.connect(self._on_tags_pasted)
        self.image_list.images_deleted.connect(self._on_images_deleted)
        self.image_list.filter_input.textChanged.connect(
            self._update_tag_highlights)

        # Tag input + autocomplete popup
        self.add_tag_btn.clicked.connect(self._add_tag)
        self.tag_input.tag_submitted.connect(self._add_tag)
        self.tag_input.textChanged.connect(self._on_tag_input_changed)
        self.tag_input.popup_navigate.connect(self._on_popup_navigate)
        self.tag_input.popup_confirm.connect(self._on_popup_confirm)
        self.tag_input.popup_cancel.connect(self._tag_popup.hide_popup)
        self.tag_input.page_navigate.connect(self._on_page_navigate)
        self.tag_input.undo_requested.connect(self._undo)
        self.tag_input.redo_requested.connect(self._redo)
        self._tag_popup.tag_selected.connect(self._on_popup_tag_selected)

        # Tag list actions
        self.remove_tag_btn.clicked.connect(self._remove_selected_tags)
        self.clear_tags_btn.clicked.connect(self._clear_tags)
        self.remove_dupes_btn.clicked.connect(self._remove_duplicates_current)
        self.tags_list.tags_changed.connect(self._on_tags_list_changed)
        self.tags_list.tags_delete_requested.connect(self._handle_tag_delete)
        self.tags_list.navigate_image.connect(self._on_page_navigate)

        # Image viewer navigation
        self.image_viewer.navigate_image.connect(self._on_page_navigate)

        # All tags
        self.all_tags_filter.textChanged.connect(self._update_all_tags_display)
        self.sort_combo.currentIndexChanged.connect(
            self._update_all_tags_display)
        self.sort_order_combo.currentIndexChanged.connect(
            self._update_all_tags_display)
        self.all_tags_list.itemDoubleClicked.connect(
            self._on_all_tag_double_clicked)
        self.all_tags_list.add_to_image_requested.connect(
            self._add_tag_to_image)
        self.all_tags_list.filter_by_tag_requested.connect(
            self._filter_by_tag)
        self.all_tags_list.delete_tags_requested.connect(
            self._delete_tags_from_all_images)

        # Captioner
        self.captioner_combo.currentIndexChanged.connect(
            self._on_captioner_changed)
        self.caption_current_btn.clicked.connect(
            lambda: self._run_captioning(mode='current'))
        self.caption_selected_btn.clicked.connect(
            lambda: self._run_captioning(mode='selected'))
        self.caption_all_btn.clicked.connect(
            lambda: self._run_captioning(mode='all'))
        self.caption_cancel_btn.clicked.connect(self._cancel_captioning)

        # Tools
        self.find_replace_btn.clicked.connect(self._show_find_replace)
        self.batch_reorder_btn.clicked.connect(self._show_batch_reorder)
        self.remove_empty_btn.clicked.connect(self._remove_empty_all)
        self.snapshot_save_btn.clicked.connect(self._save_captions_snapshot)
        self.snapshot_restore_btn.clicked.connect(self._restore_captions_snapshot)
        self.remove_dupes_all_btn.clicked.connect(
            self._remove_duplicates_all)

        # Save / separator
        self.save_captions_btn.clicked.connect(self._save_all_captions)
        self.export_captions_btn.clicked.connect(self._export_captions)
        self.reload_tags_btn.clicked.connect(self._reload_tags)

    def _setup_shortcuts(self):
        # Ctrl+S to save
        save_sc = QShortcut(QKeySequence.StandardKey.Save, self)
        save_sc.activated.connect(self._save_all_captions)

        # Ctrl+R / Ctrl+H for find & replace
        fr_sc = QShortcut(QKeySequence('Ctrl+R'), self)
        fr_sc.activated.connect(self._show_find_replace)
        fr_sc2 = QShortcut(QKeySequence('Ctrl+H'), self)
        fr_sc2.activated.connect(self._show_find_replace)

        # Ctrl+D deduplicate all
        dedup_sc = QShortcut(QKeySequence('Ctrl+D'), self)
        dedup_sc.activated.connect(self._remove_duplicates_all)

        # Ctrl+E remove empty tags
        empty_sc = QShortcut(QKeySequence('Ctrl+E'), self)
        empty_sc.activated.connect(self._remove_empty_all)

        # Ctrl+B batch reorder
        reorder_sc = QShortcut(QKeySequence('Ctrl+B'), self)
        reorder_sc.activated.connect(self._show_batch_reorder)

        # Undo / Redo
        undo_sc = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_sc.activated.connect(self._undo)
        redo_sc = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_sc.activated.connect(self._redo)
        redo_sc2 = QShortcut(QKeySequence('Ctrl+Y'), self)
        redo_sc2.activated.connect(self._redo)

        # Left/Right arrow navigation
        prev_sc = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        prev_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        prev_sc.activated.connect(self._navigate_previous)
        next_sc = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        next_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        next_sc.activated.connect(self._navigate_next)

        # PgUp/PgDown — always navigate images
        pgup_sc = QShortcut(QKeySequence('PgUp'), self)
        pgup_sc.activated.connect(lambda: self._on_page_navigate(-1))
        pgdn_sc = QShortcut(QKeySequence('PgDown'), self)
        pgdn_sc.activated.connect(lambda: self._on_page_navigate(1))

    def _navigate_previous(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit):
            return
        if (isinstance(focus, QAbstractItemView)
                and focus.state() == QAbstractItemView.State.EditingState):
            return
        self.image_list.go_to_previous()

    def _navigate_next(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit):
            return
        if (isinstance(focus, QAbstractItemView)
                and focus.state() == QAbstractItemView.State.EditingState):
            return
        self.image_list.go_to_next()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = get_settings()
        self.prob_spin.setValue(
            s.value('caption/min_probability', 0.35, type=float))
        self.max_tags_spin.setValue(
            s.value('caption/max_tags', 50, type=int))
        self.exclude_input.setText(
            s.value('caption/exclude_tags', '', type=str))
        self.caption_position_combo.setCurrentIndex(
            s.value('caption/position', 1, type=int))
        self.captioner_combo.setCurrentIndex(
            s.value('caption/captioner', 0, type=int))
        self.separator_input.setText(
            s.value('caption/tag_separator', ', ', type=str))

    def _save_setting(self, key, value):
        get_settings().setValue(f'caption/{key}', value)

    def _connect_settings_persistence(self):
        self.prob_spin.valueChanged.connect(
            lambda v: self._save_setting('min_probability', v))
        self.max_tags_spin.valueChanged.connect(
            lambda v: self._save_setting('max_tags', v))
        self.exclude_input.editingFinished.connect(
            lambda: self._save_setting('exclude_tags',
                                       self.exclude_input.text()))
        self.caption_position_combo.currentIndexChanged.connect(
            lambda v: self._save_setting('position', v))
        self.captioner_combo.currentIndexChanged.connect(
            lambda v: self._save_setting('captioner', v))
        self.separator_input.editingFinished.connect(
            lambda: self._save_setting('tag_separator',
                                       self.separator_input.text()))

    def _get_separator(self) -> str:
        """Read separator from input, decode escape sequences."""
        raw = self.separator_input.text()
        if not raw:
            return ', '
        try:
            return raw.encode('utf-8').decode('unicode_escape')
        except (UnicodeDecodeError, ValueError):
            return raw

    # ------------------------------------------------------------------
    # Auto-save
    # ------------------------------------------------------------------

    def _auto_save_image(self, image: ImageItem):
        """Save a single image's tags to disk immediately."""
        try:
            image.save_tags_to_file(separator=self._get_separator())
        except OSError as e:
            self.caption_status.setText(f'Save error: {e}')

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _push_undo(self, image_index: int | None = None,
                   snapshot: list[str] | None = None):
        """Push a snapshot to the undo stack for the given image."""
        idx = image_index if image_index is not None \
            else self._current_image_index
        if idx < 0:
            return
        if snapshot is None:
            image = self.model.get_image(idx)
            if image is None:
                return
            snapshot = list(image.tags)
        stack = self._undo_stacks.setdefault(idx, [])
        stack.append(snapshot)
        if len(stack) > self._max_undo:
            stack.pop(0)
        # Clear redo on new action
        self._redo_stacks.pop(idx, None)

    def _undo(self):
        idx = self._current_image_index
        if idx < 0:
            return
        stack = self._undo_stacks.get(idx, [])
        if not stack:
            return
        image = self.model.get_image(idx)
        if image is None:
            return
        # Push current state to redo
        redo = self._redo_stacks.setdefault(idx, [])
        redo.append(list(image.tags))
        # Restore from undo
        image.tags = stack.pop()
        self._pre_tags_snapshot = list(image.tags)
        self._skip_snapshot = True
        self.tags_list.set_tags(image.tags)
        self._skip_snapshot = False
        self._rebuild_all_tags()
        self._update_token_count()
        self._auto_save_image(image)

    def _redo(self):
        idx = self._current_image_index
        if idx < 0:
            return
        stack = self._redo_stacks.get(idx, [])
        if not stack:
            return
        image = self.model.get_image(idx)
        if image is None:
            return
        # Push current state to undo
        undo = self._undo_stacks.setdefault(idx, [])
        undo.append(list(image.tags))
        # Restore from redo
        image.tags = stack.pop()
        self._pre_tags_snapshot = list(image.tags)
        self._skip_snapshot = True
        self.tags_list.set_tags(image.tags)
        self._skip_snapshot = False
        self._rebuild_all_tags()
        self._update_token_count()
        self._auto_save_image(image)

    def _clear_undo_stacks(self):
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        self._pre_tags_snapshot = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_modify_tab(self, items: list[ImageItem]):
        self._pixmap_cache.clear()
        self._clear_undo_stacks()
        self.model.load_items(items)
        sep = self._get_separator()
        for image in self.model.images:
            image.load_tags_from_file(separator=sep)
        self._rebuild_all_tags()
        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _load_directory(self, path: str):
        directory = Path(path)
        sep = self._get_separator()

        with loading_dialog('Loading images...', self) as dlg:
            self._pixmap_cache.clear()
            self._clear_undo_stacks()
            self.model.load_directory(directory)

            # Filter out mask files
            self.model.images = [
                img for img in self.model.images
                if '-masklabel' not in img.name and '-mask' not in img.name
            ]

            dlg.set_message('Loading tags...')
            QApplication.processEvents()
            for image in self.model.images:
                image.load_tags_from_file(separator=sep)
            self._rebuild_all_tags()

        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _reload_tags(self):
        """Re-read all tag files using the current separator."""
        sep = self._get_separator()
        for image in self.model.images:
            image.load_tags_from_file(separator=sep)
        self._rebuild_all_tags()
        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self.tags_list.set_tags(image.tags)
                self._update_token_count()
        self.caption_status.setText(
            f'Reloaded tags for {len(self.model.images)} image(s)')

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def _on_image_changed(self, index: int):
        self._save_current_tags()
        self._current_image_index = index
        image = self.model.get_image(index)
        if image is None:
            return

        # Snapshot for undo tracking
        self._pre_tags_snapshot = list(image.tags)

        pixmap = self._load_cached(image.path)
        self.image_viewer.load_image(pixmap)

        # Show tags
        self._skip_snapshot = True
        self.tags_list.set_tags(image.tags)
        self._skip_snapshot = False
        self._update_token_count()
        self._update_tag_highlights()

        # Preload adjacent images after current frame finishes
        QTimer.singleShot(0, lambda idx=index: self._preload_adjacent(idx))

    def _on_images_deleted(self):
        """Refresh UI after images were deleted from disk and model."""
        self._current_image_index = -1
        current_row = self.image_list.current_source_row()
        if current_row >= 0:
            self._on_image_changed(current_row)
        else:
            self.image_viewer.load_image(None)
            self.tags_list.set_tags([])
        self._rebuild_all_tags()
        self._update_token_count()

    def _preview_max_dim(self) -> int:
        vp = self.image_viewer.viewport().size()
        return max(vp.width(), vp.height(), 800) * 2

    def _load_cached(self, path) -> object:
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
            return  # user already moved on
        for offset in (-1, 1):
            image = self.model.get_image(index + offset)
            if image and str(image.path) not in self._pixmap_cache:
                self._load_cached(image.path)

    def _update_tag_highlights(self):
        """Highlight tags in the Image Tags list that match the current filter."""
        text = self.image_list.filter_input.text().strip()
        if not text:
            self.tags_list.set_highlight_patterns([])
            return
        proxy = self.image_list.proxy
        tokens = proxy._tokenize(text)
        patterns = []
        for token in tokens:
            if token.lower() in proxy._KEYWORDS:
                continue
            if token.startswith('-') and len(token) > 1:
                token = token[1:]
            if ':' in token:
                key, _, value = token.partition(':')
                if key.lower() == 'tag' and value:
                    patterns.append(('tag', value))
            else:
                patterns.append(('text', token))
        self.tags_list.set_highlight_patterns(patterns)

    def _save_current_tags(self):
        if self._current_image_index < 0:
            return
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        image.tags = self.tags_list.get_tags()

    def _on_tags_pasted(self, tags: list[str], source_rows: list[int]):
        """Handle paste/clear from context menu on selected images."""
        for row in source_rows:
            image = self.model.get_image(row)
            if image is None:
                continue
            self._push_undo(row, list(image.tags))
            if tags:
                image.tags = image.tags + tags
            self._auto_save_image(image)

        # Refresh current image display if affected
        if self._current_image_index in source_rows:
            image = self.model.get_image(self._current_image_index)
            if image:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
                self._update_token_count()
        self._rebuild_all_tags()
        action = 'Pasted' if tags else 'Cleared'
        self.caption_status.setText(
            f'{action} tags on {len(source_rows)} image(s)')

    # ------------------------------------------------------------------
    # Tag editing (current image)
    # ------------------------------------------------------------------

    def _on_tag_input_changed(self, text: str):
        """Show autocomplete popup as user types."""
        if not text.strip():
            self._tag_popup.hide_popup()
            return
        image = self.model.get_image(self._current_image_index)
        current_tags = set(image.tags) if image else set()
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        self._tag_popup.set_session_tags(list(self._all_tags.keys()))
        self._tag_popup.show_for(self.tag_input, text, current_tags, dark=dark)

    def _on_popup_navigate(self, direction: int):
        """Handle Up/Down in tag input for popup or tags list navigation."""
        if self._tag_popup.isVisible():
            if direction > 0:
                self._tag_popup.select_next()
            else:
                self._tag_popup.select_previous()
            return
        # Navigate through image tags list
        count = self.tags_list.count()
        if count == 0:
            # No tags — move to prev/next image
            if direction > 0:
                self.image_list.go_to_next()
            else:
                self.image_list.go_to_previous()
            return
        current = self.tags_list.currentRow()
        if direction > 0:  # Down
            if current < count - 1:
                nxt = current + 1 if current >= 0 else 0
                self.tags_list.setCurrentRow(nxt)
                self.tags_list.scrollToItem(self.tags_list.item(nxt))
            else:
                # Past last tag — next image
                self.image_list.go_to_next()
        else:  # Up
            if current > 0:
                self.tags_list.setCurrentRow(current - 1)
                self.tags_list.scrollToItem(
                    self.tags_list.item(current - 1))
            else:
                # First tag or no selection — previous image
                self.image_list.go_to_previous()

    def _on_page_navigate(self, direction: int):
        """Handle PgUp/PgDown — always move to prev/next image."""
        if direction > 0:
            self.image_list.go_to_next()
        else:
            self.image_list.go_to_previous()

    def _on_popup_confirm(self):
        """Handle Enter/Tab in tag input — select from popup or submit."""
        if self._tag_popup.isVisible():
            self._tag_popup.confirm_selection()
        else:
            self._add_tag()
            self.tag_input.clear()

    def _on_popup_tag_selected(self, tag: str):
        """Insert tag from popup into the tags list (respects multi-select)."""
        self.tag_input.setText(tag)
        self._add_tag()
        self.tag_input.setFocus()

    def _add_tag(self):
        text = self.tag_input.text().strip()
        if not text:
            return
        new_tags = [t.strip() for t in text.split(',') if t.strip()]

        selected_rows = self.image_list.selected_source_rows()
        if len(selected_rows) > 1:
            n = len(selected_rows)
            reply = QMessageBox.question(
                self, 'Add tag',
                f'Add tag to {n} selected images?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._save_current_tags()
            for row in selected_rows:
                image = self.model.get_image(row)
                if image is None:
                    continue
                self._push_undo(row, list(image.tags))
                for tag in new_tags:
                    if tag not in image.tags:
                        image.tags.append(tag)
                self._auto_save_image(image)
            self.tag_input.clear()
            # Refresh displayed tags for current image
            image = self.model.get_image(self._current_image_index)
            if image:
                self.tags_list.set_tags(image.tags)
            self._rebuild_all_tags()
            self._update_token_count()
            return

        # Single image: add to tags list widget
        for tag in new_tags:
            item = QListWidgetItem(tag)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.tags_list._apply_tag_color(item, tag)
            self.tags_list.addItem(item)
        self.tag_input.clear()
        if self.tags_list.count() > 0:
            self.tags_list.scrollToBottom()
            self.tags_list.setCurrentRow(self.tags_list.count() - 1)
        self._on_tags_list_changed()

    def _remove_selected_tags(self):
        tags = [self.tags_list.item(idx.row()).text()
                for idx in self.tags_list.selectedIndexes()]
        if tags:
            self._handle_tag_delete(tags)

    def _handle_tag_delete(self, tags: list[str]):
        """Handle tag deletion — ask scope if multiple images selected."""
        selected_rows = self.image_list.selected_source_rows()
        if len(selected_rows) <= 1:
            self._do_delete_tags_current(tags)
            return

        msg = QMessageBox(self)
        msg.setWindowTitle('Delete Tags')
        msg.setText(f'Delete from all {len(selected_rows)} selected images?')
        yes_btn = msg.addButton('Yes, all selected',
                                QMessageBox.ButtonRole.YesRole)
        current_btn = msg.addButton('No, only current',
                                    QMessageBox.ButtonRole.NoRole)
        msg.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == yes_btn:
            tag_set = set(tags)
            count = 0
            for row in selected_rows:
                image = self.model.get_image(row)
                if image is None:
                    continue
                if tag_set & set(image.tags):
                    self._push_undo(row, list(image.tags))
                    image.tags = [t for t in image.tags if t not in tag_set]
                    self._auto_save_image(image)
                    count += 1
            # Refresh current image display
            image = self.model.get_image(self._current_image_index)
            if image:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
            self._rebuild_all_tags()
            self._update_token_count()
            self.caption_status.setText(
                f'Deleted tags from {count} image(s)')
        elif clicked == current_btn:
            self._do_delete_tags_current(tags)

    def _do_delete_tags_current(self, tags: list[str]):
        """Delete tags from the current image's tag list widget."""
        tag_set = set(tags)
        rows = sorted(
            [i for i in range(self.tags_list.count())
             if self.tags_list.item(i).text() in tag_set],
            reverse=True)
        for row in rows:
            self.tags_list.takeItem(row)
        # Select nearest remaining
        if rows:
            new_row = min(rows)
            if new_row >= self.tags_list.count():
                new_row = self.tags_list.count() - 1
            if new_row >= 0:
                self.tags_list.setCurrentRow(new_row)
        self._on_tags_list_changed()

    def _clear_tags(self):
        if self.tags_list.count() == 0:
            return
        reply = QMessageBox.question(
            self, 'Clear Tags',
            'Remove all tags from current image?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.tags_list.clear()
            self._on_tags_list_changed()

    def _remove_duplicates_current(self):
        tags = self.tags_list.get_tags()
        seen = set()
        unique = []
        for tag in tags:
            key = tag.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(tag)
        if len(unique) < len(tags):
            self.tags_list.set_tags(unique)
            self._on_tags_list_changed()
            self.caption_status.setText(
                f'Removed {len(tags) - len(unique)} duplicate(s)')

    def _on_tags_list_changed(self):
        """Called when tags list changes (edit, move, delete)."""
        if not self._skip_snapshot and self._pre_tags_snapshot is not None:
            self._push_undo(snapshot=list(self._pre_tags_snapshot))
        self._save_current_tags()
        # Update pre-snapshot to current state
        image = self.model.get_image(self._current_image_index)
        if image:
            self._pre_tags_snapshot = list(image.tags)
            self._auto_save_image(image)
        self._rebuild_all_tags()
        self._update_token_count()

    def _update_token_count(self):
        count = self.tags_list.count()
        text = ', '.join(self.tags_list.get_tags())
        # Rough CLIP token estimate: ~4 chars per token
        tokens = len(text) // 4 if text else 0
        label = f'Tags: {count} | ~{tokens} tokens'
        if tokens > 75:
            label += ' (over 75!)'
        self.token_count_label.setText(label)

    # ------------------------------------------------------------------
    # All tags
    # ------------------------------------------------------------------

    def _rebuild_all_tags(self):
        self._all_tags.clear()
        for image in self.model.images:
            for tag in image.tags:
                self._all_tags[tag] = self._all_tags.get(tag, 0) + 1
        self._update_all_tags_display()

    def _update_all_tags_display(self):
        filter_text = self.all_tags_filter.text().lower()
        sort_mode = self.sort_combo.currentIndex()
        ascending = self.sort_order_combo.currentIndex() == 1

        items = list(self._all_tags.items())
        if filter_text:
            items = [(tag, cnt) for tag, cnt in items
                     if filter_text in tag.lower()]

        if sort_mode == 0:  # Frequency
            items.sort(key=lambda x: x[1], reverse=not ascending)
        elif sort_mode == 1:  # Name
            items.sort(key=lambda x: x[0].lower(), reverse=not ascending)
        elif sort_mode == 2:  # Length
            items.sort(key=lambda x: len(x[0]), reverse=not ascending)

        self.all_tags_list.clear()
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        for tag, count in items:
            item = QListWidgetItem(f'{tag} ({count})')
            if self._tag_dict.is_loaded():
                color = self._tag_dict.get_color(tag, dark=dark)
                if color:
                    item.setForeground(QBrush(color))
            self.all_tags_list.addItem(item)

        filtered = len(items)
        total = len(self._all_tags)
        if filter_text:
            self.all_tags_count_label.setText(
                f'{filtered} / {total} tags')
        else:
            self.all_tags_count_label.setText(f'{total} tags')

    def _add_tag_to_image(self, tag: str):
        new_item = QListWidgetItem(tag)
        new_item.setFlags(new_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.tags_list._apply_tag_color(new_item, tag)
        self.tags_list.addItem(new_item)
        self.tags_list.scrollToBottom()
        self._on_tags_list_changed()

    def _filter_by_tag(self, tag: str):
        if ' ' in tag:
            self.image_list.filter_input.setText(f'tag:"{tag}"')
        else:
            self.image_list.filter_input.setText(f'tag:{tag}')

    def _on_all_tag_double_clicked(self, item: QListWidgetItem):
        """Double-click on all-tags: rename across all images."""
        tag = item.text().rsplit(' (', 1)[0]
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, 'Rename Tag', f'Rename "{tag}" to:', text=tag)
        if ok and new_name.strip() and new_name.strip() != tag:
            new_name = new_name.strip()
            count = 0
            for i, image in enumerate(self.model.images):
                if tag in image.tags:
                    self._push_undo(i, list(image.tags))
                    image.tags = [new_name if t == tag else t
                                  for t in image.tags]
                    self._auto_save_image(image)
                    count += 1
            self._rebuild_all_tags()
            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self._pre_tags_snapshot = list(image.tags)
                    self._skip_snapshot = True
                    self.tags_list.set_tags(image.tags)
                    self._skip_snapshot = False
            self.caption_status.setText(
                f'Renamed "{tag}" to "{new_name}" in {count} image(s)')

    def _delete_tags_from_all_images(self, tags: list[str]):
        """Delete tags from all images (triggered by Delete key in all-tags)."""
        tag_set = set(tags)
        names = ', '.join(f'"{t}"' for t in tags[:5])
        if len(tags) > 5:
            names += f' and {len(tags) - 5} more'
        reply = QMessageBox.question(
            self, 'Delete Tags',
            f'Delete {names} from all images?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        count = 0
        for i, image in enumerate(self.model.images):
            before = len(image.tags)
            if tag_set & set(image.tags):
                self._push_undo(i, list(image.tags))
                image.tags = [t for t in image.tags if t not in tag_set]
                self._auto_save_image(image)
            count += before - len(image.tags)

        self._rebuild_all_tags()
        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
        self._update_token_count()
        self.caption_status.setText(
            f'Deleted {count} tag instance(s) across all images')

    # ------------------------------------------------------------------
    # Auto-captioning
    # ------------------------------------------------------------------

    def _on_captioner_changed(self, index):
        self.wd_settings.setVisible(index == 0)
        self.comfy_settings.setVisible(index == 1)

    def _create_captioner(self):
        if self.captioner_combo.currentIndex() == 0:
            exclude_text = self.exclude_input.text().strip()
            exclude_tags = [t.strip() for t in exclude_text.split(',')
                           if t.strip()] if exclude_text else None
            return WdTaggerCaptioner(
                min_probability=self.prob_spin.value(),
                max_tags=self.max_tags_spin.value(),
                exclude_tags=exclude_tags,
            )
        else:
            workflow_text = self._caption_workflow_selector.get_workflow_text()
            if not workflow_text:
                QMessageBox.warning(self, 'Warning', 'No workflow provided.')
                return None
            return ComfyUICaptioner(workflow_text)

    def _merge_tags(self, existing: list[str], new_tags: list[str]) -> list[str]:
        """Merge new tags based on caption position setting."""
        pos = self.caption_position_combo.currentIndex()
        if pos == 0:  # Before existing
            return new_tags + existing
        elif pos == 1:  # After existing
            return existing + new_tags
        else:  # Overwrite
            return new_tags

    def _run_captioning(self, mode: str = 'all'):
        captioner = self._create_captioner()
        if captioner is None:
            return

        if mode == 'current':
            image = self.model.get_image(self._current_image_index)
            if image is None:
                return
            images = [image]
        elif mode == 'selected':
            images = self.image_list.get_selected_images()
            if not images:
                QMessageBox.information(
                    self, 'Info', 'No images selected.')
                return
        else:
            images = list(self.model.images)

        self._caption_images = images
        self._worker = CaptionWorker(captioner, images)
        self._worker.progress.connect(self._on_caption_progress)
        self._worker.status.connect(self._on_caption_status)
        self._worker.caption_complete.connect(self._on_caption_result)
        self._worker.finished_work.connect(self._on_caption_finished)
        self._worker.error.connect(self._on_caption_error)

        show_progress = len(images) > 1
        self.caption_progress.setVisible(show_progress)
        self.caption_cancel_btn.setVisible(True)
        self.caption_current_btn.setEnabled(False)
        self.caption_selected_btn.setEnabled(False)
        self.caption_all_btn.setEnabled(False)
        self._worker.start()

    def _cancel_captioning(self):
        if self._worker:
            self._worker.cancel()

    def _on_caption_progress(self, current, total):
        self.caption_progress.setMaximum(total)
        self.caption_progress.setValue(current)

    def _on_caption_status(self, text):
        self.caption_status.setText(text)

    def _on_caption_result(self, index, tags):
        images = getattr(self, '_caption_images', self.model.images)
        if 0 <= index < len(images):
            image = images[index]
            all_idx = self.model.images.index(image) \
                if image in self.model.images else -1
            self._push_undo(all_idx, list(image.tags))
            image.tags = self._merge_tags(image.tags, tags)
            self._auto_save_image(image)

            # If this is the currently viewed image, refresh
            if all_idx == self._current_image_index:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
                self._update_token_count()
        self._rebuild_all_tags()

    def _on_caption_finished(self):
        self.caption_progress.setVisible(False)
        self.caption_cancel_btn.setVisible(False)
        self.caption_current_btn.setEnabled(True)
        self.caption_selected_btn.setEnabled(True)
        self.caption_all_btn.setEnabled(True)
        self._worker = None

    def _on_caption_error(self, msg):
        QMessageBox.critical(self, 'Captioning Error', msg)

    # ------------------------------------------------------------------
    # Find & Replace
    # ------------------------------------------------------------------

    def _show_find_replace(self):
        self._save_current_tags()
        dlg = FindReplaceDialog(len(self.model.images), self)

        def do_replace():
            find_text = dlg.find_text
            replace_text = dlg.replace_text
            if not find_text:
                return

            if dlg.scope_all:
                images = self.model.images
            else:
                images = self.image_list.get_selected_images()
                if not images:
                    dlg.result_label.setText('No images selected')
                    return
            count = 0

            for image in images:
                old_tags = list(image.tags)
                new_tags = []
                for tag in image.tags:
                    if dlg.is_regex:
                        try:
                            new_tag = re.sub(find_text, replace_text, tag)
                        except re.error:
                            new_tag = tag
                    elif dlg.whole_match:
                        new_tag = replace_text if tag == find_text else tag
                    else:
                        new_tag = tag.replace(find_text, replace_text)

                    if new_tag != tag:
                        count += 1
                    new_tags.append(new_tag)

                # Remove empty tags after replacement
                image.tags = [t for t in new_tags if t.strip()]
                if image.tags != old_tags:
                    idx = self.model.images.index(image)
                    self._push_undo(idx, old_tags)
                    self._auto_save_image(image)

            # Refresh
            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self._pre_tags_snapshot = list(image.tags)
                    self._skip_snapshot = True
                    self.tags_list.set_tags(image.tags)
                    self._skip_snapshot = False
            self._rebuild_all_tags()
            self._update_token_count()
            dlg.result_label.setText(f'Replaced {count} occurrence(s)')

        dlg.replace_btn.clicked.connect(do_replace)
        dlg.exec()

    # ------------------------------------------------------------------
    # Batch Reorder
    # ------------------------------------------------------------------

    def _show_batch_reorder(self):
        self._save_current_tags()
        dlg = BatchReorderDialog(dict(self._all_tags), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            count = 0
            for i, image in enumerate(self.model.images):
                if image.tags:
                    new_tags = dlg.reorder_tags(image.tags)
                    if new_tags != image.tags:
                        self._push_undo(i, list(image.tags))
                        image.tags = new_tags
                        self._auto_save_image(image)
                        count += 1

            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self._pre_tags_snapshot = list(image.tags)
                    self._skip_snapshot = True
                    self.tags_list.set_tags(image.tags)
                    self._skip_snapshot = False
            self._rebuild_all_tags()
            self.caption_status.setText(
                f'Reordered tags in {count} image(s)')

    # ------------------------------------------------------------------
    # Bulk tools
    # ------------------------------------------------------------------

    def _remove_empty_all(self):
        count = 0
        for i, image in enumerate(self.model.images):
            before = len(image.tags)
            if any(not t.strip() for t in image.tags):
                self._push_undo(i, list(image.tags))
                image.tags = [t for t in image.tags if t.strip()]
                self._auto_save_image(image)
            count += before - len(image.tags)

        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
        self._rebuild_all_tags()
        self.caption_status.setText(f'Removed {count} empty tag(s)')

    def _remove_duplicates_all(self):
        count = 0
        for i, image in enumerate(self.model.images):
            seen = set()
            unique = []
            for tag in image.tags:
                key = tag.strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(tag)
            removed = len(image.tags) - len(unique)
            if removed:
                self._push_undo(i, list(image.tags))
                image.tags = unique
                self._auto_save_image(image)
            count += removed

        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
        self._rebuild_all_tags()
        self.caption_status.setText(
            f'Removed {count} duplicate(s) across all images')

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _save_captions_snapshot(self):
        """Save all image captions to a CSV snapshot file."""
        from datetime import datetime
        self._save_current_tags()
        if not self.model.images:
            self.caption_status.setText('No images loaded')
            return
        default_name = datetime.now().strftime('%Y-%m-%d %H-%M') + '.csv'
        default_dir = str(self.model.images[0].path.parent) \
            if self.model.images else ''
        default_path = str(Path(default_dir) / default_name) \
            if default_dir else default_name
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Captions Snapshot', default_path,
            'CSV Files (*.csv)')
        if not path:
            return
        sep = self._get_separator()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'captions'])
            for image in self.model.images:
                writer.writerow([image.filename, sep.join(image.tags)])
        self.caption_status.setText(
            f'Snapshot saved: {len(self.model.images)} image(s) → {path}')

    def _restore_captions_snapshot(self):
        """Restore captions from a CSV snapshot file."""
        if not self.model.images:
            self.caption_status.setText('No images loaded')
            return
        default_dir = str(self.model.images[0].path.parent) \
            if self.model.images else ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Restore Captions Snapshot', default_dir,
            'CSV Files (*.csv)')
        if not path:
            return
        sep = self._get_separator()
        # Parse snapshot
        snapshot: dict[str, list[str]] = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 2:
                QMessageBox.warning(self, 'Error',
                                    'Invalid snapshot file (expected CSV '
                                    'with image_name, captions columns).')
                return
            for row in reader:
                if len(row) >= 2:
                    name = row[0].strip()
                    captions_str = row[1].strip()
                    tags = [t.strip() for t in captions_str.split(
                        sep.strip() or sep) if t.strip()] \
                        if captions_str else []
                    snapshot[name] = tags

        # Apply
        matched = 0
        for i, image in enumerate(self.model.images):
            if image.filename in snapshot:
                new_tags = snapshot[image.filename]
                if new_tags != image.tags:
                    self._push_undo(i, list(image.tags))
                    image.tags = new_tags
                    self._auto_save_image(image)
                matched += 1

        # Refresh display
        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self._pre_tags_snapshot = list(image.tags)
                self._skip_snapshot = True
                self.tags_list.set_tags(image.tags)
                self._skip_snapshot = False
        self._rebuild_all_tags()
        self._update_token_count()

        unmatched = len(snapshot) - matched
        msg = f'Restored captions for {matched} image(s)'
        if unmatched > 0:
            msg += f' ({unmatched} in snapshot not found in current folder)'
        self.caption_status.setText(msg)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_all_captions(self):
        self._save_current_tags()
        sep = self._get_separator()
        count = 0
        for image in self.model.images:
            image.save_tags_to_file(separator=sep)
            count += 1
        self.caption_status.setText(f'Saved {count} caption file(s)')

    def _export_captions(self):
        self._save_current_tags()
        sep = self._get_separator()
        folder = QFileDialog.getExistingDirectory(
            self, 'Export Images + Captions To')
        if not folder:
            return
        out = Path(folder)
        count = 0
        for image in self.model.images:
            shutil.copy2(image.path, out / image.filename)
            caption_file = out / f'{image.name}.txt'
            caption_file.write_text(sep.join(image.tags), encoding='utf-8')
            count += 1
        self.caption_status.setText(
            f'Exported {count} image(s) + captions to {folder}')
