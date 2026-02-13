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
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QCompleter,
                               QDialog, QDoubleSpinBox, QGraphicsScene,
                               QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QFileDialog, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from ltd.comfyui.client import ComfyUIClient
from ltd.comfyui.workflow import (load_workflow, validate_caption_workflow,
                                  set_input_image)
from ltd.data.image_item import ImageItem
from ltd.settings import get_settings
from ltd.data.image_list_model import IMAGE_EXTENSIONS, ImageListModel
from ltd.dialogs.batch_reorder_dialog import BatchReorderDialog
from ltd.dialogs.find_replace_dialog import FindReplaceDialog
from ltd.utils.file_utils import get_temp_dir
from ltd.utils.image_utils import load_pixmap_preview
from ltd.widgets.caption_image_list import CaptionImageList
from ltd.widgets.workflow_selector import WorkflowSelector
from ltd.workers.caption_worker import CaptionWorker


# ---------------------------------------------------------------------------
# Captioner backends (unchanged)
# ---------------------------------------------------------------------------

class WdTaggerCaptioner:
    """WD Tagger auto-captioning using timm + safetensors (GPU via PyTorch)."""

    MODEL_REPO = 'SmilingWolf/wd-eva02-large-tagger-v3'
    MODELS_BASE = Path('./models/caption')
    INPUT_SIZE = 448  # model's expected input resolution

    def __init__(self, min_probability: float = 0.35, max_tags: int = 50,
                 exclude_tags: list[str] | None = None):
        self.model_dir = self.MODELS_BASE / self.MODEL_REPO
        self.min_probability = min_probability
        self.max_tags = max_tags
        self.exclude_tags = set(exclude_tags or [])
        self._model = None
        self._device = None
        self._tags = []
        self._rating_indices = []

    @property
    def name(self) -> str:
        return self.MODEL_REPO

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        import timm
        import huggingface_hub
        from safetensors.torch import load_file

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.model_dir / 'model.safetensors'
        tags_path = self.model_dir / 'selected_tags.csv'
        config_path = self.model_dir / 'config.json'

        for fname, fpath in [('model.safetensors', model_path),
                              ('selected_tags.csv', tags_path),
                              ('config.json', config_path)]:
            if not fpath.exists():
                huggingface_hub.hf_hub_download(
                    self.MODEL_REPO, filename=fname,
                    local_dir=str(self.model_dir))

        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        arch = config.get('architecture', 'eva02_large_patch14_448')
        num_classes = config.get('num_classes',
                                 config.get('num_features', 9083))

        model = timm.create_model(arch, pretrained=False,
                                  num_classes=num_classes)
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict)

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self._device)
        model.eval()
        self._model = model

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
        import torch
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

        if max_dim != self.INPUT_SIZE:
            canvas = canvas.resize((self.INPUT_SIZE, self.INPUT_SIZE),
                                   resample=PilImage.Resampling.BICUBIC)

        # RGB→BGR, normalize with mean=0.5/std=0.5 → [-1, 1], NCHW
        arr = np.array(canvas, dtype=np.float32)[:, :, ::-1] / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        results = []
        for i, (tag, prob) in enumerate(zip(self._tags, probs)):
            if i in self._rating_indices:
                continue
            if tag in self.exclude_tags:
                continue
            if prob >= self.min_probability:
                results.append((tag, float(prob)))

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

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            rows = sorted(set(idx.row() for idx in self.selectedIndexes()),
                          reverse=True)
            if rows:
                for row in rows:
                    self.takeItem(row)
                # Select nearest remaining
                new_row = min(rows) if min(rows) < self.count() \
                    else self.count() - 1
                if new_row >= 0:
                    self.setCurrentRow(new_row)
                self.tags_changed.emit()
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
            self.addItem(item)
        self.blockSignals(False)
        self._apply_highlights()
        self.setUpdatesEnabled(True)

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
    """All-tags list with delete key support."""
    delete_tags_requested = Signal(list)  # list of tag names

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)

    def keyPressEvent(self, event):
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
    """Tag input that clears properly after Enter, bypassing completer."""
    tag_submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.tag_submitted.emit()
            self.clear()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Image Viewer (auto-fit, Ctrl+scroll zoom, spacebar+drag pan)
# ---------------------------------------------------------------------------

class CaptionImageViewer(QGraphicsView):
    """Image viewer with auto-fit, Ctrl+scroll zoom, Space+drag pan."""

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

        self._setup_ui()
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
        self.tag_completer = QCompleter([])
        self.tag_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive)
        self.tag_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.tag_input.setCompleter(self.tag_completer)
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

        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel('Click:'))
        self.click_action_combo = QComboBox()
        self.click_action_combo.addItems(['Add to image', 'Filter images'])
        action_layout.addWidget(self.click_action_combo)
        all_tags_layout.addLayout(action_layout)

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

        tools_layout.addStretch()
        self.right_tabs.addTab(tools_tab, 'Tools')

        right_layout.addWidget(self.right_tabs)

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
        self.image_list.filter_input.textChanged.connect(
            self._update_tag_highlights)

        # Tag input
        self.add_tag_btn.clicked.connect(self._add_tag)
        self.tag_input.tag_submitted.connect(self._add_tag)

        # Tag list actions
        self.remove_tag_btn.clicked.connect(self._remove_selected_tags)
        self.clear_tags_btn.clicked.connect(self._clear_tags)
        self.remove_dupes_btn.clicked.connect(self._remove_duplicates_current)
        self.tags_list.tags_changed.connect(self._on_tags_list_changed)

        # All tags
        self.all_tags_filter.textChanged.connect(self._update_all_tags_display)
        self.sort_combo.currentIndexChanged.connect(
            self._update_all_tags_display)
        self.sort_order_combo.currentIndexChanged.connect(
            self._update_all_tags_display)
        self.all_tags_list.itemClicked.connect(self._on_all_tag_clicked)
        self.all_tags_list.itemDoubleClicked.connect(
            self._on_all_tag_double_clicked)
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
        self.remove_dupes_all_btn.clicked.connect(
            self._remove_duplicates_all)

        # Save
        self.save_captions_btn.clicked.connect(self._save_all_captions)
        self.export_captions_btn.clicked.connect(self._export_captions)

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

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_modify_tab(self, items: list[ImageItem]):
        self._pixmap_cache.clear()
        self.model.load_items(items)
        for image in self.model.images:
            image.load_tags_from_file()
        self._rebuild_all_tags()
        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _load_directory(self, path: str):
        self._pixmap_cache.clear()
        directory = Path(path)
        self.model.load_directory(directory)

        # Filter out mask files
        self.model.images = [
            img for img in self.model.images
            if '-masklabel' not in img.name and '-mask' not in img.name
        ]

        for image in self.model.images:
            image.load_tags_from_file()
        self._rebuild_all_tags()
        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def _on_image_changed(self, index: int):
        self._save_current_tags()
        self._current_image_index = index
        image = self.model.get_image(index)
        if image is None:
            return

        pixmap = self._load_cached(image.path)
        self.image_viewer.load_image(pixmap)

        # Show tags
        self.tags_list.set_tags(image.tags)
        self._update_token_count()
        self._update_tag_highlights()

        # Preload adjacent images after current frame finishes
        QTimer.singleShot(0, lambda idx=index: self._preload_adjacent(idx))

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
            if tags:
                # Append pasted tags
                image.tags = image.tags + tags
            # else: tags were already cleared by the caller

        # Refresh current image display if affected
        if self._current_image_index in source_rows:
            image = self.model.get_image(self._current_image_index)
            if image:
                self.tags_list.set_tags(image.tags)
                self._update_token_count()
        self._rebuild_all_tags()
        action = 'Pasted' if tags else 'Cleared'
        self.caption_status.setText(
            f'{action} tags on {len(source_rows)} image(s)')

    # ------------------------------------------------------------------
    # Tag editing (current image)
    # ------------------------------------------------------------------

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
                for tag in new_tags:
                    if tag not in image.tags:
                        image.tags.append(tag)
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
            self.tags_list.addItem(item)
        self.tag_input.clear()
        if self.tags_list.count() > 0:
            self.tags_list.scrollToBottom()
            self.tags_list.setCurrentRow(self.tags_list.count() - 1)
        self._on_tags_list_changed()

    def _remove_selected_tags(self):
        rows = sorted(set(idx.row() for idx in self.tags_list.selectedIndexes()),
                      reverse=True)
        for row in rows:
            self.tags_list.takeItem(row)
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
        self._save_current_tags()
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
        # Update autocomplete
        self.tag_completer.model().setStringList(list(self._all_tags.keys()))

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
        for tag, count in items:
            self.all_tags_list.addItem(f'{tag} ({count})')

        filtered = len(items)
        total = len(self._all_tags)
        if filter_text:
            self.all_tags_count_label.setText(
                f'{filtered} / {total} tags')
        else:
            self.all_tags_count_label.setText(f'{total} tags')

    def _on_all_tag_clicked(self, item: QListWidgetItem):
        tag = item.text().rsplit(' (', 1)[0]
        action = self.click_action_combo.currentIndex()
        if action == 0:  # Add to image
            new_item = QListWidgetItem(tag)
            new_item.setFlags(new_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.tags_list.addItem(new_item)
            self.tags_list.scrollToBottom()
            self._on_tags_list_changed()
        elif action == 1:  # Filter images (use image list filter bar)
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
            for image in self.model.images:
                if tag in image.tags:
                    image.tags = [new_name if t == tag else t
                                  for t in image.tags]
                    count += 1
            self._rebuild_all_tags()
            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self.tags_list.set_tags(image.tags)
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
        for image in self.model.images:
            before = len(image.tags)
            image.tags = [t for t in image.tags if t not in tag_set]
            count += before - len(image.tags)

        self._rebuild_all_tags()
        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self.tags_list.set_tags(image.tags)
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
            image.tags = self._merge_tags(image.tags, tags)

            # If this is the currently viewed image, refresh
            all_idx = self.model.images.index(image) \
                if image in self.model.images else -1
            if all_idx == self._current_image_index:
                self.tags_list.set_tags(image.tags)
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

            # Refresh
            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self.tags_list.set_tags(image.tags)
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
            for image in self.model.images:
                if image.tags:
                    new_tags = dlg.reorder_tags(image.tags)
                    if new_tags != image.tags:
                        image.tags = new_tags
                        count += 1

            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self.tags_list.set_tags(image.tags)
            self._rebuild_all_tags()
            self.caption_status.setText(
                f'Reordered tags in {count} image(s)')

    # ------------------------------------------------------------------
    # Bulk tools
    # ------------------------------------------------------------------

    def _remove_empty_all(self):
        count = 0
        for image in self.model.images:
            before = len(image.tags)
            image.tags = [t for t in image.tags if t.strip()]
            count += before - len(image.tags)

        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self.tags_list.set_tags(image.tags)
        self._rebuild_all_tags()
        self.caption_status.setText(f'Removed {count} empty tag(s)')

    def _remove_duplicates_all(self):
        count = 0
        for image in self.model.images:
            seen = set()
            unique = []
            for tag in image.tags:
                key = tag.strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(tag)
            count += len(image.tags) - len(unique)
            image.tags = unique

        if self._current_image_index >= 0:
            image = self.model.get_image(self._current_image_index)
            if image:
                self.tags_list.set_tags(image.tags)
        self._rebuild_all_tags()
        self.caption_status.setText(
            f'Removed {count} duplicate(s) across all images')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_all_captions(self):
        self._save_current_tags()
        count = 0
        for image in self.model.images:
            image.save_tags_to_file()
            count += 1
        self.caption_status.setText(f'Saved {count} caption file(s)')

    def _export_captions(self):
        self._save_current_tags()
        folder = QFileDialog.getExistingDirectory(
            self, 'Export Images + Captions To')
        if not folder:
            return
        out = Path(folder)
        count = 0
        for image in self.model.images:
            shutil.copy2(image.path, out / image.filename)
            caption_file = out / f'{image.name}.txt'
            caption_file.write_text(', '.join(image.tags), encoding='utf-8')
            count += 1
        self.caption_status.setText(
            f'Exported {count} image(s) + captions to {folder}')
