"""Caption tab: tag editing and auto-captioning (adapted from taggui)."""
import csv
import re
import shutil
from collections import OrderedDict
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QImageReader, QKeySequence, QPainter,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox,
                               QGraphicsScene, QGraphicsView,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QFileDialog,
                               QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSizePolicy, QSpinBox, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from ltd.comfyui.client import ComfyUIClient
from ltd.comfyui.workflow import (load_workflow, validate_caption_workflow,
                                  validate_generation_workflow,
                                  set_input_image, set_input_text,
                                  set_latent_size, set_seed)
from ltd.data.image_item import ImageItem
from ltd.data.tag_dictionary import TagDictionary
from ltd.settings import get_settings
from ltd.data.image_list_model import ImageListModel
from ltd.dialogs.batch_reorder_dialog import BatchReorderDialog
from ltd.dialogs.find_replace_dialog import FindReplaceDialog
from ltd.utils.file_utils import get_temp_dir, get_temp_dir_no_clear
from ltd.utils.image_utils import load_pixmap, load_pixmap_preview
from ltd.widgets.caption_image_list import CaptionImageList
from ltd.widgets.elided_label import ElidedLabel
from ltd.widgets.loading_dialog import loading_dialog
from ltd.widgets.tag_completer_popup import TagCompleterPopup
from ltd.widgets.workflow_selector import WorkflowSelector
from ltd.workers.caption_worker import CaptionWorker
from ltd.workers.generation_worker import GenerationWorker


# ---------------------------------------------------------------------------
# Captioner backends (unchanged)
# ---------------------------------------------------------------------------

class WdTaggerCaptioner:
    """WD Tagger auto-captioning using timm + safetensors (GPU via PyTorch)."""

    MODEL_REPO = 'SmilingWolf/wd-eva02-large-tagger-v3'
    MODELS_BASE = Path('./models/caption')
    INPUT_SIZE = 448
    # Preprocessing: SmilingWolf models expect BGR, normalized to [-1, 1]
    # (mean=0.5, std=0.5). Subclasses override for other conventions.
    _BGR = True
    _MEAN = 0.5
    _STD = 0.5

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

        import json
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

        with open(config_path, 'r') as f:
            config = json.load(f)

        arch = config['architecture']
        num_classes = config['num_classes']
        model_args = config.get('model_args', {})

        model = timm.create_model(arch, pretrained=False,
                                  num_classes=num_classes, **model_args)
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

        # Normalize per model config (BGR/mean/std), NCHW for PyTorch.
        arr = np.array(canvas, dtype=np.float32) / 255.0
        if self._BGR:
            arr = arr[:, :, ::-1]
        arr = (arr - self._MEAN) / self._STD
        tensor = torch.from_numpy(
            np.ascontiguousarray(arr, dtype=np.float32)
        ).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            if not self._model:
                raise ValueError('Failed to load model')
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


class AnimeTimmCaptioner(WdTaggerCaptioner):
    """animetimm (deepghs) tagger — RGB, ImageNet normalization, 512px input.

    Shares the timm + safetensors loading and tag-thresholding of the base
    class; only the model repo and preprocessing convention differ.
    """

    MODEL_REPO = 'animetimm/convnextv2_huge.dbv4-full'
    INPUT_SIZE = 512
    _BGR = False
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text,
                     flags=re.DOTALL | re.IGNORECASE)
    if '</think>' in cleaned.lower():
        cleaned = re.split(r'</think>', cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


class LMStudioCaptioner:
    """Natural-language captioning via an LM Studio vision model.

    Produces a single caption per image (returned split on the tag separator
    so it round-trips with the shared .txt storage). Always overwrites the
    existing caption; when ``append_context`` is set the image's current
    caption/tags are sent to the model as context to steer the description.
    """

    # Signals CaptionTab that results replace the caption rather than merging
    # via the WD-tagger Position combo.
    replaces_caption = True

    def __init__(self, model: str, system_prompt: str = '',
                 append_context: bool = False, separator: str = ', '):
        self.model = model
        self.system_prompt = system_prompt
        self.append_context = append_context
        self.separator = separator

    def _current_caption(self, image_path: Path) -> str:
        txt = image_path.with_suffix('.txt')
        if txt.exists():
            try:
                return txt.read_text(encoding='utf-8').strip()
            except OSError:
                return ''
        return ''

    def caption(self, image_path: Path) -> list[str]:
        from ltd.lmstudio.client import LMStudioClient
        client = LMStudioClient()

        user_text = None
        if self.append_context:
            existing = self._current_caption(image_path)
            if existing:
                user_text = ('Here is the current caption/tags describing '
                             f'this image, use it as context: {existing}')

        raw = client.caption(image_path, self.model,
                             system_prompt=self.system_prompt,
                             user_text=user_text)
        caption = strip_thinking(raw)
        if not caption:
            return []
        key = self.separator.strip() or self.separator
        return [t.strip() for t in caption.split(key) if t.strip()]

    def finalize(self):
        """Called once after the batch — unload the model to free memory."""
        from ltd.lmstudio.client import LMStudioClient
        try:
            LMStudioClient().unload_model(self.model)
        except Exception:
            pass


def generated_cache_name(image: ImageItem) -> str:
    """Stable, collision-safe filename for an image's cached generated render.

    Keyed by the relative path (not just the stem) so images with the same
    name in different subfolders don't clobber each other in the flat cache.
    """
    rel = image.relative_path or image.filename
    stem = Path(rel).with_suffix('').as_posix()
    return re.sub(r'[^\w.\-]+', '_', stem) + '.png'


class CaptionImageGenerator:
    """Render an image's caption to a new image via a ComfyUI workflow.

    Generates at the image's aspect ratio scaled to ~``megapixels`` MPix and
    caches the result under ``cache_dir`` (never in the source folder).
    """

    # Fixed render seed so a regenerated image only changes when the caption
    # changes (controlled comparison); also avoids rgthree's -1 randomize path.
    SEED = 42

    def __init__(self, workflow_json: str, cache_dir: Path,
                 separator: str = ', ', megapixels: float = 1.0):
        self.workflow_json = workflow_json
        self.cache_dir = Path(cache_dir)
        self.separator = separator
        self.megapixels = megapixels

    @staticmethod
    def compute_dims(w: int, h: int, megapixels: float = 1.0,
                     multiple: int = 16) -> tuple[int, int]:
        """Width/height at the w:h aspect ratio totalling ~megapixels, snapped
        to a multiple (latent-friendly)."""
        if w <= 0 or h <= 0:
            w = h = 1
        total = megapixels * 1_000_000
        aspect = w / h
        width = (total * aspect) ** 0.5
        height = total / width if width else total ** 0.5

        def snap(x: float) -> int:
            return max(multiple, int(round(x / multiple)) * multiple)

        return snap(width), snap(height)

    def _caption_text(self, image: ImageItem) -> str:
        # In-memory tags reflect edits the tab flushes before generating;
        # fall back to the .txt on disk.
        if image.tags:
            return self.separator.join(image.tags)
        txt = image.path.with_suffix('.txt')
        if txt.exists():
            try:
                return txt.read_text(encoding='utf-8').strip()
            except OSError:
                return ''
        return ''

    def _orig_size(self, image: ImageItem) -> tuple[int, int]:
        if image.width and image.height:
            return image.width, image.height
        size = QImageReader(str(image.path)).size()
        if size.isValid():
            return size.width(), size.height()
        return 1, 1

    def generate(self, image: ImageItem) -> Path:
        workflow = load_workflow(self.workflow_json)
        if workflow is None:
            raise ValueError('Invalid workflow JSON')
        valid, msg = validate_generation_workflow(workflow)
        if not valid:
            raise ValueError(msg)

        set_input_text(workflow, self._caption_text(image))
        w, h = self._orig_size(image)
        gw, gh = self.compute_dims(w, h, self.megapixels)
        set_latent_size(workflow, gw, gh)
        set_seed(workflow, self.SEED)

        client = ComfyUIClient()
        if not client.health_check():
            raise ConnectionError('Cannot connect to ComfyUI')

        out_dir = get_temp_dir('comfyui_generate_output')
        result = client.run_workflow(workflow, out_dir)
        files = result.get('files') or []
        if not files:
            raise ValueError('Workflow produced no image output')

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / generated_cache_name(image)
        shutil.copy2(files[0], dest)
        return dest

    def finalize(self):
        """Called once after the batch — free ComfyUI's models/VRAM."""
        try:
            ComfyUIClient().free()
        except Exception:
            pass


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
        for pat in self._highlight_patterns:
            kind = pat[0]
            if kind == 'tag':
                v = pat[1].lower()
                if '*' in v or '?' in v:
                    if fnmatchcase(tag_lower, v):
                        return True
                else:
                    if tag_lower == v:
                        return True
            elif kind == 'regex':
                if pat[1].search(tag_text.strip()) is not None:
                    return True
            elif kind == 'text':
                if pat[1].lower() in tag_lower:
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
        tags = []
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

class _FullResLoader(QThread):
    """Background thread that loads a full-resolution pixmap."""
    ready = Signal(str, QPixmap)  # (path_key, pixmap)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        pixmap = load_pixmap(self._path)
        if pixmap:
            self.ready.emit(str(self._path), pixmap)


class CaptionImageViewer(QGraphicsView):
    """Image viewer with auto-fit, Ctrl+scroll zoom, drag pan when zoomed."""
    navigate_image = Signal(int)  # -1 = prev, +1 = next

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap_item = None
        self._manually_zoomed = False
        self._update_drag_mode()

    def load_image(self, pixmap):
        self._scene.clear()
        self._pixmap_item = None
        if pixmap:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._pixmap_item.setTransformationMode(
                Qt.TransformationMode.SmoothTransformation)
            # Reset scene rect to this pixmap's bounds so fitInView
            # doesn't use a stale rect from a previously larger image.
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
            self._manually_zoomed = False
            self._fit_image()
        else:
            self._scene.setSceneRect(0, 0, 0, 0)

    def load_comparison(self, first: QPixmap, second: QPixmap,
                        stack_vertical: bool):
        """Show `first` (original) beside/above `second` (generated).

        `first` is placed at its native resolution (never downscaled); `second`
        is scaled to share the common dimension so the pair displays at equal
        size. Both keep their full pixmap, so zooming shows native detail.
        Landscape originals stack top/bottom; square/portrait sit left/right.
        """
        self._scene.clear()
        self._pixmap_item = None
        gap = 12
        it1 = self._scene.addPixmap(first)
        it2 = self._scene.addPixmap(second)
        it1.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        it2.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        if stack_vertical:
            scale2 = first.width() / second.width() if second.width() else 1.0
            it2.setScale(scale2)
            it1.setPos(0, 0)
            it2.setPos(0, first.height() + gap)
            total_w = first.width()
            total_h = first.height() + gap + second.height() * scale2
        else:
            scale2 = (first.height() / second.height()
                      if second.height() else 1.0)
            it2.setScale(scale2)
            it1.setPos(0, 0)
            it2.setPos(first.width() + gap, 0)
            total_w = first.width() + gap + second.width() * scale2
            total_h = first.height()
        self._scene.setSceneRect(0, 0, total_w, total_h)
        self._manually_zoomed = False
        self._fit_image()

    def replace_pixmap(self, pixmap):
        """Swap pixmap data without resetting zoom/pan state."""
        if self._pixmap_item is None:
            return
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if not self._manually_zoomed:
            self._fit_image()

    def _fit_image(self):
        rect = self._scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._manually_zoomed = False
            self._update_drag_mode()

    def _update_drag_mode(self):
        if self._manually_zoomed:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self._scene.sceneRect()
        if not self._manually_zoomed and not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            self._manually_zoomed = True
            self._update_drag_mode()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
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


# ---------------------------------------------------------------------------
# Caption Tab
# ---------------------------------------------------------------------------

class CaptionTab(QWidget):
    _PIXMAP_CACHE_MAX = 5
    # WD-style taggers shown before the 'ComfyUI Workflow' entry; they share
    # the same settings panel (min probability, max tags, exclude, position).
    _WD_MODELS = (WdTaggerCaptioner, AnimeTimmCaptioner)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self._current_image_index = -1
        self._all_tags: dict[str, int] = {}  # tag -> count
        self._worker: CaptionWorker | None = None
        self._pixmap_cache: OrderedDict[str, QPixmap | None] = OrderedDict()
        self._fullres_loader: _FullResLoader | None = None

        # Undo/redo stacks (per image index)
        # Each entry is (tag_snapshot, batch_id_or_None)
        self._undo_stacks: dict[int, list[tuple[list[str], int | None]]] = {}
        self._redo_stacks: dict[int, list[tuple[list[str], int | None]]] = {}
        self._pre_tags_snapshot: list[str] | None = None
        self._max_undo = 50
        self._skip_snapshot = False
        self._next_batch_id = 0

        # Caption (natural-language) panel state. Captions share the same
        # .txt file / image.tags as tags — the caption box is just an
        # alternate view of the joined-tag text, so auto-save, snapshots,
        # export and undo all reuse the tag machinery unchanged.
        self._panel_mode = 'tags'  # 'tags' | 'caption'
        self._loading_caption = False
        # Suppresses the LM Studio model auto-query until construction is done,
        # so restoring an LM-Studio captioner selection can't block startup on
        # a network request.
        self._lm_autorefresh_enabled = False

        # Caption→image generation (ComfyUI) + comparison cache.
        self._gen_worker: GenerationWorker | None = None
        self._generated_dir: Path | None = None
        self._caption_debounce = QTimer(self)
        self._caption_debounce.setSingleShot(True)
        self._caption_debounce.setInterval(400)

        # Tag dictionary for autocomplete + colors
        self._tag_dict = TagDictionary()
        autocomp_dir = Path('autocompletions')
        if autocomp_dir.is_dir():
            self._tag_dict.load_directory(autocomp_dir)
        elif Path('tags.csv').exists():
            self._tag_dict.load_csv(Path('tags.csv'))

        self._setup_ui()
        self._tag_popup = TagCompleterPopup(self._tag_dict, self)
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        self.tags_list.set_tag_dictionary(self._tag_dict, dark=dark)
        self._connect_signals()
        self._setup_shortcuts()
        self._restore_settings()
        self._connect_settings_persistence()
        # Construction done — LM Studio model queries may now run on demand.
        self._lm_autorefresh_enabled = True

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
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Top row: mode dropdown + shortcut info button (right-aligned)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addStretch(1)
        # Toggle between the tag tabs and the natural-language caption box.
        self.panel_mode_combo = QComboBox()
        self.panel_mode_combo.addItems(['Tags', 'Caption'])
        self.panel_mode_combo.setToolTip(
            'Tags: edit comma-separated tags.\n'
            'Caption: edit the same file as one natural-language caption.')
        header.addWidget(self.panel_mode_combo)
        from ltd.widgets.info_button import DynamicInfoButton
        self.shortcuts_info = DynamicInfoButton(self._build_shortcuts_help)
        header.addWidget(self.shortcuts_info)
        right_layout.addLayout(header)

        self.right_tabs = QTabWidget()

        # ---- Tab: Image Tags ----
        self.image_tags_tab = QWidget()
        tag_layout = QVBoxLayout(self.image_tags_tab)

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

        self.right_tabs.addTab(self.image_tags_tab, 'Image Tags')

        # ---- Tab: All Tags ----
        self.all_tags_tab = QWidget()
        all_tags_layout = QVBoxLayout(self.all_tags_tab)

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

        self.right_tabs.addTab(self.all_tags_tab, 'All Tags')

        # ---- Tab: Auto-Caption ----
        caption_tab_w = QWidget()
        caption_layout = QVBoxLayout(caption_tab_w)

        self.captioner_combo = QComboBox()
        for cls in self._WD_MODELS:
            self.captioner_combo.addItem(cls.MODEL_REPO)
        self.captioner_combo.addItem('ComfyUI Workflow')
        self.captioner_combo.addItem('LM Studio')
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

        # LM Studio settings
        self.lmstudio_settings = QWidget()
        lm_layout = QVBoxLayout(self.lmstudio_settings)
        lm_layout.setContentsMargins(0, 0, 0, 0)

        lm_model_row = QHBoxLayout()
        lm_model_row.addWidget(QLabel('Model:'))
        self.lmstudio_model_combo = QComboBox()
        self.lmstudio_model_combo.setEditable(True)
        self.lmstudio_model_combo.setPlaceholderText(
            'Refresh to load models')
        self.lmstudio_model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lm_model_row.addWidget(self.lmstudio_model_combo, 1)
        self.lmstudio_refresh_btn = QPushButton('Refresh')
        self.lmstudio_refresh_btn.setToolTip(
            'Query the LM Studio URL for loaded models')
        lm_model_row.addWidget(self.lmstudio_refresh_btn)
        lm_layout.addLayout(lm_model_row)

        lm_layout.addWidget(QLabel('System prompt:'))
        self.lmstudio_system_prompt = QPlainTextEdit()
        self.lmstudio_system_prompt.setPlaceholderText(
            'Instructions for the captioning model...')
        # Preferred (not Expanding) vertical policy so the panel stays compact
        # and the tab's bottom stretch absorbs slack — otherwise the leftover
        # space inflates the label above and opens a gap.
        self.lmstudio_system_prompt.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lmstudio_system_prompt.setMinimumHeight(90)
        self.lmstudio_system_prompt.setMaximumHeight(160)
        lm_layout.addWidget(self.lmstudio_system_prompt)

        self.lmstudio_append_check = QCheckBox('Append current caption')
        self.lmstudio_append_check.setToolTip(
            'Send this image\'s current caption/tags to the model as context. '
            'The generated caption overwrites the file either way.')
        lm_layout.addWidget(self.lmstudio_append_check)

        caption_layout.addWidget(self.lmstudio_settings)
        self.lmstudio_settings.setVisible(False)

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

        snapshot_group = QGroupBox('Caption Snapshots')
        snapshot_layout = QHBoxLayout(snapshot_group)
        self.snapshot_save_btn = QPushButton('Create')
        self.snapshot_restore_btn = QPushButton('Restore All')
        self.snapshot_restore_some_btn = QPushButton('Restore Some')
        snapshot_layout.addWidget(self.snapshot_save_btn)
        snapshot_layout.addWidget(self.snapshot_restore_btn)
        snapshot_layout.addWidget(self.snapshot_restore_some_btn)
        tools_layout.addWidget(snapshot_group)

        tools_row3 = QHBoxLayout()
        self.reload_autocompletions_btn = QPushButton('Reload Autocompletions')
        self.reload_autocompletions_btn.setToolTip(
            'Reload tag dictionary from autocompletions/ folder')
        tools_row3.addWidget(self.reload_autocompletions_btn)
        tools_row3.addStretch()
        tools_layout.addLayout(tools_row3)

        # ---- Compare original image to generated from caption ----
        gen_group = QGroupBox('Compare original image to generated from caption')
        gen_layout = QVBoxLayout(gen_group)
        self._generate_workflow_selector = WorkflowSelector(
            settings_key='generate')
        gen_layout.addWidget(self._generate_workflow_selector)
        gen_info = QLabel(
            'Renders the caption via ComfyUI at the original aspect ratio '
            '(~1 MPix) and shows it beside the original. Required nodes: '
            'LTD_Input_Text, LTD_Latent_Size, LTD_Output_Image.')
        gen_info.setWordWrap(True)
        gen_layout.addWidget(gen_info)

        gen_btn_layout = QHBoxLayout()
        gen_btn_layout.addWidget(QLabel('Generate:'))
        self.generate_current_btn = QPushButton('Current')
        self.generate_selected_btn = QPushButton('Selected')
        self.generate_all_btn = QPushButton('All')
        self.generate_cancel_btn = QPushButton('Cancel')
        self.generate_cancel_btn.setVisible(False)
        gen_btn_layout.addWidget(self.generate_current_btn)
        gen_btn_layout.addWidget(self.generate_selected_btn)
        gen_btn_layout.addWidget(self.generate_all_btn)
        gen_btn_layout.addWidget(self.generate_cancel_btn)
        gen_layout.addLayout(gen_btn_layout)

        self.generate_progress = QProgressBar()
        self.generate_progress.setVisible(False)
        gen_layout.addWidget(self.generate_progress)
        tools_layout.addWidget(gen_group)

        tools_layout.addStretch()
        self.right_tabs.addTab(tools_tab, 'Tools')

        # ---- Tab: Image Caption (natural language) ----
        # Sits inside the same tab widget; the Tags/Caption dropdown swaps it
        # in for the Image Tags + All Tags tabs while keeping Auto-Caption and
        # Tools available. Inserted at index 0 so it leads in Caption mode.
        self.caption_panel = QWidget()
        cap_panel_layout = QVBoxLayout(self.caption_panel)
        self.caption_edit = QPlainTextEdit()
        self.caption_edit.setPlaceholderText(
            'Natural-language caption for this image...')
        cap_panel_layout.addWidget(self.caption_edit)
        self.caption_count_label = QLabel('0 chars | ~0 tokens')
        cap_panel_layout.addWidget(self.caption_count_label)
        self.right_tabs.insertTab(0, self.caption_panel, 'Image Caption')

        right_layout.addWidget(self.right_tabs)

        # Start in Tags mode: hide the Image Caption tab.
        self._set_caption_mode_tabs(False)

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

        self.caption_status = ElidedLabel('')
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

        # Panel mode (Tags / Caption) + caption editing
        self.panel_mode_combo.currentIndexChanged.connect(
            self._on_panel_mode_changed)
        self.caption_edit.textChanged.connect(self._on_caption_text_changed)
        self._caption_debounce.timeout.connect(self._commit_caption_edit)

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
        self.lmstudio_refresh_btn.clicked.connect(
            self._refresh_lmstudio_models)
        self.caption_current_btn.clicked.connect(
            lambda: self._run_captioning(mode='current'))
        self.caption_selected_btn.clicked.connect(
            lambda: self._run_captioning(mode='selected'))
        self.caption_all_btn.clicked.connect(
            lambda: self._run_captioning(mode='all'))
        self.caption_cancel_btn.clicked.connect(self._cancel_captioning)

        # Generate from caption (comparison)
        self.generate_current_btn.clicked.connect(
            lambda: self._run_generation(mode='current'))
        self.generate_selected_btn.clicked.connect(
            lambda: self._run_generation(mode='selected'))
        self.generate_all_btn.clicked.connect(
            lambda: self._run_generation(mode='all'))
        self.generate_cancel_btn.clicked.connect(self._cancel_generation)

        # Tools
        self.find_replace_btn.clicked.connect(self._show_find_replace)
        self.batch_reorder_btn.clicked.connect(self._show_batch_reorder)
        self.remove_empty_btn.clicked.connect(self._remove_empty_all)
        self.snapshot_save_btn.clicked.connect(self._save_captions_snapshot)
        self.snapshot_restore_btn.clicked.connect(self._restore_captions_snapshot)
        self.snapshot_restore_some_btn.clicked.connect(
            self._restore_some_from_snapshot)
        self.remove_dupes_all_btn.clicked.connect(
            self._remove_duplicates_all)
        self.reload_autocompletions_btn.clicked.connect(
            self._reload_autocompletions)

        # Save / separator
        self.save_captions_btn.clicked.connect(self._save_all_captions)
        self.export_captions_btn.clicked.connect(self._export_captions)
        self.reload_tags_btn.clicked.connect(self._reload_tags)

    def _build_shortcuts_help(self) -> str:
        from ltd.widgets.info_button import focus_in
        from ltd.widgets.info_text import (CAPTION_SHORTCUTS_BULK,
                                            CAPTION_SHORTCUTS_GLOBAL,
                                            CAPTION_SHORTCUTS_INPUT,
                                            CAPTION_SHORTCUTS_NAV,
                                            CAPTION_SHORTCUTS_TAGS_LIST)
        sections: list[str] = []
        if focus_in(self.tag_input):
            sections.append(CAPTION_SHORTCUTS_INPUT)
        elif focus_in(self.tags_list):
            sections.append(CAPTION_SHORTCUTS_TAGS_LIST)
        elif focus_in(self.image_list):
            sections.append(CAPTION_SHORTCUTS_NAV)
        if not sections:
            sections = [CAPTION_SHORTCUTS_TAGS_LIST,
                        CAPTION_SHORTCUTS_INPUT,
                        CAPTION_SHORTCUTS_NAV]
        sections.append(CAPTION_SHORTCUTS_BULK)
        sections.append(CAPTION_SHORTCUTS_GLOBAL)
        return '<br>'.join(sections)

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
        if isinstance(focus, (QLineEdit, QPlainTextEdit)):
            return
        if (isinstance(focus, QAbstractItemView)
                and focus.state() == QAbstractItemView.State.EditingState):
            return
        self.image_list.go_to_previous()

    def _navigate_next(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QPlainTextEdit)):
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
        # LM Studio captioner settings
        self.lmstudio_system_prompt.setPlainText(
            s.value('caption/lmstudio_system_prompt', '', type=str))
        self.lmstudio_append_check.setChecked(
            s.value('caption/lmstudio_append', False, type=bool))
        saved_model = s.value('caption/lmstudio_model', '', type=str)
        if saved_model:
            self.lmstudio_model_combo.setCurrentText(saved_model)
        # Applies visibility via _on_panel_mode_changed (only fires if the
        # stored index differs from the combo's current 0).
        self.panel_mode_combo.setCurrentIndex(
            s.value('caption/panel_mode', 0, type=int))

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
        self.lmstudio_model_combo.currentTextChanged.connect(
            lambda v: self._save_setting('lmstudio_model', v))
        self.lmstudio_system_prompt.textChanged.connect(
            lambda: self._save_setting(
                'lmstudio_system_prompt',
                self.lmstudio_system_prompt.toPlainText()))
        self.lmstudio_append_check.toggled.connect(
            lambda v: self._save_setting('lmstudio_append', v))

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
    # Caption mode (natural-language view of the same .txt / image.tags)
    # ------------------------------------------------------------------

    def _parse_caption_text(self, text: str) -> list[str]:
        """Split caption box text into tags, mirroring load_tags_from_file."""
        sep = self._get_separator()
        key = sep.strip() or sep
        return [t.strip() for t in text.split(key) if t.strip()]

    def _set_caption_mode_tabs(self, is_caption: bool):
        """Show the Image Caption tab or the Image Tags + All Tags tabs.

        Auto-Caption and Tools stay visible in both modes.
        """
        tabs = self.right_tabs
        tabs.setTabVisible(tabs.indexOf(self.caption_panel), is_caption)
        tabs.setTabVisible(tabs.indexOf(self.image_tags_tab), not is_caption)
        tabs.setTabVisible(tabs.indexOf(self.all_tags_tab), not is_caption)
        tabs.setCurrentWidget(
            self.caption_panel if is_caption else self.image_tags_tab)

    def _on_panel_mode_changed(self, index: int):
        mode = 'caption' if index == 1 else 'tags'
        if mode == self._panel_mode:
            return
        # Flush the editor that's about to be hidden into the model.
        self._save_current_tags()
        self._panel_mode = mode
        is_caption = mode == 'caption'
        self._set_caption_mode_tabs(is_caption)
        image = self.model.get_image(self._current_image_index)
        if is_caption:
            self._load_caption_into_edit(image)
        else:
            # tags_list may be stale after caption edits — refresh from model.
            self._skip_snapshot = True
            self.tags_list.set_tags(image.tags if image else [])
            self._skip_snapshot = False
            self._update_token_count()
        self._save_setting('panel_mode', index)

    def _load_caption_into_edit(self, image: ImageItem | None):
        """Populate the caption box from an image's tags (as joined text)."""
        self._caption_debounce.stop()
        self._loading_caption = True
        if image is not None:
            self.caption_edit.setPlainText(self._get_separator().join(image.tags))
        else:
            self.caption_edit.setPlainText('')
        self._loading_caption = False
        self._pre_tags_snapshot = list(image.tags) if image is not None else None
        self._update_token_count()

    def _sync_caption_view(self):
        """Reload the caption box from the current image (caption mode only)."""
        if self._panel_mode == 'caption':
            self._load_caption_into_edit(
                self.model.get_image(self._current_image_index))

    def _on_caption_text_changed(self):
        if self._loading_caption:
            return
        self._update_token_count()
        self._caption_debounce.start()

    def _commit_caption_edit(self):
        """Write pending caption edits back to image.tags (undo + autosave)."""
        self._caption_debounce.stop()
        idx = self._current_image_index
        if idx < 0:
            return
        image = self.model.get_image(idx)
        if image is None:
            return
        new_tags = self._parse_caption_text(self.caption_edit.toPlainText())
        if new_tags == image.tags:
            return
        if self._pre_tags_snapshot is not None:
            self._push_undo(idx, list(self._pre_tags_snapshot))
        image.tags = new_tags
        self._pre_tags_snapshot = list(new_tags)
        self._auto_save_image(image)
        self._rebuild_all_tags()
        self._update_token_count()

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

    def _new_batch_id(self) -> int:
        """Return a unique batch ID for grouping multi-image undo entries."""
        bid = self._next_batch_id
        self._next_batch_id += 1
        return bid

    def _push_undo(self, image_index: int | None = None,
                   snapshot: list[str] | None = None,
                   batch_id: int | None = None):
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
        stack.append((snapshot, batch_id))
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

        snapshot, batch_id = stack[-1]

        if batch_id is not None:
            # Undo all images in this batch
            self._undo_batch(batch_id)
        else:
            # Single-image undo
            stack.pop()
            redo = self._redo_stacks.setdefault(idx, [])
            redo.append((list(image.tags), None))
            image.tags = snapshot
            self._auto_save_image(image)

        # Refresh current image display
        image = self.model.get_image(idx)
        if image:
            self._pre_tags_snapshot = list(image.tags)
            self._skip_snapshot = True
            self.tags_list.set_tags(image.tags)
            self._skip_snapshot = False
        self._sync_caption_view()
        self._rebuild_all_tags()
        self._update_token_count()

    def _undo_batch(self, batch_id: int):
        """Undo all entries with the given batch_id across all images."""
        for img_idx in list(self._undo_stacks.keys()):
            stack = self._undo_stacks[img_idx]
            if not stack or stack[-1][1] != batch_id:
                continue
            snapshot, _ = stack.pop()
            image = self.model.get_image(img_idx)
            if image is None:
                continue
            redo = self._redo_stacks.setdefault(img_idx, [])
            redo.append((list(image.tags), batch_id))
            image.tags = snapshot
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

        snapshot, batch_id = stack[-1]

        if batch_id is not None:
            # Redo all images in this batch
            self._redo_batch(batch_id)
        else:
            # Single-image redo
            stack.pop()
            undo = self._undo_stacks.setdefault(idx, [])
            undo.append((list(image.tags), None))
            image.tags = snapshot
            self._auto_save_image(image)

        # Refresh current image display
        image = self.model.get_image(idx)
        if image:
            self._pre_tags_snapshot = list(image.tags)
            self._skip_snapshot = True
            self.tags_list.set_tags(image.tags)
            self._skip_snapshot = False
        self._sync_caption_view()
        self._rebuild_all_tags()
        self._update_token_count()

    def _redo_batch(self, batch_id: int):
        """Redo all entries with the given batch_id across all images."""
        for img_idx in list(self._redo_stacks.keys()):
            stack = self._redo_stacks[img_idx]
            if not stack or stack[-1][1] != batch_id:
                continue
            snapshot, _ = stack.pop()
            image = self.model.get_image(img_idx)
            if image is None:
                continue
            undo = self._undo_stacks.setdefault(img_idx, [])
            undo.append((list(image.tags), batch_id))
            image.tags = snapshot
            self._auto_save_image(image)

    def _clear_undo_stacks(self):
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        self._pre_tags_snapshot = None
        self._next_batch_id = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_directory(self, path: str):
        directory = Path(path)
        sep = self._get_separator()

        # Per-folder cache for generated comparison images (persists across
        # sessions; never written into the source folder). Keyed by folder path.
        import hashlib
        dir_hash = hashlib.md5(str(directory).encode()).hexdigest()[:12]
        self._generated_dir = get_temp_dir_no_clear(f'generated_{dir_hash}')

        with loading_dialog('Loading images...', self) as dlg:
            self._cancel_fullres_loader()
            self._pixmap_cache.clear()
            self._clear_undo_stacks()
            self._current_image_index = -1
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
                self._sync_caption_view()
                self._update_token_count()
        self.caption_status.setText(
            f'Reloaded tags for {len(self.model.images)} image(s)')

    def _reload_autocompletions(self):
        """Reload the tag dictionary from autocompletions/ folder."""
        self._tag_dict = TagDictionary()
        autocomp_dir = Path('autocompletions')
        if autocomp_dir.is_dir():
            self._tag_dict.load_directory(autocomp_dir)
        elif Path('tags.csv').exists():
            self._tag_dict.load_csv(Path('tags.csv'))

        self._tag_popup.set_dictionary(self._tag_dict)
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        self.tags_list.set_tag_dictionary(self._tag_dict, dark=dark)
        count = len(self._tag_dict._tags) if self._tag_dict.is_loaded() else 0
        self.caption_status.setText(
            f'Reloaded autocompletions ({count} tags)')

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def _cancel_fullres_loader(self):
        # Detach the in-flight loader without blocking the UI thread.
        # The thread keeps running to completion; its result is dropped
        # because the signal is disconnected, and _on_fullres_ready also
        # guards against stale keys. Qt parent ownership keeps the
        # QThread object alive until it finishes.
        if self._fullres_loader is not None:
            try:
                self._fullres_loader.ready.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._fullres_loader = None

    def _on_fullres_ready(self, key: str, pixmap: QPixmap):
        """Swap in the full-res pixmap if the image is still current."""
        image = self.model.get_image(self._current_image_index)
        if image is None or str(image.path) != key:
            return
        self._pixmap_cache[key] = pixmap
        self.image_viewer.replace_pixmap(pixmap)
        self._fullres_loader = None

    def _on_image_changed(self, index: int):
        self._save_current_tags()
        self._current_image_index = index
        image = self.model.get_image(index)
        if image is None:
            return

        # Snapshot for undo tracking
        self._pre_tags_snapshot = list(image.tags)

        # Cancel any in-flight full-res load
        self._cancel_fullres_loader()

        gen_path = self._generated_path(image)
        if gen_path is not None:
            # A generated render exists — show it beside the original.
            self._show_comparison(image.path, gen_path)
        else:
            key = str(image.path)
            cached = self._pixmap_cache.get(key)
            if cached:
                # Cache hit — show immediately
                self._pixmap_cache.move_to_end(key)
                self.image_viewer.load_image(cached)
            else:
                # Show preview instantly, then swap full-res from background
                pixmap = load_pixmap_preview(image.path,
                                             self._preview_max_dim())
                self.image_viewer.load_image(pixmap)

                loader = _FullResLoader(image.path, self)
                loader.ready.connect(self._on_fullres_ready)
                loader.start()
                self._fullres_loader = loader

        # Show tags
        self._skip_snapshot = True
        self.tags_list.set_tags(image.tags)
        self._skip_snapshot = False
        if self._panel_mode == 'caption':
            self._load_caption_into_edit(image)
        self._update_token_count()
        self._update_tag_highlights()

        # Preload adjacent full-res images after current frame finishes
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
            self._sync_caption_view()
        self._rebuild_all_tags()
        self._update_token_count()

    def _preview_max_dim(self) -> int:
        # Preview is a transient flash: full-res swaps in via
        # _on_fullres_ready, and zoom uses that. Target ~half the
        # viewport so libjpeg can DCT-scale large JPEGs cheaply.
        vp = self.image_viewer.viewport().size()
        return max(vp.width(), vp.height(), 800) // 2

    def _preload_adjacent(self, index: int):
        if index != self._current_image_index:
            return  # user already moved on
        for offset in (-1, 1):
            image = self.model.get_image(index + offset)
            if image and str(image.path) not in self._pixmap_cache:
                loader = _FullResLoader(image.path, self)
                loader.ready.connect(self._on_preload_ready)
                loader.start()

    def _on_preload_ready(self, key: str, pixmap: QPixmap):
        """Cache a preloaded full-res pixmap (no UI swap)."""
        if key not in self._pixmap_cache:
            self._pixmap_cache[key] = pixmap
            while len(self._pixmap_cache) > self._PIXMAP_CACHE_MAX:
                self._pixmap_cache.popitem(last=False)

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
            if token in ('(', ')'):
                continue
            if token.lower() in proxy._KEYWORDS:
                continue
            if token.startswith('-') and len(token) > 1:
                token = token[1:]
            if ':' in token:
                key, _, value = token.partition(':')
                kl = key.lower()
                if kl == 'tag' and value:
                    patterns.append(('tag', value))
                elif kl == 'r' and value:
                    try:
                        patterns.append(('regex', re.compile(value, re.IGNORECASE)))
                    except re.error:
                        pass
            else:
                patterns.append(('text', token))
        self.tags_list.set_highlight_patterns(patterns)

    def _save_current_tags(self):
        if self._current_image_index < 0:
            return
        if self._panel_mode == 'caption':
            # Caption box is the active editor — flush it (with undo/autosave).
            self._commit_caption_edit()
            return
        image = self.model.get_image(self._current_image_index)
        if image is None:
            return
        image.tags = self.tags_list.get_tags()

    def _on_tags_pasted(self, tags: list[str], source_rows: list[int]):
        """Handle paste/clear from context menu on selected images."""
        bid = self._new_batch_id() if len(source_rows) > 1 else None
        for row in source_rows:
            image = self.model.get_image(row)
            if image is None:
                continue
            self._push_undo(row, list(image.tags), bid)
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
                self._sync_caption_view()
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
            bid = self._new_batch_id()
            for row in selected_rows:
                image = self.model.get_image(row)
                if image is None:
                    continue
                self._push_undo(row, list(image.tags), bid)
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
        first_new_row = self.tags_list.count()
        for tag in new_tags:
            item = QListWidgetItem(tag)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.tags_list._apply_tag_color(item, tag)
            self.tags_list.addItem(item)
        self.tag_input.clear()
        if self.tags_list.count() > 0:
            self.tags_list.scrollToBottom()
            self.tags_list.clearSelection()
            if len(new_tags) == 1:
                # Single tag: select only the last added
                self.tags_list.setCurrentRow(self.tags_list.count() - 1)
            else:
                # Multiple comma-separated tags: select all newly added
                for row in range(first_new_row, self.tags_list.count()):
                    self.tags_list.item(row).setSelected(True)
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
            bid = self._new_batch_id()
            for row in selected_rows:
                image = self.model.get_image(row)
                if image is None:
                    continue
                if tag_set & set(image.tags):
                    self._push_undo(row, list(image.tags), bid)
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
        # Rough CLIP token estimate: ~4 chars per token
        if self._panel_mode == 'caption':
            text = self.caption_edit.toPlainText()
            tokens = len(text) // 4 if text else 0
            label = f'{len(text)} chars | ~{tokens} tokens'
            if tokens > 75:
                label += ' (over 75!)'
            self.caption_count_label.setText(label)
            return
        count = self.tags_list.count()
        text = ', '.join(self.tags_list.get_tags())
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
            bid = self._new_batch_id()
            for i, image in enumerate(self.model.images):
                if tag in image.tags:
                    self._push_undo(i, list(image.tags), bid)
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

        # Remember selection row before dialog steals focus
        selected_rows = [idx.row() for idx in
                         self.all_tags_list.selectedIndexes()]
        restore_row = min(selected_rows) if selected_rows else -1

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
        bid = self._new_batch_id()
        cur_idx = self._current_image_index
        cur_has_batch = False
        for i, image in enumerate(self.model.images):
            before = len(image.tags)
            if tag_set & set(image.tags):
                self._push_undo(i, list(image.tags), bid)
                if i == cur_idx:
                    cur_has_batch = True
                image.tags = [t for t in image.tags if t not in tag_set]
                self._auto_save_image(image)
            count += before - len(image.tags)

        # Ensure current image has a batch entry so Ctrl+Z can find it
        if not cur_has_batch and cur_idx >= 0:
            cur_image = self.model.get_image(cur_idx)
            if cur_image is not None:
                self._push_undo(cur_idx, list(cur_image.tags), bid)

        self._rebuild_all_tags()

        # Restore selection to next tag (or last item if at end)
        if restore_row >= 0 and self.all_tags_list.count() > 0:
            new_row = min(restore_row, self.all_tags_list.count() - 1)
            self.all_tags_list.setCurrentRow(new_row)

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
        text = self.captioner_combo.currentText()
        is_comfy = text == 'ComfyUI Workflow'
        is_lm = text == 'LM Studio'
        is_wd = not is_comfy and not is_lm
        self.wd_settings.setVisible(is_wd)
        self.comfy_settings.setVisible(is_comfy)
        self.lmstudio_settings.setVisible(is_lm)
        # Auto-populate the model list the first time LM Studio is shown
        # (skipped during startup restore to avoid a blocking network call).
        if (is_lm and self._lm_autorefresh_enabled
                and self.lmstudio_model_combo.count() == 0):
            self._refresh_lmstudio_models()

    def _refresh_lmstudio_models(self):
        """Query the LM Studio URL and fill the model dropdown."""
        from ltd.lmstudio.client import LMStudioClient
        self.caption_status.setText('LM Studio: querying models...')
        QApplication.processEvents()
        try:
            models = LMStudioClient().list_models()
        except Exception as e:
            self.caption_status.setText(f'LM Studio: {e}')
            return
        # Preserve the current/saved selection across a refresh.
        target = self.lmstudio_model_combo.currentText().strip() or \
            get_settings().value('caption/lmstudio_model', '', type=str)
        self.lmstudio_model_combo.blockSignals(True)
        self.lmstudio_model_combo.clear()
        self.lmstudio_model_combo.addItems(models)
        # Default to the last-used model if it's still available, otherwise
        # leave the selector empty rather than auto-picking the first entry.
        if target and target in models:
            self.lmstudio_model_combo.setCurrentText(target)
        else:
            self.lmstudio_model_combo.setCurrentIndex(-1)
        self.lmstudio_model_combo.blockSignals(False)
        if models:
            self.caption_status.setText(
                f'LM Studio: {len(models)} vision model(s) available')
        else:
            self.caption_status.setText(
                'LM Studio: no vision-capable models found')

    def _create_captioner(self):
        text = self.captioner_combo.currentText()
        index = self.captioner_combo.currentIndex()
        if text == 'ComfyUI Workflow':
            workflow_text = self._caption_workflow_selector.get_workflow_text()
            if not workflow_text:
                QMessageBox.warning(self, 'Warning', 'No workflow provided.')
                return None
            return ComfyUICaptioner(workflow_text)
        elif text == 'LM Studio':
            model = self.lmstudio_model_combo.currentText().strip()
            if not model:
                QMessageBox.warning(
                    self, 'Warning',
                    'No LM Studio model selected. Click Refresh to load '
                    'the models from your LM Studio instance.')
                return None
            return LMStudioCaptioner(
                model=model,
                system_prompt=self.lmstudio_system_prompt.toPlainText(),
                append_context=self.lmstudio_append_check.isChecked(),
                separator=self._get_separator(),
            )
        else:
            exclude_text = self.exclude_input.text().strip()
            exclude_tags = [t.strip() for t in exclude_text.split(',')
                           if t.strip()] if exclude_text else None
            return self._WD_MODELS[index](
                min_probability=self.prob_spin.value(),
                max_tags=self.max_tags_spin.value(),
                exclude_tags=exclude_tags,
            )

    def _merge_tags(self, existing: list[str], new_tags: list[str]) -> list[str]:
        """Merge new tags based on caption position setting."""
        # Some captioners (LM Studio) produce a complete caption that replaces
        # the existing one regardless of the WD-tagger Position combo.
        if getattr(self, '_caption_replace', False):
            return list(new_tags)
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
        # Whether results overwrite the caption (LM Studio) or merge via the
        # Position combo (WD taggers / ComfyUI).
        self._caption_replace = getattr(captioner, 'replaces_caption', False)

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
                self._sync_caption_view()
                self._update_token_count()
        self._rebuild_all_tags()

    def _on_caption_finished(self):
        self.caption_progress.setVisible(False)
        self.caption_cancel_btn.setVisible(False)
        self.caption_current_btn.setEnabled(True)
        self.caption_selected_btn.setEnabled(True)
        self.caption_all_btn.setEnabled(True)
        self._worker = None
        from ltd.utils.sound import play_completion_sound
        play_completion_sound()

    def _on_caption_error(self, msg):
        QMessageBox.critical(self, 'Captioning Error', msg)

    # ------------------------------------------------------------------
    # Generate image from caption (ComfyUI) + side-by-side comparison
    # ------------------------------------------------------------------

    def _generated_path(self, image: ImageItem) -> Path | None:
        """Cached generated render for an image, or None if not generated yet."""
        if self._generated_dir is None:
            return None
        path = self._generated_dir / generated_cache_name(image)
        return path if path.exists() else None

    def _run_generation(self, mode: str = 'all'):
        if self._gen_worker is not None:
            return
        workflow_text = self._generate_workflow_selector.get_workflow_text()
        if not workflow_text:
            QMessageBox.warning(self, 'Warning', 'No workflow provided.')
            return
        if self._generated_dir is None:
            QMessageBox.information(self, 'Info', 'Load a folder first.')
            return

        # Flush any pending caption edit so the render uses the latest text.
        self._save_current_tags()

        if mode == 'current':
            image = self.model.get_image(self._current_image_index)
            if image is None:
                return
            images = [image]
        elif mode == 'selected':
            images = self.image_list.get_selected_images()
            if not images:
                QMessageBox.information(self, 'Info', 'No images selected.')
                return
        else:
            images = list(self.model.images)
        if not images:
            return

        generator = CaptionImageGenerator(
            workflow_text, self._generated_dir,
            separator=self._get_separator(), megapixels=1.0)
        self._gen_images = images
        worker = GenerationWorker(generator, images)
        worker.progress.connect(self._on_generation_progress)
        worker.status.connect(self._on_caption_status)
        worker.generated.connect(self._on_generated_result)
        worker.finished_work.connect(self._on_generation_finished)
        worker.error.connect(self._on_generation_error)
        self._gen_worker = worker

        self.generate_progress.setVisible(len(images) > 1)
        self.generate_cancel_btn.setVisible(True)
        self.generate_current_btn.setEnabled(False)
        self.generate_selected_btn.setEnabled(False)
        self.generate_all_btn.setEnabled(False)
        worker.start()

    def _cancel_generation(self):
        if self._gen_worker:
            self._gen_worker.cancel()

    def _on_generation_progress(self, current, total):
        self.generate_progress.setMaximum(total)
        self.generate_progress.setValue(current)

    def _on_generated_result(self, index: int, path: str):
        images = getattr(self, '_gen_images', self.model.images)
        if not (0 <= index < len(images)):
            return
        image = images[index]
        all_idx = self.model.images.index(image) \
            if image in self.model.images else -1
        # If the generated image is the one on screen, show the comparison now.
        if all_idx == self._current_image_index:
            self._show_comparison(image.path, Path(path))

    def _on_generation_finished(self):
        self.generate_progress.setVisible(False)
        self.generate_cancel_btn.setVisible(False)
        self.generate_current_btn.setEnabled(True)
        self.generate_selected_btn.setEnabled(True)
        self.generate_all_btn.setEnabled(True)
        self._gen_worker = None
        from ltd.utils.sound import play_completion_sound
        play_completion_sound()

    def _on_generation_error(self, msg):
        QMessageBox.critical(self, 'Generation Error', msg)

    def _show_comparison(self, original_path: Path, generated_path: Path):
        """Show the original (full resolution) beside/above the generated
        render. Landscape originals stack top/bottom; square/portrait sit
        left/right."""
        # Reuse the cached full-res original if present (from viewing/preload),
        # else load it full-res — the original must not be shown downscaled.
        key = str(original_path)
        orig = self._pixmap_cache.get(key)
        if orig is None:
            orig = load_pixmap(original_path)
            if orig is not None:
                self._pixmap_cache[key] = orig
                self._pixmap_cache.move_to_end(key)
                while len(self._pixmap_cache) > self._PIXMAP_CACHE_MAX:
                    self._pixmap_cache.popitem(last=False)
        gen = load_pixmap(generated_path)
        if gen is None:
            # Generated render unreadable — just show the original.
            self.image_viewer.load_image(orig)
            return
        if orig is None:
            self.image_viewer.load_image(gen)
            return
        stack_vertical = orig.width() > orig.height()  # landscape → top/bottom
        self.image_viewer.load_comparison(orig, gen, stack_vertical)

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
            bid = self._new_batch_id()

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
                    self._push_undo(idx, old_tags, bid)
                    self._auto_save_image(image)

            # Refresh
            if self._current_image_index >= 0:
                image = self.model.get_image(self._current_image_index)
                if image:
                    self._pre_tags_snapshot = list(image.tags)
                    self._skip_snapshot = True
                    self.tags_list.set_tags(image.tags)
                    self._skip_snapshot = False
            self._sync_caption_view()
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
            bid = self._new_batch_id()
            for i, image in enumerate(self.model.images):
                if image.tags:
                    new_tags = dlg.reorder_tags(image.tags)
                    if new_tags != image.tags:
                        self._push_undo(i, list(image.tags), bid)
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
            self._sync_caption_view()
            self._rebuild_all_tags()
            self.caption_status.setText(
                f'Reordered tags in {count} image(s)')

    # ------------------------------------------------------------------
    # Bulk tools
    # ------------------------------------------------------------------

    def _remove_empty_all(self):
        self._save_current_tags()
        count = 0
        bid = self._new_batch_id()
        for i, image in enumerate(self.model.images):
            before = len(image.tags)
            if any(not t.strip() for t in image.tags):
                self._push_undo(i, list(image.tags), bid)
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
        self._sync_caption_view()
        self._rebuild_all_tags()
        self.caption_status.setText(f'Removed {count} empty tag(s)')

    def _remove_duplicates_all(self):
        self._save_current_tags()
        count = 0
        bid = self._new_batch_id()
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
                self._push_undo(i, list(image.tags), bid)
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
        self._sync_caption_view()
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
        bid = self._new_batch_id()
        for i, image in enumerate(self.model.images):
            if image.filename in snapshot:
                new_tags = snapshot[image.filename]
                if new_tags != image.tags:
                    self._push_undo(i, list(image.tags), bid)
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
        self._sync_caption_view()
        self._rebuild_all_tags()
        self._update_token_count()

        unmatched = len(snapshot) - matched
        msg = f'Restored captions for {matched} image(s)'
        if unmatched > 0:
            msg += f' ({unmatched} in snapshot not found in current folder)'
        self.caption_status.setText(msg)

    def _restore_some_from_snapshot(self):
        """Restore only specific tags from a CSV snapshot (merge, no overwrite)."""
        if not self.model.images:
            self.caption_status.setText('No images loaded')
            return
        default_dir = str(self.model.images[0].path.parent) \
            if self.model.images else ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Restore Tags from Snapshot', default_dir,
            'CSV Files (*.csv)')
        if not path:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Tags to Restore')
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.addWidget(QLabel(
            'Enter comma-separated tag names (or a regex pattern):'
        ))
        input_edit = QLineEdit()
        dlg_layout.addWidget(input_edit)
        use_regex_cb = QCheckBox('Use regex')
        dlg_layout.addWidget(use_regex_cb)
        dlg_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        dlg_buttons.accepted.connect(dlg.accept)
        dlg_buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(dlg_buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        user_input = input_edit.text().strip()
        if not user_input:
            return
        is_regex = use_regex_cb.isChecked()

        if is_regex:
            try:
                tag_rx = re.compile(user_input, re.IGNORECASE)
            except re.error as e:
                QMessageBox.warning(self, 'Error', f'Invalid regex: {e}')
                return
        else:
            requested_tags = {t.strip() for t in user_input.split(',')
                              if t.strip()}
            if not requested_tags:
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

        # Apply — only add requested tags that are missing
        matched = 0
        bid = self._new_batch_id()
        for i, image in enumerate(self.model.images):
            if image.filename in snapshot:
                snapshot_tags = snapshot[image.filename]
                if is_regex:
                    tags_to_add = [t for t in snapshot_tags
                                   if tag_rx.search(t)
                                   and t not in image.tags]
                else:
                    tags_to_add = [t for t in snapshot_tags
                                   if t in requested_tags
                                   and t not in image.tags]
                if tags_to_add:
                    self._push_undo(i, list(image.tags), bid)
                    image.tags.extend(tags_to_add)
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
        self._sync_caption_view()
        self._rebuild_all_tags()
        self._update_token_count()

        self.caption_status.setText(
            f'Restored requested tags for {matched} image(s)')

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
