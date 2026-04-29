"""Image list for the Manage Gen Images tab.

Reuses CaptionImageList's filter grammar and extends it with terms
specific to generation metadata: full-text metadata search, exact
resolution match, width / height comparisons, and a metadata-format
selector.

Multi-select; context menu actions:
  Copy Prompt   (Ctrl+C)        — concatenated positive prompts
  Copy Image    (Ctrl+Shift+C)  — copy file(s) to chosen folder
  Move Image    (Ctrl+M)        — move file(s); refreshes model
  Delete Image  (Ctrl+D)        — moveToTrash + remove from model
  Open in Default App / Show in Explorer (single selection)

Shortcuts use ``WidgetWithChildrenShortcut`` so they only fire when the
list (or one of its inner widgets) has focus — this leaves QLineEdit
copy/paste alone.
"""
from __future__ import annotations

import operator
import re
import shutil
from fnmatch import fnmatchcase
from pathlib import Path

from PySide6.QtCore import QFile, QModelIndex, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QListView, QMenu, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel
from ltd.widgets.caption_image_list import (ImageFilterProxyModel,
                                             NavigableListView)
from ltd.widgets.info_button import InfoButton
from ltd.widgets.info_text import GEN_IMAGES_FILTER_HELP


class GenImageFilterProxyModel(ImageFilterProxyModel):
    """Filter proxy with terms relevant to generation metadata.

    New term keys (kept compatible with the parent grammar):
      - WxH literal (e.g. ``1024x1024``) → exact resolution match
      - ``size:WxH``                     → exact resolution match
      - ``w:>=N`` / ``h:<=N``            → numeric width/height comparison
      - ``meta:text``                    → substring search in raw metadata
      - ``format:forge|comfyui``         → match parsed metadata format
      - bare text term also searches positive/negative prompt + raw metadata
    """

    _RES_RE = re.compile(r'^(\d+)\s*x\s*(\d+)$', re.IGNORECASE)
    _OPS = {
        '=': operator.eq, '==': operator.eq, '!=': operator.ne,
        '<': operator.lt, '>': operator.gt,
        '<=': operator.le, '>=': operator.ge,
    }
    _NUM_RE = re.compile(r'^(==?|!=|<=?|>=?)(\d+)$')

    def _parse_term(self, token: str) -> tuple:
        m = self._RES_RE.match(token)
        if m:
            return ('size', int(m.group(1)), int(m.group(2)))
        if ':' in token:
            key, _, value = token.partition(':')
            key_lower = key.lower()
            if key_lower in ('size', 'wxh'):
                rm = self._RES_RE.match(value)
                if rm:
                    return ('size', int(rm.group(1)), int(rm.group(2)))
            elif key_lower in ('w', 'width'):
                nm = self._NUM_RE.match(value)
                if nm:
                    return ('dim', 'w', nm.group(1), int(nm.group(2)))
            elif key_lower in ('h', 'height'):
                nm = self._NUM_RE.match(value)
                if nm:
                    return ('dim', 'h', nm.group(1), int(nm.group(2)))
            elif key_lower == 'meta':
                return ('meta', value)
            elif key_lower == 'format':
                return ('format', value.lower())
        return super()._parse_term(token)

    def _term_matches(self, image: ImageItem, term: tuple) -> bool:
        kind = term[0]
        if kind == 'size':
            _, w, h = term
            return image.width == w and image.height == h
        if kind == 'dim':
            _, axis, op_str, num = term
            op_fn = self._OPS.get(op_str)
            if op_fn is None:
                return True
            value = image.width if axis == 'w' else image.height
            return op_fn(value, num)
        if kind == 'meta':
            md = image.metadata
            if md is None:
                return False
            return term[1].lower() in md.raw_text.lower()
        if kind == 'format':
            md = image.metadata
            if md is None:
                return term[1] == ''
            return md.format == term[1]
        if kind == 'text':
            needle = term[1].lower()
            if needle in image.display_name.lower():
                return True
            if any(needle in t.lower() for t in image.tags):
                return True
            md = image.metadata
            if md is not None:
                if needle in md.positive_prompt.lower():
                    return True
                if needle in md.negative_prompt.lower():
                    return True
                if needle in md.raw_text.lower():
                    return True
            return False
        if kind == 'tag':
            md = image.metadata
            tag_source = md.tags if md is not None else image.tags
            p = term[1].lower()
            if '*' in p or '?' in p:
                return any(fnmatchcase(t.lower(), p) for t in tag_source)
            return any(t.strip().lower() == p for t in tag_source)
        if kind == 'tags_num':
            md = image.metadata
            tag_source = md.tags if md is not None else image.tags
            op_str, num = term[1], term[2]
            op_fn = self._OPS.get(op_str)
            return op_fn(len(tag_source), num) if op_fn else True
        if kind == 'regex':
            md = image.metadata
            tag_source = md.tags if md is not None else image.tags
            rx = term[1]
            if rx.search(image.display_name) is not None:
                return True
            if any(rx.search(t) is not None for t in tag_source):
                return True
            if md is not None and rx.search(md.raw_text) is not None:
                return True
            return False
        return super()._term_matches(image, term)


class GenImageList(QWidget):
    """Image list with folder-load button, filter, multi-select, context menu."""

    current_changed = Signal(int)
    selection_changed = Signal(list)
    load_directory_requested = Signal(str)
    images_changed = Signal()  # rows mutated (move/delete) — refresh derived state

    def __init__(self, model: ImageListModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.proxy = GenImageFilterProxyModel(model)
        self._setup_ui()
        self._setup_actions()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton('Load Folder...')
        layout.addWidget(self.load_button)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(
            'Filter: tag:x, 1024x1024, w:>=512, meta:str, format:forge, '
            'tags:>20, name:*, r:regex, or, not/-')
        self.filter_input.setClearButtonEnabled(True)
        filter_row.addWidget(self.filter_input, stretch=1)
        filter_row.addWidget(InfoButton(GEN_IMAGES_FILTER_HELP))
        layout.addLayout(filter_row)

        self.list_view = NavigableListView()
        self.list_view.setModel(self.proxy)
        self.list_view.setViewMode(QListView.ViewMode.ListMode)
        self.list_view.setIconSize(QSize(160, 90))
        self.list_view.setSpacing(2)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setSelectionMode(
            QListView.SelectionMode.ExtendedSelection)
        self.list_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        layout.addWidget(self.list_view, stretch=1)

        self.counter_label = QLabel('0 / 0')
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.counter_label)

    def _setup_actions(self):
        self._act_select_all = self._make_action(
            'Select All', 'Ctrl+A', self.list_view.selectAll)
        self._act_open = self._make_action(
            'Open in Default App', 'Ctrl+O', self._open_image)
        self._act_show = self._make_action(
            'Show in Explorer', '', self._show_in_explorer)
        self._act_copy_prompt = self._make_action(
            'Copy Prompt', 'Ctrl+C', self._copy_prompt)
        self._act_copy_image = self._make_action(
            'Copy Image to...', 'Ctrl+Shift+C', self._copy_images_to)
        self._act_move_image = self._make_action(
            'Move Image to...', 'Ctrl+M', self._move_images_to)
        self._act_delete_image = self._make_action(
            'Delete Image', 'Ctrl+D', self._delete_images)

    def _make_action(self, text: str, shortcut: str, handler):
        # Action is owned by list_view with WidgetWithChildrenShortcut so the
        # shortcut only fires when the list (or one of its descendants) has
        # focus — this keeps Ctrl+C from clobbering QLineEdit copy etc.
        act = QAction(text, self.list_view)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut)
        act.triggered.connect(handler)
        self.list_view.addAction(act)
        return act

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.load_button.clicked.connect(self._on_load_clicked)
        self.filter_input.textChanged.connect(self._on_filter_changed)
        sm = self.list_view.selectionModel()
        sm.currentChanged.connect(self._on_current_changed)
        sm.selectionChanged.connect(self._on_selection_changed)
        self.list_view.customContextMenuRequested.connect(
            self._show_context_menu)
        self.model.modelReset.connect(self._on_model_reset)

    def _on_model_reset(self):
        sm = self.list_view.selectionModel()
        for sig, slot in (
            (sm.currentChanged, self._on_current_changed),
            (sm.selectionChanged, self._on_selection_changed),
        ):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            sig.connect(slot)
        self._update_counter()

    def _on_filter_changed(self, text: str):
        self.proxy.set_filter_text(text)
        self._update_counter()

    def _on_current_changed(self, current: QModelIndex,
                            previous: QModelIndex):
        if current.isValid():
            source_idx = self.proxy.mapToSource(current)
            self.current_changed.emit(source_idx.row())
        self._update_counter()

    def _on_selection_changed(self):
        self.selection_changed.emit(self.selected_source_rows())
        self._update_counter()
        self._update_action_labels()

    def _update_counter(self):
        current = self.list_view.currentIndex()
        row = current.row() + 1 if current.isValid() else 0
        visible = self.proxy.rowCount()
        total = self.model.rowCount()
        sel_count = len(self.selected_source_rows())
        sel_text = f'  •  {sel_count} selected' if sel_count > 1 else ''
        if visible < total:
            self.counter_label.setText(
                f'{row} / {visible} ({total} total){sel_text}')
        else:
            self.counter_label.setText(f'{row} / {total}{sel_text}')

    def _update_action_labels(self):
        n = len(self.selected_source_rows())
        s = 's' if n != 1 else ''
        self._act_copy_image.setText(f'Copy Image{s} to...')
        self._act_move_image.setText(f'Move Image{s} to...')
        self._act_delete_image.setText(f'Delete Image{s}')
        self._act_copy_prompt.setText(
            'Copy Prompts' if n > 1 else 'Copy Prompt')
        self._act_open.setVisible(n <= 1)
        self._act_show.setVisible(n <= 1)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def refresh_filter(self):
        self.proxy.invalidateFilter()
        self._update_counter()

    def select_index(self, source_row: int):
        source_idx = self.model.index(source_row)
        proxy_idx = self.proxy.mapFromSource(source_idx)
        if proxy_idx.isValid():
            self.list_view.setCurrentIndex(proxy_idx)
            self.list_view.scrollTo(proxy_idx)

    def current_source_row(self) -> int:
        current = self.list_view.currentIndex()
        if current.isValid():
            return self.proxy.mapToSource(current).row()
        return -1

    def selected_source_rows(self) -> list[int]:
        rows = set()
        for proxy_idx in self.list_view.selectionModel().selectedIndexes():
            rows.add(self.proxy.mapToSource(proxy_idx).row())
        return sorted(rows)

    def get_selected_images(self) -> list[ImageItem]:
        return [self.model.images[r] for r in self.selected_source_rows()]

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos):
        idx = self.list_view.indexAt(pos)
        if not idx.isValid():
            return
        # Make sure the right-clicked row is in the selection.
        sm = self.list_view.selectionModel()
        if not sm.isSelected(idx):
            sm.select(idx, sm.SelectionFlag.ClearAndSelect)
            self.list_view.setCurrentIndex(idx)
        self._update_action_labels()

        menu = QMenu(self)
        menu.addAction(self._act_copy_prompt)
        menu.addAction(self._act_copy_image)
        menu.addAction(self._act_move_image)
        menu.addAction(self._act_delete_image)
        menu.addSeparator()
        menu.addAction(self._act_select_all)
        if self._act_open.isVisible():
            menu.addAction(self._act_open)
        if self._act_show.isVisible():
            menu.addAction(self._act_show)
        menu.exec(self.list_view.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _open_image(self):
        images = self.get_selected_images()
        if images:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(images[0].path)))

    def _show_in_explorer(self):
        images = self.get_selected_images()
        if images:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(images[0].path.parent)))

    def _copy_prompt(self):
        images = self.get_selected_images()
        if not images:
            return
        prompts = []
        for img in images:
            md = img.metadata
            if md is not None and md.positive_prompt:
                prompts.append(md.positive_prompt)
        if prompts:
            QApplication.clipboard().setText('\n\n'.join(prompts))

    def _copy_images_to(self):
        images = self.get_selected_images()
        if not images:
            return
        dest = QFileDialog.getExistingDirectory(
            self, f'Copy {len(images)} Image(s) to...')
        if not dest:
            return
        dest_path = Path(dest)
        for img in images:
            try:
                shutil.copy2(img.path, dest_path / img.filename)
            except OSError as e:
                QMessageBox.critical(
                    self, 'Error', f'Failed to copy {img.filename}: {e}')
                return

    def _move_images_to(self):
        images = self.get_selected_images()
        if not images:
            return
        dest = QFileDialog.getExistingDirectory(
            self, f'Move {len(images)} Image(s) to...')
        if not dest:
            return
        dest_path = Path(dest)
        rows = self.selected_source_rows()
        root = self.model.root_directory
        dest_inside_root = False
        if root:
            try:
                dest_path.relative_to(root)
                dest_inside_root = True
            except ValueError:
                pass

        for img in images:
            try:
                shutil.move(str(img.path), dest_path / img.filename)
            except OSError as e:
                QMessageBox.critical(
                    self, 'Error', f'Failed to move {img.filename}: {e}')
                return

        if dest_inside_root and root is not None:
            for i, img in enumerate(images):
                old_key = str(img.path)
                new_path = dest_path / img.filename
                img.path = new_path
                img.relative_path = new_path.relative_to(root).as_posix()
                self.model._thumbnail_cache.pop(old_key, None)
                self.model.invalidate_thumbnail(rows[i])
        else:
            self.model.remove_rows(rows)
            self.images_changed.emit()
        self._update_counter()

    def _delete_images(self):
        rows = self.selected_source_rows()
        if not rows:
            return
        n = len(rows)
        s = 's' if n != 1 else ''
        reply = QMessageBox.question(
            self, 'Delete Image' + s,
            f'Move {n} image{s} to the recycle bin?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        for row in rows:
            image = self.model.images[row]
            f = QFile(str(image.path))
            if not f.moveToTrash():
                QMessageBox.critical(
                    self, 'Error', f'Failed to delete {image.filename}.')

        self.model.remove_rows(rows)
        self.images_changed.emit()
        self._update_counter()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _on_load_clicked(self):
        from ltd.settings import get_settings
        settings = get_settings()
        last_dir = settings.value('last_gen_images_directory', '', type=str)
        directory = QFileDialog.getExistingDirectory(
            self, 'Select Gen Images Directory', last_dir)
        if directory:
            settings.setValue('last_gen_images_directory', directory)
            self.load_directory_requested.emit(directory)
