"""Manage Gen Images tab.

A 3-pane viewer for AI-generated images that carry generation metadata
(forge / a1111 ``parameters`` chunk or ComfyUI ``prompt`` chunk):

  Left   — Image list with filter input, multi-select, copy/move ops.
  Center — Read-only image canvas (auto-fit, Ctrl+wheel zoom, drag pan).
  Right  — Tabs: positive prompt tags, all-tags counter+filter, settings table.

Metadata is read-only here.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication,
                               QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QPlainTextEdit,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from ltd.data.gen_metadata import GenMetadata
from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel
from ltd.data.tag_dictionary import TagDictionary
from ltd.settings import get_settings
from ltd.utils.image_utils import load_pixmap, load_pixmap_preview
from ltd.widgets.gen_image_list import GenImageList
from ltd.widgets.info_button import DynamicInfoButton, focus_in
from ltd.widgets.info_text import (GEN_SHORTCUTS_CANVAS, GEN_SHORTCUTS_LIST,
                                   GEN_SHORTCUTS_TAGS)
from ltd.widgets.loading_dialog import loading_dialog
from ltd.workers.metadata_worker import MetadataWorker


# ---------------------------------------------------------------------------
# Image canvas (auto-fit, ctrl+wheel zoom, drag pan when zoomed)
# ---------------------------------------------------------------------------

class _FullResLoader(QThread):
    ready = Signal(str, QPixmap)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        pixmap = load_pixmap(self._path)
        if pixmap:
            self.ready.emit(str(self._path), pixmap)


class GenImageCanvas(QGraphicsView):
    navigate_image = Signal(int)

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
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
            self._manually_zoomed = False
            self._fit_image()
        else:
            self._scene.setSceneRect(0, 0, 0, 0)

    def replace_pixmap(self, pixmap):
        if self._pixmap_item is None:
            return
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if not self._manually_zoomed:
            self._fit_image()

    def _fit_image(self):
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)
            self._manually_zoomed = False
            self._update_drag_mode()

    def _update_drag_mode(self):
        if self._manually_zoomed:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

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
            self._update_drag_mode()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
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
# Tab
# ---------------------------------------------------------------------------

class GenImagesTab(QWidget):
    _PIXMAP_CACHE_MAX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = ImageListModel()
        self._current_image_index = -1
        self._all_tags: dict[str, int] = {}
        self._pixmap_cache: OrderedDict[str, QPixmap | None] = OrderedDict()
        self._fullres_loader: _FullResLoader | None = None

        self._tag_dict = TagDictionary()
        autocomp_dir = Path('autocompletions')
        if autocomp_dir.is_dir():
            self._tag_dict.load_directory(autocomp_dir)
        elif Path('tags.csv').exists():
            self._tag_dict.load_csv(Path('tags.csv'))

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.image_list = GenImageList(self.model)
        splitter.addWidget(self.image_list)

        self.canvas = GenImageCanvas()
        splitter.addWidget(self.canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Top row of right panel: format label + shortcuts info button
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.format_label = QLabel('No image loaded')
        self.format_label.setStyleSheet('font-weight: bold;')
        header.addWidget(self.format_label, stretch=1)
        self.shortcuts_info = DynamicInfoButton(self._build_shortcuts_help)
        header.addWidget(self.shortcuts_info)
        right_layout.addLayout(header)

        self.right_tabs = QTabWidget()

        # ---- Image Tags ----
        tags_tab = QWidget()
        tags_layout = QVBoxLayout(tags_tab)
        self.tags_list = QListWidget()
        self.tags_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tags_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        tags_layout.addWidget(self.tags_list)
        self.tag_count_label = QLabel('Tags: 0')
        tags_layout.addWidget(self.tag_count_label)
        self.right_tabs.addTab(tags_tab, 'Image Tags')

        self._act_copy_image_tag = self._make_tag_copy_action(
            self.tags_list, lambda: self._copy_tags_from(self.tags_list))

        # ---- All Tags ----
        all_tab = QWidget()
        all_layout = QVBoxLayout(all_tab)
        self.all_tags_filter = QLineEdit()
        self.all_tags_filter.setPlaceholderText('Filter tags...')
        self.all_tags_filter.setClearButtonEnabled(True)
        all_layout.addWidget(self.all_tags_filter)
        self.all_tags_list = QListWidget()
        self.all_tags_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.all_tags_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        all_layout.addWidget(self.all_tags_list)
        self.all_tags_count_label = QLabel('0 tags')
        all_layout.addWidget(self.all_tags_count_label)
        self.right_tabs.addTab(all_tab, 'All Tags')

        self._act_copy_all_tag = self._make_tag_copy_action(
            self.all_tags_list,
            lambda: self._copy_tags_from(self.all_tags_list, strip_count=True))

        # ---- Image Settings ----
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        self.settings_table = QTableWidget(0, 2)
        self.settings_table.setHorizontalHeaderLabels(['Setting', 'Value'])
        self.settings_table.verticalHeader().setVisible(False)
        self.settings_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.settings_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.settings_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.settings_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.settings_table.setAlternatingRowColors(True)
        settings_layout.addWidget(self.settings_table)
        settings_layout.addWidget(QLabel('Prompt:'))
        self.prompt_view = QPlainTextEdit()
        self.prompt_view.setReadOnly(True)
        self.prompt_view.setMaximumHeight(140)
        settings_layout.addWidget(self.prompt_view)
        settings_layout.addWidget(QLabel('Negative prompt:'))
        self.negative_view = QPlainTextEdit()
        self.negative_view.setReadOnly(True)
        self.negative_view.setMaximumHeight(120)
        settings_layout.addWidget(self.negative_view)
        self.right_tabs.addTab(settings_tab, 'Image Settings')

        right_layout.addWidget(self.right_tabs)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([300, 800, 360])

        layout.addWidget(splitter)

    def _connect_signals(self):
        self.image_list.load_directory_requested.connect(self._load_directory)
        self.image_list.current_changed.connect(self._on_current_changed)
        self.image_list.images_changed.connect(self._on_images_changed)
        self.canvas.navigate_image.connect(self._navigate)
        self.all_tags_filter.textChanged.connect(self._update_all_tags_view)
        self.tags_list.customContextMenuRequested.connect(
            self._show_image_tag_menu)
        self.all_tags_list.customContextMenuRequested.connect(
            self._show_all_tag_menu)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_directory(self, directory: str):
        self._pixmap_cache.clear()
        self._all_tags.clear()
        self._update_all_tags_view()
        self.canvas.load_image(None)
        self._show_metadata(None)

        with loading_dialog('Loading folder...', self) as dlg:
            self.model.load_directory(Path(directory))
            QApplication.processEvents()
            if not self.model.images:
                return

            dlg.set_message(
                f'Reading metadata for {len(self.model.images)} image(s)...')
            QApplication.processEvents()

            loop = QEventLoop()
            worker = MetadataWorker(self.model.images, self)
            worker.parsed.connect(self._on_metadata_parsed)
            worker.progress.connect(dlg.set_progress)
            worker.finished_all.connect(loop.quit)
            worker.start()
            loop.exec()
            worker.wait()

        self.image_list.refresh_filter()
        self._update_all_tags_view()
        if self.model.rowCount() > 0:
            self.image_list.select_index(0)

    def _on_metadata_parsed(self, image: ImageItem, md: GenMetadata):
        image.metadata = md
        for tag in md.tags:
            self._all_tags[tag] = self._all_tags.get(tag, 0) + 1

    def _on_images_changed(self):
        # Rows may have been removed via Move-to-outside-root.
        # Recompute the all-tags counter.
        self._all_tags.clear()
        for img in self.model.images:
            md = img.metadata
            if md is None:
                continue
            for tag in md.tags:
                self._all_tags[tag] = self._all_tags.get(tag, 0) + 1
        self._update_all_tags_view()
        if not self.model.images:
            self.canvas.load_image(None)
            self._show_metadata(None)

    # ------------------------------------------------------------------
    # Selection / display
    # ------------------------------------------------------------------

    def _on_current_changed(self, source_row: int):
        self._current_image_index = source_row
        if source_row < 0 or source_row >= len(self.model.images):
            self.canvas.load_image(None)
            self._show_metadata(None)
            return
        image = self.model.images[source_row]
        self._show_image(image)
        self._show_metadata(image.metadata)

    def _show_image(self, image: ImageItem):
        path_key = str(image.path)
        cached = self._pixmap_cache.get(path_key)
        if cached is not None:
            self._pixmap_cache.move_to_end(path_key)
            self.canvas.load_image(cached)
        else:
            preview = load_pixmap_preview(image.path)
            self.canvas.load_image(preview)
            self._pixmap_cache[path_key] = preview
            while len(self._pixmap_cache) > self._PIXMAP_CACHE_MAX:
                self._pixmap_cache.popitem(last=False)
            if self._fullres_loader is not None:
                self._fullres_loader.quit()
                self._fullres_loader.wait()
            self._fullres_loader = _FullResLoader(image.path, self)
            self._fullres_loader.ready.connect(self._on_fullres_ready)
            self._fullres_loader.start()

    def _on_fullres_ready(self, path_key: str, pixmap: QPixmap):
        self._pixmap_cache[path_key] = pixmap
        if self._current_image_index >= 0:
            current = self.model.images[self._current_image_index]
            if str(current.path) == path_key:
                self.canvas.replace_pixmap(pixmap)

    def _show_metadata(self, md: GenMetadata | None):
        self.tags_list.clear()
        if md is None:
            self.format_label.setText('No image loaded')
            self.tag_count_label.setText('Tags: 0')
            self.settings_table.setRowCount(0)
            self.prompt_view.setPlainText('')
            self.negative_view.setPlainText('')
            return

        if not md.has_data:
            self.format_label.setText('No metadata found')
        else:
            res = md.resolution_str or '?'
            self.format_label.setText(f'Format: {md.format}  •  {res}')

        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        for tag in md.tags:
            item = QListWidgetItem(tag)
            if self._tag_dict.is_loaded():
                color = self._tag_dict.get_color(tag, dark=dark)
                if color:
                    item.setForeground(QBrush(color))
            self.tags_list.addItem(item)
        self.tag_count_label.setText(f'Tags: {len(md.tags)}')

        self.settings_table.setRowCount(0)
        rows = []
        if md.width and md.height:
            rows.append(('Resolution', f'{md.width}x{md.height}'))
        for k, v in md.settings.items():
            rows.append((str(k), str(v)))
        self.settings_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.settings_table.setItem(r, 0, QTableWidgetItem(k))
            self.settings_table.setItem(r, 1, QTableWidgetItem(v))

        self.prompt_view.setPlainText(md.positive_prompt)
        self.negative_view.setPlainText(md.negative_prompt)

    # ------------------------------------------------------------------
    # All Tags
    # ------------------------------------------------------------------

    def _update_all_tags_view(self):
        self.all_tags_list.clear()
        filter_text = self.all_tags_filter.text().lower()
        items = sorted(self._all_tags.items(), key=lambda kv: -kv[1])
        if filter_text:
            items = [(t, c) for t, c in items if filter_text in t.lower()]
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        for tag, count in items:
            item = QListWidgetItem(f'{tag} ({count})')
            if self._tag_dict.is_loaded():
                color = self._tag_dict.get_color(tag, dark=dark)
                if color:
                    item.setForeground(QBrush(color))
            self.all_tags_list.addItem(item)
        total = len(self._all_tags)
        if filter_text:
            self.all_tags_count_label.setText(f'{len(items)} / {total} tags')
        else:
            self.all_tags_count_label.setText(f'{total} tags')

    # ------------------------------------------------------------------
    # Shortcut help
    # ------------------------------------------------------------------

    def _build_shortcuts_help(self) -> str:
        sections: list[str] = []
        if focus_in(self.image_list.list_view):
            sections.append(GEN_SHORTCUTS_LIST)
        elif focus_in(self.tags_list) or focus_in(self.all_tags_list):
            sections.append(GEN_SHORTCUTS_TAGS)
        elif focus_in(self.canvas):
            sections.append(GEN_SHORTCUTS_CANVAS)
        if not sections:
            sections = [GEN_SHORTCUTS_LIST, GEN_SHORTCUTS_TAGS,
                        GEN_SHORTCUTS_CANVAS]
        return '<br>'.join(sections)

    # ------------------------------------------------------------------
    # Tag context menus  → "Copy tag" / "Filter images by {tag}"
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_count(text: str) -> str:
        return text.rsplit(' (', 1)[0] if ' (' in text else text

    @staticmethod
    def _quote_tag_term(tag: str) -> str:
        return f'tag:"{tag}"' if ' ' in tag else f'tag:{tag}'

    def _filter_by_tag(self, tag: str):
        self.image_list.filter_input.setText(self._quote_tag_term(tag))

    def _add_filter_by_tag(self, tag: str):
        term = self._quote_tag_term(tag)
        current = self.image_list.filter_input.text().strip()
        new_text = f'{current} and {term}' if current else term
        self.image_list.filter_input.setText(new_text)

    def _make_tag_copy_action(self, list_widget: QListWidget, handler):
        """Bind Ctrl+C on the given tag list to ``handler``."""
        act = QAction('Copy Tag', list_widget)
        act.setShortcut(QKeySequence('Ctrl+C'))
        act.setShortcutContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut)
        act.triggered.connect(handler)
        list_widget.addAction(act)
        return act

    def _copy_tags_from(self, list_widget: QListWidget,
                        strip_count: bool = False):
        items = list_widget.selectedItems()
        if not items:
            return
        texts = [self._strip_count(i.text()) if strip_count else i.text()
                 for i in items]
        QApplication.clipboard().setText(', '.join(texts))

    def _show_image_tag_menu(self, pos):
        item = self.tags_list.itemAt(pos)
        if item is None:
            return
        tag = item.text()
        n = len(self.tags_list.selectedItems())
        copy_label = 'Copy Tags' if n > 1 else 'Copy Tag'
        self._act_copy_image_tag.setText(copy_label)
        menu = QMenu(self)
        menu.addAction(self._act_copy_image_tag)
        act_filter = menu.addAction(f'Filter images by "{tag}"')
        act_add = menu.addAction(f'Add filter by "{tag}"')
        action = menu.exec(self.tags_list.mapToGlobal(pos))
        if action == act_filter:
            self._filter_by_tag(tag)
        elif action == act_add:
            self._add_filter_by_tag(tag)

    def _show_all_tag_menu(self, pos):
        item = self.all_tags_list.itemAt(pos)
        if item is None:
            return
        tag = self._strip_count(item.text())
        n = len(self.all_tags_list.selectedItems())
        copy_label = 'Copy Tags' if n > 1 else 'Copy Tag'
        self._act_copy_all_tag.setText(copy_label)
        menu = QMenu(self)
        menu.addAction(self._act_copy_all_tag)
        act_filter = menu.addAction(f'Filter images by "{tag}"')
        act_add = menu.addAction(f'Add filter by "{tag}"')
        action = menu.exec(self.all_tags_list.mapToGlobal(pos))
        if action == act_filter:
            self._filter_by_tag(tag)
        elif action == act_add:
            self._add_filter_by_tag(tag)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, direction: int):
        proxy = self.image_list.proxy
        view = self.image_list.list_view
        current = view.currentIndex()
        if not current.isValid():
            return
        new_row = current.row() + direction
        if 0 <= new_row < proxy.rowCount():
            view.setCurrentIndex(proxy.index(new_row, 0))
