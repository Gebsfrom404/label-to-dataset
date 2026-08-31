"""Image list for the Manage Duplicates tab.

Same navigation as the other image lists (thumbnails, W/S + arrows, filter
grammar, multi-select), with two additions specific to duplicate handling:

* Rows are grouped — every duplicate group gets an alternating background
  tint so the group boundaries are readable at a glance.
* Rows carry a *marked for deletion* flag, drawn in red. The mark is
  independent of the Qt selection, so navigating never loses it.

Originals (images coming from the source checked as *original*) are drawn in
blue, listed first inside their group, and can never be marked.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QModelIndex, QSize, Qt, QUrl, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont,
                           QKeySequence)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListView,
                               QMenu, QVBoxLayout, QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel
from ltd.settings import get_settings
from ltd.utils.duplicate_utils import DuplicateGroup
from ltd.widgets.caption_image_list import (ImageFilterProxyModel,
                                            NavigableListView)
from ltd.widgets.info_button import InfoButton
from ltd.widgets.info_text import DUPLICATES_FILTER_HELP

MARKED_COLOR = QColor('#e05a4e')
ORIGINAL_COLOR = QColor('#4aa3df')


@dataclass
class DuplicateRow:
    """Per-row duplicate bookkeeping, parallel to ``ImageListModel.images``."""
    group: int
    score: float
    is_original: bool = False
    marked: bool = False
    size_bytes: int = 0
    mtime: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def area(self) -> int:
        return self.width * self.height


class DuplicateListModel(ImageListModel):
    """Image list model that knows about duplicate groups and delete marks."""

    marks_changed = Signal()

    def __init__(self, thumbnail_width: int = 160, parent=None):
        super().__init__(thumbnail_width, parent)
        self.rows: list[DuplicateRow] = []
        self._by_path: dict[str, DuplicateRow] = {}
        self._group_brush: QBrush | None = None
        self._refresh_palette()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_groups(self, groups: list[DuplicateGroup],
                    label_for: Callable[[Path], str] | None = None):
        """Replace the contents with the members of every duplicate group."""
        self._refresh_palette()
        images: list[ImageItem] = []
        rows: list[DuplicateRow] = []
        for group_index, group in enumerate(groups):
            for member in group.members:
                try:
                    stat = member.path.stat()
                    size, mtime = stat.st_size, stat.st_mtime
                except OSError:
                    size, mtime = 0, 0.0
                relative = label_for(member.path) if label_for else ''
                images.append(ImageItem(path=member.path, width=member.width,
                                        height=member.height,
                                        relative_path=relative))
                rows.append(DuplicateRow(
                    group=group_index, score=member.score,
                    is_original=member.is_original, size_bytes=size,
                    mtime=mtime, width=member.width, height=member.height))
        self.rows = rows
        self._by_path = {str(img.path): row for img, row in zip(images, rows)}
        self.load_items(images)
        self.marks_changed.emit()

    def _refresh_palette(self):
        dark = get_settings().value('theme', 'dark', type=str) == 'dark'
        shade = QColor(255, 255, 255, 14) if dark else QColor(0, 0, 0, 14)
        self._group_brush = QBrush(shade)

    def row_for(self, image: ImageItem) -> DuplicateRow | None:
        return self._by_path.get(str(image.path))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def data(self, index: QModelIndex,  # pyright: ignore[reportIncompatibleMethodOverride]
             role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.rows):
            return super().data(index, role)
        image = self.images[index.row()]
        row = self.rows[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return f'{image.display_name}   [{self._score_text(row)}]'
        if role == Qt.ItemDataRole.ForegroundRole:
            if row.marked:
                return QBrush(MARKED_COLOR)
            if row.is_original:
                return QBrush(ORIGINAL_COLOR)
            return None
        if role == Qt.ItemDataRole.BackgroundRole:
            return self._group_brush if row.group % 2 else None
        if role == Qt.ItemDataRole.FontRole:
            if row.is_original or row.marked:
                font = QFont()
                font.setBold(True)
                return font
            return None
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(image, row)
        return super().data(index, role)

    @staticmethod
    def _score_text(row: DuplicateRow) -> str:
        """Near-identical scores need a decimal to stay distinguishable.

        With a 256-bit hash every meaningful match lands in the last few
        percent, so rounding to whole percent would print "100%" for a pair
        that is merely very close.
        """
        if row.is_original:
            return 'original'
        if row.score >= 0.99:
            return f'{row.score:.1%}'
        return f'{row.score:.0%}'

    def _tooltip(self, image: ImageItem, row: DuplicateRow) -> str:
        size_mb = row.size_bytes / (1024 * 1024)
        lines = [str(image.path),
                 f'{row.width}x{row.height}  •  {size_mb:.2f} MB',
                 f'Group {row.group + 1}']
        if row.is_original:
            lines.append('Original (never deleted)')
        else:
            lines.append(f'Similarity: {row.score:.2%}')
        if row.marked:
            lines.append('Marked for deletion')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    def marked_rows(self) -> list[int]:
        return [i for i, row in enumerate(self.rows) if row.marked]

    def marked_count(self) -> int:
        return sum(1 for row in self.rows if row.marked)

    def set_marked(self, rows: list[int], marked: bool):
        changed = []
        for row_index in rows:
            if not 0 <= row_index < len(self.rows):
                continue
            row = self.rows[row_index]
            if row.is_original and marked:
                continue  # originals are the reference — never deletable
            if row.marked != marked:
                row.marked = marked
                changed.append(row_index)
        self._emit_changed(changed)

    def toggle_marked(self, rows: list[int]):
        # A mixed selection marks everything; an all-marked one clears it.
        unmarked = [r for r in rows
                    if 0 <= r < len(self.rows) and not self.rows[r].marked
                    and not self.rows[r].is_original]
        if unmarked:
            self.set_marked(unmarked, True)
        else:
            self.set_marked(rows, False)

    def clear_marks(self):
        self.set_marked(list(range(len(self.rows))), False)

    def mark_all(self):
        self.set_marked(list(range(len(self.rows))), True)

    def mark_all_but(self, keep: str):
        """Keep one row per group ('biggest' or 'newest'), mark the rest.

        The keeper is picked across the whole group, originals included, so a
        group whose original is already the biggest / newest loses every copy.
        """
        def rank(row: DuplicateRow) -> tuple[float, float, float]:
            if keep == 'biggest':
                return (row.area, row.size_bytes, -row.mtime)
            return (row.mtime, row.area, row.size_bytes)

        by_group: dict[int, list[int]] = {}
        for index, row in enumerate(self.rows):
            by_group.setdefault(row.group, []).append(index)

        to_mark, to_clear = [], []
        for indices in by_group.values():
            keeper = max(indices, key=lambda i: rank(self.rows[i]))
            for index in indices:
                (to_clear if index == keeper else to_mark).append(index)
        self.set_marked(to_clear, False)
        self.set_marked(to_mark, True)

    def _emit_changed(self, rows: list[int]):
        if not rows:
            self.marks_changed.emit()
            return
        self.dataChanged.emit(self.index(min(rows)), self.index(max(rows)))
        self.marks_changed.emit()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def remove_rows(self, rows: list[int]):
        for row_index in sorted(rows, reverse=True):
            if not 0 <= row_index < len(self.images):
                continue
            self.beginRemoveRows(QModelIndex(), row_index, row_index)
            image = self.images.pop(row_index)
            self.rows.pop(row_index)
            self._by_path.pop(str(image.path), None)
            self._thumbnail_cache.pop(str(image.path), None)
            self.endRemoveRows()
        self.marks_changed.emit()

    def prune_lone_groups(self) -> int:
        """Drop rows whose group no longer holds at least two images."""
        counts: dict[int, int] = {}
        for row in self.rows:
            counts[row.group] = counts.get(row.group, 0) + 1
        orphans = [i for i, row in enumerate(self.rows)
                   if counts[row.group] < 2]
        if orphans:
            self.remove_rows(orphans)
        return len(orphans)

    def clear(self):
        super().clear()
        self.rows.clear()
        self._by_path.clear()
        self.marks_changed.emit()


class DuplicateFilterProxyModel(ImageFilterProxyModel):
    """Adds ``marked:``, ``original:`` and ``score:`` to the filter grammar."""

    _BOOLS = {'yes': True, 'true': True, '1': True,
              'no': False, 'false': False, '0': False}

    def _parse_term(self, token: str) -> tuple:
        if ':' in token:
            key, _, value = token.partition(':')
            key_lower = key.lower()
            if key_lower in ('marked', 'original'):
                flag = self._BOOLS.get(value.lower())
                if flag is not None:
                    return (key_lower, flag)
            elif key_lower == 'score':
                match = self._NUM_RE.match(value)
                if match:
                    return ('score', match.group(1), int(match.group(2)))
        return super()._parse_term(token)

    def _term_matches(self, image: ImageItem, term: tuple) -> bool:
        kind = term[0]
        if kind in ('marked', 'original', 'score'):
            source = self.sourceModel()
            row = source.row_for(image) if isinstance(
                source, DuplicateListModel) else None
            if row is None:
                return False
            if kind == 'marked':
                return row.marked == term[1]
            if kind == 'original':
                return row.is_original == term[1]
            op_fn = self._OPS.get(term[1])
            return op_fn(round(row.score * 100), term[2]) if op_fn else True
        return super()._term_matches(image, term)


class DuplicateImageList(QWidget):
    """Filterable, markable list of the images found by a duplicate search."""

    current_changed = Signal(int)     # source row
    selection_changed = Signal(list)  # source rows

    def __init__(self, model: DuplicateListModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.proxy = DuplicateFilterProxyModel(model)
        self._setup_ui()
        self._setup_actions()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(
            'Filter: marked:yes, original:no, score:>=95, name:*, path:*')
        self.filter_input.setClearButtonEnabled(True)
        filter_row.addWidget(self.filter_input, stretch=1)
        filter_row.addWidget(InfoButton(DUPLICATES_FILTER_HELP))
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

        self.counter_label = QLabel('No duplicates')
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.counter_label)

    def _setup_actions(self):
        self._act_toggle = self._make_action(
            'Mark / Unmark for Deletion', 'Space', self._toggle_marks)
        self._act_select_all = self._make_action(
            'Select All', 'Ctrl+A', self.list_view.selectAll)
        self._act_open = self._make_action(
            'Open in Default App', 'Ctrl+O', self._open_image)
        self._act_show = self._make_action(
            'Show in Explorer', '', self._show_in_explorer)

    def _make_action(self, text: str, shortcut: str, handler) -> QAction:
        act = QAction(text, self.list_view)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut)
        act.triggered.connect(handler)
        self.list_view.addAction(act)
        return act

    def _connect_signals(self):
        self.filter_input.textChanged.connect(self._on_filter_changed)
        self.list_view.customContextMenuRequested.connect(
            self._show_context_menu)
        self.model.modelReset.connect(self._on_model_reset)
        self.model.marks_changed.connect(self._update_counter)
        self._bind_selection_model()

    def _bind_selection_model(self):
        sm = self.list_view.selectionModel()
        sm.currentChanged.connect(self._on_current_changed)
        sm.selectionChanged.connect(self._on_selection_changed)

    def _on_model_reset(self):
        # Qt hands out a fresh selection model on reset in some paths.
        sm = self.list_view.selectionModel()
        for signal, slot in ((sm.currentChanged, self._on_current_changed),
                             (sm.selectionChanged, self._on_selection_changed)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            signal.connect(slot)
        self._update_counter()

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _on_filter_changed(self, text: str):
        self.proxy.set_filter_text(text)
        self._update_counter()

    def _on_current_changed(self, current: QModelIndex, _previous):
        if current.isValid():
            self.current_changed.emit(self.proxy.mapToSource(current).row())
        self._update_counter()

    def _on_selection_changed(self):
        self.selection_changed.emit(self.selected_source_rows())
        self._update_counter()

    def selected_source_rows(self) -> list[int]:
        rows = {self.proxy.mapToSource(index).row()
                for index in self.list_view.selectionModel().selectedIndexes()}
        return sorted(rows)

    def get_selected_images(self) -> list[ImageItem]:
        return [self.model.images[row] for row in self.selected_source_rows()]

    def select_index(self, source_row: int):
        proxy_index = self.proxy.mapFromSource(self.model.index(source_row))
        if proxy_index.isValid():
            self.list_view.setCurrentIndex(proxy_index)
            self.list_view.scrollTo(proxy_index)

    def refresh_filter(self):
        self.proxy.invalidateFilter()
        self._update_counter()

    def _update_counter(self):
        current = self.list_view.currentIndex()
        row = current.row() + 1 if current.isValid() else 0
        visible = self.proxy.rowCount()
        total = self.model.rowCount()
        if total == 0:
            self.counter_label.setText('No duplicates')
            return
        marked = self.model.marked_count()
        shown = f'{row} / {visible}' if visible == total \
            else f'{row} / {visible} ({total} total)'
        self.counter_label.setText(f'{shown}  •  {marked} marked')

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _toggle_marks(self):
        rows = self.selected_source_rows()
        if rows:
            self.model.toggle_marked(rows)

    def _open_image(self):
        images = self.get_selected_images()
        if images:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(images[0].path)))

    def _show_in_explorer(self):
        images = self.get_selected_images()
        if images:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(images[0].path.parent)))

    def _show_context_menu(self, pos):
        index = self.list_view.indexAt(pos)
        if not index.isValid():
            return
        sm = self.list_view.selectionModel()
        if not sm.isSelected(index):
            sm.select(index, sm.SelectionFlag.ClearAndSelect)
            self.list_view.setCurrentIndex(index)

        single = len(self.selected_source_rows()) <= 1
        menu = QMenu(self)
        menu.addAction(self._act_toggle)
        menu.addSeparator()
        menu.addAction(self._act_select_all)
        if single:
            menu.addAction(self._act_open)
            menu.addAction(self._act_show)
        menu.exec(self.list_view.mapToGlobal(pos))
