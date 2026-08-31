"""Manage Duplicates tab.

Finds duplicate images across one or more source folders and helps delete the
copies you do not want to keep:

  Left   — Sources (check one as *original*) over the duplicate list.
  Center — Read-only preview of the highlighted image.
  Right  — Tabs: Search (algorithm + tolerance) and Actions (mark / delete).

Marking is separate from the list selection: marked rows are drawn in red and
only ``Delete Selected`` touches the disk (files go to the recycle bin).
Images from the original source are the reference and are never marked.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QFile, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar,
                               QPushButton, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from ltd.settings import get_settings
from ltd.utils.duplicate_utils import (ALGORITHM_BY_LABEL, ALGORITHM_LABELS,
                                       ALGO_ORB, ALGO_PHASH, MASK_SUFFIX,
                                       DuplicateGroup, SearchCache,
                                       threshold_description)
from ltd.utils.image_utils import load_pixmap_preview
from ltd.widgets.duplicate_image_list import DuplicateImageList, DuplicateListModel
from ltd.widgets.elided_label import ElidedLabel
from ltd.widgets.image_canvas import FullResLoader, ImageCanvas
from ltd.widgets.info_button import DynamicInfoButton, InfoButton, focus_in
from ltd.widgets.info_text import (DUPLICATES_ACTIONS_HELP,
                                   DUPLICATES_SEARCH_HELP,
                                   DUPLICATES_SHORTCUTS_CANVAS,
                                   DUPLICATES_SHORTCUTS_LIST,
                                   DUPLICATES_SHORTCUTS_SOURCES)
from ltd.widgets.settings_widgets import (SettingsCheckBox, SettingsComboBox,
                                          SettingsSlider)
from ltd.workers.duplicate_worker import DuplicateWorker


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SourceListWidget(QWidget):
    """Folders to search, with an exclusive *original* checkbox per folder."""

    sources_changed = Signal()

    _SOURCES_KEY = 'duplicates/sources'
    _ORIGINAL_KEY = 'duplicates/original'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self._updating = False
        self._setup_ui()
        self._restore()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.itemDoubleClicked.connect(self._open_folder)
        layout.addWidget(self.list_widget, stretch=1)

        hint = QLabel('Check a folder to use it as the original')
        hint.setWordWrap(True)
        hint.setEnabled(False)
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        self.add_button = QPushButton('Add Folder...')
        self.remove_button = QPushButton('Remove')
        self.add_button.clicked.connect(self._add_folder)
        self.remove_button.clicked.connect(self._remove_selected)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------

    def sources(self) -> list[tuple[Path, bool]]:
        """Every folder, paired with whether it is the original."""
        result = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            result.append((Path(item.text()),
                           item.checkState() == Qt.CheckState.Checked))
        return result

    def original_folders(self) -> list[Path]:
        return [path for path, is_original in self.sources() if is_original]

    def add_folder(self, folder: str):
        if any(str(path) == folder for path, _ in self.sources()):
            return
        item = QListWidgetItem(folder)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        item.setToolTip(folder)
        self._updating = True
        self.list_widget.addItem(item)
        self._updating = False
        self._persist()
        self.sources_changed.emit()

    def _add_folder(self):
        last = self.settings.value('last_duplicates_directory', '', type=str)
        folder = QFileDialog.getExistingDirectory(
            self, 'Add Source Folder', last)
        if not folder:
            return
        self.settings.setValue('last_duplicates_directory', folder)
        self.add_folder(folder)

    def _remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))
        self._persist()
        self.sources_changed.emit()

    def _open_folder(self, item: QListWidgetItem):
        QDesktopServices.openUrl(QUrl.fromLocalFile(item.text()))

    def _on_item_changed(self, item: QListWidgetItem):
        if self._updating:
            return
        # "Original" is exclusive: checking one clears every other row.
        if item.checkState() == Qt.CheckState.Checked:
            self._updating = True
            for row in range(self.list_widget.count()):
                other = self.list_widget.item(row)
                if other is not item:
                    other.setCheckState(Qt.CheckState.Unchecked)
            self._updating = False
        self._persist()
        self.sources_changed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.list_widget.hasFocus():
            self._remove_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------

    def _persist(self):
        paths = [str(path) for path, _ in self.sources()]
        originals = self.original_folders()
        self.settings.setValue(self._SOURCES_KEY, paths)
        self.settings.setValue(self._ORIGINAL_KEY,
                               str(originals[0]) if originals else '')

    def _restore(self):
        raw = self.settings.value(self._SOURCES_KEY, [])
        # QSettings collapses a one-element string list back to a bare string.
        if isinstance(raw, str):
            stored = [raw] if raw else []
        elif isinstance(raw, (list, tuple)):
            stored = [str(entry) for entry in raw]
        else:
            stored = []
        original = self.settings.value(self._ORIGINAL_KEY, '', type=str)
        self._updating = True
        for path in stored:
            item = QListWidgetItem(str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if str(path) == original
                               else Qt.CheckState.Unchecked)
            item.setToolTip(str(path))
            self.list_widget.addItem(item)
        self._updating = False


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class DuplicatesTab(QWidget):
    _PIXMAP_CACHE_MAX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.model = DuplicateListModel()
        self._worker: DuplicateWorker | None = None
        # Outlives each worker: signatures and measured pairs are reused so a
        # re-run only redoes what actually changed (see duplicate-detection.md).
        self._cache = SearchCache()
        self._current_row = -1
        self._scanned = 0
        self._skipped = 0
        self._reuse_note = ''
        self._pixmap_cache: OrderedDict[str, QPixmap | None] = OrderedDict()
        self._fullres_loader: FullResLoader | None = None

        self._setup_ui()
        self._connect_signals()
        self._update_threshold_label()
        self._update_marked_label()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([340, 760, 340])
        layout.addWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        left = QSplitter(Qt.Orientation.Vertical)

        sources_group = QGroupBox('Sources')
        sources_layout = QVBoxLayout(sources_group)
        self.source_list = SourceListWidget()
        sources_layout.addWidget(self.source_list)
        left.addWidget(sources_group)

        results_group = QGroupBox('Duplicates')
        results_layout = QVBoxLayout(results_group)
        self.duplicate_list = DuplicateImageList(self.model)
        results_layout.addWidget(self.duplicate_list)
        left.addWidget(results_group)

        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 3)
        left.setSizes([220, 620])
        return left

    def _build_center_panel(self) -> QWidget:
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.preview_label = ElidedLabel('No image selected')
        self.preview_label.setStyleSheet('font-weight: bold;')
        header.addWidget(self.preview_label, stretch=1)
        self.shortcuts_info = DynamicInfoButton(self._build_shortcuts_help)
        header.addWidget(self.shortcuts_info)
        center_layout.addLayout(header)

        self.canvas = ImageCanvas()
        center_layout.addWidget(self.canvas, stretch=1)

        self.detail_label = QLabel('')
        self.detail_label.setWordWrap(True)
        center_layout.addWidget(self.detail_label)
        return center

    def _build_right_panel(self) -> QWidget:
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self._build_search_tab(), 'Search')
        self.right_tabs.addTab(self._build_actions_tab(), 'Actions')
        return self.right_tabs

    def _build_search_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        header = QHBoxLayout()
        header.addWidget(QLabel('Algorithm'), stretch=1)
        header.addWidget(InfoButton(DUPLICATES_SEARCH_HELP))
        layout.addLayout(header)

        self.algorithm_combo = SettingsComboBox(
            'duplicates/algorithm', ALGORITHM_LABELS[ALGO_PHASH])
        self.algorithm_combo.addItems(list(ALGORITHM_LABELS.values()))
        layout.addWidget(self.algorithm_combo)

        tolerance_row = QHBoxLayout()
        tolerance_row.addWidget(QLabel('Tolerance'), stretch=1)
        self.tolerance_value = QLabel('')
        tolerance_row.addWidget(self.tolerance_value)
        layout.addLayout(tolerance_row)

        self.tolerance_slider = SettingsSlider('duplicates/tolerance', 25, 0, 100)
        self.tolerance_slider.setTickPosition(SettingsSlider.TickPosition.TicksBelow)
        self.tolerance_slider.setTickInterval(10)
        layout.addWidget(self.tolerance_slider)

        self.threshold_label = QLabel('')
        self.threshold_label.setWordWrap(True)
        self.threshold_label.setEnabled(False)
        layout.addWidget(self.threshold_label)

        self.search_button = QPushButton('Search')
        layout.addWidget(self.search_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setVisible(False)
        layout.addWidget(self.cancel_button)

        self.status_label = ElidedLabel('')
        layout.addWidget(self.status_label)

        self.summary_label = QLabel('')
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addStretch()
        return tab

    def _build_actions_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        header = QHBoxLayout()
        self.marked_label = QLabel('0 marked for deletion')
        self.marked_label.setStyleSheet('font-weight: bold;')
        header.addWidget(self.marked_label, stretch=1)
        header.addWidget(InfoButton(DUPLICATES_ACTIONS_HELP))
        layout.addLayout(header)

        self.select_all_button = QPushButton('Select All')
        self.select_biggest_button = QPushButton('Select All But Biggest')
        self.select_newest_button = QPushButton('Select All But Newest')
        self.clear_marks_button = QPushButton('Clear Selection')
        for button in (self.select_all_button, self.select_biggest_button,
                       self.select_newest_button, self.clear_marks_button):
            layout.addWidget(button)

        layout.addSpacing(12)
        self.sidecar_check = SettingsCheckBox(
            'duplicates/delete_sidecars', False,
            'Also delete .txt caption and mask')
        layout.addWidget(self.sidecar_check)

        self.delete_button = QPushButton('Delete Selected')
        layout.addWidget(self.delete_button)

        note = QLabel('Deleted files go to the recycle bin.')
        note.setWordWrap(True)
        note.setEnabled(False)
        layout.addWidget(note)

        layout.addStretch()
        return tab

    def _connect_signals(self):
        self.search_button.clicked.connect(self._start_search)
        self.cancel_button.clicked.connect(self._cancel_search)
        self.algorithm_combo.currentTextChanged.connect(
            self._update_threshold_label)
        self.tolerance_slider.valueChanged.connect(self._update_threshold_label)

        self.duplicate_list.current_changed.connect(self._on_current_changed)
        self.canvas.navigate_image.connect(self._navigate)
        self.model.marks_changed.connect(self._update_marked_label)

        self.select_all_button.clicked.connect(self.model.mark_all)
        self.select_biggest_button.clicked.connect(
            lambda: self.model.mark_all_but('biggest'))
        self.select_newest_button.clicked.connect(
            lambda: self.model.mark_all_but('newest'))
        self.clear_marks_button.clicked.connect(self.model.clear_marks)
        self.delete_button.clicked.connect(self._delete_marked)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _algorithm(self) -> str:
        return ALGORITHM_BY_LABEL.get(self.algorithm_combo.currentText(),
                                      ALGO_PHASH)

    def _update_threshold_label(self):
        tolerance = self.tolerance_slider.value()
        self.tolerance_value.setText(str(tolerance))
        self.threshold_label.setText(
            threshold_description(self._algorithm(), tolerance))

    def _start_search(self):
        if self._worker is not None:
            return
        sources = self.source_list.sources()
        if not sources:
            QMessageBox.information(
                self, 'No Sources',
                'Add at least one source folder to search.')
            return
        if all(is_original for _, is_original in sources):
            QMessageBox.information(
                self, 'Nothing to Compare',
                'Every source is marked as the original. Add a folder to '
                'compare against it, or uncheck the original.')
            return
        if not self._confirm_slow_search(sources):
            return

        self._reset_results()
        self._worker = DuplicateWorker(sources, self._algorithm(),
                                       self.tolerance_slider.value(),
                                       self._cache, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self.status_label.setText)
        self._worker.scanned.connect(self._on_scanned)
        self._worker.skipped.connect(self._on_skipped)
        self._worker.reused.connect(self._on_reused)
        self._worker.groups_found.connect(self._on_groups_found)
        self._worker.error.connect(self._on_error)
        self._worker.finished_work.connect(self._on_search_finished)

        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.search_button.setEnabled(False)
        self.summary_label.setText('')
        self._worker.start()

    def _confirm_slow_search(self, sources: list[tuple[Path, bool]]) -> bool:
        """Descriptor matching is quadratic — warn before a long run.

        Skipped once the cache already covers this algorithm at an equal or
        looser tolerance: that run only re-thresholds existing measurements.
        """
        if self._algorithm() != ALGO_ORB:
            return True
        if any(is_original for _, is_original in sources):
            return True
        if self._cache.pairs_algorithm == ALGO_ORB \
                and self.tolerance_slider.value() <= self._cache.pairs_tolerance:
            return True
        reply = QMessageBox.question(
            self, 'Slow Search',
            'Descriptor matching compares every pair of images and can take '
            'a long time on large folders.\n\nCheck a source as the original '
            'to only compare against it, or continue anyway?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        return reply == QMessageBox.StandardButton.Yes

    def _cancel_search(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText('Cancelling...')

    def shutdown(self):
        """Stop background work before the window is destroyed.

        A search can run for minutes; letting the QThread outlive its parent
        widget crashes on exit. Called from ``MainWindow.closeEvent``.
        """
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
        if self._fullres_loader is not None:
            self._fullres_loader.quit()
            self._fullres_loader.wait()
            self._fullres_loader = None
        self._cache.clear()

    def _on_progress(self, current: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_scanned(self, count: int):
        self._scanned = count

    def _on_skipped(self, count: int):
        self._skipped = count

    def _on_reused(self, note: str):
        self._reuse_note = note

    def _on_groups_found(self, groups: list[DuplicateGroup]):
        self.model.load_groups(groups, self._source_label)
        self.duplicate_list.refresh_filter()
        if self.model.rowCount() > 0:
            self.duplicate_list.select_index(0)
        else:
            self._show_image(None)
        self._update_summary(groups)

    def _update_summary(self, groups: list[DuplicateGroup]):
        duplicates = sum(len(g.members) for g in groups) - len(groups)
        parts = [f'{len(groups)} group(s), {max(duplicates, 0)} extra copies',
                 f'{self._scanned} image(s) scanned']
        if self._skipped:
            parts.append(f'{self._skipped} unreadable / too few features')
        if self._reuse_note:
            parts.append(self._reuse_note)
        self.summary_label.setText('  •  '.join(parts))

    def _on_error(self, message: str):
        QMessageBox.critical(self, 'Duplicate Search Error', message)

    def _on_search_finished(self):
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.search_button.setEnabled(True)
        self._worker = None
        self.status_label.setText('')
        from ltd.utils.sound import play_completion_sound
        play_completion_sound()

    def _reset_results(self):
        self._scanned = 0
        self._skipped = 0
        self._reuse_note = ''
        self._current_row = -1
        self._pixmap_cache.clear()
        self.model.clear()
        self._show_image(None)

    def _source_label(self, path: Path) -> str:
        """Path shown in the list: source folder name + path inside it."""
        best: str | None = None
        for folder, _ in self.source_list.sources():
            try:
                relative = path.relative_to(folder).as_posix()
            except ValueError:
                continue
            candidate = f'{folder.name}/{relative}'
            if best is None or len(candidate) < len(best):
                best = candidate
        return best if best is not None else path.name

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_current_changed(self, source_row: int):
        self._current_row = source_row
        if not 0 <= source_row < len(self.model.images):
            self._show_image(None)
            return
        self._show_image(self.model.images[source_row])

    def _show_image(self, image):
        if image is None:
            self.canvas.load_image(None)
            self.preview_label.setText('No image selected')
            self.detail_label.setText('')
            return

        self.preview_label.setText(str(image.path))
        self.detail_label.setText(self._detail_text(image))

        path_key = str(image.path)
        cached = self._pixmap_cache.get(path_key)
        if cached is not None:
            self._pixmap_cache.move_to_end(path_key)
            self.canvas.load_image(cached)
            return

        preview = load_pixmap_preview(image.path)
        self.canvas.load_image(preview)
        self._pixmap_cache[path_key] = preview
        while len(self._pixmap_cache) > self._PIXMAP_CACHE_MAX:
            self._pixmap_cache.popitem(last=False)
        if self._fullres_loader is not None:
            self._fullres_loader.quit()
            self._fullres_loader.wait()
        loader = FullResLoader(image.path, self)
        loader.ready.connect(self._on_fullres_ready)
        self._fullres_loader = loader
        loader.start()

    def _on_fullres_ready(self, path_key: str, pixmap: QPixmap):
        self._pixmap_cache[path_key] = pixmap
        if 0 <= self._current_row < len(self.model.images):
            if str(self.model.images[self._current_row].path) == path_key:
                self.canvas.replace_pixmap(pixmap)

    def _detail_text(self, image) -> str:
        row = self.model.row_for(image)
        if row is None:
            return ''
        modified = datetime.fromtimestamp(row.mtime).strftime(
            '%Y-%m-%d %H:%M') if row.mtime else 'unknown'
        parts = [f'{row.width}x{row.height}',
                 f'{row.size_bytes / (1024 * 1024):.2f} MB',
                 f'modified {modified}',
                 f'group {row.group + 1}']
        parts.append('original' if row.is_original
                     else f'similarity {row.score:.1%}')
        if row.marked:
            parts.append('MARKED FOR DELETION')
        return '  •  '.join(parts)

    def _navigate(self, direction: int):
        view = self.duplicate_list.list_view
        current = view.currentIndex()
        if not current.isValid():
            return
        new_row = current.row() + direction
        if 0 <= new_row < self.duplicate_list.proxy.rowCount():
            view.setCurrentIndex(
                self.duplicate_list.proxy.index(new_row, 0))

    # ------------------------------------------------------------------
    # Marks / deletion
    # ------------------------------------------------------------------

    def _update_marked_label(self):
        count = self.model.marked_count()
        self.marked_label.setText(f'{count} marked for deletion')
        self.delete_button.setEnabled(count > 0)
        if 0 <= self._current_row < len(self.model.images):
            self.detail_label.setText(
                self._detail_text(self.model.images[self._current_row]))

    def _delete_marked(self):
        rows = self.model.marked_rows()
        if not rows:
            return
        sidecars = self.sidecar_check.isChecked()
        extra = ('\n\nMatching .txt captions and -masklabel.png masks go with '
                 'them.') if sidecars else ''
        reply = QMessageBox.question(
            self, 'Delete Selected',
            f'Move {len(rows)} image(s) to the recycle bin?{extra}',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        failed: list[str] = []
        deleted: list[int] = []
        for row in rows:
            image = self.model.images[row]
            if self._move_to_trash(image.path):
                deleted.append(row)
                if sidecars:
                    self._delete_sidecars(image.path)
            else:
                failed.append(image.filename)

        self.model.remove_rows(deleted)
        self.model.prune_lone_groups()
        self.duplicate_list.refresh_filter()
        self._current_row = -1
        self._show_image(None)
        if self.model.rowCount() > 0:
            self.duplicate_list.select_index(0)
        if failed:
            QMessageBox.critical(
                self, 'Delete Failed',
                'Could not delete:\n' + '\n'.join(failed[:20]))

    def _delete_sidecars(self, path: Path):
        for sidecar in (path.with_suffix('.txt'),
                        path.parent / f'{path.stem}{MASK_SUFFIX}'):
            if sidecar.exists():
                self._move_to_trash(sidecar)

    @staticmethod
    def _move_to_trash(path: Path) -> bool:
        return QFile(str(path)).moveToTrash()

    # ------------------------------------------------------------------

    def _build_shortcuts_help(self) -> str:
        sections: list[str] = []
        if focus_in(self.duplicate_list.list_view):
            sections.append(DUPLICATES_SHORTCUTS_LIST)
        elif focus_in(self.source_list):
            sections.append(DUPLICATES_SHORTCUTS_SOURCES)
        elif focus_in(self.canvas):
            sections.append(DUPLICATES_SHORTCUTS_CANVAS)
        if not sections:
            sections = [DUPLICATES_SHORTCUTS_LIST, DUPLICATES_SHORTCUTS_SOURCES,
                        DUPLICATES_SHORTCUTS_CANVAS]
        return '<br>'.join(sections)
