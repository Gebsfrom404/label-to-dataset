"""Caption-specific image list with filter bar, multi-select, context menu."""
import operator
import re
import shutil
from fnmatch import fnmatchcase
from pathlib import Path

from PySide6.QtCore import (QFile, QModelIndex, QSize, QSortFilterProxyModel,
                             Qt, QUrl, Signal)
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog,
                               QLabel, QLineEdit, QListView, QMenu,
                               QMessageBox, QPushButton, QVBoxLayout,
                               QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel


# ---------------------------------------------------------------------------
# List view with W/S navigation
# ---------------------------------------------------------------------------

class NavigableListView(QListView):
    """QListView that also navigates with W (up) and S (down) keys."""

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_PageUp):
            current = self.currentIndex()
            if current.isValid() and current.row() > 0:
                new_idx = self.model().index(current.row() - 1, 0)
                self.setCurrentIndex(new_idx)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_PageDown):
            current = self.currentIndex()
            if current.isValid() and current.row() < self.model().rowCount() - 1:
                new_idx = self.model().index(current.row() + 1, 0)
                self.setCurrentIndex(new_idx)
            event.accept()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Filter proxy model
# ---------------------------------------------------------------------------

class ImageFilterProxyModel(QSortFilterProxyModel):
    """Proxy model that filters images by a search expression.

    Syntax (space-separated terms are ANDed):
      - plain text     : substring match on filename or any tag
      - tag:pattern    : exact tag match (wildcards * ? for glob)
      - name:pattern   : filename match (substring, or glob with *)
      - path:pattern   : full path match
      - tags:>N        : tag count comparison (=, !=, <, >, <=, >=)
      - or             : separate OR groups (any group matches)
      - not / -        : negate the next term  (-tag:1boy, not tag:1boy)
      - and            : explicit AND (implicit between terms, optional)
      - "quoted text"  : preserves spaces in values
    """

    _OPS = {
        '=': operator.eq, '==': operator.eq, '!=': operator.ne,
        '<': operator.lt, '>': operator.gt,
        '<=': operator.le, '>=': operator.ge,
    }
    _NUM_RE = re.compile(r'^(==?|!=|<=?|>=?)(\d+)$')
    _KEYWORDS = {'or', 'and', 'not'}

    def __init__(self, source_model: ImageListModel, parent=None):
        super().__init__(parent)
        self.setSourceModel(source_model)
        # list of OR-groups; each group is list of (negated, term)
        self._filter_groups: list[list[tuple]] = []

    def set_filter_text(self, text: str):
        raw = text.strip()
        self._filter_groups = self._parse(raw) if raw else []
        self.invalidateFilter()

    # -- parsing --

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split on spaces but respect quoted strings."""
        tokens: list[str] = []
        buf = ''
        in_quote = ''
        for ch in text:
            if in_quote:
                if ch == in_quote:
                    in_quote = ''
                else:
                    buf += ch
            elif ch in ('"', "'"):
                in_quote = ch
            elif ch == ' ':
                if buf:
                    tokens.append(buf)
                    buf = ''
            else:
                buf += ch
        if buf:
            tokens.append(buf)
        return tokens

    def _parse_term(self, token: str) -> tuple:
        if ':' in token:
            key, _, value = token.partition(':')
            key_lower = key.lower()
            if key_lower == 'tag':
                return ('tag', value)
            elif key_lower == 'name':
                return ('name', value)
            elif key_lower == 'path':
                return ('path', value)
            elif key_lower == 'tags':
                m = self._NUM_RE.match(value)
                if m:
                    return ('tags_num', m.group(1), int(m.group(2)))
        return ('text', token)

    def _parse(self, text: str) -> list[list[tuple]]:
        """Parse into OR-groups of AND-terms.

        Returns list of groups.  Image matches if ANY group matches.
        Within a group every (negated, term) must hold.
        """
        tokens = self._tokenize(text)
        groups: list[list[tuple]] = [[]]
        negate_next = False

        for token in tokens:
            low = token.lower()
            if low == 'or':
                if groups[-1]:
                    groups.append([])
                negate_next = False
                continue
            if low == 'and':
                continue
            if low == 'not':
                negate_next = True
                continue

            negated = negate_next
            negate_next = False

            # - prefix negation
            if token.startswith('-') and len(token) > 1:
                negated = True
                token = token[1:]

            groups[-1].append((negated, self._parse_term(token)))

        return [g for g in groups if g]

    # -- matching --

    @staticmethod
    def _wild(pattern: str, value: str) -> bool:
        """Case-insensitive wildcard match. Auto-wraps in * if no wildcards."""
        p = pattern.lower()
        v = value.lower()
        if '*' in p or '?' in p:
            return fnmatchcase(v, p)
        return p in v

    def _term_matches(self, image: ImageItem, term: tuple) -> bool:
        kind = term[0]
        if kind == 'text':
            needle = term[1].lower()
            return (needle in image.display_name.lower()
                    or any(needle in t.lower() for t in image.tags))
        elif kind == 'tag':
            p = term[1].lower()
            if '*' in p or '?' in p:
                return any(fnmatchcase(t.lower(), p) for t in image.tags)
            return any(t.strip().lower() == p for t in image.tags)
        elif kind == 'name':
            return self._wild(term[1], image.display_name)
        elif kind == 'path':
            return self._wild(term[1], str(image.path))
        elif kind == 'tags_num':
            op_str, num = term[1], term[2]
            op_fn = self._OPS.get(op_str)
            return op_fn(len(image.tags), num) if op_fn else True
        return True

    def _group_matches(self, image: ImageItem, group: list) -> bool:
        """All terms in the group must hold (AND)."""
        for negated, term in group:
            result = self._term_matches(image, term)
            if negated and result:
                return False
            if not negated and not result:
                return False
        return True

    def _image_matches(self, image: ImageItem) -> bool:
        """Any group matching is enough (OR across groups)."""
        for group in self._filter_groups:
            if self._group_matches(image, group):
                return True
        return False

    def filterAcceptsRow(self, source_row: int,
                         source_parent: QModelIndex) -> bool:
        if not self._filter_groups:
            return True
        source = self.sourceModel()
        image = source.data(source.index(source_row, 0),
                            Qt.ItemDataRole.UserRole)
        if image is None:
            return False
        return self._image_matches(image)


# ---------------------------------------------------------------------------
# Caption Image List Widget
# ---------------------------------------------------------------------------

class CaptionImageList(QWidget):
    """Image list with search bar, multi-selection, and context menu."""

    current_changed = Signal(int)           # source model row
    selection_changed = Signal(list)         # list of source model rows
    load_directory_requested = Signal(str)
    tags_paste_requested = Signal(list, list)  # tags, source indices
    images_deleted = Signal()                # after images removed from disk

    def __init__(self, model: ImageListModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.proxy = ImageFilterProxyModel(model)
        self._setup_ui()
        self._setup_context_menu()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton('Load Folder...')
        layout.addWidget(self.load_button)

        # Filter bar
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(
            'Filter: tag:x, name:*, tags:>N, or, not/-')
        self.filter_input.setClearButtonEnabled(True)
        layout.addWidget(self.filter_input)

        # List view with proxy model
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

        # Counter
        self.counter_label = QLabel('0 / 0')
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.counter_label)

    def _setup_context_menu(self):
        self.context_menu = QMenu(self)

        self._act_select_all = self.context_menu.addAction('Select All\tCtrl+A')
        self._act_select_all.triggered.connect(self.list_view.selectAll)
        sc_select_all = QShortcut(QKeySequence('Ctrl+A'), self.list_view)
        sc_select_all.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc_select_all.activated.connect(self.list_view.selectAll)

        self._act_invert = self.context_menu.addAction('Invert Selection')
        self._act_invert.setShortcut('Ctrl+I')
        self._act_invert.triggered.connect(self._invert_selection)

        self.context_menu.addSeparator()

        self._act_copy_tags = self.context_menu.addAction('Copy Tags')
        self._act_copy_tags.setShortcut('Ctrl+C')
        self._act_copy_tags.triggered.connect(self._copy_tags)

        self._act_paste_tags = self.context_menu.addAction('Paste Tags')
        self._act_paste_tags.setShortcut('Ctrl+V')
        self._act_paste_tags.triggered.connect(self._paste_tags)

        self._act_copy_names = self.context_menu.addAction('Copy File Names')
        self._act_copy_names.setShortcut('Ctrl+Alt+C')
        self._act_copy_names.triggered.connect(self._copy_file_names)

        self._act_copy_paths = self.context_menu.addAction('Copy Paths')
        self._act_copy_paths.setShortcut('Ctrl+Shift+C')
        self._act_copy_paths.triggered.connect(self._copy_paths)

        self.context_menu.addSeparator()

        self._act_move_images = self.context_menu.addAction(
            'Move Image to...')
        self._act_move_images.setShortcut('Ctrl+M')
        self._act_move_images.triggered.connect(self._move_images_to)

        self._act_copy_images = self.context_menu.addAction(
            'Copy Image to...')
        self._act_copy_images.setShortcut('Ctrl+Shift+M')
        self._act_copy_images.triggered.connect(self._copy_images_to)

        self._act_delete_images = self.context_menu.addAction(
            'Delete Image with Tags')
        self._act_delete_images.setShortcut('Ctrl+Del')
        self._act_delete_images.triggered.connect(self._delete_images_with_tags)

        self._act_open = self.context_menu.addAction(
            'Open in Default App')
        self._act_open.setShortcut('Ctrl+O')
        self._act_open.triggered.connect(self._open_image)

        # Register shortcuts on list_view so they work without menu open
        for act in self.context_menu.actions():
            if not act.isSeparator():
                self.list_view.addAction(act)

    def _connect_signals(self):
        self.load_button.clicked.connect(self._on_load_clicked)
        self.filter_input.textChanged.connect(self._on_filter_changed)
        self.list_view.selectionModel().currentChanged.connect(
            self._on_current_changed)
        self.list_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed)
        self.list_view.customContextMenuRequested.connect(
            self._show_context_menu)
        self.model.modelReset.connect(self._on_model_reset)

    def _on_model_reset(self):
        self.list_view.selectionModel().currentChanged.connect(
            self._on_current_changed)
        self.list_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed)
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
        self._update_counter()
        self._update_context_actions()

    def _update_counter(self):
        current = self.list_view.currentIndex()
        if current.isValid():
            # Show position within proxy (filtered) list
            row = current.row() + 1
        else:
            row = 0
        visible = self.proxy.rowCount()
        total = self.model.rowCount()
        if visible < total:
            self.counter_label.setText(f'{row} / {visible} ({total} total)')
        else:
            self.counter_label.setText(f'{row} / {total}')

    def _update_context_actions(self):
        count = len(self.list_view.selectionModel().selectedIndexes())
        s = 's' if count != 1 else ''
        self._act_copy_names.setText(f'Copy File Name{s}')
        self._act_copy_paths.setText(f'Copy Path{s}')
        self._act_move_images.setText(f'Move Image{s} to...')
        self._act_copy_images.setText(f'Copy Image{s} to...')
        self._act_delete_images.setText(f'Delete Image{s} with Tags')
        self._act_open.setVisible(count == 1)

    def _show_context_menu(self, pos):
        self.context_menu.exec(self.list_view.mapToGlobal(pos))

    # -- public API --

    def select_index(self, source_row: int):
        """Select image by source model row."""
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
        """Return source model row indices for all selected images."""
        rows = []
        for proxy_idx in self.list_view.selectionModel().selectedIndexes():
            source_idx = self.proxy.mapToSource(proxy_idx)
            rows.append(source_idx.row())
        return sorted(rows)

    def get_selected_images(self) -> list[ImageItem]:
        return [self.model.images[r] for r in self.selected_source_rows()]

    def go_to_previous(self):
        current = self.list_view.currentIndex()
        if current.isValid() and current.row() > 0:
            new_idx = self.proxy.index(current.row() - 1, 0)
            self.list_view.setCurrentIndex(new_idx)

    def go_to_next(self):
        current = self.list_view.currentIndex()
        if current.isValid() and current.row() < self.proxy.rowCount() - 1:
            new_idx = self.proxy.index(current.row() + 1, 0)
            self.list_view.setCurrentIndex(new_idx)

    # -- load --

    def _on_load_clicked(self):
        from ltd.settings import get_settings, DEFAULT_SETTINGS
        settings = get_settings()
        last_dir = settings.value('last_label_directory',
                                  DEFAULT_SETTINGS['last_label_directory'],
                                  type=str)
        directory = QFileDialog.getExistingDirectory(
            self, 'Select Image Directory', last_dir)
        if directory:
            settings.setValue('last_label_directory', directory)
            self.load_directory_requested.emit(directory)

    # -- context menu actions --

    def _invert_selection(self):
        sel_model = self.list_view.selectionModel()
        selected = set(idx.row() for idx in sel_model.selectedIndexes())
        all_rows = set(range(self.proxy.rowCount()))
        unselected = all_rows - selected

        from PySide6.QtCore import QItemSelection, QItemSelectionRange
        selection = QItemSelection()
        for row in unselected:
            idx = self.proxy.index(row, 0)
            selection.append(QItemSelectionRange(idx))
        sel_model.select(selection,
                         sel_model.SelectionFlag.ClearAndSelect)
        if unselected:
            first = min(unselected)
            self.list_view.setCurrentIndex(self.proxy.index(first, 0))

    def _copy_tags(self):
        images = self.get_selected_images()
        if not images:
            return
        captions = [', '.join(img.tags) for img in images]
        QApplication.clipboard().setText('\n'.join(captions))

    def _paste_tags(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            return
        rows = self.selected_source_rows()
        if not rows:
            return

        tags = [t.strip() for t in text.split(',') if t.strip()]
        if len(rows) > 1:
            reply = QMessageBox.question(
                self, 'Paste Tags',
                f'Paste tags to {len(rows)} selected images?',
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.tags_paste_requested.emit(tags, rows)

    def _copy_file_names(self):
        images = self.get_selected_images()
        names = [img.filename for img in images]
        QApplication.clipboard().setText('\n'.join(names))

    def _copy_paths(self):
        images = self.get_selected_images()
        paths = [str(img.path) for img in images]
        QApplication.clipboard().setText('\n'.join(paths))

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
                if img.caption_path.exists():
                    shutil.move(str(img.caption_path),
                                dest_path / img.caption_path.name)
            except OSError as e:
                QMessageBox.critical(
                    self, 'Error', f'Failed to move {img.filename}: {e}')
                return
        if dest_inside_root:
            for i, img in enumerate(images):
                old_key = str(img.path)
                new_path = dest_path / img.filename
                img.path = new_path
                img.relative_path = new_path.relative_to(root).as_posix()
                self.model._thumbnail_cache.pop(old_key, None)
                self.model.invalidate_thumbnail(rows[i])
        else:
            self.model.remove_rows(rows)
            self.images_deleted.emit()
        self._update_counter()

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
                if img.caption_path.exists():
                    shutil.copy2(img.caption_path,
                                 dest_path / img.caption_path.name)
            except OSError as e:
                QMessageBox.critical(
                    self, 'Error', f'Failed to copy {img.filename}: {e}')

    def _open_image(self):
        images = self.get_selected_images()
        if images:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(images[0].path)))

    def _delete_images_with_tags(self):
        rows = self.selected_source_rows()
        if not rows:
            return
        n = len(rows)
        s = 's' if n != 1 else ''
        reply = QMessageBox.question(
            self, 'Delete Images',
            f'Delete {n} image{s} and '
            f'{"its" if n == 1 else "their"} tag file{s}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        for row in rows:
            image = self.model.images[row]
            f = QFile(str(image.path))
            if not f.moveToTrash():
                QMessageBox.critical(
                    self, 'Error', f'Failed to delete {image.filename}.')
            caption_file = QFile(str(image.caption_path))
            if caption_file.exists():
                caption_file.moveToTrash()

        self.model.remove_rows(rows)
        self._update_counter()
        self.images_deleted.emit()
