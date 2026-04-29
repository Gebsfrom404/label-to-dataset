"""Label-specific image list with filter bar for filtering by label count and class."""
import operator
import re
from fnmatch import fnmatchcase

from PySide6.QtCore import QModelIndex, QSize, QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QPushButton, QVBoxLayout, QWidget)

from ltd.data.image_item import ImageItem
from ltd.data.image_list_model import ImageListModel
from ltd.widgets.info_button import InfoButton
from ltd.widgets.info_text import LABEL_FILTER_HELP


class LabelFilterProxyModel(QSortFilterProxyModel):
    """Proxy model that filters images by label count and class.

    Syntax (space-separated terms are ANDed):
      - plain text     : substring match on filename
      - class:name     : has label with given class (glob with * ?)
      - name:pattern   : filename match (substring, or glob with *)
      - labels:>N      : label count comparison (=, !=, <, >, <=, >=)
      - or             : separate OR groups
      - not / -        : negate the next term
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
        self._filter_groups: list[list[tuple]] = []
        # class_id -> name mapping, set by LabelImageList
        self._class_names: dict[int, str] = {}

    def set_class_names(self, mapping: dict[int, str]):
        self._class_names = mapping

    def set_filter_text(self, text: str):
        raw = text.strip()
        self._filter_groups = self._parse(raw) if raw else []
        self.invalidateFilter()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
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
            if key_lower == 'class':
                return ('class', value)
            elif key_lower == 'name':
                return ('name', value)
            elif key_lower in ('labels', 'label'):
                m = self._NUM_RE.match(value)
                if m:
                    return ('labels_num', m.group(1), int(m.group(2)))
        return ('text', token)

    def _parse(self, text: str) -> list[list[tuple]]:
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

            if token.startswith('-') and len(token) > 1:
                negated = True
                token = token[1:]

            groups[-1].append((negated, self._parse_term(token)))

        return [g for g in groups if g]

    def _term_matches(self, image: ImageItem, term: tuple) -> bool:
        kind = term[0]
        if kind == 'text':
            needle = term[1].lower()
            return needle in image.filename.lower()
        elif kind == 'class':
            pattern = term[1].lower()
            has_wildcard = '*' in pattern or '?' in pattern
            for label in image.labels:
                class_name = self._class_names.get(label.class_id,
                                                    f'class_{label.class_id}')
                if has_wildcard:
                    if fnmatchcase(class_name.lower(), pattern):
                        return True
                else:
                    if class_name.lower() == pattern:
                        return True
            return False
        elif kind == 'name':
            p = term[1].lower()
            v = image.filename.lower()
            if '*' in p or '?' in p:
                return fnmatchcase(v, p)
            return p in v
        elif kind == 'labels_num':
            op_str, num = term[1], term[2]
            op_fn = self._OPS.get(op_str)
            return op_fn(len(image.labels), num) if op_fn else True
        return True

    def _group_matches(self, image: ImageItem, group: list) -> bool:
        for negated, term in group:
            result = self._term_matches(image, term)
            if negated and result:
                return False
            if not negated and not result:
                return False
        return True

    def _image_matches(self, image: ImageItem) -> bool:
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


class LabelImageList(QWidget):
    """Image list with filter bar for label tab."""

    current_changed = Signal(int)           # source model row
    load_directory_requested = Signal(str)

    def __init__(self, model: ImageListModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.proxy = LabelFilterProxyModel(model)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton('Load Folder...')
        layout.addWidget(self.load_button)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(
            'Filter: class:x, labels:>N, name:*, or, not/-')
        self.filter_input.setClearButtonEnabled(True)
        filter_row.addWidget(self.filter_input, stretch=1)
        filter_row.addWidget(InfoButton(LABEL_FILTER_HELP))
        layout.addLayout(filter_row)

        self.list_view = QListView()
        self.list_view.setModel(self.proxy)
        self.list_view.setViewMode(QListView.ViewMode.ListMode)
        self.list_view.setIconSize(QSize(160, 90))
        self.list_view.setSpacing(2)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setSelectionMode(
            QListView.SelectionMode.SingleSelection)
        layout.addWidget(self.list_view, stretch=1)

        self.counter_label = QLabel('0 / 0')
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.counter_label)

    def _connect_signals(self):
        self.load_button.clicked.connect(self._on_load_clicked)
        self.filter_input.textChanged.connect(self._on_filter_changed)
        self.list_view.selectionModel().currentChanged.connect(
            self._on_current_changed)
        self.model.modelReset.connect(self._on_model_reset)

    def _on_model_reset(self):
        self.list_view.selectionModel().currentChanged.connect(
            self._on_current_changed)
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

    def _update_counter(self):
        current = self.list_view.currentIndex()
        if current.isValid():
            row = current.row() + 1
        else:
            row = 0
        visible = self.proxy.rowCount()
        total = self.model.rowCount()
        if visible < total:
            self.counter_label.setText(f'{row} / {visible} ({total} total)')
        else:
            self.counter_label.setText(f'{row} / {total}')

    def update_class_names(self, classes):
        """Update class name mapping for filter. Takes list of LabelClass."""
        mapping = {c.class_id: c.name for c in classes}
        self.proxy.set_class_names(mapping)
        # Re-apply filter if active
        if self.filter_input.text().strip():
            self.proxy.invalidateFilter()

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

    def reapply_filter(self):
        """Re-run the current filter (e.g. after labels changed)."""
        if self.filter_input.text().strip():
            self.proxy.invalidateFilter()
            self._update_counter()
