"""Custom autocomplete popup for tag input with category colors."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QListWidget,
                               QListWidgetItem, QStyle, QStyledItemDelegate,
                               QStyleOptionViewItem, QWidget, QVBoxLayout)

from ltd.data.tag_dictionary import TagDictionary


class TagItemDelegate(QStyledItemDelegate):
    """Paints tag name (colored) on the left, post count (gray) on the right."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index):
        self.initStyleOption(option, index)
        painter.save()

        # Draw selection background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect,
                             option.palette.highlight().color().lighter(160))

        data = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            painter.restore()
            super().paint(painter, option, index)
            return

        text = data.get('display', '')
        count_text = data.get('count', '')
        color = data.get('color')
        is_used = data.get('used', False)

        rect = option.rect.adjusted(4, 0, -4, 0)
        font = painter.font()

        if is_used:
            font.setItalic(True)
            painter.setFont(font)
            painter.setOpacity(0.5)

        # Tag name
        if color:
            painter.setPen(QPen(color))
        elif option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(option.palette.highlightedText().color())
        else:
            painter.setPen(option.palette.text().color())

        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft |
                         Qt.AlignmentFlag.AlignVCenter, text)

        # Post count on right
        if count_text:
            painter.setOpacity(0.4 if is_used else 0.6)
            gray = QColor(150, 150, 150)
            painter.setPen(gray)
            painter.drawText(rect, Qt.AlignmentFlag.AlignRight |
                             Qt.AlignmentFlag.AlignVCenter, count_text)

        painter.restore()


class TagCompleterPopup(QWidget):
    """Frameless popup showing tag suggestions with category colors."""
    tag_selected = Signal(str)  # emits display_name

    def __init__(self, tag_dict: TagDictionary, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip)
        self._tag_dict = tag_dict
        self._session_tags: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.setItemDelegate(TagItemDelegate(self._list))
        self._list.setMouseTracking(True)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        self.setFixedWidth(350)
        self.setMaximumHeight(300)

    def set_session_tags(self, tags: list[str]):
        """Set session tags for fallback completion."""
        self._session_tags = tags

    def show_for(self, widget: QWidget, text: str,
                 current_tags: set[str], dark: bool = True):
        """Show popup with suggestions for the given text."""
        if not text or len(text.strip()) < 1:
            self.hide_popup()
            return

        text = text.strip()
        self._list.clear()

        if self._tag_dict.is_loaded():
            session_set = set(self._session_tags)
            results = self._tag_dict.search(text, limit=50,
                                            used_tags=current_tags,
                                            session_tags=session_set)
            dict_names = {dt.display_name for dt in results}
            dict_names |= {dt.name for dt in results}

            # Add session tags not in dictionary results (dataset-only tags)
            q = text.lower()
            extra = [t for t in self._session_tags
                     if q in t.lower() and t not in dict_names]
            # Sort: current image tags first, then alphabetical
            extra.sort(key=lambda t: (t not in current_tags, t.lower()))

            for tag in extra:
                is_used = tag in current_tags
                color = self._tag_dict.get_color(tag, dark=dark)
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, {
                    'display': tag,
                    'count': '',
                    'color': color,
                    'used': is_used,
                })
                item.setData(Qt.ItemDataRole.DisplayRole, tag)
                self._list.addItem(item)

            for dt in results:
                item = QListWidgetItem()
                color = self._tag_dict.get_color(dt.display_name, dark=dark)
                is_used = (dt.display_name in current_tags
                           or dt.name in current_tags)
                count_str = self._format_count(dt.post_count)
                item.setData(Qt.ItemDataRole.UserRole, {
                    'display': dt.display_name,
                    'count': count_str,
                    'color': color,
                    'used': is_used,
                })
                item.setData(Qt.ItemDataRole.DisplayRole, dt.display_name)
                self._list.addItem(item)
        else:
            # Fallback: session tags (used tags on top)
            q = text.lower()
            matches = [t for t in self._session_tags
                       if q in t.lower()]
            matches.sort(key=lambda t: (t not in current_tags, t.lower()))
            for tag in matches[:50]:
                is_used = tag in current_tags
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, {
                    'display': tag,
                    'count': '',
                    'color': None,
                    'used': is_used,
                })
                item.setData(Qt.ItemDataRole.DisplayRole, tag)
                self._list.addItem(item)

        if self._list.count() == 0:
            self.hide_popup()
            return

        # Position below the widget
        pos = widget.mapToGlobal(widget.rect().bottomLeft())
        self.move(pos)

        # Resize height to content
        row_h = self._list.sizeHintForRow(0) or 20
        h = min(row_h * self._list.count() + 4, 300)
        self.setFixedHeight(h)

        self._list.setCurrentRow(0)
        self.show()

    def hide_popup(self):
        self.hide()

    def select_next(self):
        row = self._list.currentRow()
        if row < self._list.count() - 1:
            self._list.setCurrentRow(row + 1)

    def select_previous(self):
        row = self._list.currentRow()
        if row > 0:
            self._list.setCurrentRow(row - 1)

    def confirm_selection(self):
        item = self._list.currentItem()
        if item:
            self._emit_tag(item)

    def _on_item_clicked(self, item: QListWidgetItem):
        self._emit_tag(item)

    def _emit_tag(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            self.tag_selected.emit(data['display'])
        else:
            self.tag_selected.emit(item.text())
        self.hide_popup()

    @staticmethod
    def _format_count(n: int) -> str:
        if n >= 1_000_000:
            return f'{n / 1_000_000:.1f}M'
        if n >= 1_000:
            return f'{n / 1_000:.0f}k'
        return str(n) if n > 0 else ''
