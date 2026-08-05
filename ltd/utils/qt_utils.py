"""Small Qt helpers shared across tabs."""
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox,
                               QLineEdit, QPlainTextEdit, QTextEdit, QWidget)

# Widgets that consume plain character keys, so tab shortcuts must not steal
# Space/A/D from them.
_TEXT_INPUT_TYPES = (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox)


def is_text_input(widget: QWidget | None) -> bool:
    """True if the widget is a text-entry field (editable combos included)."""
    if widget is None:
        return False
    if isinstance(widget, _TEXT_INPUT_TYPES):
        return True
    if isinstance(widget, QComboBox) and widget.isEditable():
        return True
    # QLineEdit inside an editable combo / spin box reports its own type,
    # but custom composites may nest one — walk a couple of parents up.
    parent = widget.parentWidget()
    if isinstance(parent, (QComboBox, QAbstractSpinBox)):
        return True
    return False


def text_input_has_focus() -> bool:
    """True while the user is typing into a text field anywhere in the app.

    Tabs use this to bail out of key event filters that would otherwise
    swallow ordinary characters (Space, A, D) meant for the text field.
    """
    return is_text_input(QApplication.focusWidget())
