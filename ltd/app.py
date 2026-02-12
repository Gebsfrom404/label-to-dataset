from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImageReader, QPalette
from PySide6.QtWidgets import QApplication

from ltd.settings import get_settings, DEFAULT_SETTINGS


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    p.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    p.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
               QColor(127, 127, 127))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
               QColor(127, 127, 127))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
               QColor(127, 127, 127))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight,
               QColor(80, 80, 80))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText,
               QColor(127, 127, 127))
    return p


def _light_palette() -> QPalette:
    return QApplication.style().standardPalette()


def create_application() -> QApplication:
    app = QApplication([])
    app.setApplicationName('Label-to-Dataset')
    app.setApplicationDisplayName('Label-to-Dataset')
    app.setStyle('Fusion')
    QImageReader.setAllocationLimit(0)

    settings = get_settings()
    font = app.font()
    font_size = settings.value('font_size',
                               defaultValue=DEFAULT_SETTINGS['font_size'],
                               type=int)
    font.setPointSize(font_size)
    app.setFont(font)

    theme = settings.value('theme', defaultValue=DEFAULT_SETTINGS['theme'],
                           type=str)
    if theme == 'dark':
        app.setPalette(_dark_palette())
    else:
        app.setPalette(_light_palette())

    return app


def apply_theme(app: QApplication, theme: str):
    if theme == 'dark':
        app.setPalette(_dark_palette())
    else:
        app.setPalette(_light_palette())
