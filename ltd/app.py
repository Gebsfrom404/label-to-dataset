import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImageReader, QPalette
from PySide6.QtWidgets import QApplication

if sys.platform == 'win32':
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        'label-to-dataset.label-to-dataset'
    )

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
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    p.setColor(QPalette.ColorRole.WindowText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Text, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Button, QColor(228, 228, 228))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    p.setColor(QPalette.ColorRole.Link, QColor(0, 102, 204))
    p.setColor(QPalette.ColorRole.Highlight, QColor(51, 140, 230))
    p.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
               QColor(160, 160, 160))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
               QColor(160, 160, 160))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
               QColor(160, 160, 160))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight,
               QColor(200, 200, 200))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText,
               QColor(160, 160, 160))
    return p


def create_application() -> QApplication:
    app = QApplication([])
    app.setApplicationName('Label-to-Dataset')
    app.setApplicationDisplayName('Label-to-Dataset')
    app.setStyle('Fusion')
    icon_path = Path(__file__).resolve().parent.parent / 'icon.png'
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
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
