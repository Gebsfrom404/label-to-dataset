"""Read-only image canvas: auto-fit, Ctrl+wheel zoom, drag pan when zoomed.

Used by the viewer tabs (Manage Gen Images, Manage Duplicates). A preview
pixmap can be shown immediately and swapped for the full-resolution one when
``FullResLoader`` finishes, without losing the current zoom.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from ltd.utils.image_utils import load_pixmap


class FullResLoader(QThread):
    """Loads one image at full resolution off the GUI thread."""
    ready = Signal(str, QPixmap)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        pixmap = load_pixmap(self._path)
        if pixmap:
            self.ready.emit(str(self._path), pixmap)


class ImageCanvas(QGraphicsView):
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
