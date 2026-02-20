"""Before/after comparison slider widget."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ComparisonSlider(QGraphicsView):
    """Horizontal curtain wipe: left = before (original), right = after (modified).

    The divider line is at slider_pos.  Everything left of the divider shows
    the *before* image; everything right shows the *after* image.

    Dragging left reveals more of the after (modified) image.
    Dragging right reveals more of the before (original) image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._before_pixmap: QPixmap | None = None
        self._after_pixmap: QPixmap | None = None
        self._before_item: QGraphicsPixmapItem | None = None
        self._after_item: QGraphicsPixmapItem | None = None
        self._slider_pos = 0.5  # 0.0 = all after, 1.0 = all before
        self._dragging = False
        self._image_width = 0
        self._image_height = 0

    def set_before(self, pixmap: QPixmap):
        self._before_pixmap = pixmap
        self._image_width = pixmap.width()
        self._image_height = pixmap.height()
        self._rebuild()

    def set_after(self, pixmap: QPixmap):
        # Scale to match before dimensions if they differ
        if (self._before_pixmap and
                (pixmap.width() != self._image_width or
                 pixmap.height() != self._image_height)):
            pixmap = pixmap.scaled(self._before_pixmap.size(),
                                   Qt.AspectRatioMode.IgnoreAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self._after_pixmap = pixmap
        self._rebuild()

    def set_images(self, before: QPixmap, after: QPixmap):
        self._before_pixmap = before
        # Scale after to match before dimensions so comparison is pixel-aligned
        if (after.width() != before.width() or
                after.height() != before.height()):
            after = after.scaled(before.size(),
                                 Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
        self._after_pixmap = after
        self._image_width = before.width()
        self._image_height = before.height()
        self._rebuild()

    def _rebuild(self):
        self.scene_.clear()
        self._before_item = None
        self._after_item = None

        # After (modified) on bottom — visible where before is cropped away
        if self._after_pixmap:
            self._after_item = self.scene_.addPixmap(self._after_pixmap)

        # Before (original) on top — cropped to show only the right portion
        if self._before_pixmap:
            self._before_item = self.scene_.addPixmap(self._before_pixmap)
            self._before_item.setZValue(1)

        self._update_clip()
        if self._image_width > 0:
            self.setSceneRect(QRectF(0, 0, self._image_width, self._image_height))
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _update_clip(self):
        """Crop the before (original) image to show only the right portion.

        slider_pos controls where the divider is:
          0.0 = divider at left edge  → all after (modified) visible
          1.0 = divider at right edge → all before (original) visible
        """
        if self._before_item and self._before_pixmap:
            # before covers from 0 to (slider_pos * width)
            clip_w = int(self._slider_pos * self._image_width)
            if clip_w <= 0:
                self._before_item.setPixmap(QPixmap())
                self._before_item.setOffset(0, 0)
            else:
                cropped = self._before_pixmap.copy(
                    0, 0, clip_w, self._image_height)
                self._before_item.setPixmap(cropped)
                self._before_item.setOffset(0, 0)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_slider_from_event(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_slider_from_event(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def _update_slider_from_event(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._image_width > 0:
            self._slider_pos = max(0.0, min(1.0,
                                             scene_pos.x() / self._image_width))
            self._update_clip()
            self.viewport().update()

    def drawForeground(self, painter: QPainter, rect: QRectF):
        """Draw the slider line."""
        super().drawForeground(painter, rect)
        if self._image_width > 0:
            x = self._slider_pos * self._image_width
            pen = QPen(Qt.GlobalColor.white, 2)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, 0), QPointF(x, self._image_height))

    def wheelEvent(self, event: QWheelEvent):
        modifiers = event.modifiers()
        delta = event.angleDelta().y()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() - delta)
