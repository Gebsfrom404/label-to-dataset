"""Before/after comparison slider widget with scrollbar, crop and split overlays."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (QGraphicsLineItem, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QScrollBar,
                               QVBoxLayout, QWidget)


class _SliderView(QGraphicsView):
    """Internal view that handles zoom and draws the divider line."""

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
        self._slider_pos = 0.5
        self._image_width = 0
        self._image_height = 0

        # Split mode state
        self._split_mode = False
        self._split_orientation = 'V'  # 'V' = vertical line, 'H' = horizontal
        self._split_pos = 0.5
        self._split_line: QGraphicsLineItem | None = None
        self._split_dragging = False

        # Crop mode state
        self._crop_mode = False
        self._crop_rect = QRectF()  # in image coords
        self._crop_border: QGraphicsRectItem | None = None
        self._crop_handles: list[QGraphicsRectItem] = []
        self._crop_dim_rects: list[QGraphicsRectItem] = []
        self._crop_dragging_handle = -1  # index of handle being dragged, -1 = none
        self._crop_min_size = 10

    @property
    def slider_pos(self):
        return self._slider_pos

    @slider_pos.setter
    def slider_pos(self, value):
        self._slider_pos = max(0.0, min(1.0, value))
        self._update_clip()
        self.viewport().update()

    def set_before(self, pixmap: QPixmap):
        self._before_pixmap = pixmap
        self._recalc_dimensions()
        self._rebuild()

    def set_after(self, pixmap: QPixmap):
        self._after_pixmap = pixmap
        self._recalc_dimensions()
        self._rebuild()

    def set_images(self, before: QPixmap, after: QPixmap):
        self._before_pixmap = before
        self._after_pixmap = after
        self._recalc_dimensions()
        self._rebuild()

    def _recalc_dimensions(self):
        """Set scene dimensions to the larger of before/after."""
        bw = self._before_pixmap.width() if self._before_pixmap else 0
        bh = self._before_pixmap.height() if self._before_pixmap else 0
        aw = self._after_pixmap.width() if self._after_pixmap else 0
        ah = self._after_pixmap.height() if self._after_pixmap else 0
        self._image_width = max(bw, aw)
        self._image_height = max(bh, ah)

    def _rebuild(self):
        self.scene_.clear()
        self._before_item = None
        self._after_item = None
        self._split_line = None
        self._crop_border = None
        self._crop_handles.clear()
        self._crop_dim_rects.clear()

        if self._after_pixmap:
            self._after_item = self.scene_.addPixmap(self._after_pixmap)
            # Center in scene if smaller than scene rect
            ox = (self._image_width - self._after_pixmap.width()) / 2
            oy = (self._image_height - self._after_pixmap.height()) / 2
            self._after_item.setOffset(ox, oy)

        if self._before_pixmap:
            self._before_item = self.scene_.addPixmap(self._before_pixmap)
            self._before_item.setZValue(1)
            # Before offset is handled by _update_clip

        self._update_clip()
        if self._image_width > 0:
            self.setSceneRect(QRectF(0, 0, self._image_width, self._image_height))
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Rebuild overlays
        if self._split_mode:
            self._create_split_line()
        if self._crop_mode:
            self._create_crop_overlay()

    def _update_clip(self):
        if self._before_item and self._before_pixmap:
            clip_w = int(self._slider_pos * self._image_width)
            if clip_w <= 0:
                self._before_item.setPixmap(QPixmap())
                self._before_item.setOffset(0, 0)
            else:
                cropped = self._before_pixmap.copy(0, 0, clip_w, self._image_height)
                self._before_item.setPixmap(cropped)
                self._before_item.setOffset(0, 0)

    def drawForeground(self, painter: QPainter, rect: QRectF):
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

    # --- Split mode ---

    def set_split_mode(self, enabled: bool, orientation: str = 'V'):
        self._split_mode = enabled
        self._split_orientation = orientation
        if enabled:
            self._create_split_line()
        else:
            if self._split_line and self._split_line.scene():
                self.scene_.removeItem(self._split_line)
            self._split_line = None
        self.viewport().update()

    def set_split_pos(self, pos: float):
        self._split_pos = max(0.0, min(1.0, pos))
        self._update_split_line()
        self.viewport().update()

    def get_split_pos(self) -> float:
        return self._split_pos

    def _create_split_line(self):
        if self._split_line and self._split_line.scene():
            self.scene_.removeItem(self._split_line)
        pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
        self._split_line = QGraphicsLineItem()
        self._split_line.setPen(pen)
        self._split_line.setZValue(10)
        self.scene_.addItem(self._split_line)
        self._update_split_line()

    def _update_split_line(self):
        if not self._split_line or self._image_width <= 0:
            return
        if self._split_orientation == 'V':
            x = self._split_pos * self._image_width
            self._split_line.setLine(x, 0, x, self._image_height)
        else:
            y = self._split_pos * self._image_height
            self._split_line.setLine(0, y, self._image_width, y)

    # --- Crop mode ---

    def set_crop_mode(self, enabled: bool):
        self._crop_mode = enabled
        if enabled and self._image_width > 0:
            # Always reset crop rect to full image bounds
            self._crop_rect = QRectF(0, 0, self._image_width, self._image_height)
            self._create_crop_overlay()
        else:
            self._remove_crop_overlay()
        self.viewport().update()

    def get_crop_rect(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) in image pixel coords."""
        r = self._crop_rect
        return (int(r.x()), int(r.y()), int(r.width()), int(r.height()))

    def _create_crop_overlay(self):
        self._remove_crop_overlay()
        if self._image_width <= 0:
            return

        # Dim overlay: 4 rects outside crop area
        dim_brush = QBrush(QColor(0, 0, 0, 120))
        for _ in range(4):
            rect = QGraphicsRectItem()
            rect.setBrush(dim_brush)
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.setZValue(8)
            self.scene_.addItem(rect)
            self._crop_dim_rects.append(rect)

        # Crop border
        border_pen = QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine)
        self._crop_border = QGraphicsRectItem()
        self._crop_border.setPen(border_pen)
        self._crop_border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._crop_border.setZValue(9)
        self.scene_.addItem(self._crop_border)

        # 8 handles: TL, TC, TR, ML, MR, BL, BC, BR
        handle_brush = QBrush(QColor(255, 255, 255))
        handle_pen = QPen(QColor(0, 0, 0), 1)
        for _ in range(8):
            handle = QGraphicsRectItem(-5, -5, 10, 10)
            handle.setBrush(handle_brush)
            handle.setPen(handle_pen)
            handle.setZValue(11)
            handle.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIgnoresTransformations)
            self.scene_.addItem(handle)
            self._crop_handles.append(handle)

        self._update_crop_overlay()

    def _remove_crop_overlay(self):
        for rect in self._crop_dim_rects:
            if rect.scene():
                self.scene_.removeItem(rect)
        self._crop_dim_rects.clear()
        if self._crop_border and self._crop_border.scene():
            self.scene_.removeItem(self._crop_border)
        self._crop_border = None
        for handle in self._crop_handles:
            if handle.scene():
                self.scene_.removeItem(handle)
        self._crop_handles.clear()

    def _update_crop_overlay(self):
        if not self._crop_border or self._image_width <= 0:
            return
        r = self._crop_rect
        iw, ih = self._image_width, self._image_height

        # Border
        self._crop_border.setRect(r)

        # Dim rects: top, bottom, left, right
        if len(self._crop_dim_rects) == 4:
            # Top
            self._crop_dim_rects[0].setRect(QRectF(0, 0, iw, r.top()))
            # Bottom
            self._crop_dim_rects[1].setRect(QRectF(0, r.bottom(), iw, ih - r.bottom()))
            # Left
            self._crop_dim_rects[2].setRect(QRectF(0, r.top(), r.left(), r.height()))
            # Right
            self._crop_dim_rects[3].setRect(QRectF(r.right(), r.top(), iw - r.right(), r.height()))

        # Handle positions: TL, TC, TR, ML, MR, BL, BC, BR
        positions = [
            QPointF(r.left(), r.top()),        # 0 TL
            QPointF(r.center().x(), r.top()),   # 1 TC
            QPointF(r.right(), r.top()),        # 2 TR
            QPointF(r.left(), r.center().y()),  # 3 ML
            QPointF(r.right(), r.center().y()), # 4 MR
            QPointF(r.left(), r.bottom()),      # 5 BL
            QPointF(r.center().x(), r.bottom()),# 6 BC
            QPointF(r.right(), r.bottom()),     # 7 BR
        ]
        for i, handle in enumerate(self._crop_handles):
            handle.setPos(positions[i])

    def _hit_test_crop_handle(self, scene_pos: QPointF) -> int:
        """Return handle index if scene_pos is near a handle, else -1."""
        if not self._crop_handles:
            return -1
        tolerance = 10  # pixels in view coords
        for i, handle in enumerate(self._crop_handles):
            handle_view_pos = self.mapFromScene(handle.pos())
            mouse_view_pos = self.mapFromScene(scene_pos)
            dx = abs(handle_view_pos.x() - mouse_view_pos.x())
            dy = abs(handle_view_pos.y() - mouse_view_pos.y())
            if dx <= tolerance and dy <= tolerance:
                return i
        return -1

    def _hit_test_split_line(self, scene_pos: QPointF) -> bool:
        """Return True if scene_pos is near the split line."""
        if not self._split_line or self._image_width <= 0:
            return False
        tolerance = 10  # view pixels
        if self._split_orientation == 'V':
            line_x = self._split_pos * self._image_width
            line_view = self.mapFromScene(QPointF(line_x, 0))
            mouse_view = self.mapFromScene(scene_pos)
            return abs(line_view.x() - mouse_view.x()) <= tolerance
        else:
            line_y = self._split_pos * self._image_height
            line_view = self.mapFromScene(QPointF(0, line_y))
            mouse_view = self.mapFromScene(scene_pos)
            return abs(line_view.y() - mouse_view.y()) <= tolerance

    def _move_crop_handle(self, handle_idx: int, scene_pos: QPointF):
        """Move a crop handle, adjusting the crop rect accordingly."""
        r = QRectF(self._crop_rect)
        x = max(0, min(scene_pos.x(), self._image_width))
        y = max(0, min(scene_pos.y(), self._image_height))
        mn = self._crop_min_size

        if handle_idx == 0:    # TL
            r.setLeft(min(x, r.right() - mn))
            r.setTop(min(y, r.bottom() - mn))
        elif handle_idx == 1:  # TC
            r.setTop(min(y, r.bottom() - mn))
        elif handle_idx == 2:  # TR
            r.setRight(max(x, r.left() + mn))
            r.setTop(min(y, r.bottom() - mn))
        elif handle_idx == 3:  # ML
            r.setLeft(min(x, r.right() - mn))
        elif handle_idx == 4:  # MR
            r.setRight(max(x, r.left() + mn))
        elif handle_idx == 5:  # BL
            r.setLeft(min(x, r.right() - mn))
            r.setBottom(max(y, r.top() + mn))
        elif handle_idx == 6:  # BC
            r.setBottom(max(y, r.top() + mn))
        elif handle_idx == 7:  # BR
            r.setRight(max(x, r.left() + mn))
            r.setBottom(max(y, r.top() + mn))

        # Clamp to image bounds
        r = r.intersected(QRectF(0, 0, self._image_width, self._image_height))
        if r.width() >= mn and r.height() >= mn:
            self._crop_rect = r
            self._update_crop_overlay()

    # --- Mouse events ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())

            # Crop handle drag
            if self._crop_mode:
                handle = self._hit_test_crop_handle(scene_pos)
                if handle >= 0:
                    self._crop_dragging_handle = handle
                    return

            # Split line drag
            if self._split_mode and self._hit_test_split_line(scene_pos):
                self._split_dragging = True
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())

        if self._crop_dragging_handle >= 0:
            self._move_crop_handle(self._crop_dragging_handle, scene_pos)
            return

        if self._split_dragging and self._image_width > 0:
            if self._split_orientation == 'V':
                self._split_pos = max(0.0, min(1.0, scene_pos.x() / self._image_width))
            else:
                self._split_pos = max(0.0, min(1.0, scene_pos.y() / self._image_height))
            self._update_split_line()
            # Emit signal to parent to sync UI
            parent = self.parent()
            if hasattr(parent, '_on_split_pos_changed_from_view'):
                parent._on_split_pos_changed_from_view(self._split_pos)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._crop_dragging_handle = -1
            self._split_dragging = False
        super().mouseReleaseEvent(event)


class ComparisonSlider(QWidget):
    """Horizontal curtain wipe with scrollbar control.

    Left = before (original), right = after (modified).
    Scrollbar at bottom controls slider position.
    """

    split_pos_changed = Signal(float)  # emitted when split line dragged

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = _SliderView(self)
        layout.addWidget(self._view)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setRange(0, 1000)
        self._scrollbar.setValue(500)
        self._scrollbar.setPageStep(100)
        layout.addWidget(self._scrollbar)

        self._scrollbar.valueChanged.connect(self._on_scrollbar_changed)

    def _on_scrollbar_changed(self, value: int):
        self._view.slider_pos = value / 1000.0

    def _on_split_pos_changed_from_view(self, pos: float):
        """Called by _SliderView when split line is dragged."""
        self.split_pos_changed.emit(pos)

    # --- Proxy methods ---

    def viewport(self):
        return self._view.viewport()

    def set_before(self, pixmap: QPixmap):
        self._view.set_before(pixmap)

    def set_after(self, pixmap: QPixmap):
        self._view.set_after(pixmap)

    def set_images(self, before: QPixmap, after: QPixmap):
        self._view.set_images(before, after)

    # --- Split mode proxies ---

    def set_split_mode(self, enabled: bool, orientation: str = 'V'):
        self._view.set_split_mode(enabled, orientation)

    def set_split_pos(self, pos: float):
        self._view.set_split_pos(pos)

    def get_split_pos(self) -> float:
        return self._view.get_split_pos()

    # --- Crop mode proxies ---

    def set_crop_mode(self, enabled: bool):
        self._view.set_crop_mode(enabled)

    def get_crop_rect(self) -> tuple[int, int, int, int]:
        return self._view.get_crop_rect()
