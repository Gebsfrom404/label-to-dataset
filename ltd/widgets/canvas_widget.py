"""Canvas widget: QGraphicsView with zoom/pan/draw tools for labeling."""
from enum import Enum, auto
from math import hypot

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPen, QPixmap,
                            QPolygonF, QWheelEvent)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsLineItem, QGraphicsPixmapItem,
                               QGraphicsPolygonItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView)

from ltd.data.label_data import DEFAULT_COLORS, Label


class Tool(Enum):
    HAND = auto()
    POINTER = auto()
    BBOX = auto()
    POLYGON = auto()
    MARKER = auto()


class DrawMode(Enum):
    NEW = auto()
    COMBINE = auto()
    ERASE = auto()


# Tools that use the brush cursor
_BRUSH_TOOLS = (Tool.MARKER,)


class CanvasWidget(QGraphicsView):
    """Main canvas for image display and label editing."""
    label_created = Signal(object)  # emits Label
    label_selected = Signal(int)  # emits label index
    label_modified = Signal(int, object)  # emits (index, Label)
    mask_updated = Signal(object)  # emits DrawMode
    brush_size_changed = Signal(int)  # emits new brush size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # State
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._mask_overlay_item: QGraphicsPixmapItem | None = None
        self._label_items: list[QGraphicsItem] = []
        self._current_tool = Tool.HAND
        self._current_class_id = 0
        self._class_colors: list[str] = list(DEFAULT_COLORS)
        self._image_width = 0
        self._image_height = 0
        self._zoom_factor = 1.0
        self._brush_size = 20

        # Draw mode (New / Combine / Erase) for polygon & brush tools
        self._draw_mode = DrawMode.NEW

        # Drawing state
        self._drawing = False
        self._draw_start: QPointF | None = None
        self._temp_rect: QGraphicsRectItem | None = None
        self._polygon_points: list[QPointF] = []
        self._polygon_markers: list[QGraphicsEllipseItem] = []
        self._polygon_lines: list[QGraphicsLineItem] = []

        # Brush cursor (visual circle showing brush size)
        self._brush_cursor: QGraphicsEllipseItem | None = None
        self._brush_cursor_outer: QGraphicsEllipseItem | None = None

        # Mask buffer (painted at image resolution)
        self._mask_buffer: QImage | None = None

        # Selected label
        self._selected_label_index = -1

        # Edit handles for dragging labels/points in pointer mode
        self._edit_handles: list[QGraphicsEllipseItem] = []
        self._dragging_handle: int = -1  # index of handle being dragged, -1 = none
        self._dragging_label: bool = False  # dragging entire label
        self._drag_start: QPointF | None = None
        self._drag_orig_points: list[QPointF] = []  # original points at drag start

        # Throttle mask overlay updates during drawing
        self._draw_stroke_count = 0

        # Spacebar pan: temporarily override to Hand tool
        self._space_held = False
        self._tool_before_space: Tool | None = None

    @property
    def current_tool(self) -> Tool:
        return self._current_tool

    @current_tool.setter
    def current_tool(self, tool: Tool):
        self._current_tool = tool
        self._finish_polygon()
        self._update_brush_cursor_visibility()
        if tool == Tool.HAND:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif tool == Tool.POINTER:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif tool == Tool.BBOX:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif tool == Tool.POLYGON:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif tool in _BRUSH_TOOLS:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.BlankCursor)

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @brush_size.setter
    def brush_size(self, size: int):
        new_size = max(1, min(200, size))
        if new_size != self._brush_size:
            self._brush_size = new_size
            self._update_brush_cursor_size()
            self.brush_size_changed.emit(new_size)

    @property
    def draw_mode(self) -> DrawMode:
        return self._draw_mode

    @draw_mode.setter
    def draw_mode(self, mode: DrawMode):
        self._draw_mode = mode

    def set_class_id(self, class_id: int):
        self._current_class_id = class_id

    def set_class_colors(self, colors: list[str]):
        self._class_colors = colors

    def load_image(self, pixmap: QPixmap):
        """Load a new image onto the canvas."""
        self._pixmap_item = None
        self._mask_overlay_item = None
        self._temp_rect = None
        self._brush_cursor = None
        self._brush_cursor_outer = None
        self._label_items.clear()
        self._edit_handles.clear()
        self._polygon_points.clear()
        self._polygon_markers.clear()
        self._polygon_lines.clear()
        self._selected_label_index = -1
        self._dragging_handle = -1
        self._dragging_label = False

        self.scene_.clear()

        self._pixmap_item = self.scene_.addPixmap(pixmap)
        self._image_width = pixmap.width()
        self._image_height = pixmap.height()

        # Create mask buffer (starts empty — no overlay needed)
        self._mask_buffer = QImage(self._image_width, self._image_height,
                                   QImage.Format.Format_Grayscale8)
        self._mask_buffer.fill(Qt.GlobalColor.black)

        # Create brush cursor
        self._create_brush_cursor()

        self.setSceneRect(QRectF(0, 0, self._image_width, self._image_height))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = 1.0

    def clear_canvas(self):
        self._pixmap_item = None
        self._mask_overlay_item = None
        self._temp_rect = None
        self._brush_cursor = None
        self._label_items.clear()
        self._edit_handles.clear()
        self._mask_buffer = None
        self.scene_.clear()

    def display_labels(self, labels: list[Label], class_colors: list[str]):
        """Draw label overlays on the canvas."""
        self._remove_edit_handles()
        for item in self._label_items:
            if item.scene() is not None:
                self.scene_.removeItem(item)
        self._label_items.clear()
        self._class_colors = class_colors

        for i, label in enumerate(labels):
            color = QColor(class_colors[label.class_id % len(class_colors)])
            color.setAlpha(60)
            pen_color = QColor(class_colors[label.class_id % len(class_colors)])
            pen = QPen(pen_color, 2)

            if label.has_polygon:
                points = [QPointF(x * self._image_width, y * self._image_height)
                          for x, y in label.polygon]
                polygon = QPolygonF(points)
                item = self.scene_.addPolygon(polygon, pen, QBrush(color))
                item.setData(0, i)  # label index
                item.setData(1, label.class_id)  # class id
                self._label_items.append(item)
            elif label.has_bbox:
                cx, cy, w, h = label.bbox
                x = (cx - w / 2) * self._image_width
                y = (cy - h / 2) * self._image_height
                rw = w * self._image_width
                rh = h * self._image_height
                item = self.scene_.addRect(QRectF(x, y, rw, rh), pen,
                                           QBrush(color))
                item.setData(0, i)  # label index
                item.setData(1, label.class_id)  # class id
                self._label_items.append(item)

    def highlight_label(self, index: int):
        """Highlight a specific label on the canvas."""
        self._selected_label_index = index
        self._remove_edit_handles()
        for i, item in enumerate(self._label_items):
            label_idx = item.data(0)
            if label_idx is None:
                continue
            class_id = item.data(1) or 0
            base_color = QColor(self._class_colors[
                class_id % len(self._class_colors)])
            if label_idx == index:
                # Selected: brighter fill, thicker pen
                pen = QPen(QColor(255, 255, 255), 3)
                fill = QColor(base_color)
                fill.setAlpha(100)
            else:
                # Normal
                pen = QPen(base_color, 2)
                fill = QColor(base_color)
                fill.setAlpha(60)
            if isinstance(item, QGraphicsRectItem):
                item.setPen(pen)
                item.setBrush(QBrush(fill))
            elif isinstance(item, QGraphicsPolygonItem):
                item.setPen(pen)
                item.setBrush(QBrush(fill))
        # Show edit handles on selected label
        if index >= 0:
            self._show_edit_handles(index)

    def set_mask(self, mask_qimage: QImage):
        """Set the mask buffer from an external QImage."""
        if (self._mask_buffer is not None
                and mask_qimage.size() == self._mask_buffer.size()):
            self._mask_buffer = mask_qimage.convertToFormat(
                QImage.Format.Format_Grayscale8)
            self._update_mask_overlay_fast()

    def get_mask_image(self) -> QImage | None:
        """Return current mask buffer."""
        return self._mask_buffer

    def clear_mask(self):
        """Clear the mask buffer."""
        if self._mask_buffer:
            self._mask_buffer.fill(Qt.GlobalColor.black)
            self._update_mask_overlay_fast()

    # --- Brush cursor ---

    def _create_brush_cursor(self):
        """Create the brush size indicator circle with contrasting outline."""
        s = self._brush_size
        # Outer black ring
        pen_outer = QPen(QColor(0, 0, 0, 200), 3)
        pen_outer.setCosmetic(True)
        self._brush_cursor_outer = self.scene_.addEllipse(
            0, 0, s * 2, s * 2, pen_outer)
        self._brush_cursor_outer.setZValue(100)
        # Inner white ring
        pen_inner = QPen(QColor(255, 255, 255, 200), 1)
        pen_inner.setCosmetic(True)
        self._brush_cursor = self.scene_.addEllipse(
            0, 0, s * 2, s * 2, pen_inner)
        self._brush_cursor.setZValue(101)
        visible = self._current_tool in _BRUSH_TOOLS
        self._brush_cursor.setVisible(visible)
        self._brush_cursor_outer.setVisible(visible)

    def _update_brush_cursor_visibility(self):
        """Show/hide brush cursor based on current tool."""
        if self._brush_cursor is not None:
            visible = self._current_tool in _BRUSH_TOOLS
            self._brush_cursor.setVisible(visible)
            if self._brush_cursor_outer is not None:
                self._brush_cursor_outer.setVisible(visible)
            if visible:
                self.setCursor(Qt.CursorShape.BlankCursor)

    def _update_brush_cursor_size(self):
        """Update brush cursor circle diameter."""
        if self._brush_cursor is not None:
            rect = self._brush_cursor.rect()
            cx = rect.center().x()
            cy = rect.center().y()
            s = self._brush_size
            new_rect = QRectF(cx - s, cy - s, s * 2, s * 2)
            self._brush_cursor.setRect(new_rect)
            if self._brush_cursor_outer is not None:
                self._brush_cursor_outer.setRect(new_rect)

    def _move_brush_cursor(self, scene_pos: QPointF):
        """Move brush cursor to follow the mouse."""
        if self._brush_cursor is not None and self._brush_cursor.isVisible():
            s = self._brush_size
            new_rect = QRectF(scene_pos.x() - s, scene_pos.y() - s,
                              s * 2, s * 2)
            self._brush_cursor.setRect(new_rect)
            if self._brush_cursor_outer is not None:
                self._brush_cursor_outer.setRect(new_rect)

    # --- Mask overlay ---

    def _remove_mask_overlay(self):
        """Safely remove the mask overlay item from the scene."""
        if self._mask_overlay_item is not None:
            if self._mask_overlay_item.scene() is not None:
                self.scene_.removeItem(self._mask_overlay_item)
            self._mask_overlay_item = None

    def _update_mask_overlay_fast(self):
        """Fast mask overlay update using numpy."""
        if self._mask_buffer is None:
            return
        try:
            import numpy as np
            from ltd.utils.mask_utils import mask_from_qimage

            mask_arr = mask_from_qimage(self._mask_buffer)
            h, w = mask_arr.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            if self._draw_mode == DrawMode.ERASE:
                color = QColor(255, 60, 60)  # Red tint for eraser
            else:
                color = QColor(self._class_colors[
                    self._current_class_id % len(self._class_colors)])
            mask_bool = mask_arr > 127
            overlay[mask_bool, 0] = color.blue()
            overlay[mask_bool, 1] = color.green()
            overlay[mask_bool, 2] = color.red()
            overlay[mask_bool, 3] = 80

            qimage = QImage(overlay.data, w, h, w * 4,
                            QImage.Format.Format_ARGB32)
            qimage = qimage.copy()

            self._remove_mask_overlay()
            self._mask_overlay_item = self.scene_.addPixmap(
                QPixmap.fromImage(qimage))
            self._mask_overlay_item.setZValue(1)
        except ImportError:
            self._update_mask_overlay_slow()

    def _update_mask_overlay_slow(self):
        """Slow fallback mask overlay update (pixel-by-pixel, no numpy)."""
        if self._mask_buffer is None:
            return
        overlay = QImage(self._image_width, self._image_height,
                         QImage.Format.Format_ARGB32)
        overlay.fill(Qt.GlobalColor.transparent)
        color = QColor(self._class_colors[
            self._current_class_id % len(self._class_colors)])
        color.setAlpha(80)

        for y in range(self._image_height):
            for x in range(self._image_width):
                if self._mask_buffer.pixelColor(x, y).red() > 127:
                    overlay.setPixelColor(x, y, color)

        self._remove_mask_overlay()
        pixmap = QPixmap.fromImage(overlay)
        self._mask_overlay_item = self.scene_.addPixmap(pixmap)
        self._mask_overlay_item.setZValue(1)

    # --- Events ---

    def wheelEvent(self, event: QWheelEvent):
        modifiers = event.modifiers()
        delta = event.angleDelta().y()

        if self._current_tool in _BRUSH_TOOLS and not modifiers:
            # Scroll without modifier = change brush size
            step = max(1, self._brush_size // 10)
            if delta > 0:
                self.brush_size = self._brush_size + step
            else:
                self.brush_size = self._brush_size - step
            event.accept()
            return

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if delta > 0 else 1 / 1.15
            self._zoom_factor *= factor
            self.scale(factor, factor)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            dx = delta * 0.5
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(dx))
        else:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta)

    def mousePressEvent(self, event):
        if self._current_tool == Tool.HAND or self._space_held:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if not self._is_in_image(scene_pos):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._current_tool == Tool.POINTER:
                self._handle_pointer_press(scene_pos)
            elif self._current_tool == Tool.BBOX:
                self._start_bbox(scene_pos)
            elif self._current_tool == Tool.POLYGON:
                self._handle_polygon_click(scene_pos)
            elif self._current_tool == Tool.MARKER:
                self._start_drawing(scene_pos,
                                    erase=(self._draw_mode == DrawMode.ERASE))
        elif event.button() == Qt.MouseButton.RightButton:
            if self._current_tool == Tool.POLYGON:
                self._finish_polygon()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())

        # Always update brush cursor position
        self._move_brush_cursor(scene_pos)

        if self._current_tool == Tool.HAND or self._space_held:
            super().mouseMoveEvent(event)
            return

        if self._current_tool == Tool.POINTER and (
                self._dragging_handle >= 0 or self._dragging_label):
            self._handle_pointer_drag(scene_pos)
        elif self._drawing and self._current_tool in _BRUSH_TOOLS:
            self._continue_drawing(scene_pos,
                                   erase=(self._draw_mode == DrawMode.ERASE))
        elif self._drawing and self._current_tool == Tool.BBOX:
            self._update_bbox(scene_pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._current_tool == Tool.HAND or self._space_held:
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._current_tool == Tool.POINTER and (
                    self._dragging_handle >= 0 or self._dragging_label):
                self._finish_pointer_drag()
            elif self._current_tool == Tool.BBOX and self._drawing:
                scene_pos = self.mapToScene(event.position().toPoint())
                self._finish_bbox(scene_pos)
            elif self._current_tool in _BRUSH_TOOLS and self._drawing:
                self._stop_drawing()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """Hide brush cursor when mouse leaves the view."""
        if self._brush_cursor is not None:
            self._brush_cursor.setVisible(False)
        if self._brush_cursor_outer is not None:
            self._brush_cursor_outer.setVisible(False)
        super().leaveEvent(event)

    def enterEvent(self, event):
        """Show brush cursor when mouse enters the view."""
        if (self._brush_cursor is not None
                and self._current_tool in _BRUSH_TOOLS):
            self._brush_cursor.setVisible(True)
            if self._brush_cursor_outer is not None:
                self._brush_cursor_outer.setVisible(True)
            self.setCursor(Qt.CursorShape.BlankCursor)
        super().enterEvent(event)

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and not self._space_held):
            self._space_held = True
            self._tool_before_space = self._current_tool
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if self._brush_cursor is not None:
                self._brush_cursor.setVisible(False)
            if self._brush_cursor_outer is not None:
                self._brush_cursor_outer.setVisible(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if (event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and self._space_held):
            self._space_held = False
            if self._tool_before_space is not None:
                # Restore tool without going through setter to preserve
                # in-progress polygon drawing
                tool = self._tool_before_space
                self._current_tool = tool
                self._tool_before_space = None
                self._update_brush_cursor_visibility()
                if tool == Tool.HAND:
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                elif tool == Tool.POLYGON:
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.setCursor(Qt.CursorShape.CrossCursor)
                elif tool == Tool.BBOX:
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.setCursor(Qt.CursorShape.CrossCursor)
                elif tool == Tool.POINTER:
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                elif tool in _BRUSH_TOOLS:
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.setCursor(Qt.CursorShape.BlankCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # --- Tool implementations ---

    def _is_in_image(self, pos: QPointF) -> bool:
        return (0 <= pos.x() < self._image_width and
                0 <= pos.y() < self._image_height)

    # --- Edit handles for pointer tool ---

    def _remove_edit_handles(self):
        """Remove all edit handle markers from the scene."""
        for h in self._edit_handles:
            if h.scene() is not None:
                self.scene_.removeItem(h)
        self._edit_handles.clear()
        self._dragging_handle = -1
        self._dragging_label = False

    def _show_edit_handles(self, label_index: int):
        """Show draggable point handles on the selected label."""
        self._remove_edit_handles()
        points = self._get_label_scene_points(label_index)
        if not points:
            return
        # Determine color from the label item
        item = self._label_item_by_index(label_index)
        if item is None:
            return
        class_id = item.data(1) or 0
        color = QColor(self._class_colors[class_id % len(self._class_colors)])
        for pt in points:
            marker = self.scene_.addEllipse(
                pt.x() - 4, pt.y() - 4, 8, 8,
                QPen(Qt.GlobalColor.white, 1),
                QBrush(color))
            marker.setZValue(20)
            self._edit_handles.append(marker)

    def _label_item_by_index(self, label_index: int) -> QGraphicsItem | None:
        """Find the QGraphicsItem for a given label index."""
        for item in self._label_items:
            if item.data(0) == label_index:
                return item
        return None

    def _get_label_scene_points(self, label_index: int) -> list[QPointF]:
        """Get the corner/vertex points of a label in scene coords."""
        item = self._label_item_by_index(label_index)
        if item is None:
            return []
        if isinstance(item, QGraphicsPolygonItem):
            poly = item.polygon()
            return [QPointF(poly[i]) for i in range(poly.count())]
        elif isinstance(item, QGraphicsRectItem):
            r = item.rect()
            return [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
        return []

    def _hit_test_handle(self, pos: QPointF) -> int:
        """Return index of handle near pos, or -1."""
        for i, h in enumerate(self._edit_handles):
            center = h.rect().center()
            if hypot(pos.x() - center.x(), pos.y() - center.y()) < 8:
                return i
        return -1

    def _handle_pointer_press(self, pos: QPointF):
        """Handle mouse press in pointer mode — start drag or select."""
        # First check if clicking on an edit handle
        if self._edit_handles:
            hi = self._hit_test_handle(pos)
            if hi >= 0:
                self._dragging_handle = hi
                self._drag_start = pos
                self._drag_orig_points = self._get_label_scene_points(
                    self._selected_label_index)
                return

        # Check if clicking on the selected label body (drag whole shape)
        if self._selected_label_index >= 0:
            item = self._label_item_by_index(self._selected_label_index)
            if item is not None and item.contains(
                    item.mapFromScene(pos)):
                self._dragging_label = True
                self._drag_start = pos
                self._drag_orig_points = self._get_label_scene_points(
                    self._selected_label_index)
                return

        # Otherwise, try to select a label
        self._handle_pointer_click(pos)

    def _handle_pointer_click(self, pos: QPointF):
        items = self.scene_.items(pos)
        for item in items:
            idx = item.data(0)
            if idx is not None:
                self._selected_label_index = idx
                self.label_selected.emit(idx)
                return
        self._selected_label_index = -1
        self.label_selected.emit(-1)

    def _handle_pointer_drag(self, pos: QPointF):
        """Update label shape and handles while dragging."""
        if self._drag_start is None or not self._drag_orig_points:
            return
        idx = self._selected_label_index
        item = self._label_item_by_index(idx)
        if item is None:
            return

        if self._dragging_handle >= 0:
            # Dragging a single point
            hi = self._dragging_handle
            new_points = list(self._drag_orig_points)
            new_points[hi] = self._clamp_to_image(pos)
            self._apply_points_to_item(item, new_points)
            self._update_handle_positions(new_points)
        elif self._dragging_label:
            # Dragging entire shape
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            new_points = []
            for pt in self._drag_orig_points:
                new_points.append(self._clamp_to_image(
                    QPointF(pt.x() + dx, pt.y() + dy)))
            self._apply_points_to_item(item, new_points)
            self._update_handle_positions(new_points)

    def _finish_pointer_drag(self):
        """Finalize drag — emit label_modified with updated label."""
        idx = self._selected_label_index
        was_dragging = self._dragging_handle >= 0 or self._dragging_label
        self._dragging_handle = -1
        self._dragging_label = False
        self._drag_start = None
        self._drag_orig_points = []

        if not was_dragging or idx < 0:
            return

        item = self._label_item_by_index(idx)
        if item is None:
            return

        # Build updated Label from current item geometry
        if isinstance(item, QGraphicsPolygonItem):
            poly = item.polygon()
            points = [(poly[i].x() / self._image_width,
                        poly[i].y() / self._image_height)
                       for i in range(poly.count())]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            class_id = item.data(1) or 0
            label = Label(class_id=class_id,
                          bbox=(cx, cy, w, h), polygon=points)
        elif isinstance(item, QGraphicsRectItem):
            r = item.rect()
            cx = r.center().x() / self._image_width
            cy = r.center().y() / self._image_height
            w = r.width() / self._image_width
            h = r.height() / self._image_height
            class_id = item.data(1) or 0
            label = Label(class_id=class_id, bbox=(cx, cy, w, h))
        else:
            return

        self.label_modified.emit(idx, label)

    def _clamp_to_image(self, pos: QPointF) -> QPointF:
        """Clamp a point to image boundaries."""
        x = max(0, min(self._image_width - 1, pos.x()))
        y = max(0, min(self._image_height - 1, pos.y()))
        return QPointF(x, y)

    def _apply_points_to_item(self, item: QGraphicsItem,
                               points: list[QPointF]):
        """Update a label item's geometry from a list of points."""
        if isinstance(item, QGraphicsPolygonItem):
            item.setPolygon(QPolygonF(points))
        elif isinstance(item, QGraphicsRectItem):
            # Rebuild rect from 4 corner points (TL, TR, BR, BL)
            xs = [p.x() for p in points]
            ys = [p.y() for p in points]
            item.setRect(QRectF(min(xs), min(ys),
                                max(xs) - min(xs), max(ys) - min(ys)))

    def _update_handle_positions(self, points: list[QPointF]):
        """Move edit handle markers to match new point positions."""
        for i, pt in enumerate(points):
            if i < len(self._edit_handles):
                self._edit_handles[i].setRect(
                    pt.x() - 4, pt.y() - 4, 8, 8)

    # --- BBox tool ---

    def _start_bbox(self, pos: QPointF):
        self._drawing = True
        self._draw_start = pos
        pen = QPen(QColor(self._class_colors[
            self._current_class_id % len(self._class_colors)]), 2,
            Qt.PenStyle.DashLine)
        self._temp_rect = self.scene_.addRect(QRectF(pos, pos), pen)
        self._temp_rect.setZValue(10)

    def _update_bbox(self, pos: QPointF):
        if self._temp_rect and self._draw_start:
            rect = QRectF(self._draw_start, pos).normalized()
            self._temp_rect.setRect(rect)

    def _finish_bbox(self, pos: QPointF):
        self._drawing = False
        if self._temp_rect:
            if self._temp_rect.scene() is not None:
                self.scene_.removeItem(self._temp_rect)
            self._temp_rect = None
        if self._draw_start:
            rect = QRectF(self._draw_start, pos).normalized()
            cx = (rect.center().x()) / self._image_width
            cy = (rect.center().y()) / self._image_height
            w = rect.width() / self._image_width
            h = rect.height() / self._image_height
            if w > 0.001 and h > 0.001:
                label = Label(class_id=self._current_class_id,
                              bbox=(cx, cy, w, h))
                self.label_created.emit(label)
        self._draw_start = None

    # --- Polygon tool ---

    def _handle_polygon_click(self, pos: QPointF):
        """Handle left click in polygon mode — add point or close polygon."""
        if len(self._polygon_points) >= 3:
            start = self._polygon_points[0]
            dist = hypot(pos.x() - start.x(), pos.y() - start.y())
            if dist < 10:
                self._finish_polygon()
                return
        self._add_polygon_point(pos)

    def _add_polygon_point(self, pos: QPointF):
        color = QColor(self._class_colors[
            self._current_class_id % len(self._class_colors)])

        if self._polygon_points:
            prev = self._polygon_points[-1]
            line = self.scene_.addLine(
                prev.x(), prev.y(), pos.x(), pos.y(),
                QPen(color, 2, Qt.PenStyle.DashLine))
            line.setZValue(10)
            self._polygon_lines.append(line)

        self._polygon_points.append(pos)

        marker = self.scene_.addEllipse(
            pos.x() - 4, pos.y() - 4, 8, 8,
            QPen(Qt.GlobalColor.white, 1),
            QBrush(color))
        marker.setZValue(11)
        self._polygon_markers.append(marker)

    def _finish_polygon(self):
        if len(self._polygon_points) >= 3:
            points = [(p.x() / self._image_width, p.y() / self._image_height)
                      for p in self._polygon_points]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            label = Label(class_id=self._current_class_id,
                          bbox=(cx, cy, w, h), polygon=points)
            self.label_created.emit(label)

        for marker in self._polygon_markers:
            if marker.scene() is not None:
                self.scene_.removeItem(marker)
        for line in self._polygon_lines:
            if line.scene() is not None:
                self.scene_.removeItem(line)
        self._polygon_markers.clear()
        self._polygon_lines.clear()
        self._polygon_points.clear()

    # --- Freehand drawing (Brush) ---

    def _start_drawing(self, pos: QPointF, erase: bool = False):
        self._drawing = True
        self._draw_stroke_count = 0
        self._last_draw_pos = pos
        self._draw_on_mask(pos, pos)

    def _continue_drawing(self, pos: QPointF, erase: bool = False):
        self._draw_on_mask(self._last_draw_pos, pos)
        self._last_draw_pos = pos
        self._draw_stroke_count += 1
        if self._draw_stroke_count % 6 == 0:
            self._update_mask_overlay_fast()

    def _stop_drawing(self):
        self._drawing = False
        self._update_mask_overlay_fast()
        self.mask_updated.emit(self._draw_mode)

    def _draw_on_mask(self, from_pos: QPointF, to_pos: QPointF):
        """Draw a continuous stroke between two points on the mask."""
        if self._mask_buffer is None:
            return
        painter = QPainter(self._mask_buffer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(Qt.GlobalColor.white, self._brush_size * 2,
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(from_pos.toPoint(), to_pos.toPoint())
        painter.end()
