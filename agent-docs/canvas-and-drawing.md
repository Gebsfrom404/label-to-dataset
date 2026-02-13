# Canvas & Drawing

## Overview (`ltd/widgets/canvas_widget.py`)

QGraphicsView-based canvas. Image loaded as QGraphicsPixmapItem. Labels rendered as semi-transparent QGraphicsRectItem/QGraphicsPolygonItem.

## Tools

| Tool | Key | Enum |
|------|-----|------|
| Hand (pan) | M, Space (hold) | `Tool.HAND` |
| Pointer (select) | P | `Tool.POINTER` |
| BBox | R | `Tool.BBOX` |
| Polygon | V | `Tool.POLYGON` |
| Brush (draw mask) | B | `Tool.MARKER` |
| Eraser | E | `Tool.ERASER` |

Spacebar temporarily switches to Hand while held, restoring previous tool on release (`_space_held`, `_tool_before_space`).

## Mask Buffer

- `_mask_buffer`: QImage at full image resolution (Format_Grayscale8)
- Brush draws onto buffer with QPainter
- Overlay pixmap generated from buffer via numpy, composited as semi-transparent colored layer
- Overlay update throttled: every 6 strokes (`_draw_stroke_count`) for performance

## Brush Drawing — Line Interpolation

Fast mouse movement caused mask tearing. Fixed by drawing lines between consecutive mouse positions instead of individual circles:

```python
pen = QPen(Qt.white, self._brush_size * 2,
           SolidLine, RoundCap, RoundJoin)
painter.drawLine(from_pos.toPoint(), to_pos.toPoint())
```

`mouseMoveEvent` stores `_last_draw_pos` and passes both from/to positions to `_draw_on_mask()`.

## Brush Cursor

Dual-ring for visibility on any background:
- Outer: black, 3px width
- Inner: white, 1px width
- Both are QGraphicsEllipseItem following mouse position

## QImage ↔ Numpy: Row Padding Gotcha

**Critical**: QImage rows are padded to 4-byte alignment. `bytesPerLine()` can exceed `width`.

```python
# WRONG — crashes on images where width % 4 != 0
arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))

# CORRECT
bpl = qimage.bytesPerLine()
arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bpl))
arr = arr[:, :width].copy()
```

This applies in `ltd/utils/mask_utils.py:mask_from_qimage()`.

## Zoom & Pan

- Zoom: `wheelEvent` with `scale()` transform, anchored under mouse
- Pan: ScrollHandDrag mode when Hand tool active
- `fitInView` on initial load

## Signals

- `label_created(Label)` — new bbox/polygon drawn
- `label_selected(int)` — label clicked with Pointer
- `label_modified(int, Label)` — label moved/resized
- `mask_updated(bool)` — mask buffer changed (bool = erase mode)
- `brush_size_changed(int)` — for external spinbox sync
