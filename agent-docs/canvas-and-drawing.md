# Canvas & Drawing

## Overview (`ltd/widgets/canvas_widget.py`)

QGraphicsView-based canvas. Image loaded as QGraphicsPixmapItem. Labels rendered as semi-transparent QGraphicsRectItem/QGraphicsPolygonItem.

## Tools

| Tool | Key | Enum |
|------|-----|------|
| Hand (pan) | M, Space (hold) | `Tool.HAND` |
| Pointer (select/drag) | P | `Tool.POINTER` |
| BBox | R | `Tool.BBOX` |
| Polygon | V | `Tool.POLYGON` |
| Brush (draw mask) | B | `Tool.MARKER` |

There is no separate Eraser tool. Erasing is handled by the `DrawMode.ERASE` mode while using the Brush tool.

Spacebar temporarily switches to Hand while held, restoring previous tool on release (`_space_held`, `_tool_before_space`).

## Draw Modes

The `DrawMode` enum controls how mask drawing and polygon creation behave:

| Mode | Behavior |
|------|----------|
| `DrawMode.NEW` | Default drawing — adds to mask |
| `DrawMode.COMBINE` | Combines with existing mask |
| `DrawMode.ERASE` | Erases from mask (overlay tinted red) |

## Pointer Tool — Select & Drag

The Pointer tool supports both selection and in-place editing:

- **Click on label**: Selects it (`label_selected` signal), shows edit handles at corners/vertices
- **Drag a handle**: Moves a single vertex/corner, emits `label_modified` on release
- **Drag label body**: Moves entire shape, clamped to image bounds
- Edit handles: small colored circles (8px) at each point, white border, Z=20

## Mask Buffer

- `_mask_buffer`: QImage at full image resolution (Format_Grayscale8)
- Brush draws onto buffer with QPainter
- Overlay pixmap generated from buffer via numpy, composited as semi-transparent colored layer
- Overlay update throttled: every 6 strokes (`_draw_stroke_count`) for performance
- Eraser mode: overlay uses red tint (255, 60, 60) instead of class color

## Brush Drawing — Line Interpolation

Fast mouse movement caused mask tearing. Fixed by drawing lines between consecutive mouse positions instead of individual circles:

```python
pen = QPen(Qt.white, self._brush_size * 2,
           SolidLine, RoundCap, RoundJoin)
painter.drawLine(from_pos.toPoint(), to_pos.toPoint())
```

`mouseMoveEvent` stores `_last_draw_pos` and passes both from/to positions to `_draw_on_mask()`.

## Brush Size — Scroll Wheel

When using brush tools (Marker), scrolling without modifiers changes brush size:
- Step size: `max(1, brush_size // 10)` — scales with current size
- Range: 1–200 pixels
- Ctrl+Scroll still zooms the canvas

## Brush Cursor

Dual-ring for visibility on any background:
- Outer: black, 3px width
- Inner: white, 1px width
- Both are QGraphicsEllipseItem following mouse position
- Hidden on `leaveEvent`, shown on `enterEvent` (when in brush tool)

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

- Zoom: Ctrl+Scroll with `scale()` transform, anchored under mouse
- Vertical scroll: Scroll without modifiers
- Horizontal scroll: Shift+Scroll
- Pan: ScrollHandDrag mode when Hand tool active
- `fitInView` on initial load

## Signals

- `label_created(Label)` — new bbox/polygon drawn
- `label_selected(int)` — label clicked with Pointer
- `label_modified(int, Label)` — label moved/resized via Pointer drag
- `mask_updated(DrawMode)` — mask buffer changed (emits DrawMode: NEW, COMBINE, or ERASE)
- `brush_size_changed(int)` — for external spinbox sync
