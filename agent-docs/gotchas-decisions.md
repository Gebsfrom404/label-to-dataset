# Gotchas & Past Decisions

## QImage Row Padding

QImage rows are padded to 4-byte alignment. Always use `bytesPerLine()` when converting to numpy:
```python
bpl = qimage.bytesPerLine()
arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bpl))[:, :width].copy()
```
Without this: `ValueError: cannot reshape array` on images where `width % 4 != 0`.

## Brush Tearing on Fast Movement

`mouseMoveEvent` fires at discrete intervals. Drawing circles at each position leaves gaps. Fix: use `QPainter.drawLine()` with `RoundCap` between consecutive positions.

## Mask Overlay Update Throttling

Rebuilding the numpy mask overlay on every stroke is expensive. Throttle to every 6 strokes during drawing, then final update on mouse release.

## Slow Image Switching

Don't call `_update_mask_overlay_fast()` when loading a new image — the mask buffer starts empty, so the overlay rebuild is wasted work.

## Label Persistence Across Sessions

Two requirements:
1. `get_temp_dir_no_clear()` (not `get_temp_dir()`) for label storage — otherwise contents cleared on every folder load
2. `cleanup_all_temp()` must skip `labels_*` directories — otherwise labels lost on app exit

Temp dir keyed by MD5 hash of source directory path for stability.

## Labels in Temp Dir, Not In-Place

Labels stored in temp folder to avoid filename collision with caption `.txt` files. Both use `{imagename}.txt` pattern but different content.

## ComparisonSlider — Currently Unused

The `ComparisonSlider` widget (`ltd/widgets/comparison_slider.py`) exists but is **not imported by any tab**. ModifyTab now uses the CanvasWidget's built-in split overlay instead. The widget is kept as a utility but the following two entries describe its internal behavior, not current app behavior.

## Comparison Slider — Layout

Horizontal curtain wipe: left = before (original), right = after (modified). The divider line is at `slider_pos`:
- `slider_pos` 0.0 = divider at left edge → all after (modified) visible
- `slider_pos` 1.0 = divider at right edge → all before (original) visible

Before image on top (Z=1), cropped from left edge (0 to `slider_pos * width`). After image sits as base layer (full, uncropped).

## Comparison Slider — Image Size Mismatch

Modified images may have different dimensions than originals (e.g., from ComfyUI workflows). `set_after()` and `set_images()` auto-scale the after image to match the before image's dimensions using `IgnoreAspectRatio` + `SmoothTransformation`. Without this, the slider shows misaligned or clipped comparisons.

## WD Tagger: BGR Not RGB

SmilingWolf models trained on OpenCV BGR input. Omitting the channel flip produces wrong color-related tags. The `[::-1]` flip creates a negative stride — must `.copy()` before `torch.from_numpy()`.

## WD Tagger: Must Normalize

Input must be normalized to `[-1, 1]` via `(pixel/255 - 0.5) / 0.5`. Passing raw `[0, 255]` or `[0, 1]` produces irrelevant tags.

## Module `prepare()` for Thread Safety

Plugin modules have Qt widget settings panels. These widgets cannot be accessed from worker threads. The `prepare()` method runs on the GUI thread before work starts — read all widget values into plain attributes there.

## Spacebar Pan

Canvas needs `StrongFocus` policy and `keyPressEvent`/`keyReleaseEvent` handlers. The label tab must call `canvas.setFocus()` after image switching so spacebar works immediately.

## No Separate Eraser Tool

There is no `Tool.ERASER` enum value. Erasing is handled by setting `DrawMode.ERASE` while using the Brush/Marker tool. The eraser mode is controlled by the draw mode selector in the Label tab UI, not by switching tools.

## Canvas Split Overlay — No-op Handler

The canvas emits `split_pos_changed(float)` when the user drags the split divider. ModifyTab connects it to `_on_split_pos_from_canvas` which is currently a no-op (`pass`). The signal and connection exist for future use — don't assume it's broken.

## ModifyTab Crop/Split Tools Are Strings, Not Tool Enum

ModifyTab defines `_TOOL_CROP = 'crop'`, `_TOOL_SPLIT_V = 'split_v'`, `_TOOL_SPLIT_H = 'split_h'` as plain string constants. These are NOT members of the canvas `Tool` enum. ModifyTab manages them alongside actual `Tool` enum values in its own `QButtonGroup`. When adding new tools to ModifyTab, use the same string-constant pattern.

## ModificationWorker — Module Unload

After batch modification completes (or is cancelled), the worker calls `module.unload()` if it exists. This allows modules to free GPU memory (e.g., LaMa model). The `unload()` method is optional — not part of the ABC.
