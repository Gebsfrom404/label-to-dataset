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

## Deleting an Image — Cached Row Index Goes Stale

`QItemSelectionModel` relocates the current index **during** `beginRemoveRows` (before the item is popped), landing on `row - 1`, or `row + 1` when deleting row 0. Two consequences:

1. `current_changed` fires against the **pre-removal** model.
2. Its stored index is persistent, so after the pop it may already sit on the row you were about to select — `setCurrentIndex()` then emits nothing, and a tab's cached `_current_image_index` keeps its pre-removal value.

Symptom: the canvas shows the right image, but every `self.model.get_image(self._current_image_index)` (mask save, crop, split, label edits) hits the **neighbouring** image.

Pattern used by `LabelTab._delete_current_image` / `ModifyTab._delete_current_image`:

```python
self.image_list.blockSignals(True)   # suppress the mid-removal callback
try:
    self.model.remove_rows([row])
finally:
    self.image_list.blockSignals(False)
self._current_image_index = -1       # force the resync to be detectable
new_row = self.image_list.select_proxy_row(new_proxy_row)  # or select_index()
if self._current_image_index != new_row:
    self._on_image_changed(new_row)  # select_*() was a no-op — resync by hand
```

`CaptionTab._on_images_deleted` achieves the same by re-reading `current_source_row()` after the removal and calling `_on_image_changed` explicitly.

Related: with a filter proxy (Label / Caption / Gen lists), the next **source** row may be hidden. Pick the next row in *proxy* space instead — `LabelImageList.current_proxy_row()` / `select_proxy_row()`.

## Labels Are Normalized — Crop/Split Invalidates Them

`Label.bbox` (cx, cy, w, h) and `Label.polygon` are normalized to *their own image*, so any geometry change silently re-interprets them: after a crop the same numbers get stretched across the smaller image. The symptom hides until the next image switch, because `_reload_and_record` used not to redraw labels at all.

`_crop_labels(labels, img_w, img_h, ax, ay, aw, ah)` (module level in `modify_tab.py`) re-normalizes onto the crop rect and drops labels that fall outside; `_apply_crop` and both halves of `_apply_split` go through it. Polygons are **clamped** to the crop box, not properly clipped — labels are display-only in ModifyTab, so an approximate outline beats a Sutherland-Hodgman pass.

## ModifyTab Deletes the Source, Not Just the Working Copy

Images handed over by "Proceed to Modify" live in the `modify` temp dir; `path` is the temp copy and `original_path` is the real file (`LabelTab._copy_to_modify`). `_delete_current_image` deletes **both**, plus `modified_path`, `mask_path`, and each one's sibling `.txt` — deleting only `path` left the source in place, which is not what "delete image" means to a user.

The delete does **not** notify the Label tab, so its list still shows the (now missing) image until reloaded.

## Per-Image State Keyed by Row Breaks on Deletion

`LabelTab._undo_stacks` and `CaptionTab._undo_stacks` / `_redo_stacks` are `dict[row_index, ...]`. Removing a row shifts every later image down one, so the stacks silently attach to the wrong images. `LabelTab._delete_current_image` remaps them (`{i - 1 if i > row else i}`); **CaptionTab does not** — its delete path leaves undo history misaligned. ModifyTab sidesteps this entirely by keying `_image_histories` on `id(image)`.

## ModificationWorker — Module Unload

After batch modification completes (or is cancelled), the worker calls `module.unload()` if it exists. This allows modules to free GPU memory (e.g., LaMa model). The `unload()` method is optional — not part of the ABC.
