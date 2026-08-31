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

Canvas needs `StrongFocus` policy and `keyPressEvent`/`keyReleaseEvent` handlers. The label tab must call `canvas.setFocus()` after image switching so spacebar works immediately — but `_on_image_changed()` skips this when `image_list.list_view.hasFocus()` (in addition to the existing `filter_input.hasFocus()` check). Without that guard, arrow-key navigation in the image list (native QListView Up/Down) would switch the image, `_on_image_changed()` would steal focus to the canvas, and a second Down press would land on the canvas (which doesn't handle it) instead of continuing to navigate the list — "press Down once, list goes dead."

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

## Run All Lost History for Every Image Except the Displayed One

`_on_mod_result` used to record a history entry only when the finished image was the one on the canvas; for the rest it set `modified_path` and stopped. The image *looked* modified, but:

- never-visited images got their `Start` entry built from the already-modified state on the next switch, so the step could not be undone;
- previously-visited images kept a stale saved history whose entries all carry `modified_path=None` — clicking any of them silently reverted the modification.

Fixed by `_record_offscreen()` (lazy, path-only entries materialized on switch) — see "Off-screen entries" in data-formats-storage.md. The same change dropped `_single_mode`: "Run Current" wrote its result to `model.get_image(self._current_image_index)`, i.e. to whatever image the user had navigated to while the run was in flight, instead of the one that was actually processed.

## ModificationWorker — Module Unload

After batch modification completes (or is cancelled), the worker calls `module.unload()` if it exists. This allows modules to free GPU memory (e.g., LaMa model). The `unload()` method is optional — not part of the ABC.

## Tab Key Filters Swallowed Typing in Text Fields

`LabelTab` and `ModifyTab` call `_install_space_filter()`, which installs the tab as an event filter on **every** child widget (`self.findChildren(QWidget)`). Their `eventFilter` then intercepted Space (canvas pan toggle) and — in ModifyTab — `A`/`D`/PgUp/PgDown (image navigation) unconditionally, returning `True`. Any text field inside the tab therefore lost those keys: typing in the ComfyUI `LTD_Input_Prompt` box or the Custom-JSON area moved to the next image instead of inserting a character.

Both filters now bail out via `text_input_has_focus()` (`ltd/utils/qt_utils.py`) before touching the key, so the focused `QLineEdit` / `QPlainTextEdit` / `QTextEdit` / `QAbstractSpinBox` / editable `QComboBox` gets the event. With focus anywhere else the old pan/navigation behaviour is unchanged.

`QShortcut`-based single-letter bindings (`M`/`R`/`V`/`B`/`C`/`1`/`2` in ModifyTab, `A`/`D`/`W`/`S` in LabelTab) never had this problem — `QWidgetTextControl` accepts the `ShortcutOverride` event for text-inserting keys, so an editable text widget wins before the shortcut fires. Only hand-rolled event filters need the guard.

## Caption Tab: Focus Escaped Into the Separator Field

`CaptionTab.separator_input` sits below `right_tabs` and was the next widget in the focus chain after the last panel field. Tab from the bottom of a right-panel tab page — or Qt's focus-chain walk when `_set_caption_mode_tabs()` hides the page that holds focus — silently landed there, and unnoticed typing rewrote the separator used by every `save_tags_to_file()` / `_parse_caption_text()` call, corrupting tag files on the next save.

Two guards: the field is `Qt.FocusPolicy.ClickFocus` (deliberate mouse click only, never Tab or the focus-chain walk), and `_set_caption_mode_tabs()` snapshots `QApplication.focusWidget()` before toggling tab visibility, then `_restore_panel_focus()` hands focus back — to the same widget if it is still visible, otherwise to `caption_edit` / `tag_input` for the new mode. Focus that never moved (e.g. `panel_mode_combo` during keyboard navigation) is left alone.

## Caption Tab: Image-List QActions Were Window-Wide Shortcuts

`CaptionImageList._setup_context_menu()` registers its context-menu actions on `list_view` so their shortcuts work without opening the menu — but `QAction` defaults to `Qt.WindowShortcut`, so Ctrl+C / Ctrl+M / Ctrl+O / Ctrl+Shift+Del fired from anywhere in the Caption tab, not just the image list. That silently swallowed Ctrl+C in the tag lists (`QListWidget` doesn't accept `ShortcutOverride` for it, unlike text widgets, which is why the tag input and caption box were never affected).

They now get `Qt.ShortcutContext.WidgetWithChildrenShortcut`, matching `GenImagesTab._make_tag_copy_action`. Consequence: those shortcuts require focus inside the image list. Ctrl+C is now context-sensitive — image list copies captions, tag list copies the selected tags.

## SAM3 Text-Prompt Masks Have a Channel Dim — `cv::copyMakeBorder` Crash

`Sam3Processor._forward_grounding()` (`modules/sam3/vendor/utils.py`) builds `state['masks']` via `interpolate(out_masks.unsqueeze(1), ...)`, giving shape `[N, 1, H, W]`. Iterating it per-detection without dropping that dim yields `[1, H, W]` masks; `mask_to_polygons()` → `cv2.findContours()` pads its input internally via `cv::copyMakeBorder`, which asserts `_src.dims() <= 2` — so a 3D mask fails with `Assertion failed: top >= 0 && bottom >= 0 && left >= 0 && right >= 0 && _src.dims() <= 2 in function 'cv::copyMakeBorder'`, a cryptic error with no obvious connection to the real cause. Fix: `Sam3Engine.segment_text()` (`modules/sam3/engine.py`) does `.squeeze(1)` right after pulling `state['masks']` to numpy. `segment_point()`'s masks (from `SAM3InteractiveImagePredictor.predict()`, which already does its own `.squeeze(0)`) were never affected.

## Magic Wand: Preview-Scaled Canvas vs Full-Resolution Source File

Both tabs display large images via `load_pixmap_preview(path, max_dim=1920)` — any source image with a side over 1920px is downscaled for the canvas. Magic Wand's click handler originally re-read the *source file* from disk (`image.path` / `image.display_path`) and segmented that, while the `(x, y)` click came from `scene_pos` in *canvas/preview* coordinate space. For any image over the preview cap this desynced two ways at once: the point landed on the wrong region of the full-resolution image, and the returned mask's shape didn't match `_mask_buffer` (built at preview resolution) — `CanvasWidget.apply_mask_result()` used to silently no-op on a shape mismatch, so the symptom was "click, wait, nothing happens," with no error.

Fixed by `CanvasWidget.get_image_rgb()`, which converts the canvas's *currently displayed* `QPixmap` to a numpy array instead of re-reading the file — guaranteed to be exactly `image_width x image_height`, the same space the click coordinates are in. `Sam3PointWorker` now takes this array directly rather than a path. `apply_mask_result()` also gained a `cv2.resize` fallback for the (should-no-longer-happen) mismatched-shape case, so a future regression here fails loudly (wrong-looking mask) rather than silently.

## Batch Workers: Memory Cleanup Gated Behind `BATCH_SIZE` Let VRAM Ratchet Up

`DetectionWorker`/`ModificationWorker`/`CaptionWorker`/`GenerationWorker` only called `_cleanup_memory()` (`gc.collect()` + `torch.cuda.empty_cache()`) every `BaseWorker.BATCH_SIZE` (100) images — fine for light models, but for a folder under 100 images it meant `empty_cache()` never ran until the whole batch finished. SAM3's text-grounding path upsamples every surviving detection's mask to the *original* image resolution (not the fixed 1008×1008 it processes internally at — `Sam3Processor._forward_grounding` in `modules/sam3/vendor/utils.py`), so one image's peak VRAM scales with image resolution × surviving-detection count × number of text-prompt phrases (each phrase is a separate forward pass). `torch.cuda.empty_cache()` only returns *reserved-but-unused* memory to the driver — reserved memory otherwise only grows to match the highest peak seen so far in the run and stays there. Measured: a 10-phrase, low-confidence run peaked at ~8GB allocated for one busy 4000×2670 image; without per-image cleanup, "reserved" (what Task Manager/nvidia-smi show) ratcheted from ~3.6GB idle up past 12GB across just 4 images and stayed there; with `empty_cache()` after each image it dropped back to ~3.7GB every time. Fixed by removing the `BATCH_SIZE` gate entirely — cleanup now runs after every single item in all four workers (`BaseWorker.BATCH_SIZE` removed, unused).

## SAM3 Shared Engine: Singleton Creation Wasn't Thread-Safe

`get_shared_engine()`'s `if _shared_engine is None: _shared_engine = Sam3Engine(...)` was an unguarded check-then-act. Magic Wand runs on its own worker thread (`Sam3PointWorker`) independently of the Label tab's Auto-Label buttons (sync for "Current", a `DetectionWorker` thread for batch) — clicking into two of these features close together could call `get_shared_engine()` from two threads at once, both see `None`, and each construct (and separately load) its own `Sam3Engine` — two full checkpoints in VRAM/RAM simultaneously, one of them orphaned and never freed since only the last-assigned instance is reachable as the "shared" one afterward. `Sam3Engine._ensure_model()`'s own lock only protects *within* one instance; it can't prevent two instances from existing in the first place. Fixed by guarding singleton creation with the same module-level `_lock` `_ensure_model()` uses for the build itself (sequential reuse of one lock across the two call sites — construction is cheap and always finishes, so no deadlock risk). Verified with a 20-thread concurrent-construction stress test.

## Magic Wand History Mislabeled "Brush"

`ModifyTab._on_mask_updated()` is the shared handler for both a finished brush stroke and (once added) Magic Wand — both go through `CanvasWidget.mask_updated`. It unconditionally logged the history entry as `'Brush'`. Fixed by adding a `source` argument to `mask_updated` (`Signal(object, str)`) and to `apply_mask_result(mask, mode, source='Magic Wand')`; `_on_mask_updated(self, draw_mode, source='Brush')` now records `source`. Label tab's own `_on_mask_updated(self, draw_mode=None)` needed no change — Qt lets a connected slot declare fewer parameters than the signal emits.

## LabelTab: `prepare()` Must Run Before `get_class_names()`

Detection modules whose `get_class_names()` depends on GUI-thread state populated by `prepare()` (SAM3's phrases from its text area; potentially others in future) return stale or empty results if `_get_class_map()` runs before `module.prepare()`. `_run_auto_detection`/`_run_auto_detection_unlabeled` now call `prepare()` first, then build the class map. `_run_auto_detection_current` didn't call `prepare()` at all — SAM3's `self._phrases` stayed whatever a *previous* `prepare()` call had left it as (empty on first-ever use, since `Sam3DetectionModule.run()` returns `[]` when phrases are empty — no crash, just silently does nothing). Fixed by adding the same `module.prepare()` call there too, before `run()`.

## Duplicate search reported "100%" for visibly different images

**Symptom:** two artworks differing by an added speech bubble (and an open
mouth) were grouped as duplicates at **tolerance 0**, listed at `[100%]`.

**Cause:** the pHash was 8×8 = 64 bits, the common default. That keeps only
the 64 lowest-frequency DCT coefficients of a 32×32 thumbnail, and a change
covering ~15% of the frame flips none of them — the two files hashed
*identically*, so distance 0 and score `1 - 0/64` = 100%. Not a threshold
problem: no tolerance setting could have separated them. On the reporting
folder, **all five** pairs at distance 0 were different images.

**Fix:** `HASH_SIZE = 16` (256 bits). The variant pair moves to 20 bits while
the worst genuine-duplicate transform tested (q20 JPEG, eighth-size, blur,
small watermark) stays at 4 — a 5× margin. `_PHASH_MAX_BITS` was re-measured
to 64/256 rather than scaled from the old 25/64, which had been accepting
5293 of 78606 pairs at tolerance 100.

**Lesson:** pick hash width from measured separation on real data, not from
what the reference implementations default to. A hash too coarse to represent
the difference cannot be rescued by tuning the threshold.

## Descriptor mode grouped the same edited pair the hash did

**Symptom:** after the pHash width fix, "perceptual hash, then descriptor
verify" still grouped two images differing by an added speech bubble.

**Cause:** a second, independent bug in the same feature. The descriptor
score was the share of keypoints that matched and survived RANSAC, which
*inverts* the ranking: the edited pair scored 0.83 while an 80% centre crop
of one image scored 0.52 and an 8° rotation 0.72 — both genuine duplicates.
A crop removes a fifth of the frame and with it a fifth of the keypoints,
while a local edit leaves most keypoints untouched. No threshold separates a
0.83 non-duplicate from a 0.52 duplicate.

**Fix:** `DescriptorComparer` now uses the homography only to *align*, then
warps one image onto the other and scores by the share of a 16×16 cell grid
that disagrees. Duplicates (crop, rotation, rescale, q20, brightness) all
score ≥0.996; the edited pair scores 0.961 and needs tolerance 34+.

**Lesson:** when a similarity score has to rank several kinds of difference
against each other, check the *ordering* on real examples before tuning the
threshold. Both bugs in this feature presented as "wrong threshold" and
neither was.
