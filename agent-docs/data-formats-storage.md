# Data Formats & Storage

## YOLO Labels

Standard YOLO `.txt` format, one line per annotation:
- Detection: `class_id cx cy w h` (normalized 0-1)
- Segmentation: `class_id x1 y1 x2 y2 ...` (normalized polygon points)

Read/write utilities in `ltd/utils/yolo_format.py`.

## Label Storage — Temp Directory

Labels are NOT stored in-place with images (avoids conflicts with caption `.txt` files). Instead:

```python
# Hash source directory path for stable temp dir
dir_hash = hashlib.md5(str(directory).encode()).hexdigest()[:12]
labels_dir = get_temp_dir_no_clear(f'labels_{dir_hash}')
```

- Same source folder → same temp dir across sessions
- `cleanup_all_temp()` skips `labels_*` directories (they survive app restarts)
- On load: checks temp dir first, falls back to in-place `.txt` for initial import

## ImageItem Dataclass (`ltd/data/image_item.py`)

```python
@dataclass
class ImageItem:
    path: Path              # Original image path
    width: int = 0
    height: int = 0
    labels: list[Label]     # YOLO labels
    tags: list[str]         # Caption tags
    mask_path: Path | None  # Binary mask path
    modified_path: Path | None  # Modified image from Modify tab
    relative_path: str = '' # Set when loaded from subfolders
    _thumbnail: object = None  # Cached thumbnail (QPixmap)
```

Key properties:
- `name` → `path.stem` (filename without extension)
- `filename` → `path.name` (filename with extension)
- `display_name` → `relative_path` if set, else `filename` (used in image list views for subfolder display)
- `display_path` → returns `modified_path` if available, else `path` (used for showing current state)
- `caption_path` → `path.with_suffix('.txt')` (caption file location)
- `load_tags_from_file(separator=', ')` / `save_tags_to_file(separator=', ')` — read/write tags with configurable separator

## Masks

Binary PNG files: white = labeled area, black = background.
- Filename: `{imagename}-masklabel.png`
- All classes merged into single mask
- Generated from labels/brush in Label tab, consumed by Modify tab
- Utilities in `ltd/utils/mask_utils.py`

## Generation Metadata (Manage Gen Images tab)

Read-only parser in `ltd/data/gen_metadata.py`. PNG text chunks are read via Pillow.

- **forge / a1111**: single `parameters` chunk. Format is `<positive>\nNegative prompt: <negative>\nSteps: 30, Sampler: ..., Size: WxH, ...`. The settings line is split on `, ` boundaries that look like `Key: ` (allows spaces in keys); quoted `"..."` and JSON `{...}` values are kept intact.
- **ComfyUI**: `prompt` chunk holds the executable graph (JSON dict of `{node_id: {class_type, inputs, _meta}}`); `workflow` chunk holds the editor layout. Parser walks back from sampler nodes' `positive`/`negative` link inputs to find `*TextEncode` nodes; if no sampler is recognized it falls back to title-based detection ("Positive"/"Negative" in `_meta.title`). Loras / VAE / model / latent size are scanned across all nodes.

`GenMetadata.tags` is derived by `split_prompt_tags()`: commas are the primary separator, plus `<lora:...>` blocks are forced into their own tag (commas are injected on both sides before splitting), and the keywords `AND`, `BREAK`, `ADDBASE`, `ADDCOMM`, `ADDCOL`, `ADDROW` are treated as commas (they vanish but split the surrounding text). Settings are stored verbatim in `GenMetadata.settings: dict`. Parsing runs via `ltd/workers/metadata_worker.py`; on directory load the tab opens `loading_dialog`, runs the worker inside a `QEventLoop`, and pipes the worker's `progress(current, total)` signal into `LoadingDialog.set_progress` so the UI shows N/total. The worker emits `parsed(ImageItem, GenMetadata)` (keyed by reference, not row, so it's row-shift safe).

`ImageItem.metadata: GenMetadata | None` carries the parsed result on the shared image dataclass.

### Pull from metadata (Caption tab → Tools)

`_pull_metadata(mode)` reuses the same parser to turn an image's embedded positive prompt into caption tags, with the usual **Current / Selected / All** buttons (target resolution mirrors `_run_captioning`). Options: a Position combo (Before / After / Overwrite, persisted at `caption/metadata_position`, default After) and a "Skip duplicates" checkbox (`caption/metadata_skip_existing`, default on).

- `_parse_metadata_for()` fills `image.metadata` only for images where it is still `None` — a single image is parsed inline, several go through `MetadataWorker` under a `loading_dialog` + `QEventLoop`. Since `MetadataWorker` skips already-parsed images (and emits no `parsed` signal for them), the apply loop reads `image.metadata` directly instead of relying on the signal, so re-pulling works off the cache.
- `_merge_metadata_tags()` collapses duplicates inside the prompt itself always, and drops tags the image already has when "Skip duplicates" is on — except in Overwrite mode, where the existing tags are discarded anyway.
- Writes go through `_push_undo` (one batch id when >1 image, so Ctrl+Z reverts the whole pull) + `_auto_save_image`, then `_sync_caption_view()` / `_rebuild_all_tags()`. Images with no metadata are counted and reported in the status line.

## Captions

Plain `.txt` files with tags, stored alongside images.
- Separator configurable in Caption tab UI (default: `, `). Supports escape sequences (`\n`, `\t`).
- Auto-saved to disk on every tag edit (no manual save needed, but "Save All" still available)
- Undo/redo per-image (Ctrl+Z / Ctrl+Y) with snapshot-based stacks
- Tag dictionary from `autocompletions/` folder provides autocomplete and category colors

### Zip Selected (Tools tab)

`_zip_selected()` (Caption tab → Tools) archives the selected images plus their caption `.txt` files into a **flat** `.zip` (no folders). Flushes pending edits via `_save_current_tags()`, builds the `(source → flat name)` mapping, resolves the tar binary, prompts with `QFileDialog.getSaveFileName` (forces `.zip` suffix), then hands the work to `ZipWorker` (`ltd/workers/zip_worker.py`) driven by a `loading_dialog` + local `QEventLoop` so the UI stays responsive (see [workers-threading.md](workers-threading.md)).

**Flat naming + conflict resolution** (in `_zip_selected`): one stem is resolved per image so its image and caption stay paired. First use of a basename keeps the original; later collisions become `N_<stem>` (`cat.png`→`1_cat.png`, and its caption `1_cat.txt`).

**`ZipWorker.do_work()`** stages each mapped file into `get_temp_dir('zip_staging')` under its flat name via `os.link` (hardlink — instant, no extra disk) with a `shutil.copy2` fallback (cross-volume / no-hardlink FS), emitting `progress`, then runs tar and rmtree's the staging dir in `finally`. tar gotchas baked in:
- **tar binary** (`CaptionTab._resolve_tar()`, passed to the worker): on Windows tries `Sysnative\tar.exe` then `System32\tar.exe` (bsdtar) before `shutil.which('tar')`. `Sysnative` handles WOW64 redirection for a 32-bit Python; the explicit path avoids a GNU `tar` on PATH (from Git) that would silently write a *tar* with a `.zip` name instead of a real zip.
- **Command-line length**: flat member names go in a `-T` list file inside the staging dir, not on the command line — a large selection would overflow the ~32 KB Windows limit (`WinError 206`). Invoked as `tar -a -c -f <dest> -C <staging> -T <list>` (`-a` picks zip format from the `.zip` extension).
- **Non-ASCII filenames**: the `-T` list is written in `mbcs` (OS code page) on Windows, because bsdtar decodes it with the code page, not UTF-8 — otherwise Cyrillic etc. names fail with "Couldn't visit directory".

### Tags vs Caption view mode

The right-panel header has a **Tags / Caption** dropdown (`panel_mode_combo`, persisted at `caption/panel_mode`). Both modes edit the **same** `.txt` file / `image.tags` — there is no separate caption file. A natural-language caption is simply the file text; tags are that text split on the separator. So the two modes are just alternate *views* of `image.tags`.

The `Image Caption` panel is a tab inside `right_tabs` (a `QPlainTextEdit` named `caption_edit`, inserted at index 0, holding `separator.join(image.tags)`). Above it sits a `caption_search` line edit that highlights all (case-insensitive) matches in the box via `setExtraSelections` (`_update_caption_search_highlights`) — display-only, so it never mutates the document/undo. Re-applied on caption load and on edit, so the term stays sticky across image navigation. `_set_caption_mode_tabs(is_caption)` toggles per-tab visibility via `QTabWidget.setTabVisible`:
- **Tags** (`_panel_mode == 'tags'`): Image Tags + All Tags visible, Image Caption hidden.
- **Caption** (`_panel_mode == 'caption'`): Image Caption visible, Image Tags + All Tags hidden.
- **Auto-Caption**, **Tools**, **Generate** and **Fast Insertion** stay visible in both modes (those that write to `image.tags` — `_on_caption_result`, the metadata pull, the snapshot-restore paths — call `_sync_caption_view()`).

### Fast Insertion tab

A `Fast Insertion` tab in `right_tabs` holds ten editable tag slots (`fast_insert_inputs`) mapped to number keys `1..9,0` (slot index `i` → key `str((i+1)%10)`). Each slot has a button (`fast_insert_buttons`) that always inserts on click, plus a `QLineEdit`. An **Enable** checkbox (`fast_insert_enable`) gates only the keyboard shortcuts, and a **Mode** combo (`fast_insert_mode`) picks Append vs Prepend.

`_fast_insert(slot)` mirrors the context-menu paste (`_on_tags_pasted`): it targets `selected_source_rows()`, falling back to the current image; flushes pending edits with `_save_current_tags()`; then per row pushes undo (batched when >1) and does `image.tags = [tag] + tags` (prepend) or `tags + [tag]` (append), auto-saving each. Because captions are just `separator.join(image.tags)`, prepend/append become `"tag, "` / `", tag"` for free — works in both Tags and Caption modes (calls `_sync_caption_view()`). No dedup, matching paste semantics.

Ten `QShortcut`s (context `WidgetWithChildrenShortcut`) are created disabled; `_on_fast_insert_enabled_changed` toggles their `setEnabled` with the checkbox, so digits type normally when off. While on, `QLineEdit`/`QPlainTextEdit` emit ShortcutOverride for digit keys, so typing in text fields is never swallowed. Persisted at `caption/fast_insert_enabled`, `caption/fast_insert_mode`, `caption/fast_insert_tag_{0..9}`. The tab stays visible in both panel modes (only Image Tags/All Tags/Image Caption are toggled by `_set_caption_mode_tabs`).

### Tags vs Caption shared machinery

Because both views share `image.tags`, auto-save, snapshots, and export work unchanged. Key sync points in `caption_tab.py`:
- `_save_current_tags()` is **mode-aware**: in caption mode it delegates to `_commit_caption_edit()` (which parses the box via `_parse_caption_text()`, pushes undo, autosaves, rebuilds all-tags). This is the single flush called before navigation and every bulk op.
- `caption_edit.textChanged` → 400 ms debounce (`_caption_debounce`) → `_commit_caption_edit()`. `_loading_caption` guards programmatic `setPlainText`.
- `_sync_caption_view()` reloads the box from `image.tags` after any external mutation of the current image (undo/redo, paste, reload-tags, and the global-shortcut bulk ops Ctrl+R/D/E/B) so a later keystroke can't commit stale text.
- Arrow-key image navigation (`_navigate_previous/next`) bails when focus is a `QPlainTextEdit` (as it already did for `QLineEdit`) so arrows work inside the caption box.

## Tag Dictionary (`autocompletions/`)

Tags loaded from `autocompletions/` directory by `ltd/data/tag_dictionary.py`:
- `*.parquet` files: downloaded via Extras tab script `download_autocompletions_list.py` from HuggingFace (`deepghs/site_tags`). Columns: `name`, `category`, `post_count`. Loaded first (higher priority).
- `custom_tags.txt`: user-defined tags, one per line. Category 0 (General), post_count 0. Won't override parquet tags.
- Falls back to legacy `tags.csv` if `autocompletions/` folder doesn't exist.
- Categories: 0=General, 1=Artist, 3=Copyright, 4=Character, 5=Meta (each has dark/light theme colors)
- Used for autocomplete popup (`ltd/widgets/tag_completer_popup.py`) and tag coloring in lists

## Temp Folder Management (`ltd/utils/file_utils.py`)

```python
get_temp_dir(subdir)          # Create temp dir, CLEAR existing contents
get_temp_dir_no_clear(subdir) # Create temp dir, KEEP existing contents
cleanup_all_temp()            # On exit: remove all except labels_* and generated_* dirs
```

Root: `{tempdir}/label-to-dataset/`

### Generated-image comparison cache (`generated_*`)

The Caption tab's **Generate** tab ("Compare original image to generated from caption") renders a caption via ComfyUI and caches the result at `{tempdir}/label-to-dataset/generated_{md5(folder)[:12]}/`, keyed by the loaded folder path (same hashing scheme as `labels_*`). Filenames come from `caption_tab.generated_cache_name(image)` — the image's relative path with unsafe chars replaced, `.png` extension — so images with the same stem in different subfolders don't collide. Never written into the source folder. `cleanup_all_temp()` preserves these dirs so comparisons survive restarts.

## Settings Persistence

QSettings stored platform-natively (Windows registry, Unix ~/.config):
```python
QSettings('LabelToDataset', 'LabelToDataset')
```

`DEFAULT_SETTINGS` in `ltd/settings.py` defines all keys with defaults. Auto-persisting widgets in `ltd/widgets/settings_widgets.py` bind directly to QSettings keys.

Extras tab uses `extras/{script_stem}/{param_name}` keys for per-script parameter persistence.

## Undo System (Label Tab)

Per-image undo stacks: `_undo_stacks[image_index] = [snapshot, ...]` (keyed by image index, not path)
- `_push_undo()` called before every label mutation (deep copy of labels + mask)
- Ctrl+Z pops and restores
- Stacks are keyed by image index, persist while folder is open, cleared on directory load
- Max 50 undo levels per image

## Undo System (Caption Tab)

Per-image undo/redo stacks: `_undo_stacks[image_index] = [tag_list_snapshot, ...]`
- Pre-snapshot strategy: `_pre_tags_snapshot` captured on image switch, pushed to undo on mutation
- Bulk ops (find/replace, batch reorder, rename, etc.) push undo per affected image
- `_skip_snapshot` flag prevents double-pushing when programmatically setting tags
- Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z), max 50 undo levels per image
- Stacks cleared on directory load

## Mask History (Modify Tab)

Per-image mask edit history: `_image_histories[id(ImageItem)] = (history_list, history_pos)`
- Keyed by `id(ImageItem)` (not row index) so history follows the object through row shifts (insert/delete/split) and doesn't leak across folder loads
- `_current_image_id` tracks the currently loaded image's id so `_on_image_changed` saves the outgoing history under the right key even if rows shifted
- `_load_directory` / `load_from_label_tab` clear `_image_histories` and reset `_current_image_id` (old ImageItems are GC'd, so their ids could be reused)
- `_delete_current_image` pops by `id(image)` and sets `_current_image_id = None` to prevent re-saving history for a deleted image
- Entries capture full state (mask QImage, pixmap, paths, dims, **labels**); max 50 entries. Labels are in the snapshot because crop/split re-normalize them — read old entries with `entry.get('labels', ...)`, they predate the key
- `_history_navigate` re-displays labels only when `need_image_reload` is set (otherwise the existing canvas items are still correct)

## Restore Base State (Modify Tab)

`_base_state[id(ImageItem)] = (mask_path, width, height, labels)` — the as-loaded state, captured by `_remember_base_state()` in `load_from_label_tab`, `_load_directory`, and for the two items `_apply_split` inserts.

Crop/split overwrite `mask_path` with a crop-sized mask, resize `width`/`height`, and re-normalize `labels`; `modified_path = None` alone would leave that crop-sized mask stretched across the full-size image. `_restore_base_geometry()` puts all four back, and is a **no-op when the dimensions still match** — a same-size modification (inpaint) keeps whatever mask edits the user made.

Safe to store just the mask *path*: as-loaded masks live in the `modify` temp dir or the source folder, while `_save_mask_buffer` always writes to the `masks` temp dir, so an edit never clobbers the base file.
