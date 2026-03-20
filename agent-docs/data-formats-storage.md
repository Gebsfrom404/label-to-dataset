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

## Captions

Plain `.txt` files with tags, stored alongside images.
- Separator configurable in Caption tab UI (default: `, `). Supports escape sequences (`\n`, `\t`).
- Auto-saved to disk on every tag edit (no manual save needed, but "Save All" still available)
- Undo/redo per-image (Ctrl+Z / Ctrl+Y) with snapshot-based stacks
- Tag dictionary from `autocompletions/` folder provides autocomplete and category colors

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
cleanup_all_temp()            # On exit: remove all except labels_* dirs
```

Root: `{tempdir}/label-to-dataset/`

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
