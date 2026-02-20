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
```

Key properties:
- `display_path` → returns `modified_path` if available, else `path` (used for showing current state)
- `caption_path` → `path.with_suffix('.txt')` (caption file location)
- `load_tags_from_file()` / `save_tags_to_file()` — read/write comma-separated tags

## Masks

Binary PNG files: white = labeled area, black = background.
- Filename: `{imagename}-masklabel.png`
- All classes merged into single mask
- Generated from labels/brush in Label tab, consumed by Modify tab
- Utilities in `ltd/utils/mask_utils.py`

## Captions

Plain `.txt` files with comma-separated tags, stored alongside images.
- Saved by Caption tab's "Save All Captions" or "Export Images + Captions..."
- Internal tag separator configurable via `tag_separator` setting

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

Per-image undo stacks: `_undo_stacks[image_path] = [snapshot, ...]`
- `_push_undo()` called before every label mutation (deep copy of labels + mask)
- Ctrl+Z pops and restores
- Stacks are keyed by image path, persist while folder is open
