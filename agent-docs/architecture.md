# Architecture

## App Structure

PySide6 desktop app. Entry: `main.py` → `ltd/app.py:create_application()` → `ltd/main_window.py:MainWindow`.

Warnings suppressed unless `LTD_ENVIRONMENT=development`.

## Tabs

```
Label → Modify → Caption → Train YOLO → Manage Gen Images → Manage Duplicates → Extras
```

Each tab is a QWidget in `ltd/tabs/`. MainWindow holds them in a QTabWidget and mediates inter-tab communication via signals. Data passes through temp folders (`ltd/utils/file_utils.py`).

`Manage Gen Images` is a standalone viewer (no inter-tab signals): browses AI-generated PNGs and shows their embedded generation metadata. Read-only.

`Manage Duplicates` is likewise standalone: it searches its own source folders and only writes when the user deletes. See [duplicate-detection.md](duplicate-detection.md).

## Key Components

| Component | File | Role |
|-----------|------|------|
| App setup | `ltd/app.py` | QApplication, Fusion style, dark/light palette, fonts |
| Settings | `ltd/settings.py` | `QSettings('LabelToDataset', 'LabelToDataset')` + `DEFAULT_SETTINGS` dict |
| Image model | `ltd/data/image_list_model.py` | QAbstractListModel, lazy thumbnail loading via `QImageReader.scaledToWidth()` |
| Image item | `ltd/data/image_item.py` | ImageItem dataclass (path, thumbnail, labels, tags, mask, modified_path) |
| Label data | `ltd/data/label_data.py` | LabelClass, Label dataclasses, DEFAULT_COLORS |
| Canvas | `ltd/widgets/canvas_widget.py` | QGraphicsView: zoom/pan/draw, mask buffer, label overlay, pointer drag editing |
| Comparison | `ltd/widgets/comparison_slider.py` | Before/after horizontal wipe slider (auto-scales mismatched sizes). **Currently unused** — ModifyTab uses CanvasWidget split overlay instead |
| Image list (Modify) | `ltd/widgets/image_list_widget.py` | Simple image list widget used by ModifyTab |
| Module selector | `ltd/widgets/module_selector.py` | Plugin dropdown + dynamic settings panel, QSettings persistence |
| Settings widgets | `ltd/widgets/settings_widgets.py` | Auto-persisting QSettings-bound Qt widgets |
| Toolbar | `ltd/widgets/toolbar_widget.py` | ComfyUI URL, LM Studio URL, theme toggle, font size |
| Collapsible panel | `ltd/widgets/collapsible_panel.py` | Expandable/collapsible QFrame with header button (used by Extras tab) |
| Loading dialog | `ltd/widgets/loading_dialog.py` | Frameless modal dialog. Indeterminate by default; call `set_progress(current, total)` to switch to determinate. Shared by every tab that does mass loading (Label, Modify, Caption, Manage Gen Images) |
| Info button | `ltd/widgets/info_button.py` | Small "ⓘ" QToolButton with rich-text tooltip. `InfoButton(html)` is static; `DynamicInfoButton(provider)` calls a `() → str` each time so content can change with focus. `focus_in(widget)` helper checks if a widget contains `QApplication.focusWidget()` |
| Info text | `ltd/widgets/info_text.py` | All filter-grammar and shortcut help strings, one per filter widget / tab section (e.g. `LABEL_FILTER_HELP`, `CAPTION_SHORTCUTS_INPUT`, `GEN_SHORTCUTS_LIST`). Tabs build focus-aware help by concatenating sections that match `focus_in()` |
| Label image list | `ltd/widgets/label_image_list.py` | Image list with filter proxy (by label count, class, filename) |
| Caption image list | `ltd/widgets/caption_image_list.py` | Image list with filter, multi-select, context menu (copy/move/delete, Open in Default App, Open in Modify). Menu shortcuts are `WidgetWithChildrenShortcut`-scoped — they need focus in the list |
| Image canvas | `ltd/widgets/image_canvas.py` | Read-only QGraphicsView (auto-fit, Ctrl+wheel zoom, drag pan) + `FullResLoader` thread. Shared by Manage Gen Images and Manage Duplicates |
| Duplicate list | `ltd/widgets/duplicate_image_list.py` | `DuplicateListModel` (duplicate groups, red delete marks, blue originals) + list widget; filter grammar adds `marked:`, `original:`, `score:` |
| Source list | `ltd/tabs/duplicates_tab.py` (inline) | Folders to search, with an exclusive "original" checkbox; persists to `duplicates/sources` |
| Gen image list | `ltd/widgets/gen_image_list.py` | Image list for Manage Gen Images tab; multi-select; context menu with Copy Prompt (Ctrl+C), Copy Image to..., Move Image to... (Ctrl+M); extends CaptionImageList's filter grammar with `WxH`, `size:`, `w:`, `h:`, `meta:`, `format:` terms |
| Tag dictionary | `ltd/data/tag_dictionary.py` | CSV tag database loader (danbooru/e621), category colors, autocomplete search |
| Tag completer popup | `ltd/widgets/tag_completer_popup.py` | Custom autocomplete popup with colored tags and post counts |
| Workflow selector | `ltd/widgets/workflow_selector.py` | Dropdown for ComfyUI `.json` workflows with validation |
| Find & replace | `ltd/dialogs/find_replace_dialog.py` | Find/replace dialog for caption tags |
| Batch reorder | `ltd/dialogs/batch_reorder_dialog.py` | Batch reorder tags dialog |
| Settings dialog | `ltd/dialogs/settings_dialog.py` | Application settings dialog |

## Inter-Tab Data Flow

- **Label → Modify**: Copies images + generated masks via `copy_to_modify_requested` signal
- **Modify → Caption**: Copies modified images via `copy_to_caption_requested` signal
- **Caption → Modify**: Context-menu "Open Image(s) in Modify" on the selection. `CaptionImageList.open_in_modify_requested` → `CaptionTab.open_in_modify_requested` → `MainWindow._on_open_in_modify()` → `ModifyTab.load_from_label_tab(items)`. No copying: fresh `ImageItem`s point at the original files (so Modify's "Save In Place" writes back to the caption folder) and pick up sibling `-masklabel.png` masks
- **Label → Train**: `dataset_saved` sets dataset path; `copy_to_train_requested` sets path + model type and switches tab
- Signals emitted by tabs, connected in `MainWindow._connect_tab_signals()`
- "Include unlabeled images" checkbox in Label tab controls all output ops

## Caption Tab Tag Lists

Both right-panel tag lists (`EditableTagsList` = Image Tags, `AllTagsList` = All Tags) support multi-select with the same two actions, via context menu or Ctrl+C:

- `filter_by_tags_requested(list, bool)` → `CaptionTab._filter_by_tags(tags, append)` — writes `tag:a and tag:b …` into the image-list filter box, so a multi-select filters on **all** selected tags. `append=True` ANDs onto the existing filter (wrapping it in parens first when it contains `or`, and skipping terms already present).
- `copy_tags_requested(list)` → `CaptionTab._copy_tags_to_clipboard()` — comma-joined, matching `CaptionImageList._copy_tags`, so copied tags paste back through its Paste Tags.

`selected_tags()` on each list returns the selection in list order; `AllTagsList` strips the ` (count)` suffix. A right-click outside the selection falls back to the clicked tag alone.

## Manage Duplicates Tab

`ltd/tabs/duplicates_tab.py` — 3 panes: sources + result list (left), preview (center), Search / Actions tabs (right).

Search runs in `DuplicateWorker`; results are duplicate **groups**, not a flat list. Marking a row for deletion is independent of the list selection, and images from a source checked as *original* can never be marked. `Delete Selected` moves files to the recycle bin (optionally with `.txt` caption and `-masklabel.png` mask).

Algorithms, tolerance mapping, and grouping rules: [duplicate-detection.md](duplicate-detection.md).

## Extras Tab

`ltd/tabs/extras_tab.py` — Standalone utility scripts. Auto-discovers Python scripts from `extras_scripts/` directory.

Scripts must export: `SCRIPT_INFO` dict (name, parameters), `check_available()` function (returns `(bool, str)` — required), and either `run()` (in-process threaded) or `build_command()` (spawns terminal).

Parameter types: `str`, `bool`, `folder`, `combo`. Settings persisted via QSettings under `extras/{script_stem}/{param_name}`.

## Directory Layout

```
ltd/              - Main app package
modules/          - Plugin modules (detection/ and modifications/)
extras_scripts/   - Extras tab scripts (gallery_dl, edge_masks, ffmpeg, etc.)
models/           - Model weights (gitignored; subdirs: yolo/, lama/, caption/)
Comfy-workflows/  - ComfyUI workflow JSONs (tracked)
references/       - Reference implementations (gitignored)
```
