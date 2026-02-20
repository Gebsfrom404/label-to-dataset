# Architecture

## App Structure

PySide6 desktop app. Entry: `main.py` → `ltd/app.py:create_application()` → `ltd/main_window.py:MainWindow`.

Warnings suppressed unless `LTD_ENVIRONMENT=development`.

## 5-Tab Pipeline

```
Label → Modify → Caption → Train YOLO → Extras
```

Each tab is a QWidget in `ltd/tabs/`. MainWindow holds them in a QTabWidget and mediates inter-tab communication via signals. Data passes through temp folders (`ltd/utils/file_utils.py`).

## Key Components

| Component | File | Role |
|-----------|------|------|
| App setup | `ltd/app.py` | QApplication, Fusion style, dark/light palette, fonts |
| Settings | `ltd/settings.py` | `QSettings('LabelToDataset', 'LabelToDataset')` + `DEFAULT_SETTINGS` dict |
| Image model | `ltd/data/image_list_model.py` | QAbstractListModel, lazy thumbnail loading via `QImageReader.scaledToWidth()` |
| Image item | `ltd/data/image_item.py` | ImageItem dataclass (path, thumbnail, labels, tags, mask, modified_path) |
| Label data | `ltd/data/label_data.py` | LabelClass, Label dataclasses, DEFAULT_COLORS |
| Canvas | `ltd/widgets/canvas_widget.py` | QGraphicsView: zoom/pan/draw, mask buffer, label overlay, pointer drag editing |
| Comparison | `ltd/widgets/comparison_slider.py` | Before/after horizontal wipe slider (auto-scales mismatched sizes) |
| Module selector | `ltd/widgets/module_selector.py` | Plugin dropdown + dynamic settings panel, QSettings persistence |
| Settings widgets | `ltd/widgets/settings_widgets.py` | Auto-persisting QSettings-bound Qt widgets |
| Toolbar | `ltd/widgets/toolbar_widget.py` | ComfyUI URL, theme toggle, font size |
| Collapsible panel | `ltd/widgets/collapsible_panel.py` | Expandable/collapsible QFrame with header button (used by Extras tab) |
| Loading dialog | `ltd/widgets/loading_dialog.py` | Frameless modal dialog with indeterminate progress bar |
| Label image list | `ltd/widgets/label_image_list.py` | Image list with filter proxy (by label count, class, filename) |
| Caption image list | `ltd/widgets/caption_image_list.py` | Image list with filter, multi-select, context menu |
| Workflow selector | `ltd/widgets/workflow_selector.py` | Dropdown for ComfyUI `.json` workflows with validation |
| Find & replace | `ltd/dialogs/find_replace_dialog.py` | Find/replace dialog for caption tags |
| Batch reorder | `ltd/dialogs/batch_reorder_dialog.py` | Batch reorder tags dialog |
| Settings dialog | `ltd/dialogs/settings_dialog.py` | Application settings dialog |

## Inter-Tab Data Flow

- **Label → Modify**: Copies images + generated masks via `copy_to_modify_requested` signal
- **Modify → Caption**: Copies modified images via `copy_to_caption_requested` signal
- **Label → Train**: `dataset_saved` sets dataset path; `copy_to_train_requested` sets path + model type and switches tab
- Signals emitted by tabs, connected in `MainWindow._connect_tab_signals()`
- "Include unlabeled images" checkbox in Label tab controls all output ops

## Extras Tab

`ltd/tabs/extras_tab.py` — Standalone utility scripts. Auto-discovers Python scripts from `extras_scripts/` directory.

Scripts must export: `SCRIPT_INFO` dict (name, parameters), `check_available()` function, and either `run()` (in-process threaded) or `build_command()` (spawns terminal).

Parameter types: `str`, `bool`, `folder`. Settings persisted via QSettings under `extras/{script_stem}/{param_name}`.

## Directory Layout

```
ltd/              - Main app package
modules/          - Plugin modules (detection/ and modifications/)
extras_scripts/   - Extras tab scripts (gallery_dl, edge_masks, ffmpeg, etc.)
models/           - Model weights (gitignored; subdirs: yolo/, lama/, caption/)
Comfy-workflows/  - ComfyUI workflow JSONs (tracked)
references/       - Reference implementations (gitignored)
```
