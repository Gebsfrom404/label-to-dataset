# Architecture

## App Structure

PySide6 desktop app. Entry: `main.py` → `ltd/app.py:create_application()` → `ltd/main_window.py:MainWindow`.

Warnings suppressed unless `LTD_ENVIRONMENT=development`.

## 4-Tab Pipeline

```
Label → Modify → Caption → Train YOLO
```

Each tab is a QWidget in `ltd/tabs/`. MainWindow holds them in a QTabWidget and mediates inter-tab communication via signals. Data passes through temp folders (`ltd/utils/file_utils.py`).

## Key Components

| Component | File | Role |
|-----------|------|------|
| App setup | `ltd/app.py` | QApplication, Fusion style, dark/light palette, fonts |
| Settings | `ltd/settings.py` | `QSettings('LabelToDataset', 'LabelToDataset')` + `DEFAULT_SETTINGS` dict |
| Image model | `ltd/data/image_list_model.py` | QAbstractListModel, lazy thumbnail loading via `QImageReader.scaledToWidth()` |
| Image item | `ltd/data/image_item.py` | ImageItem dataclass (path, thumbnail, labels, tags, mask) |
| Label data | `ltd/data/label_data.py` | LabelClass, Label dataclasses, DEFAULT_COLORS |
| Canvas | `ltd/widgets/canvas_widget.py` | QGraphicsView: zoom/pan/draw, mask buffer, label overlay |
| Comparison | `ltd/widgets/comparison_slider.py` | Before/after horizontal wipe slider |
| Module selector | `ltd/widgets/module_selector.py` | Plugin dropdown + dynamic settings panel, QSettings persistence |
| Settings widgets | `ltd/widgets/settings_widgets.py` | Auto-persisting QSettings-bound Qt widgets |
| Toolbar | `ltd/widgets/toolbar_widget.py` | ComfyUI URL, theme toggle, font size |

## Inter-Tab Data Flow

- **Label → Modify**: Copies images + generated masks to temp `modify` dir
- **Modify → Caption**: Copies modified images to temp `caption` dir
- Signals emitted by tabs, connected in MainWindow
- "Include unlabeled images" checkbox in Label tab controls all output ops

## Directory Layout

```
ltd/           - Main app package
modules/       - Plugin modules (detection/ and modifications/)
models/        - Model weights (gitignored; subdirs: yolo/, lama/, caption/)
Comfy-workflows/ - ComfyUI workflow JSONs (tracked)
references/    - Reference implementations (gitignored)
```
