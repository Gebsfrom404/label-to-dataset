# Plugin System

## ABCs (`modules/base.py`)

Two base classes:

**`BaseDetectionModule`** — for object detection (YOLO, ComfyUI workflows)
- `name` property → display name for dropdown
- `get_settings_widget() → QWidget` → module-specific UI
- `prepare()` → capture widget state before threading (called on GUI thread)
- `run(image_path, **kwargs) → list[dict]` → returns detections with keys: `class_id`, `class_name`, `confidence`, `bbox`, `polygon`, `mask`
- `get_class_names() → list[str] | None`

**`BaseModificationModule`** — for image modification (LaMa inpaint, ComfyUI)
- Same pattern: `name`, `get_settings_widget()`, `prepare()`
- `run(image_path, mask_path, **kwargs) → Path` → returns path to modified image
- `wants_mask() → bool` (default `True`) → queried by the Modify tab **after** `prepare()`. Return `False` when the module ignores masks for the current configuration (e.g. a ComfyUI workflow with no `LTD_Input_Mask` node); the tab then passes `mask_path=None` and includes maskless images in the run. See comfyui-integration.md for the full matrix.

## Discovery (`modules/__init__.py`)

```python
discover_modules(base_class, package_path) → list[instance]
```

Uses `pkgutil.iter_modules` + `inspect.getmembers` to find all non-abstract subclasses. Skips `_`-prefixed files. Instantiates each found class.

Called from tabs:
- Label tab: `discover_modules(BaseDetectionModule, 'modules/detection')`
- Modify tab: `discover_modules(BaseModificationModule, 'modules/modifications')`

## Built-in Plugins

- `modules/detection/yolo_detection.py` — Ultralytics YOLO
- `modules/detection/comfyui_detection.py` — ComfyUI workflow
- `modules/modifications/lama_inpaint.py` — big-lama torch.jit inpainting
- `modules/modifications/comfyui_modification.py` — ComfyUI workflow

## Adding a New Plugin

1. Create file in `modules/detection/` or `modules/modifications/`
2. Subclass the appropriate ABC
3. Implement all abstract methods
4. Plugin is auto-discovered at next launch

## `prepare()` Pattern

Critical for thread safety. GUI widgets must NOT be accessed from worker threads. The `prepare()` method is called from the main thread before starting the worker — read all widget values into plain attributes there.

```python
def prepare(self):
    self._confidence = self._confidence_spin.value()
    self._model_path = self._model_combo.currentText()

def run(self, image_path, **kwargs):
    # Use self._confidence, self._model_path — never touch widgets here
```
