# ComfyUI Integration

## Client (`ltd/comfyui/client.py`)

HTTP + WebSocket client. Assumes ComfyUI is already running externally.

Flow: `upload_image()` → `queue_prompt(workflow_json)` → websocket wait → `get_history()` → `download_output()`

Main entry point: `run_workflow(workflow, output_dir) → dict` with keys `'files'` (list of Paths) and `'texts'` (list of str). This is used by both detection and modification modules.

WebSocket completion: client waits for an `executing` message with `node=None` to detect workflow completion.

Base URL from settings: `comfyui_url` (default `http://127.0.0.1:8188`).

## Workflows (`ltd/comfyui/workflow.py`)

Workflow JSON files stored in `Comfy-workflows/` (tracked in git). Users can add custom workflows.

### LTD Node Title Convention

Workflows must contain nodes with `_meta.title` matching these prefixes:

| Title | Direction | Used For |
|-------|-----------|----------|
| `LTD_Input_Image` | Input | Source image (`inputs['image']`) |
| `LTD_Input_Mask` | Input | Binary mask (`inputs['image']`) |
| `LTD_Input_Text` | Input | Prompt/caption text (`inputs['text']`) |
| `LTD_Latent_Size` | Input | Output width/height (`inputs['width']`/`['height']`; literal overrides a linked value) |
| `LTD_Output_Image` | Output | Modified / generated image |
| `LTD_Output_Mask` | Output | Detected mask |
| `LTD_Output_Text` | Output | Caption/tags text |

Not all nodes required — depends on workflow type (detection, modification, caption, generation).

Setters in `workflow.py`: `set_input_image`, `set_input_mask`, `set_input_text`, `set_latent_size` (all operate on every node whose `_meta.title` matches the prefix).

### Validation

- `validate_detection_workflow(json)` — checks for required input/output nodes
- `validate_modification_workflow(json)`
- `validate_caption_workflow(json)`
- `validate_generation_workflow(json)` — requires `LTD_Input_Text`, `LTD_Latent_Size`, `LTD_Output_Image`

### Caption→Image Generation (Caption tab, Tools)

The Caption tab's Tools tab has a "Compare original image to generated from caption" subsection: a `WorkflowSelector(settings_key='generate')` + Current/Selected/All buttons. `CaptionImageGenerator` (in `caption_tab.py`) renders each image's caption:
- injects the caption via `set_input_text` and, via `set_latent_size`, the output dimensions computed by `CaptionImageGenerator.compute_dims()` — the original's aspect ratio scaled to ~1 MPix and snapped to a multiple of 16;
- runs through `ComfyUIClient.run_workflow` and copies the first output image into a per-folder cache (see data-formats-storage.md), never into the source folder.
- Runs in `GenerationWorker` (`ltd/workers/generation_worker.py`), the same progress/cancel/batch pattern as `CaptionWorker`. After the batch the worker calls the generator's `finalize()` hook, which calls `ComfyUIClient.free()` (`POST /free` `{"unload_models": true, "free_memory": true}`) to drop models from VRAM — best-effort.
- Shipped workflow: `Comfy-workflows/Generate from caption (krea2 turbo).json` (adapted from a krea2-turbo text-to-image graph with the three LTD nodes renamed).

When a cached render exists for the displayed image, `CaptionTab._on_image_changed` shows it beside the original via `CaptionImageViewer.load_comparison`: two `QGraphicsPixmapItem`s — the **original at native resolution** (never downscaled; reused from `_pixmap_cache` when available) and the generated scaled to share the common dimension so the pair displays equal-sized while both retain full detail on zoom. Landscape originals stack top/bottom; square/portrait sit left/right.

`ComfyUIClient.run_workflow` downloads outputs honoring each file's `subfolder`/`type` from history (a SaveImage `filename_prefix` like `folder/name` saves into a subfolder), and `download_output` raises on HTTP error rather than writing a broken file.

### UI-to-API Format Conversion

`load_workflow()` auto-detects UI-format workflows (exported from ComfyUI's "Save" button) and converts them to API format via `_convert_ui_to_api()`. This calls ComfyUI's `/object_info` endpoint to map node types to their inputs. The `_SEED_CONTROL_VALUES` constant (`{'fixed', 'increment', 'decrement', 'randomize'}`) is used to skip `control_after_generate` widget values during conversion.

### Workflow Selector Widget

`ltd/widgets/workflow_selector.py` — dropdown listing `.json` files from `Comfy-workflows/`, with validation feedback.
