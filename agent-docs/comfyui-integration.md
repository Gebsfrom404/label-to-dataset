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
| `LTD_Input_Image` | Input | Source image |
| `LTD_Input_Mask` | Input | Binary mask |
| `LTD_Output_Image` | Output | Modified image |
| `LTD_Output_Mask` | Output | Detected mask |
| `LTD_Output_Text` | Output | Caption/tags text |

Not all nodes required — depends on workflow type (detection, modification, caption).

### Validation

- `validate_detection_workflow(json)` — checks for required input/output nodes
- `validate_modification_workflow(json)`
- `validate_caption_workflow(json)`

### UI-to-API Format Conversion

`load_workflow()` auto-detects UI-format workflows (exported from ComfyUI's "Save" button) and converts them to API format via `_convert_ui_to_api()`. This calls ComfyUI's `/object_info` endpoint to map node types to their inputs. The `_SEED_CONTROL_VALUES` constant (`{'fixed', 'increment', 'decrement', 'randomize'}`) is used to skip `control_after_generate` widget values during conversion.

### Workflow Selector Widget

`ltd/widgets/workflow_selector.py` — dropdown listing `.json` files from `Comfy-workflows/`, with validation feedback.
