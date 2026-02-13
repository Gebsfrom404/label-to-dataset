# ComfyUI Integration

## Client (`ltd/comfyui/client.py`)

HTTP + WebSocket client. Assumes ComfyUI is already running externally.

Flow: `upload_image()` → `queue_prompt(workflow_json)` → websocket wait → `get_history()` → `download_output()`

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

### Workflow Selector Widget

`ltd/widgets/workflow_selector.py` — dropdown listing `.json` files from `Comfy-workflows/`, with validation feedback.
