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
| `LTD_Input_Mask` | Input | Binary mask (`inputs['image']`) — **optional** |
| `LTD_Input_Prompt` | Input | User-editable instruction — **optional**, see below |
| `LTD_Input_Text` | Input | App-supplied text, i.e. the caption in caption→image generation (`inputs['text']`) |
| `LTD_Latent_Size` | Input | Output width/height (`inputs['width']`/`['height']`; literal overrides a linked value) |
| `LTD_Seed` | Input | Pinned seed (`inputs['seed']`/`['noise_seed']`) |
| `LTD_Output_Image` | Output | Modified / generated image |
| `LTD_Output_Mask` | Output | Detected mask |
| `LTD_Output_Text` | Output | Caption/tags text |

Only `LTD_Input_Image` + an output node are required; everything else is optional. `LTD_Input_Text` vs `LTD_Input_Prompt`: `_Text` is filled by the app (the caption), `_Prompt` is typed by the user in the UI. They are distinct nodes so a generation workflow can use both.

Setters in `workflow.py`: `set_input_image`, `set_input_mask`, `set_input_text`, `set_input_prompt`, `set_latent_size`, `set_seed` (all operate on every node whose `_meta.title` matches the prefix). Predicates: `has_input_mask`, `has_input_prompt`, `get_input_prompt` (returns `None` when there is no prompt node, `''` when the node's value is linked rather than literal).

`set_input_prompt` / `get_input_prompt` pick the node's input key from `_PROMPT_KEYS = ('text', 'prompt', 'string', 'value')` — first one holding a `str`, else first one present, else `'text'`. That way a `CLIPTextEncode`, a primitive string node, or a custom node all work.

### Validation

- `validate_detection_workflow(json)` — requires `LTD_Input_Image` + (`LTD_Output_Mask` or `LTD_Output_Image`)
- `validate_modification_workflow(json)` — requires **only** `LTD_Input_Image` + `LTD_Output_Image`
- `validate_caption_workflow(json)` — requires `LTD_Input_Image` + `LTD_Output_Text`
- `validate_generation_workflow(json)` — requires `LTD_Input_Text`, `LTD_Latent_Size`, `LTD_Output_Image`

### Optional mask (Modify tab)

A modification workflow only consumes a mask if it declares `LTD_Input_Mask`. `BaseModificationModule.wants_mask()` (default `True`; overridden by `ComfyUIModificationModule` to mirror `has_input_mask` on the prepared workflow) is queried by `ModifyTab._run_modification` **after** `prepare()`:

| Mask node | Image has mask | Behaviour |
|-----------|----------------|-----------|
| yes | yes | normal inpaint run |
| yes | no | critical box `"<module name> requires mask"`, run aborted |
| no | yes | warning `_confirm_ignore_mask` with **Proceed without mask** / **Cancel** |
| no | no | runs on every loaded image |

`ModificationWorker(..., use_mask=False)` passes `mask_path=None` to `run()` and stops skipping maskless images, so `BaseModificationModule.run` takes `mask_path: Path | None`. Because the run list is no longer "all images with masks", `ModifyTab` stores it in `self._run_images` and `_on_mod_result` indexes into that (recomputing the filter would mis-map results).

### Prompt override (`LTD_Input_Prompt`)

`WorkflowSelector` shows a prompt box whenever the selected workflow has an `LTD_Input_Prompt` node, so every consumer (detection, modification, caption, generate) gets it for free:

- pre-filled from the workflow's own prompt value, or from a per-workflow override persisted at `workflow_selector/<settings_key>/prompt/<sanitized workflow name>`;
- `↺` resets to the workflow's value and clears the override;
- `get_workflow_text()` injects the current prompt into a parsed copy and returns re-serialized JSON — **the workflow file on disk is never written**. When the prompt is untouched it returns the raw file text verbatim.

Detection uses a cheap `json.loads` + `is_api_format` check (`WorkflowSelector._parse`), never `load_workflow`, because it runs on every combo change and must not hit `/object_info` over the network. Consequence: a UI-format workflow gets no prompt box until it is saved in API format. Custom-JSON typing re-scans on a 500 ms `QTimer` debounce; `_prompt_source` guards the box so re-scans don't wipe what the user typed.

Shipped example: `Comfy-workflows/Remove with Klein.json` has no mask node and its positive `CLIPTextEncode` is titled `LTD_Input_Prompt`, so it runs as a prompt-driven editor over maskless images.

Typing into the prompt box requires the Label/Modify tab key filters to stand down — see "Tab Key Filters Swallowed Typing in Text Fields" in gotchas-decisions.md.

### Caption→Image Generation (Caption tab, Generate)

The Caption tab's **Generate** tab ("Compare original image to generated from caption") holds a `WorkflowSelector(settings_key='generate')`, a **Show** checkbox, and Current/Selected/All buttons. `CaptionImageGenerator` (in `caption_tab.py`) renders each image's caption:
- injects the caption via `set_input_text` and, via `set_latent_size`, the output dimensions computed by `CaptionImageGenerator.compute_dims()` — the original's aspect ratio scaled to ~1 MPix and snapped to a multiple of 16;
- runs through `ComfyUIClient.run_workflow` and copies the first output image into a per-folder cache (see data-formats-storage.md), never into the source folder.
- Runs in `GenerationWorker` (`ltd/workers/generation_worker.py`), the same progress/cancel/batch pattern as `CaptionWorker`. After the batch the worker calls the generator's `finalize()` hook, which calls `ComfyUIClient.free()` (`POST /free` `{"unload_models": true, "free_memory": true}`) to drop models from VRAM — best-effort.
- Shipped workflow: `Comfy-workflows/Generate from caption (krea2 turbo).json` (adapted from a krea2-turbo text-to-image graph with the three LTD nodes renamed).

When a cached render exists for the displayed image, `CaptionTab._display_image` (called by `_on_image_changed`) shows it beside the original via `CaptionImageViewer.load_comparison`: two `QGraphicsPixmapItem`s — the **original at native resolution** (never downscaled; reused from `_pixmap_cache` when available) and the generated scaled to share the common dimension so the pair displays equal-sized while both retain full detail on zoom. Landscape originals stack top/bottom; square/portrait sit left/right.

The **Show** checkbox (`generate_show_check`, persisted at `caption/generate_show`, default on) gates that comparison: with it off `_display_image` ignores the cached render and shows the original alone, and `_on_generated_result` doesn't swap a freshly finished render onto the screen. Renders stay on disk either way, so toggling it back on re-displays them — the handler `_on_generate_show_toggled` re-runs `_display_image` on the current image so the switch is immediate.

`ComfyUIClient.run_workflow` downloads outputs honoring each file's `subfolder`/`type` from history (a SaveImage `filename_prefix` like `folder/name` saves into a subfolder), and `download_output` raises on HTTP error rather than writing a broken file.

### UI-to-API Format Conversion

`load_workflow()` auto-detects UI-format workflows (exported from ComfyUI's "Save" button) and converts them to API format via `_convert_ui_to_api()`. This calls ComfyUI's `/object_info` endpoint to map node types to their inputs. The `_SEED_CONTROL_VALUES` constant (`{'fixed', 'increment', 'decrement', 'randomize'}`) is used to skip `control_after_generate` widget values during conversion.

### Workflow Selector Widget

`ltd/widgets/workflow_selector.py` — dropdown listing `.json` files from `Comfy-workflows/`, plus the Custom JSON area and the optional prompt box described above.
