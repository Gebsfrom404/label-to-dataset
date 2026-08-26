# ML Models

## YOLO Detection

Uses Ultralytics library. Models stored in `models/yolo/`.

Train tab offers YOLO26 presets for both detect and segment tasks:
- Detect: `YOLO26n/s/m/l/x` (`yolo26n.pt` … `yolo26x.pt`)
- Segment: `YOLO26n-seg/s-seg/m-seg/l-seg/x-seg`
- Custom models: any `.pt` file placed in `./models/yolo/` is also listed in the dropdown

Training via `ltd/tabs/train_tab.py` → `ltd/workers/training_worker.py` using `ultralytics.YOLO.train()`.

For detection in Label tab, any user-supplied `.pt` model can be loaded (not restricted to YOLO26).

## LaMa Inpainting

`modules/modifications/lama_inpaint.py` — big-lama model loaded via `torch.jit.load()`.

Model file: `models/lama/big-lama.pt` (user downloads separately).

Mask dilation controlled by `mask_grow` setting (default 5px).

## SAM3 (Magic Wand + text-prompt detection)

Meta's SAM3 (Segment Anything Model 3), vendored as plain PyTorch — not via
the official gated `facebook/sam3` HF repo, but through the public,
ungated mirror `apozz/sam3-safetensors` (same one the ComfyUI-SAM3
reference downloads from), single checkpoint file `sam3.safetensors`.

**Two entry points, one shared model:**
- **Magic Wand** — a canvas tool (`Tool.MAGIC_WAND` in `ltd/widgets/canvas_widget.py`, used by both Label and Modify tabs) that segments the object under a single click via SAM3's SAM2-style point predictor.
- **`modules/detection/sam3_detection.py`** (`Sam3DetectionModule`) — a `BaseDetectionModule` in the Label tab's Auto-Detection dropdown. Text area = one open-vocabulary phrase per non-empty line, each queried **independently** against the image (own class name, own detections) — not sent as one combined string. Confidence slider gates `Sam3Processor.confidence_threshold` (default 0.20).

Both go through `modules/sam3/engine.py`'s `Sam3Engine` (`segment_point()` / `segment_text()`), accessed via the process-wide singleton `get_shared_engine()` so the checkpoint is loaded (and its VRAM held) only once regardless of which feature is used first. `Sam3Engine._prepare_image()` calls `Sam3Processor.set_image()` fresh on every call (no cross-call feature caching — simplicity over a minor perf win, see `modules/sam3/engine.py` docstring).

**Model path**: single shared QSettings key `sam3/model_path`, same "Browse... / Auto-download" UX as `LamaInpaintModule` — configured from the SAM3 detection module's settings panel (Magic Wand has no settings panel of its own, so it just reads the same key via the shared engine). Unset → `download_ckpt_from_hf()` pulls the public mirror on first use.

**Vendored code** (`modules/sam3/vendor/`): `model.py`, `attention.py`, `text_encoder.py`, `tokenizer.py`, `perflib.py`, `utils.py`, `bpe_simple_vocab_16e6.txt.gz`, and `__init__.py` are copied close to verbatim from `references/comfyui-sam3/nodes/sam3/` — this is ~13k lines of model math, hand-transcribing it would be far riskier than copying it. The **only** hand edit: `vendor/__init__.py` had its video/tracking builders (`build_sam3_video_model`, `build_sam3_video_predictor`) and the `predictor.py` import removed (video/tracking is unused here, and `predictor.py` pulls in `psutil`, which isn't otherwise a dependency).

**No ComfyUI carried over.** The vendored files' only ComfyUI touchpoints are a handful of thin torch wrappers (`comfy.ops.manual_cast.{Linear,Conv2d,ConvTranspose2d,LayerNorm,Embedding,GroupNorm}`, `comfy.ops.cast_to_input`, `comfy.model_management.get_torch_device`, `comfy.utils.load_torch_file`/`ProgressBar`, `comfy.ldm.modules.attention.optimized_attention_for_device`/`attention_pytorch`) — no node classes, no ModelPatcher, no `comfy-env` installer. `modules/sam3/comfy_shim.py` implements these as plain PyTorch and registers them into `sys.modules` (must run — via `comfy_shim.install()` — before any `modules.sam3.vendor.*` import; `engine.py` does this at module load time), so the vendored files' `import comfy.ops` etc. resolve transparently without ComfyUI installed. If a vendored file starts raising `AttributeError` on a `comfy.*` symbol after a future re-vendor, the missing symbol needs adding to the shim — grep the vendor dir for `comfy\.` to find what's actually used before guessing.

**New pip deps**: `ftfy`, `regex` (SAM3's CLIP-style BPE tokenizer needs the `regex` package, not stdlib `re`, plus `ftfy` for text cleanup). Everything else (`torch`, `torchvision`, `numpy`, `huggingface_hub`, `safetensors`) was already a dependency.

**Point-prompt coordinates**: pixel space (not normalized), matching the canvas's own `scene_pos` coordinates — no conversion needed between `CanvasWidget.magic_wand_requested(x, y)` and `Sam3Engine.segment_point(image_rgb, x, y)`. Label convention is SAM2-standard: `1` = foreground. `multimask_output=True` returns 3 candidate masks; `segment_point()` picks the one with the highest IoU prediction, then keeps only the mask's largest connected component (`engine.py:_largest_component()`) — point prompts occasionally scatter a few stray-pixel islands around the main object, which a single click shouldn't produce. `segment_text()` does **not** apply this — a text-prompt detection's mask can legitimately be multi-part (e.g. an object split by an occluder), and its detections are already one-object-per-entry.

Verified against the real checkpoint (not just a mocked engine): point-click and multi-phrase text-prompt detection both produce correct, well-separated masks/boxes on a real photo. `model.py`'s optional mask-postprocessing step warns and skips itself if `scikit-image` isn't installed (`No module named 'skimage'`) — harmless per the upstream code's own comment ("OK to ignore... doesn't affect results in most cases"), confirmed in testing; add `scikit-image` as a dependency only if a specific case turns up where it matters.

**Gotcha — `state['masks']` has a channel dim.** `Sam3Processor._forward_grounding()` (vendor/utils.py) builds `state['masks']` via `interpolate(out_masks.unsqueeze(1), ...)`, so its shape is `[N, 1, H, W]`, not `[N, H, W]`. `Sam3Engine.segment_text()` must `.squeeze(1)` before iterating per-detection masks — without it, each mask is `[1, H, W]` (3D), which `cv2.findContours` (via `mask_to_polygons()`) rejects with `cv::copyMakeBorder` `_src.dims() <= 2` assertion failures (findContours pads its input internally). See gotchas-decisions.md.

## WD Tagger (Caption Tab)

Auto-captioning using timm-based tagger models. Implemented in caption_tab.py.
`WdTaggerCaptioner` is the base (SmilingWolf `wd-eva02-large-tagger-v3`);
`AnimeTimmCaptioner` subclasses it for `animetimm/convnextv2_huge.dbv4-full`.
The `CaptionTab._WD_MODELS` tuple lists the WD-style taggers shown in the
Auto-Caption dropdown, followed by the `ComfyUI Workflow` and `LM Studio`
entries. Dispatch (`_on_captioner_changed` / `_create_captioner`) is by the
combo's current *text* for the two special entries, falling back to
`_WD_MODELS[index]` for the taggers; each captioner has its own settings panel
(`wd_settings` / `comfy_settings` / `lmstudio_settings`), one visible at a time.

### Stack: timm + safetensors (PyTorch)

Uses `timm.create_model()` with safetensors weights for inference. GPU-accelerated via PyTorch CUDA.

```python
import timm
from safetensors.torch import load_file
model = timm.create_model(arch, pretrained=False, num_classes=num_classes, **model_args)
state_dict = load_file(str(model_path))
model.load_state_dict(state_dict)
```

**Critical**: Must pass `model_args` from `config.json` (includes `ref_feat_shape`) to `timm.create_model()`. Without it, EVA02 attention produces different outputs.

### Auto-download

Models downloaded via `huggingface_hub.hf_hub_download()` to `models/caption/`. Files: `model.safetensors`, `selected_tags.csv`, `config.json`.

### Preprocessing — configurable per model, NCHW

Preprocessing is driven by class attributes so subclasses only override what
differs: `INPUT_SIZE`, `_BGR`, `_MEAN`, `_STD`. Image is padded to a square with
white background, resized to `INPUT_SIZE`, optionally flipped to BGR, then
normalized. `_MEAN`/`_STD` may be a scalar or a per-channel numpy array (RGB
order); broadcasting handles both.

```python
arr = np.array(canvas, dtype=np.float32) / 255.0
if self._BGR:
    arr = arr[:, :, ::-1]
arr = (arr - self._MEAN) / self._STD
tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)) \
    .permute(2, 0, 1).unsqueeze(0)  # NCHW
```

- **SmilingWolf** (`WdTaggerCaptioner`): BGR, mean=0.5/std=0.5 → [-1, 1], 448px.
- **animetimm** (`AnimeTimmCaptioner`): RGB, ImageNet mean/std
  (`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`), 512px. Repo is HF-gated —
  `hf_hub_download` needs the user's HF token/login to fetch weights.

Both use the same `selected_tags.csv` convention (`name`, `category` with
`9` = rating filtered out). Thresholding uses the UI `min_probability` slider
for both (animetimm's `best_threshold` column is ignored).

## LM Studio Captioner (Caption Tab)

Natural-language captioning via a local LM Studio vision model. Client:
`ltd/lmstudio/client.py` (`LMStudioClient`) — uses LM Studio's OpenAI-compatible
REST API (`requests`, no extra SDK), mirroring `ltd/comfyui/client.py`:
- URL from the `lmstudio_url` setting (toolbar field, default
  `http://localhost:1234`); `/v1` is appended if absent.
- `list_models(vision_only=True)` fills the **Model** dropdown with only
  vision-capable models. It queries LM Studio's **native** `GET /api/v0/models`
  (which reports a per-model `type` of `llm`/`vlm`/`embeddings`) and keeps only
  `type == 'vlm'`. If that endpoint is missing (older LM Studio), it falls back
  to `GET /v1/models`, which carries no capability info, so filtering is skipped
  there. Triggered by the **Refresh** button or the first on-demand switch to
  LM Studio — never during startup restore (gated by `_lm_autorefresh_enabled`),
  so a down server can't block launch.
- `caption(path, model, system_prompt, user_text)` → `POST /v1/chat/completions`
  with the image inlined as a base64 `data:` URI and the system prompt as a
  `system` message.

`LMStudioCaptioner` (in `caption_tab.py`) implements the standard
`caption(path) -> list[str]` captioner interface:
- **System prompt** and **Append current caption** switch are persisted at
  `caption/lmstudio_system_prompt`, `caption/lmstudio_append`,
  `caption/lmstudio_model`.
- "Append current caption" = send the image's current `.txt` content to the
  model as context (it does **not** append to the output).
- `strip_thinking()` removes `<think>…</think>` reasoning blocks from the reply.
- The result is split on the tag separator and **replaces** the existing
  caption: the captioner sets `replaces_caption = True`, which `CaptionTab`
  reads into `_caption_replace` so `_merge_tags` overwrites instead of honoring
  the WD-tagger Position combo.
- After the batch, `CaptionWorker` calls the captioner's optional `finalize()`
  hook (in the worker thread, so no UI stall). `LMStudioCaptioner.finalize()`
  unloads the model via `LMStudioClient.unload_model()` →
  `POST /api/v1/models/unload` `{"instance_id": <model>}` (LM Studio 0.4.0+ v1
  API) to free VRAM. Best-effort: silently no-ops on older builds / 404 /
  connection errors. WD and ComfyUI captioners have no `finalize`, so the hook
  is skipped for them.
