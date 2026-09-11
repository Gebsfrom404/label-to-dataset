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

Reads the image with `IMREAD_COLOR` and writes RGB — transparency is handled
outside the module by `ModificationWorker` (see workers-threading.md).

`edge_step` setting (default 0px, `lama_inpaint/edge_step`) zeroes the mask
within N px of the image border. Applied *after* the 1536px downscale, so the
margin is measured in the pixels LaMa actually sees. Use it when a mask touches
the border: the crop is padded with `BORDER_REFLECT_101`, which mirrors the
opposite side in, and keeping a thin strip of original pixels usually looks
better. If the step-away empties the mask, inpainting is skipped entirely.

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
Auto-Caption dropdown, followed by the `ComfyUI Workflow` and `Local server`
entries. Dispatch (`_on_captioner_changed` / `_create_captioner`) is by the
combo's current *text* for the two special entries, falling back to
`_WD_MODELS[index]` for the taggers; each captioner has its own settings panel
(`wd_settings` / `comfy_settings` / `local_server_settings`), one visible at a time.

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

### Embeddings from the tagger

`convnextv2_huge.dbv4-full` is a plain timm `convnextv2_huge` with a
`NormMlpClassifierHead`, so the head splits cleanly and **tags and embeddings
come from one forward pass**:

```python
feats  = model.forward_features(tensor)             # (N, 2816, 16, 16)
embeds = model.forward_head(feats, pre_logits=True) # (N, 2816)  <- embedding
logits = model.forward_head(feats)                  # (N, 12476) <- tags
```

The 2816-d pre-logits vector is post-pool, post-LayerNorm, pre-`head.fc` — the
right layer for cosine similarity. It is **not** L2-normalized; normalize
before comparing. `model.reset_classifier(0)` yields a bit-identical vector but
destroys the tag head, so prefer `pre_logits=True`.

**Never run this model in fp16** — ConvNeXtV2's GRN takes a spatial L2 norm
that overflows half range at 512px and the embeddings come out **all-NaN**. Use
bf16 (same exponent range as fp32, cosine similarity 0.9996 vs fp32, ~2×
faster) or plain fp32. See [gotchas-decisions.md](gotchas-decisions.md).

Consumed by `extras_scripts/dataset_clustering.py` —
see [dataset-clustering.md](dataset-clustering.md).

## Local Server Captioner (Caption Tab)

Natural-language captioning via a vision model on any local OpenAI-compatible
server (was "LM Studio" before; the user runs Unsloth Studio). Client:
`ltd/localserver/client.py` (`LocalServerClient`) — `requests`, no SDK,
mirroring `ltd/comfyui/client.py`:
- URL + API key from the `local_server_url` / `local_server_api_key` settings
  (Settings tab; default `http://localhost:1234`, key empty). `/v1` is optional
  in the URL. A non-empty key is sent as `Authorization: Bearer` on **every**
  request — Unsloth Studio returns 401 on `/v1/*` without one (unless keyless
  API access is enabled in Studio), and so does llama.cpp started with
  `--api-key`. 401/403 raise `PermissionError` with a "set the API key" hint.
- **Flavor detection** (`detect_flavor(refresh=False)`), cached per
  (URL, key) in a module dict. `GET /api/health` is probed first and doubles as
  the reachability check (connection failure → `ConnectionError`, no further
  probes). Then, in order: `Server` header containing `unsloth` (Studio sends
  `unsloth-studio`) or health `service` containing "Unsloth" → Unsloth Studio;
  `Server: llama.cpp` → llama.cpp; `GET /api/v0/models` has `data` → LM Studio;
  `GET /api/version` has `version` → Ollama; `GET /props` has
  `default_generation_settings`/`modalities` → llama.cpp; `GET /version` has
  `version` → vLLM; else generic OpenAI. Order matters: Unsloth answers
  `/props` and `/version` with 401, so it must be matched before those probes.
- `list_models(vision_only=True)` → `(ids, vision_filtered)`. Per flavor:
  LM Studio native `/api/v0/models` `type == 'vlm'` (filtered); Ollama
  `/api/tags` + `POST /api/show` `capabilities` contains `vision`; llama.cpp
  `/v1/models` + `/props` `modalities.vision`; **Unsloth** `/v1/models` lists
  every downloaded model with a `loaded` flag — loaded first, and the loaded
  model is dropped only when `/v1/status` says `is_vision: false` (capabilities
  are unknown for the others, so `vision_filtered` is False); vLLM / generic:
  `/v1/models` unfiltered. The Caption tab status line only says "vision
  model(s)" when `vision_filtered` is True.
- `caption(path, model, system_prompt, user_text, max_megapixels)` →
  `POST /v1/chat/completions`, image inline as a base64 `data:` URI via
  `encode_image_data_uri()`: downscaled (LANCZOS) to the megapixel budget
  (default 1 MP; `<= 0` = original). An image already in budget and in
  png/jpg/webp is sent byte-for-byte; otherwise re-encoded as JPEG q95, or PNG
  when it has transparency (converted to RGB/RGBA *before* resizing — Pillow
  resizes palette images nearest-neighbour).
- `unload_model(id)` — best-effort, returns False on no endpoint / any error:
  Unsloth `POST /v1/unload {"model_path"}`, LM Studio
  `POST /api/v1/models/unload {"instance_id"}` (0.4.0+), Ollama
  `POST /api/generate {"model", "keep_alive": 0}`, llama.cpp
  `POST /models/unload {"model"}` (router mode only); vLLM / generic: none.

The Settings tab **Test** button runs `detect_flavor(refresh=True)` +
`list_models()`; the Caption tab **Refresh** button also re-detects, so
switching apps behind the same URL is picked up without a restart.

`LocalServerCaptioner` (in `caption_tab.py`) implements the standard
`caption(path) -> list[str]` captioner interface:
- Settings persisted at `caption/local_server_model`,
  `caption/local_server_system_prompt`, `caption/local_server_append`,
  `caption/local_server_max_mp`, `caption/local_server_unload`, and the
  system-prompt box height `caption/local_server_prompt_height` (set by the
  `HeightResizeGrip` drag bar under the box, default 120 px). The old
  `lmstudio_url` / `caption/lmstudio_*` keys are moved to the new names once by
  `settings.migrate_settings()` (called from `create_application()`). The
  Auto-Caption combo is persisted by **index**, so the rename kept the entry.
- Model list auto-populates on the first on-demand switch to Local server —
  never during startup restore (gated by `_lm_autorefresh_enabled`), so a down
  server can't block launch.
- "Append current caption" = send the image's current `.txt` content to the
  model as context (it does **not** append to the output).
- `strip_thinking()` removes `<think>…</think>` reasoning blocks from the reply.
- The result is split on the tag separator and **replaces** the existing
  caption: the captioner sets `replaces_caption = True`, which `CaptionTab`
  reads into `_caption_replace` so `_merge_tags` overwrites instead of honoring
  the WD-tagger Position combo. Splitting keeps `image.tags` a fixed point of
  the caption box's split-then-join, which `_commit_caption_edit()`'s
  "did the user edit?" check relies on — a captioner that stores a multi-line
  reply as one tag would need that check to compare text instead.
- After the batch, `CaptionWorker` calls the captioner's optional `finalize()`
  hook (in the worker thread, so no UI stall); `LocalServerCaptioner.finalize()`
  unloads the model when **Unload model after batch** is checked (default on).
  WD and ComfyUI captioners have no `finalize`, so the hook is skipped for them.
