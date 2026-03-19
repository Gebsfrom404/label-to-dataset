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

## WD Tagger (Caption Tab)

Auto-captioning using SmilingWolf WD tagger models. Implemented in caption_tab.py.

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

### Preprocessing — BGR, normalized, NCHW

**Critical**: SmilingWolf models expect BGR channel order. Normalize with mean=0.5/std=0.5, NCHW format for PyTorch.

```python
arr = np.array(canvas, dtype=np.float32)[:, :, ::-1] / 255.0  # RGB→BGR + scale
arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)  # NCHW
```

Input dimension hardcoded to 448 (from config).
