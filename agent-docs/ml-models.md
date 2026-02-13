# ML Models

## YOLO Detection

Uses Ultralytics library. Models stored in `models/yolo/`. Supports YOLO v8/v10/v11 variants (n/s/m/l/x).

Training via `ltd/tabs/train_tab.py` → `ltd/workers/training_worker.py` using `ultralytics.YOLO.train()`.

## LaMa Inpainting

`modules/modifications/lama_inpaint.py` — big-lama model loaded via `torch.jit.load()`.

Model file: `models/lama/big-lama.pt` (user downloads separately).

Mask dilation controlled by `mask_grow` setting (default 5px).

## WD Tagger (Caption Tab)

Auto-captioning using SmilingWolf WD tagger models. Implemented in caption_tab.py.

### Stack: timm + safetensors + PyTorch (NOT ONNX)

Originally used ONNX Runtime but migrated to native PyTorch for GPU support.

```python
model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
state_dict = load_file(str(model_path))  # safetensors
model.load_state_dict(state_dict)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
```

### Auto-download

Models downloaded via `huggingface_hub.hf_hub_download()` to `models/caption/`. Files: `model.safetensors`, `config.json`, `selected_tags.csv`.

### Preprocessing — BGR + Normalize

**Critical**: SmilingWolf models were trained with OpenCV (BGR channel order), not RGB.

```python
# Resize to 448x448, convert to numpy
arr = np.array(canvas, dtype=np.float32)

# BGR flip (model trained on OpenCV BGR input)
arr = arr[:, :, ::-1] / 255.0

# Normalize to [-1, 1]  (mean=0.5, std=0.5)
arr = (arr - 0.5) / 0.5

# .copy() required because negative stride from [::-1] is not contiguous
tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
```

Getting this wrong causes: wrong color tags (no BGR flip) or irrelevant tags (no normalization).
