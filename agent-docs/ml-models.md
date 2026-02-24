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

### Stack: ONNX Runtime

Uses `onnxruntime.InferenceSession` for inference. This matches the original SmilingWolf model format and produces correct results.

```python
from onnxruntime import InferenceSession
self._model = InferenceSession(str(model_path))
probs = self._model.run([output_name], {input_name: arr})[0][0]
```

### Auto-download

Models downloaded via `huggingface_hub.hf_hub_download()` to `models/caption/`. Files: `model.onnx`, `selected_tags.csv`.

### Preprocessing — BGR, no normalization

**Critical**: SmilingWolf ONNX models expect BGR channel order (OpenCV convention) with raw float32 values (NOT normalized to [-1,1]).

```python
arr = np.array(canvas, dtype=np.float32)[:, :, ::-1]  # RGB→BGR
arr = np.expand_dims(arr, axis=0)  # add batch dim (NHWC)
```

Input dimension is read from the model: `self._model.get_inputs()[0].shape`.
