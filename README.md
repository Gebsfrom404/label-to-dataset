# Label-to-Dataset (LTD)

A desktop application for building image datasets end-to-end: label images, modify them (inpaint/remove objects), caption them with tags, and train YOLO models — all in a single pipeline.

Built with PySide6 (Qt) and PyTorch.

### [You can tip me on Boosty](https://boosty.to/gebsfrom404)

## Pipeline

```
Label  -->  Modify  -->  Caption  -->  Train YOLO
```

Each stage is a tab. Data flows forward via the "Copy to..." buttons, or each tab can load its own folder independently. Two more tabs sit alongside: **Manage Gen Images** (browse generated images by their prompt/metadata) and **Extras** (run your own scripts from `./extras_scripts/`).

Tools, shortcuts and options are documented in-app — hover the ⓘ buttons.

## Installation

**Requirements:** Python 3.13+ (the uv installer below can fetch it for you). An NVIDIA GPU is strongly recommended — PyTorch is installed from the CUDA 12.8 index; everything still falls back to CPU, it's just will be slower.

### 1. Install

With [uv](https://docs.astral.sh/uv/) — recommended, installs uv and Python 3.13 automatically if missing:

```
install_uv.bat        REM Windows
./install_uv.sh       # Linux/macOS
```

Or with pip, if you already have Python 3.13+ on PATH:

```
install.bat           REM Windows
./install.sh          # Linux/macOS
```

Either one creates `.venv`, installs all dependencies and creates the model directories. Re-run it later to update.

### 2. Launch

```
launch.bat            REM Windows
./launch.sh           # Linux/macOS
```

(`chmod +x install_uv.sh launch.sh` first on Linux/macOS.)

### Manual installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# requirements.txt already points torch/torchvision at the CUDA 12.8 index
pip install -r requirements.txt

python main.py
```

### Models

**Everything downloads automatically on first use** — there is nothing to fetch by hand.

| Model | Directory | Notes |
|---|---|---|
| YOLO26 base models | `./models/yolo/` | Downloaded by ultralytics when training starts. Custom and freshly trained `.pt` files land here too and show up in the dropdowns. |
| big-lama | `./models/lama/` | `big-lama.safetensors` from [michaelgold/big-lama](https://huggingface.co/michaelgold/big-lama). "Browse..." in the module settings overrides it with your own checkpoint. |
| WD Tagger | `./models/caption/` | Model, tag list and config pulled from the selected HuggingFace tagger repo. |

Ready-made watermark detection models: [yolo-watermark-detection-collection](https://huggingface.co/Gebsfrom404/yolo-watermark-detection-collection) — drop the `.pt` files into `./models/yolo/`.

#### HuggingFace login for the animetimm (dbv4) tagger

The **animetimm ConvNeXtV2 Huge (dbv4-full)** tagger is served from a repo that may require authentication. If the first download fails with a `401`/gated-repo error, authenticate once inside the project's virtual environment:

```bash
# Activate the venv first
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

hf auth login                   # paste a token from https://huggingface.co/settings/tokens
```

The token is stored in `~/.cache/huggingface/` and reused on every subsequent run. Alternatively, set the `HF_TOKEN` environment variable before launching (e.g. in `launch.bat`). A read-only token is enough; if the model card shows an "Agree and access" gate, accept it on the model page first.

### ComfyUI (optional)

Detection, modification and captioning can run through a ComfyUI instance instead of the built-in modules.

1. Start ComfyUI (`--highvram` recommended, so models aren't reloaded for every image).
2. Set the ComfyUI URL in the app toolbar (default: `http://127.0.0.1:8188`).
3. Drop API-format `.json` workflows into `./Comfy-workflows/` — they show up in the workflow dropdowns. UI-format workflows (Ctrl+S in ComfyUI) are converted automatically.

LTD passes images in and out by **node title**, so the relevant LoadImage/SaveImage/PreviewImage nodes must be titled:

| Workflow type | Required node titles |
|---|---|
| Detection | `LTD_Input_Image`, `LTD_Output_Mask` or `LTD_Output_Image` |
| Modification | `LTD_Input_Image`, `LTD_Input_Mask`, `LTD_Output_Image` |
| Caption | `LTD_Input_Image`, `LTD_Output_Text` |
| Generation (from a caption) | `LTD_Input_Text`, `LTD_Latent_Size`, `LTD_Output_Image` |

## References

- [yolo-watermark-detection-collection](https://huggingface.co/Gebsfrom404/yolo-watermark-detection-collection) — my YOLO watermark detection models
- [Boosty](https://boosty.to/gebsfrom404) — support the project

## License

MIT License. See [LICENSE](LICENSE) for details.
