# Label-to-Dataset (LTD)
A desktop application for building image datasets end-to-end: label images, modify them (inpaint/remove objects), caption them with tags, and train YOLO models — all in a single pipeline.

Built with PySide6 (Qt) and PyTorch.

## Pipeline

```
Label  -->  Modify  -->  Caption  -->  Train
```

Each stage is a tab in the application. Data flows forward via "Copy to..." buttons, or each tab can load its own folder independently.

## Features

### Label Tab

Create and edit object annotations on images using a full-featured canvas.

**Drawing tools:**
- **Hand (M)** — pan the canvas (also available via Spacebar hold)
- **Pointer (P)** — select existing labels
- **BBox (R)** — draw bounding boxes
- **Polygon (V)** — draw polygon annotations (click to add points, click near first point to close)
- **Brush (B)** — freehand paint to create/extend polygon labels
- **Eraser (E)** — subtract from existing labels

**Navigation:** A/D to cycle images, W/S to cycle classes, Delete to remove selected label, Ctrl+Z to undo, scroll wheel to resize brush.

**Auto-detection:** Run YOLO models or ComfyUI workflows to auto-label all images in batch.

**Output:**
- Save as YOLO dataset (detection or segmentation format)
- Save binary masks
- Copy to Modify tab (images + masks)
- Copy to Train tab (auto-split 80/20 with data.yaml)

### Modify Tab

Apply modifications to images using their label masks (e.g., inpaint/remove detected objects).

**Built-in modules:**
- **big-lama** — LaMa inpainting model (torch.jit). Configurable mask dilation.
- **ComfyUI Workflow** — run any ComfyUI workflow that accepts an image and mask as input.

**Before/after comparison slider** to inspect results. Can chain modifications by re-running on already-modified images.

### Caption Tab

Tag images with text captions for training text-to-image or classification models.

**Captioning methods:**
- **WD Tagger** — automatic tagging using WD Eva02 Large Tagger v3 or animetimm ConvNeXtV2 Huge (dbv4-full), auto-downloaded from HuggingFace. Configurable probability threshold, max tags, and exclude list. The animetimm (dbv4) model may require a HuggingFace login — see [Models](#models).
- **ComfyUI Workflow** — run any ComfyUI captioning workflow.

**Tag editing:**
- Inline edit, drag-and-drop reorder, autocomplete from existing tags
- All-tags panel with frequency counts, sorting, and filtering
- Find & replace across all images (regex supported)
- Batch reorder tags by drag-and-drop priority
- Remove empty tags, remove duplicates

**Output:** Export images + caption `.txt` files to a folder.

### Train Tab

Train YOLO models directly from labeled datasets.

- Select base model: YOLO11 n/s/m/l/x or any custom `.pt` from `./models/yolo/`
- Detection or segmentation mode (auto-detected from dataset)
- Configurable epochs, image size, batch size
- Real-time training log output
- Trained models saved to `./models/yolo/` and immediately available in the Label tab

## ComfyUI Integration

LTD connects to a running ComfyUI instance for detection, modification, and captioning workflows.

**Setup:**
1. Start ComfyUI (recommended: `--highvram` flag to avoid model reloading per image)
2. Set the ComfyUI URL in the app toolbar (default: `http://127.0.0.1:8188`)

**Workflows:**
- Place ComfyUI API-format `.json` files in the `./Comfy-workflows/` folder — they appear in dropdown selectors
- UI-format workflows (saved with Ctrl+S in ComfyUI) are auto-converted to API format
- Custom workflows can be pasted directly in the "Custom" option

**Required node titles in workflows:**
| Workflow Type | Required Nodes |
|---|---|
| Detection | `LTD_Input_Image`, `LTD_Output_Mask` or `LTD_Output_Image` |
| Modification | `LTD_Input_Image`, `LTD_Input_Mask`, `LTD_Output_Image` |
| Caption | `LTD_Input_Image`, `LTD_Output_Text` |

Set these titles on the relevant LoadImage/SaveImage/PreviewImage nodes in your ComfyUI workflow.

## Installation

**Requirements:** Python 3.12+, NVIDIA GPU recommended (CUDA 12.1)

### Quick Start (Windows)

```
launch.bat
```

### Quick Start (Linux/macOS)

```
chmod +x launch.sh
./launch.sh
```

The launcher script will:
1. Create a `.venv` virtual environment
2. Install PyTorch with CUDA support
3. Install all dependencies
4. Create model directories
5. Start the application

### Manual Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Models

Place models in the corresponding directories:

| Model | Directory | Notes |
|---|---|---|
| YOLO (.pt) | `./models/yolo/` | For detection and training. YOLO11 base models download automatically. |
| big-lama (.pt) | `./models/lama/` | For inpainting. Download `big-lama.pt` manually. |
| WD Tagger | `./models/caption/` | Auto-downloaded from HuggingFace on first use. |

### HuggingFace login for the animetimm (dbv4) tagger

The **animetimm ConvNeXtV2 Huge (dbv4-full)** tagger is served from a HuggingFace repo that may require you to be authenticated. If the first download fails with a `401`/gated-repo error, authenticate once inside the project's virtual environment:

```bash
# Activate the venv first
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

hf auth login                   # paste a token from https://huggingface.co/settings/tokens
```

This stores the token in `~/.cache/huggingface/`, and `huggingface_hub` reads it automatically on every subsequent run — no need to re-login. Alternatively, set the `HF_TOKEN` environment variable before launching (e.g. in `launch.bat`). A read-only token is sufficient; if the model card shows an "Agree and access" gate, accept it on the model page first.

## Project Structure

```
label-to-dataset/
├── main.py                  # Entry point
├── launch.bat / launch.sh   # Launcher scripts
├── requirements.txt
├── Comfy-workflows/         # ComfyUI workflow JSON files
├── models/                  # Model weights (gitignored)
│   ├── yolo/
│   ├── lama/
│   └── caption/
├── modules/                 # Plugin system
│   ├── base.py              # ABC interfaces
│   ├── detection/           # Detection modules (YOLO, ComfyUI)
│   └── modifications/       # Modification modules (big-lama, ComfyUI)
└── ltd/                     # Main application package
    ├── app.py               # QApplication setup
    ├── main_window.py       # Main window with tab widget
    ├── settings.py          # QSettings wrapper
    ├── comfyui/             # ComfyUI client and workflow utilities
    ├── data/                # Data classes (ImageItem, Label, etc.)
    ├── tabs/                # Tab implementations
    ├── widgets/             # Reusable Qt widgets
    ├── workers/             # QThread workers for batch operations
    ├── dialogs/             # Find/replace, batch reorder, settings
    └── utils/               # Image, mask, YOLO format, file utilities
```

## Plugin System

Detection and modification modules are auto-discovered at startup from `modules/detection/` and `modules/modifications/`. To add a custom module:

1. Create a Python file in the appropriate directory
2. Define a class inheriting from `BaseDetectionModule` or `BaseModificationModule`
3. Implement `name`, `get_settings_widget()`, and `run()` methods

The module will automatically appear in the dropdown selector.

## License

MIT License. See [LICENSE](LICENSE) for details.
