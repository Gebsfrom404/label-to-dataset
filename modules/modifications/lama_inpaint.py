"""Built-in big-lama inpainting module."""
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from modules.base import BaseModificationModule

_HF_REPO = 'michaelgold/big-lama'
_HF_FILENAME = 'big-lama.safetensors'
_DEFAULT_MODEL_DIR = Path('./models/lama')


class LamaInpaintModule(BaseModificationModule):

    def __init__(self):
        self._model = None
        self._model_path = ''
        self._mask_grow = 5
        self._settings_widget = None
        self._model_path_label = None
        self._device = None
        self._is_jit = False
        self._restore_settings()

    @property
    def name(self) -> str:
        return 'Remove with big-lama'

    def get_settings_widget(self) -> QWidget:
        if self._settings_widget is not None:
            return self._settings_widget

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel('Model:'))
        label_text = (Path(self._model_path).name if self._model_path
                      else 'Auto-download')
        self._model_path_label = QLabel(label_text)
        self._model_path_label.setWordWrap(True)
        path_layout.addWidget(self._model_path_label, stretch=1)
        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Mask grow
        grow_layout = QHBoxLayout()
        grow_layout.addWidget(QLabel('Mask grow:'))
        self._grow_spin = QSpinBox()
        self._grow_spin.setRange(0, 50)
        self._grow_spin.setValue(self._mask_grow)
        self._grow_spin.setSuffix(' px')
        self._grow_spin.valueChanged.connect(self._on_grow_changed)
        grow_layout.addWidget(self._grow_spin)
        layout.addLayout(grow_layout)

        layout.addStretch()
        self._settings_widget = widget
        return widget

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            None, 'Select big-lama Model', str(_DEFAULT_MODEL_DIR),
            'Model Files (*.safetensors *.pt);;All Files (*)')
        if path:
            self._model_path = path
            self._model_path_label.setText(Path(path).name)
            self._model = None
            self._save_settings()

    def _on_grow_changed(self, value):
        self._mask_grow = value
        self._save_settings()

    def _save_settings(self):
        from ltd.settings import get_settings
        settings = get_settings()
        settings.setValue('lama_inpaint/model_path', self._model_path)
        settings.setValue('lama_inpaint/mask_grow', self._mask_grow)

    def _restore_settings(self):
        from ltd.settings import get_settings
        settings = get_settings()
        path = settings.value('lama_inpaint/model_path', '', type=str)
        if path and Path(path).exists():
            self._model_path = path
        self._mask_grow = settings.value('lama_inpaint/mask_grow', 5, type=int)

    def _download_model(self) -> Path:
        """Download big-lama.safetensors from HuggingFace if not present."""
        import huggingface_hub

        model_dir = _DEFAULT_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / _HF_FILENAME

        if not model_path.exists():
            huggingface_hub.hf_hub_download(
                _HF_REPO, filename=_HF_FILENAME,
                local_dir=str(model_dir))

        return model_path

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch

        # Auto-download if no model path configured
        if not self._model_path:
            safetensors_path = self._download_model()
            self._model_path = str(safetensors_path)
            if self._model_path_label is not None:
                self._model_path_label.setText(safetensors_path.name)
            self._save_settings()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device

        if self._model_path.endswith('.safetensors'):
            self._load_safetensors(self._model_path, device)
            self._is_jit = False
        else:
            self._model = torch.jit.load(self._model_path,
                                          map_location=device).eval()
            self._is_jit = True

    def _load_safetensors(self, path: str, device):
        import torch
        from safetensors.torch import load_file
        from modules.modifications.lama_arch import FFCResNetGenerator

        state_dict = load_file(path)

        # Strip 'generator.' prefix (HuggingFace big-lama format)
        prefix = 'generator.'
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v
                          for k, v in state_dict.items()}

        # Auto-detect block count from state dict keys
        resnet_indices = {int(k.split('.')[1]) for k in state_dict
                          if k.startswith('model.') and k.split('.')[1].isdigit()
                          and '.conv1.ffc.' in k}
        n_blocks = len(resnet_indices) if resnet_indices else 18

        has_lfu = any('lfu' in k for k in state_dict)

        model = FFCResNetGenerator(input_nc=4, output_nc=3, ngf=64,
                                   n_downsampling=3, n_blocks=n_blocks,
                                   enable_lfu=has_lfu)
        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(device).eval()

    def run(self, image_path: Path, mask_path: Path | None, **kwargs) -> Path:
        self._ensure_model()
        if mask_path is None:
            raise ValueError('LaMa inpainting requires a mask')

        from ltd.utils.file_utils import cv_imread
        image = cv_imread(image_path)
        mask = cv_imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {mask_path}')

        # Resize mask to match image if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Dilate mask
        if self._mask_grow > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel, iterations=self._mask_grow)

        # BGR -> RGB (big-lama was trained on RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Downscale large images to fit in VRAM (cap shortest side at 1536px)
        orig_h, orig_w = image.shape[:2]
        max_side = 1536
        short_side = min(orig_h, orig_w)
        if short_side > max_side:
            scale = max_side / short_side
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Process each disconnected mask region as an independent crop
        result = image.copy()
        regions = self._find_mask_regions(mask)
        for (x1, y1, x2, y2) in regions:
            result = self._inpaint_region(
                result, mask, x1, y1, x2, y2)

        # Upscale back to original resolution if we downscaled
        if short_side > max_side:
            result = cv2.resize(result, (orig_w, orig_h),
                                interpolation=cv2.INTER_LANCZOS4)

        # RGB -> BGR for cv2.imwrite
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Save result to temp dir (not next to original)
        from ltd.utils.file_utils import get_temp_dir_no_clear
        output_dir = get_temp_dir_no_clear('lama_output')
        output_path = output_dir / f'{image_path.stem}_inpainted.png'
        from ltd.utils.file_utils import cv_imwrite
        cv_imwrite(output_path, result)
        return output_path

    def _find_mask_regions(self, mask: np.ndarray, context: int = 128):
        """Find bounding boxes of disconnected mask regions with context."""
        h, w = mask.shape[:2]
        binary = (mask > 127).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)

        boxes = []
        for label_id in range(1, num_labels):
            ys, xs = np.where(labels == label_id)
            y1 = max(0, ys.min() - context)
            y2 = min(h, ys.max() + 1 + context)
            x1 = max(0, xs.min() - context)
            x2 = min(w, xs.max() + 1 + context)
            boxes.append((x1, y1, x2, y2))

        # Merge overlapping boxes
        merged = True
        while merged:
            merged = False
            new_boxes = []
            used = set()
            for i, a in enumerate(boxes):
                if i in used:
                    continue
                ax1, ay1, ax2, ay2 = a
                for j, b in enumerate(boxes):
                    if j <= i or j in used:
                        continue
                    bx1, by1, bx2, by2 = b
                    if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used.add(j)
                        merged = True
                new_boxes.append((ax1, ay1, ax2, ay2))
                used.add(i)
            boxes = new_boxes

        return boxes if boxes else [(0, 0, w, h)]

    def _inpaint_region(self, image, mask, x1, y1, x2, y2):
        """Inpaint a single cropped region, padding edges that touch borders."""
        import torch
        img_h, img_w = image.shape[:2]
        crop_img = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()

        # Pad sides that touch the image border so LaMa treats them as interior
        pad_size = 128
        pad_top = pad_size if y1 == 0 else 0
        pad_bot = pad_size if y2 == img_h else 0
        pad_left = pad_size if x1 == 0 else 0
        pad_right = pad_size if x2 == img_w else 0

        if any((pad_top, pad_bot, pad_left, pad_right)):
            # Pre-fill masked pixels with local colors so reflect padding
            # doesn't mirror artifacts into the padded area
            if crop_mask.any():
                crop_img = cv2.inpaint(
                    crop_img, (crop_mask > 127).astype(np.uint8),
                    inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            crop_img = cv2.copyMakeBorder(
                crop_img, pad_top, pad_bot, pad_left, pad_right,
                cv2.BORDER_REFLECT_101)
            crop_mask = cv2.copyMakeBorder(
                crop_mask, pad_top, pad_bot, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0)

        # Run LaMa on this crop
        img_tensor = self._prepare_image(crop_img)
        mask_tensor = self._prepare_mask(crop_mask)

        with torch.inference_mode():
            if self._is_jit:
                inpainted = self._model(img_tensor, mask_tensor)
            else:
                masked_image = img_tensor * (1 - mask_tensor)
                inp = torch.cat([masked_image, mask_tensor], dim=1)
                inpainted = self._model(inp)
                inpainted = (img_tensor * (1 - mask_tensor)
                             + inpainted * mask_tensor)
            patch = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            patch = np.clip(patch * 255, 0, 255).astype(np.uint8)
        del img_tensor, mask_tensor, inpainted

        # Remove modulo padding (from _pad_to_modulo)
        patch = patch[:crop_img.shape[0], :crop_img.shape[1]]

        # Remove edge padding
        ph, pw = patch.shape[:2]
        patch = patch[pad_top:ph - pad_bot if pad_bot else ph,
                      pad_left:pw - pad_right if pad_right else pw]

        # Paste back into the result image
        image[y1:y2, x1:x2] = patch
        return image

    def _prepare_image(self, image: np.ndarray):
        import torch
        img = image.astype(np.float32) / 255
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = self._pad_to_modulo(img, 8)
        return torch.from_numpy(img).unsqueeze(0).to(self._device)

    def _prepare_mask(self, mask: np.ndarray):
        import torch
        m = mask.astype(np.float32) / 255
        m = m[np.newaxis, ...]  # HW -> 1HW
        m = self._pad_to_modulo(m, 8)
        t = torch.from_numpy(m).unsqueeze(0).to(self._device)
        return (t > 0).float()

    @staticmethod
    def _pad_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
        _, h, w = img.shape
        out_h = ((h + mod - 1) // mod) * mod
        out_w = ((w + mod - 1) // mod) * mod
        return np.pad(img, ((0, 0), (0, out_h - h), (0, out_w - w)),
                      mode='symmetric')

    def unload(self):
        self._model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
