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

    def run(self, image_path: Path, mask_path: Path, **kwargs) -> Path:
        import torch
        self._ensure_model()

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

        # Prevent mask from touching border
        mask[0:1, :] = 0
        mask[-1:, :] = 0
        mask[:, 0:1] = 0
        mask[:, -1:] = 0

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

        # Prepare tensors
        img_tensor = self._prepare_image(image)
        mask_tensor = self._prepare_mask(mask)

        with torch.inference_mode():
            if self._is_jit:
                inpainted = self._model(img_tensor, mask_tensor)
            else:
                masked_image = img_tensor * (1 - mask_tensor)
                inp = torch.cat([masked_image, mask_tensor], dim=1)
                inpainted = self._model(inp)
                # Composite: keep original pixels outside mask
                inpainted = img_tensor * (1 - mask_tensor) + inpainted * mask_tensor
            result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
        del img_tensor, mask_tensor, inpainted

        # Crop padding back to inference size
        result = result[:image.shape[0], :image.shape[1]]

        # Upscale back to original resolution if we downscaled
        if short_side > max_side:
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # RGB -> BGR for cv2.imwrite
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Save result to temp dir (not next to original)
        from ltd.utils.file_utils import get_temp_dir_no_clear
        output_dir = get_temp_dir_no_clear('lama_output')
        output_path = output_dir / f'{image_path.stem}_inpainted.png'
        from ltd.utils.file_utils import cv_imwrite
        cv_imwrite(output_path, result)
        return output_path

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
