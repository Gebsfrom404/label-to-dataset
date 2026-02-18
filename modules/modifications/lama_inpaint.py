"""Built-in big-lama inpainting module."""
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from modules.base import BaseModificationModule


class LamaInpaintModule(BaseModificationModule):

    def __init__(self):
        self._model = None
        self._model_path = ''
        self._mask_grow = 5
        self._settings_widget = None
        self._model_path_label = None
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
                      else 'No model selected')
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
            None, 'Select big-lama Model', './models/lama',
            'PyTorch Models (*.pt);;All Files (*)')
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

    def _ensure_model(self):
        if self._model is None:
            if not self._model_path:
                raise ValueError('No big-lama model selected. '
                                 'Download big-lama.pt to ./models/lama/')
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = torch.jit.load(self._model_path,
                                          map_location=device).eval()
            self._device = device

    def run(self, image_path: Path, mask_path: Path, **kwargs) -> Path:
        import torch
        self._ensure_model()

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

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
            inpainted = self._model(img_tensor, mask_tensor)
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

        # Save result
        output_path = image_path.parent / f'{image_path.stem}_inpainted.png'
        cv2.imwrite(str(output_path), result)
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
