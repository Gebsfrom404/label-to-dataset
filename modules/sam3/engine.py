"""Shared SAM3 inference engine: point-prompt (Magic Wand) and text-prompt
(open-vocabulary detection) segmentation, on top of the vendored model in
``modules/sam3/vendor``.

Not a plugin module — used directly by ``modules/detection/sam3_detection.py``
and by the Label/Modify tabs' Magic Wand tool, via the shared singleton
returned by :func:`get_shared_engine`, so the (large) checkpoint is only ever
loaded once per process.
"""
import threading

import cv2
import numpy as np
import torch
from PIL import Image

from . import comfy_shim

comfy_shim.install()

from .vendor import build_sam3_image_model  # noqa: E402
from .vendor.model import Sam3Image  # noqa: E402
from .vendor.utils import Sam3Processor  # noqa: E402

_lock = threading.Lock()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a uint8 0/255 mask.

    Point-prompt segmentation occasionally scatters a few small stray
    islands around the main object; a single click means "this object",
    so only the biggest blob is kept.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num_labels <= 2:  # 0 = background, at most one foreground component
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


class Sam3Engine:
    """Lazily-loaded SAM3 model shared by point and text prompting."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Sam3Image | None = None
        self._processor: Sam3Processor | None = None

    def set_model_path(self, model_path: str | None):
        model_path = model_path or None
        if model_path != self._model_path:
            self._model_path = model_path
            self.unload()

    def _ensure_model(self):
        if self._model is not None:
            return
        with _lock:
            if self._model is not None:
                return
            model = build_sam3_image_model(
                checkpoint_path=self._model_path,
                device=self._device,
                load_from_HF=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,
            )
            model.to(self._device)
            model.eval()
            self._model = model
            self._processor = Sam3Processor(model, device=self._device)

    def _prepare_image(self, image_rgb: np.ndarray) -> dict:
        self._ensure_model()
        pil_image = Image.fromarray(image_rgb)
        return self._processor.set_image(pil_image)

    def segment_point(self, image_rgb: np.ndarray, x: float, y: float) -> np.ndarray:
        """Single positive-point segmentation. Returns a uint8 0/255 mask at
        the image's native resolution."""
        state = self._prepare_image(image_rgb)
        with torch.inference_mode():
            masks, ious, _ = self._model.predict_inst(
                state,
                point_coords=[[x, y]],
                point_labels=[1],
                multimask_output=True,
                normalize_coords=True,
            )
        best = masks[int(np.argmax(ious))] > 0
        return _largest_component(best.astype(np.uint8) * 255)

    def segment_text(self, image_rgb: np.ndarray, phrases: list[str],
                      confidence: float) -> list[dict]:
        """Open-vocabulary text-prompt detection, one independent query per
        phrase. Returns a list of {'phrase', 'mask', 'bbox_xyxy', 'score'}."""
        state = self._prepare_image(image_rgb)
        self._processor.confidence_threshold = confidence
        results = []
        with torch.inference_mode():
            for phrase in phrases:
                state = self._processor.set_text_prompt(phrase, state)
                # state['masks'] is [N, 1, H, W] (channel dim from the
                # interpolate() call in Sam3Processor._forward_grounding) —
                # drop it so each per-detection mask below is plain [H, W],
                # not [1, H, W] (which crashes cv2.findContours downstream).
                masks = state['masks'].detach().cpu().numpy().squeeze(1)
                boxes = state['boxes'].detach().cpu().numpy()
                scores = state['scores'].detach().cpu().numpy()
                for mask, box, score in zip(masks, boxes, scores):
                    results.append({
                        'phrase': phrase,
                        'mask': (mask.astype(np.uint8) * 255),
                        'bbox_xyxy': box,
                        'score': float(score),
                    })
        return results

    def unload(self):
        self._model = None
        self._processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_shared_engine: Sam3Engine | None = None


def get_shared_engine() -> Sam3Engine:
    """Return the process-wide SAM3 engine, creating it on first use.

    Magic Wand and the SAM3 detection module both call this so the checkpoint
    is loaded (and its VRAM held) only once, regardless of which feature is
    used first.
    """
    global _shared_engine
    if _shared_engine is None:
        from ltd.settings import get_settings
        model_path = get_settings().value('sam3/model_path', '', type=str)
        _shared_engine = Sam3Engine(model_path or None)
    return _shared_engine
