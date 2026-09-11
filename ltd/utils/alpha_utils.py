"""Carry image transparency through RGB-only modification modules.

big-lama and ComfyUI's LoadImage/SaveImage work in RGB. Dropping the alpha
channel exposes whatever colour the file stores under fully transparent
pixels — undefined, often junk — so the module inpaints from it and writes it
back out as opaque pixels.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ltd.utils.file_utils import get_temp_dir_no_clear

# Relative aspect-ratio change still treated as a resize (e.g. a workflow
# rounding sides to a multiple of 8) rather than a crop or pad.
_ASPECT_TOLERANCE = 0.02


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ('RGBA', 'LA', 'PA') or 'transparency' in img.info


def _bleed_into_transparent(rgb: np.ndarray, transparent: np.ndarray):
    """Give every fully transparent pixel the colour of its nearest opaque one."""
    _, labels = cv2.distanceTransformWithLabels(
        transparent.astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    # DIST_LABEL_PIXEL numbers the opaque pixels in raster order from 1.
    opaque = np.flatnonzero(~transparent)
    hidden = np.flatnonzero(transparent)
    flat = rgb.reshape(-1, 3)
    nearest = labels.ravel()[hidden].astype(np.intp) - 1
    flat[hidden] = flat[opaque[nearest]]


def strip_transparency(image_path: Path) -> tuple[Path, np.ndarray | None]:
    """Prepare an image for a module that ignores alpha.

    Returns ``(path, alpha)``. Fully opaque images come back unchanged with
    ``alpha=None``. Otherwise the colour under fully transparent pixels is
    replaced by the nearest opaque colour, written to a temp PNG that keeps
    the input's stem (modules name their output after it) and its alpha
    (ComfyUI's LoadImage still gets its MASK).
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        if not _has_alpha(img):
            return image_path, None
        rgba = np.array(ImageOps.exif_transpose(img).convert('RGBA'))
    alpha = rgba[..., 3].copy()
    if (alpha == 255).all():
        return image_path, None

    transparent = alpha == 0
    if not transparent.any() or transparent.all():
        return image_path, alpha

    rgb = np.ascontiguousarray(rgba[..., :3])
    _bleed_into_transparent(rgb, transparent)
    out_path = get_temp_dir_no_clear('alpha_input') / f'{image_path.stem}.png'
    Image.fromarray(np.dstack([rgb, alpha]), 'RGBA').save(out_path)
    return out_path, alpha


def restore_transparency(output_path: Path, alpha: np.ndarray) -> Path:
    """Put the source alpha back on a module's output; returns its path.

    Outputs that already carry alpha (e.g. a background-removal workflow) are
    left alone, as are outputs whose aspect ratio changed — a crop or pad the
    alpha can't be lined up with. A resized output gets the alpha resized.
    Non-PNG outputs are replaced by a PNG of the same stem.
    """
    output_path = Path(output_path)
    with Image.open(output_path) as img:
        if _has_alpha(img):
            return output_path
        rgb = np.array(img.convert('RGB'))

    out_h, out_w = rgb.shape[:2]
    src_h, src_w = alpha.shape
    if (out_h, out_w) != (src_h, src_w):
        src_aspect = src_w / src_h
        if abs(out_w / out_h - src_aspect) > _ASPECT_TOLERANCE * src_aspect:
            return output_path
        interp = cv2.INTER_AREA if out_w < src_w else cv2.INTER_LINEAR
        alpha = cv2.resize(alpha, (out_w, out_h), interpolation=interp)

    png_path = output_path.with_suffix('.png')
    Image.fromarray(np.dstack([rgb, alpha]), 'RGBA').save(png_path)
    if png_path != output_path:
        output_path.unlink()
    return png_path
