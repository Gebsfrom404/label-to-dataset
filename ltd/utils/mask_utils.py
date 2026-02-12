"""Mask manipulation utilities."""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


def create_empty_mask(height: int, width: int) -> np.ndarray:
    """Create an empty (black) mask."""
    return np.zeros((height, width), dtype=np.uint8)


def polygon_to_mask(polygon: list[tuple[float, float]], width: int,
                    height: int) -> np.ndarray:
    """Convert normalized polygon points to a binary mask."""
    mask = create_empty_mask(height, width)
    if len(polygon) < 3:
        return mask
    pts = np.array([(int(x * width), int(y * height)) for x, y in polygon],
                   dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def bbox_to_mask(bbox: tuple[float, float, float, float], width: int,
                 height: int) -> np.ndarray:
    """Convert normalized bbox (cx, cy, w, h) to a binary mask."""
    mask = create_empty_mask(height, width)
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def merge_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Merge multiple binary masks into one via bitwise OR."""
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    result = masks[0].copy()
    for m in masks[1:]:
        result = cv2.bitwise_or(result, m)
    return result


def dilate_mask(mask: np.ndarray, iterations: int = 3,
                kernel_size: int = 7) -> np.ndarray:
    """Dilate a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def mask_to_bbox_normalized(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Infer normalized bbox (cx, cy, w, h) from a binary mask."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    height, width = mask.shape[:2]
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return (cx, cy, w / width, h / height)


def mask_to_polygons(mask: np.ndarray,
                     epsilon_factor: float = 0.004,
                     min_area: int = 50) -> list[list[tuple[float, float]]]:
    """Convert binary mask to list of normalized polygon point lists.

    Uses cv2.findContours + approxPolyDP for contour approximation.
    Returns list of polygons, each polygon is [(x_norm, y_norm), ...].
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            points = [(float(pt[0][0]) / w, float(pt[0][1]) / h)
                      for pt in approx]
            polygons.append(points)
    return polygons


def label_to_mask(label, width: int, height: int) -> Optional[np.ndarray]:
    """Convert a Label (with polygon or bbox) to a binary mask."""
    if hasattr(label, 'has_polygon') and label.has_polygon:
        return polygon_to_mask(label.polygon, width, height)
    elif hasattr(label, 'has_bbox') and label.has_bbox:
        return bbox_to_mask(label.bbox, width, height)
    return None


def masks_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """Check if two binary masks have any overlapping pixels."""
    return cv2.bitwise_and(mask_a, mask_b).any()


def save_mask(mask: np.ndarray, path: Path):
    """Save a binary mask as PNG."""
    cv2.imwrite(str(path), mask)


def load_mask(path: Path) -> np.ndarray | None:
    """Load a mask from a PNG file."""
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask


def mask_from_qimage(qimage) -> np.ndarray:
    """Convert a QImage (Format_Grayscale8) to numpy mask."""
    from PySide6.QtGui import QImage
    qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))
    return arr.copy()
