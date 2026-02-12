from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DetectionType(Enum):
    BBOX = 'bbox'
    POLYGON = 'polygon'


@dataclass
class LabelClass:
    name: str
    color: str  # hex color e.g. '#FF0000'
    detection_type: DetectionType = DetectionType.BBOX
    class_id: int = 0

    def __eq__(self, other):
        if isinstance(other, LabelClass):
            return self.name == other.name and self.class_id == other.class_id
        return False

    def __hash__(self):
        return hash((self.name, self.class_id))


@dataclass
class Label:
    class_id: int
    bbox: Optional[tuple[float, float, float, float]] = None  # cx, cy, w, h (normalized)
    polygon: Optional[list[tuple[float, float]]] = None  # [(x, y), ...] normalized
    mask_path: Optional[str] = None

    @property
    def has_bbox(self) -> bool:
        return self.bbox is not None

    @property
    def has_polygon(self) -> bool:
        return self.polygon is not None and len(self.polygon) >= 3

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None


DEFAULT_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FF8000', '#8000FF', '#0080FF', '#FF0080', '#80FF00', '#00FF80',
    '#FF4040', '#40FF40', '#4040FF', '#FFAA00', '#AA00FF', '#00AAFF',
    '#FF6060', '#60FF60', '#6060FF', '#FFCC00', '#CC00FF', '#00CCFF',
]
