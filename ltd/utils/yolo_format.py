"""YOLO format .txt read/write utilities."""
from pathlib import Path

from ltd.data.label_data import Label


def parse_yolo_line(line: str) -> Label | None:
    """Parse a single YOLO annotation line."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None  # not a YOLO label (e.g. caption text)

    if len(values) == 4:
        # Detection format: class cx cy w h
        return Label(class_id=class_id, bbox=tuple(values))
    elif len(values) >= 6 and len(values) % 2 == 0:
        # Segmentation format: class x1 y1 x2 y2 ...
        polygon = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
        # Infer bbox from polygon
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return Label(class_id=class_id, bbox=(cx, cy, w, h), polygon=polygon)
    return None


def read_yolo_labels(txt_path: Path) -> list[Label]:
    """Read all labels from a YOLO .txt file."""
    if not txt_path.exists():
        return []
    labels = []
    for line in txt_path.read_text(encoding='utf-8').strip().splitlines():
        label = parse_yolo_line(line)
        if label is not None:
            labels.append(label)
    return labels


def format_yolo_bbox(label: Label) -> str:
    """Format a label as YOLO detection line: class cx cy w h"""
    if label.bbox is None:
        return ''
    cx, cy, w, h = label.bbox
    return f'{label.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'


def format_yolo_polygon(label: Label) -> str:
    """Format a label as YOLO segmentation line: class x1 y1 x2 y2 ...

    If the label has no polygon but has a bbox, converts bbox to 4-corner polygon.
    """
    if label.polygon is not None and len(label.polygon) >= 3:
        parts = [str(label.class_id)]
        for x, y in label.polygon:
            parts.append(f'{x:.6f}')
            parts.append(f'{y:.6f}')
        return ' '.join(parts)

    # Convert bbox to 4-corner polygon
    if label.bbox is not None:
        cx, cy, w, h = label.bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        parts = [str(label.class_id)]
        for x, y in corners:
            parts.append(f'{x:.6f}')
            parts.append(f'{y:.6f}')
        return ' '.join(parts)

    return ''


def write_yolo_labels(txt_path: Path, labels: list[Label],
                      force_format: str | None = None):
    """Write labels to a YOLO .txt file.

    Args:
        txt_path: Output file path.
        labels: Labels to write.
        force_format: If None, each label uses its native format.
                      If 'detection', all labels as bbox format.
                      If 'segmentation', all labels as polygon format
                      (bbox→4-corner polygon if no polygon data).
    """
    lines = []
    for label in labels:
        if force_format == 'detection':
            line = format_yolo_bbox(label)
        elif force_format == 'segmentation':
            line = format_yolo_polygon(label)
        else:
            # Native: polygon if available, otherwise bbox
            if label.has_polygon:
                line = format_yolo_polygon(label)
            else:
                line = format_yolo_bbox(label)
        if line:
            lines.append(line)
    txt_path.write_text('\n'.join(lines) + '\n' if lines else '',
                        encoding='utf-8')
