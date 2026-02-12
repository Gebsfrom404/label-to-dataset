from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage, QImageReader, QPixmap


def load_full_image(path: Path) -> QImage | None:
    """Load a full-resolution image from disk."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    return image if not image.isNull() else None


def load_pixmap(path: Path) -> QPixmap | None:
    """Load a full-resolution pixmap from disk."""
    image = load_full_image(path)
    if image is None:
        return None
    return QPixmap.fromImage(image)


def generate_thumbnail(path: Path, width: int) -> QPixmap:
    """Generate a thumbnail with the given width, preserving aspect ratio."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and size.width() > 0:
        scale = width / size.width()
        reader.setScaledSize(QSize(width, int(size.height() * scale)))
    image = reader.read()
    if image.isNull():
        pixmap = QPixmap(width, width)
        pixmap.fill()
        return pixmap
    return QPixmap.fromImage(image)
