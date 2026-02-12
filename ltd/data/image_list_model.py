from pathlib import Path

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QSize, Qt,
                             Signal)
from PySide6.QtGui import QImage, QImageReader, QPixmap

from ltd.data.image_item import ImageItem
from ltd.settings import DEFAULT_SETTINGS, get_settings

IMAGE_EXTENSIONS = {'.bmp', '.gif', '.jpg', '.jpeg', '.png', '.tif', '.tiff',
                    '.webp'}


class ImageListModel(QAbstractListModel):
    images_loaded = Signal()

    def __init__(self, thumbnail_width: int = 160, parent=None):
        super().__init__(parent)
        self.images: list[ImageItem] = []
        self.thumbnail_width = thumbnail_width
        self._thumbnail_cache: dict[str, QPixmap] = {}

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.images)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.images):
            return None
        image = self.images[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return image.filename
        elif role == Qt.ItemDataRole.DecorationRole:
            return self._get_thumbnail(image)
        elif role == Qt.ItemDataRole.UserRole:
            return image
        elif role == Qt.ItemDataRole.ToolTipRole:
            return f'{image.filename}\n{image.width}x{image.height}'
        return None

    def _get_thumbnail(self, image: ImageItem) -> QPixmap:
        key = str(image.path)
        if key in self._thumbnail_cache:
            return self._thumbnail_cache[key]
        reader = QImageReader(str(image.path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and size.width() > 0:
            scale = self.thumbnail_width / size.width()
            reader.setScaledSize(
                QSize(self.thumbnail_width, int(size.height() * scale)))
        qimage = reader.read()
        if qimage.isNull():
            pixmap = QPixmap(self.thumbnail_width, self.thumbnail_width)
            pixmap.fill(Qt.GlobalColor.darkGray)
        else:
            pixmap = QPixmap.fromImage(qimage)
        self._thumbnail_cache[key] = pixmap
        return pixmap

    def load_directory(self, directory: Path):
        """Load all images from a directory."""
        self.beginResetModel()
        self.images.clear()
        self._thumbnail_cache.clear()

        if not directory.exists():
            self.endResetModel()
            return

        image_files = sorted(
            [f for f in directory.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda f: f.name.lower()
        )

        for f in image_files:
            try:
                import imagesize
                w, h = imagesize.get(str(f))
                if w <= 0 or h <= 0:
                    w, h = 0, 0
            except Exception:
                w, h = 0, 0
            item = ImageItem(path=f, width=w, height=h)
            self.images.append(item)

        self.endResetModel()
        self.images_loaded.emit()

    def load_items(self, items: list[ImageItem]):
        """Load pre-built ImageItems (e.g. from another tab)."""
        self.beginResetModel()
        self.images = list(items)
        self._thumbnail_cache.clear()
        self.endResetModel()
        self.images_loaded.emit()

    def get_image(self, index: int) -> ImageItem | None:
        if 0 <= index < len(self.images):
            return self.images[index]
        return None

    def clear(self):
        self.beginResetModel()
        self.images.clear()
        self._thumbnail_cache.clear()
        self.endResetModel()

    def invalidate_thumbnail(self, index: int):
        """Clear cached thumbnail for an image so it reloads."""
        if 0 <= index < len(self.images):
            key = str(self.images[index].path)
            self._thumbnail_cache.pop(key, None)
            model_index = self.index(index)
            self.dataChanged.emit(model_index, model_index)
