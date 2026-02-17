from pathlib import Path

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QSize, Qt,
                             QThread, Signal)
from PySide6.QtGui import QImage, QImageReader, QPixmap

from ltd.data.image_item import ImageItem
from ltd.settings import DEFAULT_SETTINGS, get_settings

IMAGE_EXTENSIONS = {'.bmp', '.gif', '.jpg', '.jpeg', '.png', '.tif', '.tiff',
                    '.webp'}


class ThumbnailWorker(QThread):
    """Generates thumbnails in a background thread.

    QImage is thread-safe; QPixmap conversion happens on the main thread.
    """
    batch_ready = Signal(list)  # list of (str_key, QImage)
    finished_all = Signal()

    BATCH_SIZE = 20

    def __init__(self, paths: list[tuple[str, str]], width: int,
                 parent=None):
        """paths: list of (cache_key, file_path_str)"""
        super().__init__(parent)
        self._paths = paths
        self._width = width
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        batch = []
        for key, path_str in self._paths:
            if self._cancelled:
                break
            reader = QImageReader(path_str)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and size.width() > 0:
                scale = self._width / size.width()
                reader.setScaledSize(
                    QSize(self._width, int(size.height() * scale)))
            qimage = reader.read()
            if qimage.isNull():
                qimage = QImage(self._width, self._width,
                                QImage.Format.Format_RGB32)
                qimage.fill(Qt.GlobalColor.darkGray)
            batch.append((key, qimage))
            if len(batch) >= self.BATCH_SIZE:
                self.batch_ready.emit(batch)
                batch = []
        if batch:
            self.batch_ready.emit(batch)
        self.finished_all.emit()


class ImageListModel(QAbstractListModel):
    images_loaded = Signal()

    def __init__(self, thumbnail_width: int = 160, parent=None):
        super().__init__(parent)
        self.images: list[ImageItem] = []
        self.thumbnail_width = thumbnail_width
        self._thumbnail_cache: dict[str, QPixmap] = {}
        self._placeholder: QPixmap | None = None
        self._thumb_worker: ThumbnailWorker | None = None

    def _get_placeholder(self) -> QPixmap:
        if self._placeholder is None:
            self._placeholder = QPixmap(self.thumbnail_width,
                                        self.thumbnail_width)
            self._placeholder.fill(Qt.GlobalColor.darkGray)
        return self._placeholder

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.images)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.images):
            return None
        image = self.images[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return image.filename
        elif role == Qt.ItemDataRole.DecorationRole:
            key = str(image.path)
            if key in self._thumbnail_cache:
                return self._thumbnail_cache[key]
            return self._get_placeholder()
        elif role == Qt.ItemDataRole.UserRole:
            return image
        elif role == Qt.ItemDataRole.ToolTipRole:
            return f'{image.filename}\n{image.width}x{image.height}'
        return None

    def _start_thumbnail_worker(self):
        """Start background thumbnail generation for all uncached images."""
        self._cancel_thumbnail_worker()
        paths = []
        for image in self.images:
            key = str(image.path)
            if key not in self._thumbnail_cache:
                paths.append((key, str(image.path)))
        if not paths:
            return
        self._thumb_worker = ThumbnailWorker(
            paths, self.thumbnail_width, self)
        self._thumb_worker.batch_ready.connect(self._on_thumbnails_ready)
        self._thumb_worker.finished_all.connect(self._on_thumbnails_done)
        self._thumb_worker.start()

    def _cancel_thumbnail_worker(self):
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.batch_ready.disconnect(
                self._on_thumbnails_ready)
            self._thumb_worker.finished_all.disconnect(
                self._on_thumbnails_done)
            self._thumb_worker.wait()
            self._thumb_worker = None

    def _on_thumbnails_ready(self, batch: list):
        """Receive a batch of (key, QImage), convert to QPixmap, update."""
        # Build index lookup for dataChanged
        key_to_row = {}
        for i, image in enumerate(self.images):
            k = str(image.path)
            if k not in self._thumbnail_cache:
                key_to_row[k] = i

        first_row = None
        last_row = None
        for key, qimage in batch:
            self._thumbnail_cache[key] = QPixmap.fromImage(qimage)
            row = key_to_row.get(key)
            if row is not None:
                if first_row is None or row < first_row:
                    first_row = row
                if last_row is None or row > last_row:
                    last_row = row

        if first_row is not None and last_row is not None:
            self.dataChanged.emit(self.index(first_row),
                                  self.index(last_row))

    def _on_thumbnails_done(self):
        self._thumb_worker = None

    def load_directory(self, directory: Path):
        """Load all images from a directory."""
        self._cancel_thumbnail_worker()
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
        self._start_thumbnail_worker()

    def load_items(self, items: list[ImageItem]):
        """Load pre-built ImageItems (e.g. from another tab)."""
        self._cancel_thumbnail_worker()
        self.beginResetModel()
        self.images = list(items)
        self._thumbnail_cache.clear()
        self.endResetModel()
        self.images_loaded.emit()
        self._start_thumbnail_worker()

    def get_image(self, index: int) -> ImageItem | None:
        if 0 <= index < len(self.images):
            return self.images[index]
        return None

    def clear(self):
        self._cancel_thumbnail_worker()
        self.beginResetModel()
        self.images.clear()
        self._thumbnail_cache.clear()
        self.endResetModel()

    def invalidate_thumbnail(self, index: int):
        """Clear cached thumbnail for an image so it reloads."""
        if 0 <= index < len(self.images):
            key = str(self.images[index].path)
            self._thumbnail_cache.pop(key, None)
            # Regenerate immediately (single image, fast enough)
            image = self.images[index]
            reader = QImageReader(str(image.path))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and size.width() > 0:
                scale = self.thumbnail_width / size.width()
                reader.setScaledSize(
                    QSize(self.thumbnail_width, int(size.height() * scale)))
            qimage = reader.read()
            if not qimage.isNull():
                self._thumbnail_cache[key] = QPixmap.fromImage(qimage)
            model_index = self.index(index)
            self.dataChanged.emit(model_index, model_index)
