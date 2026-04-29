"""Background worker that parses generation metadata for a list of images."""
from PySide6.QtCore import QThread, Signal

from ltd.data.gen_metadata import GenMetadata, parse_metadata
from ltd.data.image_item import ImageItem


class MetadataWorker(QThread):
    parsed = Signal(object, object)  # ImageItem, GenMetadata
    progress = Signal(int, int)      # current, total
    finished_all = Signal()

    def __init__(self, images: list[ImageItem], parent=None):
        super().__init__(parent)
        # Snapshot the references — keyed by identity so row shifts in the
        # source model don't desync the worker.
        self._jobs: list[ImageItem] = list(images)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self._jobs)
        for i, image in enumerate(self._jobs):
            if self._cancelled:
                break
            if image.metadata is None:
                try:
                    md = parse_metadata(image.path)
                except Exception:
                    md = GenMetadata()
                self.parsed.emit(image, md)
            self.progress.emit(i + 1, total)
        self.finished_all.emit()
