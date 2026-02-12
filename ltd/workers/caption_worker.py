"""Worker thread for batch captioning."""
from pathlib import Path

from PySide6.QtCore import Signal

from ltd.data.image_item import ImageItem
from ltd.workers.base_worker import BaseWorker


class CaptionWorker(BaseWorker):
    """Run captioning on a list of images."""
    caption_complete = Signal(int, list)  # image_index, list[str] tags

    def __init__(self, captioner, images: list[ImageItem], parent=None):
        super().__init__(parent)
        self.captioner = captioner
        self.images = images

    def do_work(self):
        total = len(self.images)
        errors = []
        for i, image in enumerate(self.images):
            if self.is_cancelled:
                break

            self.status.emit(f'Captioning {i + 1}/{total}: {image.filename}')
            self.progress.emit(i, total)

            try:
                tags = self.captioner.caption(image.path)
                self.caption_complete.emit(i, tags)
            except Exception as e:
                msg = f'{image.filename}: {e}'
                errors.append(msg)
                self.status.emit(f'Error: {msg}')

            if (i + 1) % self.BATCH_SIZE == 0:
                self._cleanup_memory()

        self.progress.emit(total, total)
        if errors:
            self.error.emit('Captioning errors:\n' + '\n'.join(errors))
        else:
            self.status.emit('Captioning complete')
