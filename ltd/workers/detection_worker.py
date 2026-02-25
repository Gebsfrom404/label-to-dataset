"""Worker thread for batch detection."""

from PySide6.QtCore import Signal

from ltd.data.image_item import ImageItem
from ltd.data.label_data import Label
from ltd.workers.base_worker import BaseWorker


class DetectionWorker(BaseWorker):
    """Run detection module on a list of images."""
    detection_complete = Signal(int, list)  # image_index, list[Label]

    def __init__(self, module, images: list[ImageItem], class_map: dict = None,
                 parent=None):
        super().__init__(parent)
        self.module = module
        self.images = images
        self.class_map = class_map or {}

    def do_work(self):
        total = len(self.images)
        errors = []
        for i, image in enumerate(self.images):
            if self.is_cancelled:
                break

            self.status.emit(f'Detecting {i + 1}/{total}: {image.filename}')
            self.progress.emit(i, total)

            try:
                results = self.module.run(image.path)
                labels = []
                for det in results:
                    class_id = det['class_id']
                    # Remap class if needed
                    if self.class_map:
                        class_name = det.get('class_name', '')
                        if class_name in self.class_map:
                            class_id = self.class_map[class_name]

                    label = Label(
                        class_id=class_id,
                        bbox=det.get('bbox'),
                        polygon=det.get('polygon'),
                    )
                    labels.append(label)

                self.detection_complete.emit(i, labels)
            except Exception as e:
                msg = f'{image.filename}: {e}'
                errors.append(msg)
                self.status.emit(f'Error: {msg}')

            # Batch memory cleanup
            if (i + 1) % self.BATCH_SIZE == 0:
                self._cleanup_memory()

        self.progress.emit(total, total)
        if errors:
            self.error.emit('Detection errors:\n' + '\n'.join(errors))
        else:
            self.status.emit('Detection complete')
