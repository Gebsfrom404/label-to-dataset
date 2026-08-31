"""Worker thread for generating images from captions via ComfyUI."""

from PySide6.QtCore import Signal

from ltd.data.image_item import ImageItem
from ltd.workers.base_worker import BaseWorker


class GenerationWorker(BaseWorker):
    """Run a ComfyUI generation workflow on a list of images.

    The generator is any object with ``generate(image) -> Path`` that renders
    the image's caption and returns the path of the cached generated image.
    """
    generated = Signal(int, str)  # image_index, generated_path

    def __init__(self, generator, images: list[ImageItem], parent=None):
        super().__init__(parent)
        self.generator = generator
        self.images = images

    def do_work(self):
        total = len(self.images)
        errors = []
        for i, image in enumerate(self.images):
            if self.is_cancelled:
                break

            self.status.emit(f'Generating {i + 1}/{total}: {image.filename}')
            self.progress.emit(i, total)

            try:
                path = self.generator.generate(image)
                self.generated.emit(i, str(path))
            except Exception as e:
                msg = f'{image.filename}: {e}'
                errors.append(msg)
                self.status.emit(f'Error: {msg}')

            # Cleanup after every image — see detection_worker.py.
            self._cleanup_memory()

        self.progress.emit(total, total)

        # Let the generator release resources after the batch (ComfyUI /free
        # to drop models from VRAM). Best-effort — never fail the run.
        finalize = getattr(self.generator, 'finalize', None)
        if finalize is not None:
            self.status.emit('Finishing up (freeing ComfyUI)...')
            try:
                finalize()
            except Exception:
                pass

        if errors:
            self.error.emit('Generation errors:\n' + '\n'.join(errors))
        else:
            self.status.emit('Generation complete')
