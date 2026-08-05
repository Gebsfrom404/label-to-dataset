"""Worker thread for batch image modification."""

from PySide6.QtCore import Signal

from ltd.data.image_item import ImageItem
from ltd.workers.base_worker import BaseWorker


class ModificationWorker(BaseWorker):
    """Run modification module on a list of images."""
    modification_complete = Signal(int, str)  # image_index, output_path

    def __init__(self, module, images: list[ImageItem],
                 use_current: bool = False, use_mask: bool = True,
                 parent=None):
        super().__init__(parent)
        self.module = module
        self.images = images
        self.use_current = use_current
        # False when the module ignores masks (e.g. a ComfyUI workflow with
        # no LTD_Input_Mask node) — then maskless images still get processed.
        self.use_mask = use_mask

    def do_work(self):
        total = len(self.images)
        errors = []
        try:
            for i, image in enumerate(self.images):
                if self.is_cancelled:
                    break

                self.status.emit(f'Modifying {i + 1}/{total}: {image.filename}')
                self.progress.emit(i, total)

                try:
                    source = image.display_path if self.use_current else image.path
                    if self.use_mask and image.mask_path is None:
                        self.status.emit(f'Skipping {image.filename}: no mask')
                        continue

                    mask = image.mask_path if self.use_mask else None
                    output_path = self.module.run(source, mask)
                    self.modification_complete.emit(i, str(output_path))
                except Exception as e:
                    msg = f'{image.filename}: {e}'
                    errors.append(msg)
                    self.status.emit(f'Error: {msg}')

                if (i + 1) % self.BATCH_SIZE == 0:
                    self._cleanup_memory()
        finally:
            if hasattr(self.module, 'unload'):
                self.module.unload()

        self.progress.emit(total, total)
        if errors:
            self.error.emit('Modification errors:\n' + '\n'.join(errors))
        else:
            self.status.emit('Modification complete')
