"""Worker thread for single-point SAM3 segmentation (Magic Wand)."""
from ltd.workers.base_worker import BaseWorker


class Sam3PointWorker(BaseWorker):
    """Run SAM3 point-prompt segmentation for one click on one image.

    Takes the image as an RGB numpy array (grabbed from the canvas's
    currently-displayed pixmap via CanvasWidget.get_image_rgb()) rather
    than a file path — the click's (x, y) is in canvas coordinate space,
    which may be a downscaled preview of the source file, so segmenting
    the re-read full-resolution file would desync coordinates and mask
    shape from what the user actually clicked on.
    """

    def __init__(self, engine, image_rgb, x: float, y: float, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.image_rgb = image_rgb
        self.x = x
        self.y = y
        self.mask = None

    def do_work(self):
        if self.image_rgb is None:
            self.error.emit('No image loaded on canvas')
            return
        self.mask = self.engine.segment_point(self.image_rgb, self.x, self.y)
        self.status.emit('Magic Wand: segmented')
