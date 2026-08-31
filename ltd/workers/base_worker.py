import gc
import traceback

from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """Base worker thread with progress, status, error, and cancel support."""
    progress = Signal(int, int)  # current, total
    status = Signal(str)
    error = Signal(str)
    finished_work = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        try:
            self._is_cancelled = False
            self.do_work()
        except Exception as e:
            self.error.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        finally:
            self._cleanup_memory()
            self.finished_work.emit()

    def do_work(self):
        """Override this method in subclasses."""
        raise NotImplementedError

    def _cleanup_memory(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
