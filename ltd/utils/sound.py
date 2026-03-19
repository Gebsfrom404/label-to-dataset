"""Simple completion sound cue for finished batch processes."""

import logging
import sys

logger = logging.getLogger(__name__)


def play_completion_sound():
    """Play a short system notification sound (non-blocking)."""
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        else:
            # Fallback: PySide6 system beep
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                app.beep()
    except Exception:
        logger.debug('Could not play completion sound', exc_info=True)
