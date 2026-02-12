import shutil
import tempfile
from pathlib import Path

_TEMP_ROOT = Path(tempfile.gettempdir()) / 'label-to-dataset'


def get_temp_dir(subdir: str) -> Path:
    """Get or create a temp subdirectory, clearing any existing contents."""
    path = _TEMP_ROOT / subdir
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temp_dir_no_clear(subdir: str) -> Path:
    """Get or create a temp subdirectory without clearing."""
    path = _TEMP_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_all_temp():
    """Remove all temp data on exit, preserving persistent label dirs."""
    if not _TEMP_ROOT.exists():
        return
    for child in _TEMP_ROOT.iterdir():
        if child.name.startswith('labels_'):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
