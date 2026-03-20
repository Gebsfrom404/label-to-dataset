import os
from pathlib import Path

import cv2
import numpy as np

SCRIPT_INFO = {
    'name': 'Generate Masks for Black Edges',
    'type': 'masks',
    'description': 'Detect black borders in images and generate binary mask files (white=keep, black=border)',
    'parameters': [
        {'name': 'input_folder', 'type': 'folder', 'label': 'Input Folder', 'default': ''},
        {'name': 'output_folder', 'type': 'folder', 'label': 'Output Folder', 'default': '',
         'placeholder': 'Leave empty to save next to source images'},
        {'name': 'threshold', 'type': 'str', 'label': 'Black Threshold (0-255)', 'default': '30',
         'placeholder': '30'},
        {'name': 'invert', 'type': 'bool', 'label': 'Inverted Masks', 'default': False},
        {'name': 'overwrite', 'type': 'bool', 'label': 'Overwrite Existing Masks', 'default': False},
    ],
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def check_available() -> tuple[bool, str]:
    try:
        import cv2 as _cv2  # noqa: F401
    except ImportError:
        return False, 'opencv-python (cv2) not installed'
    return True, ''


def _detect_black_borders(image: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    """Return border thickness (top, bottom, left, right)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    top = bottom = left = right = 0

    for i in range(h):
        if np.mean(gray[i, :]) > threshold:
            top = i
            break

    for i in range(h - 1, -1, -1):
        if np.mean(gray[i, :]) > threshold:
            bottom = h - i - 1
            break

    for i in range(w):
        if np.mean(gray[:, i]) > threshold:
            left = i
            break

    for i in range(w - 1, -1, -1):
        if np.mean(gray[:, i]) > threshold:
            right = w - i - 1
            break

    return top, bottom, left, right


def run(params: dict, progress_callback) -> None:
    """Run edge mask generation. Called on a worker thread.

    Args:
        params: Parameter values from the UI.
        progress_callback(current, total, message): Report progress.
    """
    input_folder = params.get('input_folder', '').strip()
    output_folder = params.get('output_folder', '').strip()
    invert = params.get('invert', False)
    overwrite = params.get('overwrite', False)

    try:
        threshold = int(params.get('threshold', '30'))
    except ValueError:
        threshold = 30

    if not input_folder or not os.path.isdir(input_folder):
        raise ValueError(f'Input folder does not exist: {input_folder}')

    if not output_folder:
        output_folder = input_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith(IMAGE_EXTENSIONS) and not f.endswith('-masklabel.png')]
    total = len(files)

    if total == 0:
        progress_callback(0, 0, 'No images found')
        return

    created = 0
    skipped = 0
    for i, filename in enumerate(files):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f'{base_name}-masklabel.png')

        if not overwrite and os.path.exists(output_path):
            skipped += 1
            progress_callback(i + 1, total, f'Skipped {filename} (mask exists)')
            continue

        image = cv2.imread(input_path)
        if image is None:
            progress_callback(i + 1, total, f'Could not read {filename}')
            continue

        h, w = image.shape[:2]
        top, bottom, left, right = _detect_black_borders(image, threshold)

        # Skip if no borders detected
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            skipped += 1
            progress_callback(i + 1, total, f'No borders in {filename}')
            continue

        mask = np.full((h, w), 255, dtype=np.uint8)
        if top > 0:
            mask[:top, :] = 0
        if bottom > 0:
            mask[-bottom:, :] = 0
        if left > 0:
            mask[:, :left] = 0
        if right > 0:
            mask[:, -right:] = 0

        if invert:
            mask = cv2.bitwise_not(mask)

        cv2.imwrite(output_path, mask)
        created += 1
        progress_callback(i + 1, total, f'Created mask for {filename}')

    progress_callback(total, total, f'Done — {created} masks created, {skipped} skipped')
