import math
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT_INFO = {
    'name': 'Extract Sharp Frames from Videos',
    'description': (
        'Use sharp-frames (via uvx) to extract the sharpest frames from videos. '
        'Frame count = ceil(duration / ln(duration)), capped at 100.'
    ),
    'project_url': 'https://github.com/reflct/sharp-frames-python',
    'parameters': [
        {'name': 'input_folder', 'type': 'folder', 'label': 'Input Folder', 'default': ''},
        {'name': 'output_folder', 'type': 'folder', 'label': 'Output Folder', 'default': ''},
    ],
}

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv')


def check_available() -> tuple[bool, str]:
    if shutil.which('uvx') is None:
        return False, 'uvx not found in PATH (install uv: https://docs.astral.sh/uv/)'
    if shutil.which('ffmpeg') is None:
        return False, 'ffmpeg not found in PATH'
    return True, ''


def _get_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def _calc_num_frames(duration: float) -> int:
    """Calculate number of frames: ceil(duration / ln(duration)), capped at 100."""
    if duration <= 1:
        return 1
    return min(100, max(1, math.ceil(duration / math.log(duration))))


def run(params: dict, progress_callback) -> None:
    input_folder = params.get('input_folder', '').strip()
    output_folder = params.get('output_folder', '').strip()

    if not input_folder or not os.path.isdir(input_folder):
        raise ValueError(f'Input folder does not exist: {input_folder}')
    if not output_folder:
        raise ValueError('Output folder is required')

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    videos = [f for f in os.listdir(input_folder)
              if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
    total = len(videos)

    if total == 0:
        progress_callback(0, 0, 'No video files found')
        return

    total_frames = 0
    for vi, filename in enumerate(videos):
        video_path = os.path.join(input_folder, filename)
        stem = os.path.splitext(filename)[0]

        progress_callback(vi, total, f'Getting duration of {filename}...')
        try:
            duration = _get_duration(video_path)
        except (ValueError, subprocess.SubprocessError):
            progress_callback(vi + 1, total, f'Could not read duration of {filename}, skipping')
            continue

        num_frames = _calc_num_frames(duration)
        progress_callback(vi, total,
                          f'Extracting {num_frames} frames from {filename} ({duration:.1f}s)...')

        # Run sharp-frames into output folder
        result = subprocess.run(
            ['uvx', 'sharp-frames', video_path, output_folder,
             '--selection-method', 'best-n', '--num-frames', str(num_frames),
             '--force-overwrite'],
            capture_output=True, text=True, stdin=subprocess.DEVNULL
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            progress_callback(vi + 1, total, f'sharp-frames failed for {filename}: {err}')
            continue

        # Rename frame_N.jpg -> stem-frame_N.jpg
        renamed = 0
        for f in sorted(os.listdir(output_folder)):
            if f.startswith('frame_') and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                new_name = f'{stem}-{f}'
                os.rename(
                    os.path.join(output_folder, f),
                    os.path.join(output_folder, new_name)
                )
                renamed += 1

        total_frames += renamed
        progress_callback(vi + 1, total, f'{filename}: {renamed} frames extracted')

    progress_callback(total, total, f'Done - {total_frames} frames from {total} videos')
