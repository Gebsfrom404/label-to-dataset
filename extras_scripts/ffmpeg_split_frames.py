import shutil


SCRIPT_INFO = {
    'name': 'Split GIFs and Videos into Frames with ffmpeg',
    'description': 'Extract frames from .gif, .webm, .mp4 files using ffmpeg',
    'parameters': [
        {'name': 'source_folder', 'type': 'folder', 'label': 'Source Folder', 'default': ''},
        {'name': 'output_folder', 'type': 'folder', 'label': 'Output Folder', 'default': '',
         'placeholder': 'Leave empty to save next to source files'},
        {'name': 'extensions', 'type': 'str', 'label': 'File Extensions', 'default': '*.gif *.webm *.mp4',
         'placeholder': '*.gif *.webm *.mp4'},
        {'name': 'fps', 'type': 'str', 'label': 'Frame Rate (fps)', 'default': '',
         'placeholder': 'e.g. 1 = one frame/sec, empty = all frames'},
        {'name': 'output_format', 'type': 'str', 'label': 'Output Format', 'default': 'png',
         'placeholder': 'png, jpg, bmp'},
    ],
}


def check_available() -> tuple[bool, str]:
    """Return (available, reason). If not available, reason shown to user."""
    if shutil.which('ffmpeg') is None:
        return False, 'ffmpeg not found in PATH'
    return True, ''


def build_command(params: dict) -> str:
    """Build a cmd for-loop string. Returned as str for shell execution."""
    source = params.get('source_folder', '.').replace('/', '\\')
    output = params.get('output_folder', '').replace('/', '\\')
    extensions = params.get('extensions', '*.gif *.webm *.mp4').strip()
    fmt = params.get('output_format', 'png').strip().lstrip('.')
    fps = params.get('fps', '').strip()

    # Build ffmpeg flags
    vf = f'-vf "fps={fps}"' if fps else ''

    if output:
        # Output to separate folder: output_folder/filename_0001.png
        out_tpl = f'"{output}\\%~nv_%04d.{fmt}"'
    else:
        # Output next to source file
        out_tpl = f'"%~dpv%~nv_%04d.{fmt}"'

    parts = ['for', '/r', f'"{source}"', '%v', 'in', f'({extensions})', 'do',
             'ffmpeg', '-i', '"%v"']
    if vf:
        parts.append(vf)
    parts.extend([out_tpl, '&', 'pause'])

    return ' '.join(parts)
