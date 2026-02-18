import shutil


SCRIPT_INFO = {
    'name': 'Download with gallery-dl',
    'description': 'Download images from URLs using gallery-dl',
    'parameters': [
        {'name': 'download_link', 'type': 'str', 'label': 'Download Link', 'default': '', 'placeholder': 'https://...'},
        {'name': 'save_folder', 'type': 'folder', 'label': 'Save Folder', 'default': ''},
        {'name': 'proxy_url', 'type': 'str', 'label': 'Proxy URL', 'default': '', 'placeholder': 'http://localhost:2080'},
        {'name': 'filter_string', 'type': 'str', 'label': 'Filter String', 'default': ''},
        {'name': 'write_tags', 'type': 'bool', 'label': 'Write Tags', 'default': False},
    ],
}


def check_available() -> tuple[bool, str]:
    """Return (available, reason). If not available, reason shown to user."""
    if shutil.which('gallery-dl') is None:
        return False, 'gallery-dl not found in PATH'
    return True, ''


def build_command(params: dict) -> list[str]:
    """Build command list for subprocess. Spawned in a new terminal."""
    cmd = ['gallery-dl']

    if params.get('proxy_url'):
        cmd.extend(['--proxy', params['proxy_url']])

    if params.get('save_folder'):
        cmd.extend(['-d', params['save_folder']])

    if params.get('filter_string'):
        cmd.extend(['--filter', params['filter_string']])

    if params.get('write_tags'):
        cmd.append('--write-tags')

    if params.get('download_link'):
        cmd.append(params['download_link'])

    return cmd
