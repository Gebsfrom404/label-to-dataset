import os
from pathlib import Path

SCRIPT_INFO = {
    'name': 'Rename && Reformat gallery-dl e621 Tag Files',
    'type': 'tags',
    'description': 'Remove .png/.jpg from txt filenames and convert newline-delimited tags to comma-separated',
    'parameters': [
        {'name': 'input_folder', 'type': 'folder', 'label': 'Input Folder', 'default': ''},
        {'name': 'replace_underscores', 'type': 'bool', 'label': 'Replace Underscores with Spaces', 'default': True},
    ],
}


def check_available() -> tuple[bool, str]:
    return True, ''


def run(params: dict, progress_callback) -> None:
    input_folder = params.get('input_folder', '').strip()

    if not input_folder or not os.path.isdir(input_folder):
        raise ValueError(f'Input folder does not exist: {input_folder}')

    replace_underscores = params.get('replace_underscores', True)
    folder = Path(input_folder)
    txt_files = list(folder.glob('*.txt'))
    total = len(txt_files)

    if total == 0:
        progress_callback(0, 0, 'No .txt files found')
        return

    renamed = 0
    for i, txt_file in enumerate(txt_files):
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        tags = [tag.strip() for tag in content.strip().split('\n') if tag.strip()]
        if replace_underscores:
            tags = [tag.replace('_', ' ') for tag in tags]
        new_content = ', '.join(tags)

        # Remove .png or .jpg from stem if present
        original_name = txt_file.stem
        new_name = original_name
        if original_name.endswith('.png'):
            new_name = original_name[:-4]
        elif original_name.endswith('.jpg'):
            new_name = original_name[:-4]

        new_file = txt_file.parent / f'{new_name}.txt'

        with open(new_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        if new_file != txt_file:
            os.remove(txt_file)
            renamed += 1

        progress_callback(i + 1, total, f'Processed {txt_file.name} ({len(tags)} tags)')

    progress_callback(total, total, f'Done — {total} files processed, {renamed} renamed')
