from PySide6.QtCore import QSettings

DEFAULT_SETTINGS = {
    'font_size': 12,
    'theme': 'dark',
    'comfyui_url': 'http://127.0.0.1:8188',
    'image_list_image_width': 160,
    'image_list_file_formats': 'bmp, gif, jpg, jpeg, png, tif, tiff, webp',
    'tag_separator': ', ',
    'last_label_directory': '',
    'last_modify_directory': '',
    'last_caption_directory': '',
    'last_dataset_directory': '',
    'detection_confidence': 0.25,
    'mask_grow': 5,
    'train_split': 80,
}


def get_settings() -> QSettings:
    return QSettings('LabelToDataset', 'LabelToDataset')
