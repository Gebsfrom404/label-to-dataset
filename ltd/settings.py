from __future__ import annotations

from typing import overload

from PySide6.QtCore import QSettings

DEFAULT_SETTINGS = {
    'font_size': 12,
    'theme': 'dark',
    'comfyui_url': 'http://127.0.0.1:8188',
    'local_server_url': 'http://localhost:1234',
    'local_server_api_key': '',
    'image_list_image_width': 160,
    'image_list_file_formats': 'bmp, gif, jpg, jpeg, png, tif, tiff, webp',
    'tag_separator': ', ',
    'last_label_directory': '',
    'last_modify_directory': '',
    'last_caption_directory': '',
    'last_dataset_directory': '',
    'last_duplicates_directory': '',
    'detection_confidence': 0.25,
    'mask_grow': 5,
    'train_split': 80,
}

# Keys renamed when the LM Studio captioner became the generic "Local server".
_RENAMED_KEYS = {
    'lmstudio_url': 'local_server_url',
    'caption/lmstudio_model': 'caption/local_server_model',
    'caption/lmstudio_system_prompt': 'caption/local_server_system_prompt',
    'caption/lmstudio_append': 'caption/local_server_append',
}


class TypedSettings(QSettings):
    @overload
    def value(self, key: str, defaultValue: bool, *, type: type[bool]) -> bool: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def value(self, key: str, defaultValue: int, *, type: type[int]) -> int: ...
    @overload
    def value(self, key: str, defaultValue: float, *, type: type[float]) -> float: ...
    @overload
    def value(self, key: str, defaultValue: str, *, type: type[str]) -> str: ...
    @overload
    def value(self, key: str, defaultValue: object = ..., type: type | None = ...) -> object: ...

    def value(self, key, defaultValue=None, type=None):  # type: ignore[override]
        if type is not None:
            return super().value(key, defaultValue, type=type)
        if defaultValue is not None:
            return super().value(key, defaultValue)
        return super().value(key)


def get_settings() -> TypedSettings:
    return TypedSettings('LabelToDataset', 'LabelToDataset')


def migrate_settings():
    """Move values stored under renamed keys to their new names (runs at startup)."""
    settings = get_settings()
    for old, new in _RENAMED_KEYS.items():
        if not settings.contains(old):
            continue
        value = settings.value(old)
        if value not in (None, '') and not settings.contains(new):
            settings.setValue(new, value)
        settings.remove(old)
