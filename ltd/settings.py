from __future__ import annotations

from typing import overload

from PySide6.QtCore import QSettings

DEFAULT_SETTINGS = {
    'font_size': 12,
    'theme': 'dark',
    'comfyui_url': 'http://127.0.0.1:8188',
    'lmstudio_url': 'http://localhost:1234',
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
