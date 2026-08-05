"""Abstract base classes for detection and modification modules."""
from abc import ABC, abstractmethod
from pathlib import Path

from PySide6.QtWidgets import QWidget


class BaseDetectionModule(ABC):
    """ABC for detection modules (YOLO, ComfyUI, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the dropdown."""
        ...

    @abstractmethod
    def get_settings_widget(self) -> QWidget:
        """Return a QWidget with module-specific settings."""
        ...

    def prepare(self):
        """Capture widget state before worker thread starts.

        Called from the main/GUI thread. Override to read widget values
        into plain attributes so run() never touches Qt widgets.
        """

    @abstractmethod
    def run(self, image_path: Path, **kwargs) -> list[dict]:
        """Run detection on a single image.

        Returns list of dicts with keys:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: tuple[float, float, float, float] | None  (cx, cy, w, h normalized)
            - polygon: list[tuple[float, float]] | None  (normalized points)
            - mask: numpy.ndarray | None  (binary mask at image resolution)
        """
        ...

    def get_class_names(self) -> list[str] | None:
        """Return class names if the module defines them (e.g. YOLO model classes)."""
        return None


class BaseModificationModule(ABC):
    """ABC for image modification modules (LaMa, ComfyUI, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the dropdown."""
        ...

    @abstractmethod
    def get_settings_widget(self) -> QWidget:
        """Return a QWidget with module-specific settings."""
        ...

    def prepare(self):
        """Capture widget state before worker thread starts.

        Called from the main/GUI thread. Override to read widget values
        into plain attributes so run() never touches Qt widgets.
        """

    def wants_mask(self) -> bool:
        """Whether run() consumes the mask. Queried after prepare().

        Modules that always need one (e.g. LaMa) keep the default. Modules
        whose mask use depends on configuration — such as a ComfyUI workflow
        with or without an ``LTD_Input_Mask`` node — override this. When it
        returns False the tab passes ``mask_path=None`` and runs images that
        have no mask at all.
        """
        return True

    @abstractmethod
    def run(self, image_path: Path, mask_path: Path | None, **kwargs) -> Path:
        """Run modification on a single image.

        Args:
            image_path: Path to the original image
            mask_path: Path to the binary mask (white=area to modify), or
                None when the module reported ``wants_mask() == False``

        Returns:
            Path to the modified image
        """
        ...
