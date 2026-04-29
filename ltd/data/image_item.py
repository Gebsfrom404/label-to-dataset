from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ltd.data.label_data import Label

if TYPE_CHECKING:
    from ltd.data.gen_metadata import GenMetadata


@dataclass
class ImageItem:
    path: Path
    width: int = 0
    height: int = 0
    labels: list[Label] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    mask_path: Optional[Path] = None
    modified_path: Optional[Path] = None
    original_path: Optional[Path] = None
    relative_path: str = ''
    metadata: Optional['GenMetadata'] = None
    _thumbnail: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def display_name(self) -> str:
        """Filename with subfolder prefix when loaded recursively."""
        return self.relative_path if self.relative_path else self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def display_path(self) -> Path:
        """Return modified path if available, otherwise original."""
        return self.modified_path if self.modified_path else self.path

    @property
    def caption_path(self) -> Path:
        base = self.original_path if self.original_path else self.path
        return base.with_suffix('.txt')

    def load_tags_from_file(self, separator: str = ', '):
        """Load tags from the caption .txt file if it exists."""
        if self.caption_path.exists():
            text = self.caption_path.read_text(encoding='utf-8').strip()
            if text:
                self.tags = [t.strip() for t in text.split(separator.strip() or separator)
                             if t.strip()]
            else:
                self.tags = []

    def save_tags_to_file(self, separator: str = ', '):
        """Save tags to the caption .txt file."""
        self.caption_path.write_text(
            separator.join(self.tags), encoding='utf-8')
