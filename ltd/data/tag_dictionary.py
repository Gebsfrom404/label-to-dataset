"""Tag dictionary loaded from CSV for autocomplete and category colors."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QColor


@dataclass
class DictTag:
    name: str            # original form: "long_hair"
    display_name: str    # spaces: "long hair"
    category: int
    post_count: int
    aliases: list[str] = field(default_factory=list)  # display form (spaces)


class TagDictionary:
    """Loads a danbooru/e621-format CSV and provides search + color lookup."""

    CATEGORY_COLORS_DARK = {
        0: QColor('#9ECFFF'),   # General - light blue
        1: QColor('#CD5C5C'),   # Artist - indian red
        3: QColor('#EE82EE'),   # Copyright - violet
        4: QColor('#90EE90'),   # Character - light green
        5: QColor('#FFA500'),   # Meta - orange
    }
    CATEGORY_COLORS_LIGHT = {
        0: QColor('#1E90FF'),   # General - dodger blue
        1: QColor('#B22222'),   # Artist - firebrick
        3: QColor('#9932CC'),   # Copyright - dark orchid
        4: QColor('#006400'),   # Character - dark green
        5: QColor('#FF8C00'),   # Meta - dark orange
    }

    def __init__(self):
        self._tags: list[DictTag] = []
        self._by_display: dict[str, DictTag] = {}   # "long hair" -> DictTag
        self._by_name: dict[str, DictTag] = {}       # "long_hair" -> DictTag
        self._loaded = False

    def load_csv(self, path: Path):
        """Load tags from a danbooru-format CSV.

        Format: tag_name,category_id,post_count,"alias1,alias2"
        """
        self._tags.clear()
        self._by_display.clear()
        self._by_name.clear()
        self._loaded = False

        if not path.exists():
            return

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                name = row[0].strip()
                try:
                    category = int(row[1])
                except (ValueError, IndexError):
                    category = 0
                try:
                    post_count = int(row[2])
                except (ValueError, IndexError):
                    post_count = 0

                display_name = name.replace('_', ' ')
                aliases = []
                if len(row) > 3 and row[3].strip():
                    aliases = [a.strip().replace('_', ' ')
                               for a in row[3].split(',') if a.strip()]

                tag = DictTag(
                    name=name,
                    display_name=display_name,
                    category=category,
                    post_count=post_count,
                    aliases=aliases,
                )
                self._tags.append(tag)
                self._by_display[display_name.lower()] = tag
                self._by_name[name.lower()] = tag

        self._loaded = bool(self._tags)

    def search(self, query: str, limit: int = 50,
               used_tags: set[str] | None = None,
               session_tags: set[str] | None = None) -> list[DictTag]:
        """Search tags by substring match on name + aliases.

        Priority: current image tags > dataset tags > by post_count.
        """
        if not query or not self._tags:
            return []
        q = query.lower().replace('_', ' ')
        results: list[DictTag] = []
        used = used_tags or set()
        session = session_tags or set()

        for tag in self._tags:
            if len(results) >= limit * 3:
                break
            dn = tag.display_name.lower()
            if q in dn:
                results.append(tag)
                continue
            if any(q in alias.lower() for alias in tag.aliases):
                results.append(tag)

        # Sort: current image tags first, then dataset tags, then rest
        def sort_key(t):
            in_current = t.display_name in used or t.name in used
            in_session = t.display_name in session or t.name in session
            return (not in_current, not in_session, -t.post_count)
        results.sort(key=sort_key)
        return results[:limit]

    def get_tag(self, display_name: str) -> DictTag | None:
        """Look up a tag by its display name (spaces)."""
        key = display_name.lower()
        tag = self._by_display.get(key)
        if tag:
            return tag
        return self._by_name.get(key.replace(' ', '_'))

    def get_color(self, tag_name: str, dark: bool = True) -> QColor | None:
        """Get the category color for a tag, or None if not in dictionary."""
        tag = self.get_tag(tag_name)
        if tag is None:
            return None
        colors = self.CATEGORY_COLORS_DARK if dark else self.CATEGORY_COLORS_LIGHT
        return colors.get(tag.category)

    def is_loaded(self) -> bool:
        return self._loaded
