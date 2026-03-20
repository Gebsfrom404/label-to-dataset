"""Tag dictionary loaded from CSV/parquet/txt for autocomplete and category colors."""
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

    def load_parquet(self, path: Path):
        """Load tags from a parquet file with columns: name, category, post_count, aliases."""
        if not path.exists():
            return
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return

        table = pq.read_table(path)
        columns = table.column_names
        if 'name' not in columns:
            return

        names = table.column('name').to_pylist()
        categories = table.column('category').to_pylist() if 'category' in columns else [0] * len(names)
        post_counts = table.column('post_count').to_pylist() if 'post_count' in columns else [0] * len(names)
        alias_col = table.column('aliases').to_pylist() if 'aliases' in columns else [''] * len(names)

        for name, cat, count, alias_str in zip(names, categories, post_counts, alias_col):
            name = str(name).strip()
            if not name:
                continue
            display_name = name.replace('_', ' ')
            aliases = []
            if alias_str and str(alias_str).strip():
                aliases = [a.strip().replace('_', ' ')
                           for a in str(alias_str).split(',') if a.strip()]
            tag = DictTag(
                name=name,
                display_name=display_name,
                category=int(cat) if cat else 0,
                post_count=int(count) if count else 0,
                aliases=aliases,
            )
            self._tags.append(tag)
            key = display_name.lower()
            if key not in self._by_display:
                self._by_display[key] = tag
                self._by_name[name.lower()] = tag

        self._loaded = bool(self._tags)

    def load_custom_txt(self, path: Path):
        """Load user-defined tags from a newline-separated text file.

        Tags loaded here get category 0 (General) and post_count 0.
        They won't override tags already loaded from parquet/csv.
        """
        if not path.exists():
            return

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                display_name = name.replace('_', ' ')
                key = display_name.lower()
                if key in self._by_display:
                    continue
                tag = DictTag(
                    name=name,
                    display_name=display_name,
                    category=0,
                    post_count=0,
                )
                self._tags.append(tag)
                self._by_display[key] = tag
                self._by_name[name.lower()] = tag

        self._loaded = bool(self._tags)

    def load_directory(self, directory: Path):
        """Load tags from the autocompletions directory.

        Loads parquet files first (higher priority), then custom_tags.txt.
        Custom tags won't override parquet tags.
        """
        if not directory.is_dir():
            return

        self._tags.clear()
        self._by_display.clear()
        self._by_name.clear()
        self._loaded = False

        for parquet_file in sorted(directory.glob('*.parquet')):
            self.load_parquet(parquet_file)

        custom_path = directory / 'custom_tags.txt'
        self.load_custom_txt(custom_path)

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

        # Sort: exact match first, then current image tags, dataset tags, rest
        def sort_key(t):
            dn = t.display_name.lower()
            exact = dn == q or t.name.lower() == q
            starts = dn.startswith(q) or t.name.lower().startswith(q.replace(' ', '_'))
            in_current = t.display_name in used or t.name in used
            in_session = t.display_name in session or t.name in session
            return (not exact, not starts, not in_current, not in_session, -t.post_count)
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
