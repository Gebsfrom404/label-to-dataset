"""Centralized help text shown by InfoButton tooltips.

Filter-grammar help is per filter widget; shortcut help is per tab and is
broken into sections so a focus-aware info button can surface only the
section relevant to the currently focused area.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Filter grammar help (shown next to a filter QLineEdit)
# ---------------------------------------------------------------------------

LABEL_FILTER_HELP = """\
<b>Image filter</b><br>
Space-separated terms are ANDed.<br>
<br>
<b>Terms</b>
<ul style="margin: 0 0 0 -16px;">
<li><i>plain text</i> &mdash; substring match on filename</li>
<li><code>class:NAME</code> &mdash; has label of given class (glob: <code>*</code> <code>?</code>)</li>
<li><code>name:PATTERN</code> &mdash; filename match (substring or glob)</li>
<li><code>labels:&gt;N</code> &mdash; label count (<code>=</code> <code>!=</code> <code>&lt;</code> <code>&gt;</code> <code>&lt;=</code> <code>&gt;=</code>)</li>
<li><code>"quoted text"</code> &mdash; preserves spaces in values</li>
</ul>
<b>Combinators</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>and</code> &mdash; implicit between terms (optional)</li>
<li><code>or</code> &mdash; separate OR groups (any group matches)</li>
<li><code>not</code> or <code>-</code> &mdash; negate the next term</li>
</ul>
<b>Examples</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>class:dog labels:&gt;2</code></li>
<li><code>not class:cat or labels:0</code></li>
<li><code>class:"long name" name:*.png</code></li>
</ul>"""


CAPTION_FILTER_HELP = """\
<b>Image filter</b><br>
Space-separated terms are ANDed.<br>
<br>
<b>Terms</b>
<ul style="margin: 0 0 0 -16px;">
<li><i>plain text</i> &mdash; substring match on filename or any tag</li>
<li><code>tag:NAME</code> &mdash; exact tag match (glob: <code>*</code> <code>?</code>)</li>
<li><code>r:REGEX</code> &mdash; regex (filename or tags)</li>
<li><code>name:PATTERN</code> &mdash; filename match</li>
<li><code>path:PATTERN</code> &mdash; full path match</li>
<li><code>tags:&gt;N</code> &mdash; tag count (<code>=</code> <code>!=</code> <code>&lt;</code> <code>&gt;</code> <code>&lt;=</code> <code>&gt;=</code>)</li>
<li><code>"quoted text"</code> &mdash; preserves spaces</li>
</ul>
<b>Combinators</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>and</code>, <code>or</code>, <code>not</code> / <code>-</code>, <code>(...)</code> grouping</li>
</ul>
<b>Examples</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>tag:1girl and not (tag:1boy or tag:3boys)</code></li>
<li><code>sun and r:"shadow$"</code></li>
<li><code>tag:long_hair tags:&gt;=10</code></li>
</ul>"""


GEN_IMAGES_FILTER_HELP = """\
<b>Image filter</b><br>
Space-separated terms are ANDed. Searches filename, prompt, and raw metadata.<br>
<br>
<b>Terms</b>
<ul style="margin: 0 0 0 -16px;">
<li><i>plain text</i> &mdash; substring on name, tags, prompts, raw metadata</li>
<li><code>tag:NAME</code> &mdash; exact tag match (glob: <code>*</code> <code>?</code>)</li>
<li><code>r:REGEX</code> &mdash; regex (name, tags, raw metadata)</li>
<li><code>name:PATTERN</code> &mdash; filename match</li>
<li><code>path:PATTERN</code> &mdash; full path match</li>
<li><code>tags:&gt;N</code> &mdash; tag count comparison</li>
<li><code>WxH</code> e.g. <code>1024x1024</code> &mdash; exact resolution</li>
<li><code>size:WxH</code> &mdash; same as bare WxH</li>
<li><code>w:&gt;=N</code> &nbsp; <code>h:&lt;=N</code> &mdash; width/height comparison</li>
<li><code>meta:TEXT</code> &mdash; substring in raw metadata</li>
<li><code>format:forge</code> &nbsp; <code>format:comfyui</code> &mdash; metadata source</li>
<li><code>"quoted text"</code> &mdash; preserves spaces</li>
</ul>
<b>Combinators</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>and</code>, <code>or</code>, <code>not</code> / <code>-</code>, <code>(...)</code> grouping</li>
</ul>
<b>Examples</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>format:forge 1024x1024</code></li>
<li><code>tag:1girl w:&gt;=1024 not tag:nsfw</code></li>
<li><code>meta:"Euler a" or meta:dpmpp</code></li>
</ul>"""


# ---------------------------------------------------------------------------
# Tab shortcut help. Sections may have a "scope" key — the section is shown
# only when the focused widget is inside (or equals) that scope. Sections
# without a scope are global to the tab.
# ---------------------------------------------------------------------------


def _section(title: str, rows: list[tuple[str, str]]) -> str:
    body = '\n'.join(
        f'<tr><td style="padding-right: 12px;"><code>{k}</code></td>'
        f'<td>{d}</td></tr>'
        for k, d in rows)
    return (f'<b>{title}</b>'
            f'<table cellspacing="0" cellpadding="1">{body}</table>')


# Each tab provides an OrderedDict-like list of (scope_key, html). The
# scope_key is a string the tab can map to a widget at lookup time. The
# DynamicInfoButton in the tab calls a provider that walks focus and picks
# matching sections.

LABEL_SHORTCUTS_GLOBAL = _section('Tools', [
    ('M', 'Hand (pan / select)'),
    ('P', 'Pointer'),
    ('R', 'Bounding box'),
    ('V', 'Polygon'),
    ('B', 'Marker'),
]) + _section('Mode', [
    ('1', 'New label'),
    ('2', 'Combine into existing'),
    ('3', 'Erase'),
]) + _section('Navigation', [
    ('A / PgUp', 'Previous image'),
    ('D / PgDown', 'Next image'),
    ('W', 'Next class'),
    ('S', 'Previous class'),
]) + _section('Edit', [
    ('Delete', 'Delete selected label'),
    ('Ctrl+Z', 'Undo'),
    ('Ctrl+C', 'Copy labels'),
    ('Ctrl+V', 'Paste labels'),
    ('Space (hold)', 'Temporary pan'),
])


MODIFY_SHORTCUTS_GLOBAL = _section('Tools', [
    ('M', 'Hand'),
    ('R', 'Bounding box'),
    ('V', 'Polygon'),
    ('B', 'Marker'),
    ('C', 'Crop'),
]) + _section('Draw mode', [
    ('1', 'Draw / new'),
    ('2', 'Erase'),
]) + _section('Apply / cancel', [
    ('Enter', 'Apply current overlay (crop / split)'),
    ('Esc', 'Cancel current action'),
]) + _section('Edit', [
    ('Ctrl+Z', 'Undo last mask edit'),
    ('Ctrl+Delete', 'Delete current image'),
    ('Ctrl+S', 'Save modified images'),
    ('Ctrl+Shift+S', 'Save in place'),
    ('Space (hold)', 'Temporary pan'),
])


CAPTION_SHORTCUTS_GLOBAL = _section('Files', [
    ('Ctrl+S', 'Save tags to disk'),
])
CAPTION_SHORTCUTS_TAGS_LIST = _section('Image tags list', [
    ('Up / Down', 'Move (rolls over to next image at edges)'),
    ('PgUp / PgDown', 'Previous / next image'),
    ('Delete / Backspace', 'Delete selected tag(s)'),
    ('Double-click', 'Inline edit'),
    ('Drag', 'Reorder'),
])
CAPTION_SHORTCUTS_INPUT = _section('Tag input', [
    ('Enter / Tab', 'Confirm completion / add tag'),
    ('Up / Down', 'Navigate completer'),
    ('Esc', 'Cancel completer'),
    ('PgUp / PgDown', 'Previous / next image'),
    ('Ctrl+Z / Ctrl+Y', 'Undo / redo'),
])
CAPTION_SHORTCUTS_BULK = _section('Bulk operations', [
    ('Ctrl+R / Ctrl+H', 'Find &amp; replace'),
    ('Ctrl+B', 'Batch reorder'),
    ('Ctrl+D', 'Remove duplicates'),
    ('Ctrl+E', 'Remove empty'),
])
CAPTION_SHORTCUTS_NAV = _section('Image navigation', [
    ('Left / Right', 'Previous / next image'),
    ('PgUp / PgDown', 'Previous / next image'),
])


GEN_SHORTCUTS_LIST = _section('Image list', [
    ('Up / Down / W / S / PgUp / PgDown', 'Navigate'),
    ('Ctrl+A', 'Select all visible images'),
    ('Ctrl+C', 'Copy prompt(s) of selection'),
    ('Ctrl+Shift+C', 'Copy image file(s) to folder…'),
    ('Ctrl+M', 'Move image file(s) to folder…'),
    ('Ctrl+D', 'Delete image(s) (recycle bin)'),
    ('Ctrl+O', 'Open in default app'),
])
GEN_SHORTCUTS_TAGS = _section('Tag lists (Image / All)', [
    ('Ctrl+C', 'Copy selected tag(s)'),
    ('Right-click', 'Filter images by tag'),
])
GEN_SHORTCUTS_CANVAS = _section('Canvas', [
    ('Ctrl+Wheel', 'Zoom'),
    ('Ctrl+0', 'Fit to view'),
    ('Drag (when zoomed)', 'Pan'),
    ('PgUp / PgDown', 'Previous / next image'),
])
