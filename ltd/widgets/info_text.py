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


DUPLICATES_FILTER_HELP = """\
<b>Duplicate filter</b><br>
Space-separated terms are ANDed. Filters the search results only.<br>
<br>
<b>Terms</b>
<ul style="margin: 0 0 0 -16px;">
<li><i>plain text</i> &mdash; substring on the listed path</li>
<li><code>marked:yes</code> &nbsp; <code>marked:no</code> &mdash; marked for deletion</li>
<li><code>original:yes</code> &nbsp; <code>original:no</code> &mdash; from the original source</li>
<li><code>score:&gt;=95</code> &mdash; similarity in percent (<code>=</code> <code>!=</code> <code>&lt;</code> <code>&gt;</code> <code>&lt;=</code> <code>&gt;=</code>)</li>
<li><code>name:PATTERN</code> &mdash; filename match</li>
<li><code>path:PATTERN</code> &mdash; full path match</li>
<li><code>r:REGEX</code> &mdash; regex on the listed path</li>
<li><code>"quoted text"</code> &mdash; preserves spaces</li>
</ul>
<b>Combinators</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>and</code>, <code>or</code>, <code>not</code> / <code>-</code>, <code>(...)</code> grouping</li>
</ul>
<b>Examples</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>marked:yes</code></li>
<li><code>original:no score:&gt;=98</code></li>
<li><code>path:*\\batch_02\\* not marked:yes</code></li>
</ul>"""


DUPLICATES_SEARCH_HELP = """\
<b>Duplicate search</b><br>
<br>
<b>Sources</b>
<ul style="margin: 0 0 0 -16px;">
<li>Every source folder is scanned recursively.</li>
<li>Check <i>original</i> on one source to make it the reference: every other
image is compared against it and originals are never deleted.</li>
<li>With no original checked, all images are compared against each other.</li>
</ul>
<b>Algorithms</b>
<ul style="margin: 0 0 0 -16px;">
<li><b>Perceptual hash</b> &mdash; one 64-bit DCT hash per image, compared by
bit distance. Fast on thousands of images. Finds rescaled, re-encoded and
lightly edited copies; <i>misses crops and rotations</i>.</li>
<li><b>Local feature descriptors</b> &mdash; ORB keypoints matched with a ratio
test and verified with a RANSAC homography. Finds cropped, rotated and
reframed copies. Every pair is real work, so runtime grows with the square of
the image count.</li>
<li><b>Hash, then descriptor verify</b> &mdash; the hash builds a loose
shortlist that descriptors confirm. Much faster than a full descriptor pass
and fewer false positives than the hash alone, but it inherits the hash's
blind spot for rotations.</li>
</ul>
<b>Tolerance</b>
<ul style="margin: 0 0 0 -16px;">
<li><code>0</code> &mdash; near-identical copies only.</li>
<li><code>100</code> &mdash; loose; expect unrelated images to be grouped.</li>
<li>The line under the slider shows what the current value means for the
selected algorithm.</li>
</ul>"""


DUPLICATES_ACTIONS_HELP = """\
<b>Marking and deleting</b><br>
<br>
Marking is separate from the list selection: marked rows are drawn in
<span style="color:#e05a4e;">red</span> and survive navigation. Images from an
<span style="color:#4aa3df;">original</span> source are never marked.<br>
<br>
<b>Buttons</b>
<ul style="margin: 0 0 0 -16px;">
<li><b>Select All</b> &mdash; marks every duplicate found.</li>
<li><b>Select All But Biggest</b> &mdash; per group, keeps the largest by pixel
count (file size breaks ties) and marks the rest.</li>
<li><b>Select All But Newest</b> &mdash; per group, keeps the most recently
modified file and marks the rest.</li>
<li><b>Delete Selected</b> &mdash; moves every marked file to the recycle bin,
optionally with its <code>.txt</code> caption and
<code>-masklabel.png</code> mask.</li>
</ul>
Groups left with a single image after a deletion drop off the list."""


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
    ('Ctrl+Delete', 'Delete current image (from disk)'),
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
CAPTION_SHORTCUTS_TAGS_LIST = _section('Tag lists (Image / All)', [
    ('Up / Down', 'Move (rolls over to next image at edges)'),
    ('PgUp / PgDown', 'Previous / next image'),
    ('Delete / Backspace', 'Delete selected tag(s)'),
    ('Ctrl+C', 'Copy selected tag(s)'),
    ('Right-click', 'Filter images by selected tag(s) — ANDed'),
    ('Double-click', 'Inline edit (image tags) / rename (all tags)'),
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
CAPTION_SHORTCUTS_FAST_INSERT = _section('Fast insertion', [
    ('1 2 3 … 9 0', 'Add slot tag to selected (or current) image(s)'),
    ('Enable', 'Toggle the number-key shortcuts on/off'),
    ('Mode', 'Append (", tag") or prepend ("tag, ")'),
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


DUPLICATES_SHORTCUTS_LIST = _section('Duplicate list', [
    ('Up / Down / W / S / PgUp / PgDown', 'Navigate'),
    ('Space', 'Mark / unmark selection for deletion'),
    ('Ctrl+A', 'Select all visible rows'),
    ('Ctrl+O', 'Open in default app'),
    ('Right-click', 'Mark, open, show in explorer'),
])
DUPLICATES_SHORTCUTS_SOURCES = _section('Sources', [
    ('Checkbox', 'Mark the source as original (only one at a time)'),
    ('Double-click', 'Open the folder in explorer'),
    ('Del', 'Remove the selected source'),
])
DUPLICATES_SHORTCUTS_CANVAS = _section('Preview', [
    ('Ctrl+Wheel', 'Zoom'),
    ('Ctrl+0', 'Fit to view'),
    ('Drag (when zoomed)', 'Pan'),
    ('PgUp / PgDown', 'Previous / next duplicate'),
])
