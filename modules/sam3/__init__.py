"""Shared SAM3 (Segment Anything Model 3) engine.

Not a plugin package — ``modules/detection`` and ``modules/modifications``
are the only paths ``discover_modules()`` scans (see
``modules/__init__.py``), so this package is never picked up as a detection
or modification module itself. It's imported directly by
``modules/detection/sam3_detection.py`` and by the Label/Modify tabs' Magic
Wand tool.
"""
