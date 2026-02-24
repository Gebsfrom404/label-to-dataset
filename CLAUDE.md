# CLAUDE.md
## Overview

PySide6 desktop app: **Label → Modify → Caption → Train YOLO** pipeline (4 tabs) plus **Extras**.
## Agent Docs
Read these on-demand — only when working on the relevant area.

**Maintenance rule**: When you change code that is covered by an agent doc, update that doc in the same session. This includes: new patterns, changed APIs, renamed files, new gotchas, fixed bugs, added components, or removed features. If you introduce something entirely new that doesn't fit existing docs, create a new `agent-docs/*.md` and add it to the table below.

| Doc | When to read |
|-----|-------------|
| [architecture.md](agent-docs/architecture.md) | App structure, tabs, data flow, key components |
| [plugin-system.md](agent-docs/plugin-system.md) | Adding/modifying detection or modification modules |
| [canvas-and-drawing.md](agent-docs/canvas-and-drawing.md) | Canvas widget, drawing tools, mask buffer, QImage↔numpy |
| [workers-threading.md](agent-docs/workers-threading.md) | Background workers, cancel pattern, memory cleanup |
| [data-formats-storage.md](agent-docs/data-formats-storage.md) | YOLO labels, masks, captions, temp dirs, undo, settings |
| [comfyui-integration.md](agent-docs/comfyui-integration.md) | ComfyUI client, workflows, LTD_* node convention |
| [ml-models.md](agent-docs/ml-models.md) | YOLO, LaMa, WD tagger (timm/safetensors preprocessing) |
| [gotchas-decisions.md](agent-docs/gotchas-decisions.md) | Past bugs and fixes — read before debugging |
