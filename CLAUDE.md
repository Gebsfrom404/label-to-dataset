# CLAUDE.md
## Agent Docs
Read these on-demand — only when working on the relevant area.

## Code Intelligence

Prefer LSP over Grep/Read for code navigation — it's faster, precise, and avoids reading entire files:
- `workspaceSymbol` to find where something is defined
- `findReferences` to see all usages across the codebase
- `goToDefinition` / `goToImplementation` to jump to source
- `hover` for type info without reading the file

Use Grep only when LSP isn't available or for text/pattern searches (comments, strings, config).

After writing or editing code, check LSP diagnostics and fix errors before proceeding.

## Maintenance rule  
When you change code that is covered by an agent doc, update that doc in the same session. This includes: new patterns, changed APIs, renamed files, new gotchas, fixed bugs, added components, or removed features. If you introduce something entirely new that doesn't fit existing docs, create a new `agent-docs/*.md` and add it to the table below.

| Doc | When to read |
|-----|-------------|
| [architecture.md](agent-docs/architecture.md) | App structure, tabs, data flow, key components |
| [plugin-system.md](agent-docs/plugin-system.md) | Adding/modifying detection or modification modules |
| [canvas-and-drawing.md](agent-docs/canvas-and-drawing.md) | Canvas widget, drawing tools, mask buffer, QImage↔numpy |
| [workers-threading.md](agent-docs/workers-threading.md) | Background workers, cancel pattern, memory cleanup |
| [data-formats-storage.md](agent-docs/data-formats-storage.md) | YOLO labels, masks, captions, temp dirs, undo, settings |
| [comfyui-integration.md](agent-docs/comfyui-integration.md) | ComfyUI client, workflows, LTD_* node convention |
| [ml-models.md](agent-docs/ml-models.md) | YOLO, LaMa, WD tagger (timm/safetensors preprocessing) |
| [duplicate-detection.md](agent-docs/duplicate-detection.md) | Duplicate search: pHash / ORB, tolerance, originals, delete marks |
| [gotchas-decisions.md](agent-docs/gotchas-decisions.md) | Past bugs and fixes — read before debugging |