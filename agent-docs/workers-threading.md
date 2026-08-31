# Workers & Threading

## Base Pattern (`ltd/workers/base_worker.py`)

```python
class BaseWorker(QThread):
    progress = Signal(int, int)   # current, total
    status = Signal(str)
    error = Signal(str)
    finished_work = Signal()
```

Subclass and override `do_work()`. The `run()` method wraps it with try/except/finally, emitting errors and cleaning up memory.

## Cancel Pattern

```python
self._is_cancelled = False  # set in run() before do_work()

# In do_work() loop:
for i, item in enumerate(items):
    if self.is_cancelled:
        break
```

Call `worker.cancel()` from GUI thread.

## Memory Cleanup

`_cleanup_memory()` runs after every `do_work()` (in `run()`'s `finally`) **and after every single item** in each worker's loop (`DetectionWorker`, `ModificationWorker`, `CaptionWorker`, `GenerationWorker`):
```python
gc.collect()
torch.cuda.empty_cache()  # if torch available
```
Originally gated behind a `BATCH_SIZE = 100` counter (cleanup only every 100th image). Removed: `torch.cuda.empty_cache()` only returns *reserved* memory to the driver — it doesn't free anything still referenced — so calling it is always safe, and the cost (a few ms) is negligible next to per-image inference time. The batching mattered for cheap, low-VRAM modules, but a heavy per-image model (e.g. SAM3, whose peak VRAM scales with image resolution × surviving-detection count × text-prompt count — see ml-models.md) would otherwise ratchet PyTorch's caching allocator up to the highest peak seen anywhere in the run and never give it back until the batch fully finished — for a folder under 100 images, that's effectively "runs the whole batch, cleans up once at the end," which is what produced the "memory keeps growing during Auto-Label" symptom (see gotchas-decisions.md).

## Workers

| Worker | File | Purpose |
|--------|------|---------|
| DetectionWorker | `detection_worker.py` | Batch auto-detection via plugin module |
| ModificationWorker | `modification_worker.py` | Batch image modification via plugin |
| CaptionWorker | `caption_worker.py` | Batch captioning, emits `caption_complete(index, tags)` |
| TrainingWorker | `training_worker.py` | YOLO training via ultralytics |
| DuplicateWorker | `duplicate_worker.py` | Duplicate search: scan → signatures → pairwise compare → group. Emits `groups_found(list)`, `scanned(int)`, `skipped(int)` |
| _ScriptWorker | `extras_tab.py` (inline) | Runs extras script `run()` in thread with progress callback |

## Module Unload

ModificationWorker calls `module.unload()` (if it exists) in a `finally` block after processing. This is optional — not part of the ABC — but allows modules to free GPU memory after batch work.

## Extras _ScriptWorker

The Extras tab has its own lightweight worker (`_ScriptWorker`) that doesn't extend `BaseWorker`. It takes a `run_func` and `params` dict, calls `run_func(params, progress_callback)` in a thread. The progress callback signature: `(current: int, total: int, message: str = '')`.

## Thread Safety Rules

1. **Never touch Qt widgets from worker threads** — use module `prepare()` to capture state beforehand
2. **Emit signals for all GUI updates** — progress, status, results
3. **Check `is_cancelled` between items** — don't block on long ops without checking
4. **Cleanup every item** — call `_cleanup_memory()` after each item, not just at the end of the batch, to prevent GPU memory ratcheting up over a long run
