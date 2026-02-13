# Workers & Threading

## Base Pattern (`ltd/workers/base_worker.py`)

```python
class BaseWorker(QThread):
    progress = Signal(int, int)   # current, total
    status = Signal(str)
    error = Signal(str)
    finished_work = Signal()
    BATCH_SIZE = 100
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

`_cleanup_memory()` runs after every `do_work()` and every `BATCH_SIZE` items:
```python
gc.collect()
torch.cuda.empty_cache()  # if torch available
```

## Workers

| Worker | File | Purpose |
|--------|------|---------|
| DetectionWorker | `detection_worker.py` | Batch auto-detection via plugin module |
| ModificationWorker | `modification_worker.py` | Batch image modification via plugin |
| CaptionWorker | `caption_worker.py` | Batch captioning, emits `caption_complete(index, tags)` |
| TrainingWorker | `training_worker.py` | YOLO training via ultralytics |

## Thread Safety Rules

1. **Never touch Qt widgets from worker threads** — use module `prepare()` to capture state beforehand
2. **Emit signals for all GUI updates** — progress, status, results
3. **Check `is_cancelled` between items** — don't block on long ops without checking
4. **Batch cleanup** — call `_cleanup_memory()` every BATCH_SIZE items to prevent GPU OOM
