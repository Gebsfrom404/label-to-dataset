# Duplicate Detection

Powers the **Manage Duplicates** tab. Pure OpenCV + numpy — no extra
dependency, no model weights.

| File | Role |
|------|------|
| `ltd/utils/duplicate_utils.py` | Algorithms, dataclasses, tolerance→threshold mapping, grouping, `SearchCache` |
| `ltd/workers/duplicate_worker.py` | `DuplicateWorker` — scan → signatures → compare → group |
| `ltd/widgets/duplicate_image_list.py` | `DuplicateListModel` (groups + delete marks), filter proxy, list widget |
| `ltd/tabs/duplicates_tab.py` | Tab, `SourceListWidget`, actions, deletion |

## Algorithms

Selected by the `ALGO_*` constants; the combo box shows `ALGORITHM_LABELS`
and maps back through `ALGORITHM_BY_LABEL`.

| Key | What it does | Finds | Misses |
|-----|--------------|-------|--------|
| `phash` | 256-bit DCT hash of a 64×64 grayscale copy, Hamming distance | rescales, re-encodes, mild colour edits | crops, rotations |
| `orb` | ≤500 ORB keypoints → RANSAC homography → **aligned pixel comparison** | crops, rotations, reframes; separates a copy from an edited variant | nothing structural, but slow |
| `hybrid` | pHash shortlist (≤96 bits) verified by ORB | what pHash shortlists, minus false positives | crops and rotations — they never reach the shortlist |

Both families emit a **0.0–1.0 score** so one number works in the UI: pHash
uses `1 - distance / HASH_BITS`, ORB one minus the share of the aligned frame
that disagrees. The list prints a decimal above 99%,
because at 256 bits every real match lands in the last few percent and whole
percentages would render a merely-close pair as "100%".

### Why 256 bits and not the usual 64

Measured on a real 397-image folder (78606 pairs):

| | 64-bit | 256-bit |
|---|---|---|
| pairs at distance 0 | 5 — **all of them different images** | 0 |
| worst genuine duplicate (q20 JPEG, ⅛ size, blur, watermark) | 0 | 4 |
| a variant differing by an added speech bubble | **0** | 20 |
| pairs accepted at tolerance 100 | 5293 | 120 |

An 8×8 hash keeps only the 64 coarsest DCT coefficients, and a change over
~15% of the frame moves none of them — so two visibly different images hashed
identically and the UI reported "100%" at tolerance 0. 16×16 leaves a 5×
margin between the worst genuine duplicate (4 bits) and that variant pair
(20 bits). 32×32 was also tried and starts picking up compression noise.

`_PHASH_MAX_BITS = 64` (25% of the hash) is the tolerance-100 ceiling.
Unrelated images sit at ~50% by construction, so a quarter of the bits is
already generous.

### Cost

pHash is one packed 8-byte row per image and `hamming_distances()` compares a
row against the whole rest of the array at once — an all-pairs sweep over
thousands of images is fine. ORB has to *match* each pair (`knnMatch` +
RANSAC), so it is genuinely quadratic in work, not just in comparisons. The
tab warns before an all-pairs ORB run; checking a source as original reduces
it to `originals × others`.

ORB images are downscaled to `ORB_MAX_DIM = 640` on the longest side before
detection, and `ImageSignature.width/height` still records the *original*
resolution (the list and "select all but biggest" depend on that).

### Why the descriptor score is not keypoint agreement

The obvious ORB score — the share of keypoints that match and survive RANSAC
— ranks duplicates *below* edited copies, so no threshold can separate them.
Measured on real files against one original:

| pair | keypoint share | aligned disagreement |
|---|---|---|
| edited copy (speech bubble added) — **different** | **0.83** | 3.9% |
| other artwork from the same post — different | 0.15 | 57.5% |
| half-size q70 copy — duplicate | 0.95 | 0.0% |
| 80% centre crop — duplicate | **0.52** | 0.0% |
| rotated 8° — duplicate | **0.72** | 0.0% |

A crop drops 20% of the frame, so a fifth of the keypoints have no partner
and a perfect duplicate scores 0.52; a local edit leaves 83% of keypoints
untouched and scores 0.83. Ordering is inverted.

The fix is to treat the homography as *alignment*, not as the answer: warp
one image onto the other and measure how much of the overlap actually
disagrees. All three duplicates then score 1.00 and the edited copy 0.96.

Pooling into a `_GRID`×`_GRID` grid matters as much as the alignment. A plain
mean difference does not work — an 8° rotation scored a *higher* mean
difference (3.04) than the genuinely edited copy (2.38), because resampling
raises every pixel slightly while an edit ruins a small region completely.
Counting whole cells past `_CELL_DIFF` ignores the diffuse kind and catches
the concentrated kind. Exposure is equalised over the overlap first, so a
brightness tweak is not a difference.

**Cost:** the alignment stage is descriptors only and touches no files, so it
can run over every pair (~0.6 ms). Only a pair that actually aligns reaches
the pixel stage — measured at 2.0% of pairs on a real folder — which is why
reading images there is affordable.

## Caching — what a re-run actually redoes

`SearchCache` lives on the tab and is handed to every worker, so it outlives
each search. Two layers:

**Signatures**, keyed by path and validated against the file's
`(mtime, size)` stamp. A file is read again only when it is new, edited, or
when the algorithm needs a signature type that was never computed for it
(switching pHash → ORB). `is_original` comes from the source list rather than
the file, so the worker re-stamps it on every run instead of trusting the
cached value. ORB descriptors are big (~20 KB/image), so they are held under
`ORB_BUDGET_BYTES` (256 MB) and dropped oldest-first — the signature row
survives, only the descriptor payload goes, and it is recomputed on demand.

**Pairs**: `PairScore` keeps the raw measurements (`hash_distance`,
`orb_score`) rather than a match verdict, tagged with the algorithm, a
`fingerprint_entries()` hash of the file set, and the tolerance they were
computed at. Because every threshold is *monotonic* in tolerance, the set
accepted at `t` is a subset of the set accepted at any `T >= t` — so a re-run
at an equal or tighter tolerance calls `build_groups()` on the cached pairs
and skips the signature and comparison phases outright. This is why
`build_groups()` and the threshold predicates are pure functions in
`duplicate_utils.py` rather than worker methods.

Measured on a 6-image folder (`match` = descriptor comparisons):

| Re-run | Images read | Comparisons | Note |
|--------|-------------|-------------|------|
| tolerance tightened | 0 | 0 | pairs re-thresholded |
| tolerance loosened | 0 | re-run | signatures reused |
| algorithm pHash → ORB | all (ORB only) | re-run | pHash rows kept |
| algorithm ORB → pHash | 0 | re-run | both signature types now cached |
| one file touched | 1 | re-run | stamp mismatch |

Whenever you change how a threshold maps from tolerance, **check that it stays
monotonic** — a non-monotonic mapping would silently return stale results on
the re-filter path. The equivalence property to preserve: a warm re-run must
produce exactly what a cold run produces.

## Tolerance

One 0–100 slider maps to a per-algorithm threshold, so the number means
"looser", not a unit:

- `phash_max_distance()` → `0..25` bits of the 64 may differ.
- `orb_min_score()` → keypoint agreement from `0.45` (strict) down to `0.05`.
- `hybrid_shortlist_distance()` → never tighter than 24 bits, so the
  shortlist stays generous and ORB does the rejecting.

`threshold_description()` renders the current setting as one line under the
slider — keep it in sync when the mapping changes.

## Originals

A source checked as *original* changes three things at once:

1. Only original-vs-other pairs are compared (`_iter_pairs`).
2. Each duplicate keeps its **best** original match, so an image never lands
   in two groups (`_build_original_groups`).
3. `DuplicateListModel.set_marked()` refuses to mark an original, so no
   button — including *Select All* — can ever delete the reference copy.

With no original checked, matches are unioned into groups by
`group_pairs()` (union-find) and every member is deletable.

## Marks vs. selection

Rows carry a `marked` flag that is **independent of the Qt selection**, so
navigating never loses it. Marked rows draw red (`MARKED_COLOR`), originals
blue (`ORIGINAL_COLOR`), and alternating groups get a background tint so the
group boundaries are visible.

`mark_all_but('biggest'|'newest')` picks the keeper across the whole group,
originals included — so if the original is already the biggest, its group
loses every copy; if it is not, the group keeps two files (the keeper and the
protected original). That is intended.

## Deletion

`Delete Selected` moves marked files to the recycle bin via
`QFile.moveToTrash()`, optionally taking the sibling `.txt` caption and
`-masklabel.png` mask (`duplicates/delete_sidecars`). Afterwards
`prune_lone_groups()` drops groups that no longer hold at least two images.

Masks are skipped during scanning (`MASK_SUFFIX`), so a mask is never listed
as a duplicate of the image it belongs to.

## Gotchas

- **Overlapping / nested sources** are de-duplicated by normalised absolute
  path during the scan; if the same file arrives through both a plain and an
  original source, original wins.
- `SourceListWidget` persists to `duplicates/sources` — QSettings collapses a
  one-element string list back to a bare string, hence the `isinstance` dance
  in `_restore()`.
- `DuplicatesTab.shutdown()` is called from `MainWindow.closeEvent`; a search
  can run for minutes and a live QThread outliving its parent widget crashes
  on exit. It also clears the cache, releasing the ORB descriptors.
- The scan phase always runs — it is what produces the fingerprint, and it is
  how a file added since the last search gets noticed. Only the phases after
  it are cacheable.
- `DescriptorComparer` scores 0 when it cannot align a pair at all — no
  homography, too few RANSAC inliers, or under `_MIN_COVERAGE` overlap. A
  small patch aligns perfectly against anything it was cut from, so thin
  evidence must not read as a perfect match.
- The comparer holds a bounded LRU of grey images at detection scale.
  Keypoints are stored in that downscaled frame, so the warp must use the
  same scale — `load_orb_grey()` is the single place that rule lives.
- Changing `HASH_SIZE` invalidates nothing on disk (the cache is in-memory
  only), but it does change what every tolerance means — re-measure the
  ceiling against a real folder rather than scaling the old constant.
