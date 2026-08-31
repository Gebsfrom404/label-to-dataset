# Duplicate Detection

Powers the **Manage Duplicates** tab. Pure OpenCV + numpy — no extra
dependency, no model weights.

| File | Role |
|------|------|
| `ltd/utils/duplicate_utils.py` | Algorithms, dataclasses, tolerance→threshold mapping |
| `ltd/workers/duplicate_worker.py` | `DuplicateWorker` — scan → signatures → compare → group |
| `ltd/widgets/duplicate_image_list.py` | `DuplicateListModel` (groups + delete marks), filter proxy, list widget |
| `ltd/tabs/duplicates_tab.py` | Tab, `SourceListWidget`, actions, deletion |

## Algorithms

Selected by the `ALGO_*` constants; the combo box shows `ALGORITHM_LABELS`
and maps back through `ALGORITHM_BY_LABEL`.

| Key | What it does | Finds | Misses |
|-----|--------------|-------|--------|
| `phash` | 64-bit DCT hash of a 32×32 grayscale copy, Hamming distance | rescales, re-encodes, mild colour edits | crops, rotations |
| `orb` | ≤500 ORB keypoints, Lowe ratio test, RANSAC homography | crops, rotations, reframes | nothing structural, but slow |
| `hybrid` | pHash shortlist (≤24 bits) verified by ORB | what pHash shortlists, minus false positives | rotations — they never reach the shortlist |

Both families emit a **0.0–1.0 score** so one number works in the UI: pHash
uses `1 - distance / 64`, ORB the share of keypoints that survive matching
*and* geometric verification.

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
  on exit.
- `orb_similarity()` falls back to the plain ratio-test share when RANSAC
  cannot fit a homography (flat or heavily repetitive images), rather than
  scoring them 0.
