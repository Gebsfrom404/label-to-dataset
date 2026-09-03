# Dataset Clustering

`extras_scripts/dataset_clustering.py` — an Extras tab script that shrinks an
overrepresented dataset so a style LoRA sees a balanced spread of scenes
instead of 200 near-identical frames of one, without discarding the rare
material that keeps the LoRA flexible.

Self-contained: imports nothing from `ltd/`, matching the other extras
scripts. Deps are all already in `requirements.txt` (`torch`, `timm`,
`safetensors`, `Pillow`, `huggingface-hub`, plus **scipy**, which arrives
transitively via `ultralytics`).

## Pipeline

1. `_iter_images()` — recursive scan. `-masklabel.png` files are sidecars, not
   dataset images, and are excluded from the image list.
2. `_embed_all()` — 2816-d embedding per image from
   `animetimm/convnextv2_huge.dbv4-full` (see [ml-models.md](ml-models.md)).
   Disk-cached.
3. `_cluster()` — complete-linkage agglomerative clustering on cosine
   distance, cut at the **Cluster Tolerance** threshold.
4. Clusters below **Min Cluster Size** are outliers.
5. Cap = `round(median_cluster_size × cap_percent / 100)`. Clusters above it
   are trimmed by `_farthest_point_keep()`.
6. Moves are planned into `_Plan.build()`, then executed (or not, under
   **Dry Run**).

## Strategies

| Strategy | Behavior |
|----------|----------|
| `Cluster by folder` | Every kept image moves into `{output}/{repeats}_{name}/`. Outliers **and** over-cap trim go to the leftovers folder. Input subfolder structure is flattened. |
| `Drop outliers` | Only outliers move to leftovers. Everything else stays in place with its original subfolder structure. **The cap is not applied** — this is deliberate, not an oversight. |

## Balancing: trim the big, repeat the small

Both directions are used, so the *effective* image count per cluster converges:

- Big clusters are trimmed down to the cap.
- `repeats = clamp(round(median / kept_size), 1, max_repeats)` boosts small
  clusters, following the kohya `{repeats}_{name}` folder convention, so the
  output folder is drop-in trainable.

Measured on the 504-image `references/dataset_clustering` set at the defaults:
42 clusters, effective per-cluster counts land in a ~1.8× band instead of the
raw 38:3 spread.

## Why complete linkage

Measured on that same 504-image set:

| Method | Behavior |
|--------|----------|
| `average` | **Chains badly.** Between tolerance 0.55 and 0.60 the largest cluster jumped 152 → 339 of 504 images while the median cluster stayed at 5. No usable setting exists. |
| `complete` | **Chosen.** Bounds each cluster's diameter, so size grows gradually and the tolerance knob is actually tunable. |
| `ward` | Balances best (22 clusters, median 19, no singletons at t=2.0) but forces every image into a cluster, so it can never surface an outlier — fatal for the `Drop outliers` strategy. |

Cosine distances in this embedding space are wide (median pairwise 0.645), so
the tolerance scale is **not** the same as the Duplicates tab's. Usable range
is ~0.45–0.80; below ~0.45 almost everything is a singleton.

## Trimming keeps the spread, not the centre

`_farthest_point_keep()` is farthest-point sampling: start at the cluster
medoid, then repeatedly add the image farthest from everything already kept.
Near-duplicates are dropped first and the cluster's edges survive. Keeping the
images *closest* to the centroid would do the opposite — it collapses the
cluster to one look, which is exactly the failure mode this feature exists to
prevent.

## Cluster naming

Names come from the tagger's own tag head, read from the **same forward pass**
as the embedding (free). A tag scores by how much more common it is inside the
cluster than across the dataset (`local_share - global_share`), keeping only
tags present in ≥50% of cluster members; the top 3 form the name. Falls back to
the medoid image's filename stem. Produces names like
`1_blue_hair_wings_single_horn`.

## Sidecars, collisions, safety

- `.txt` captions and `-masklabel.png` masks move with their image.
- Flattening subfolders can collide filenames. `_unique_dest()` suffixes
  `_1`, `_2`, … and the sidecars are **renamed to match the image's new stem**,
  so an image never gets separated from its own caption.
- **Dry Run defaults to on.** It writes the report and moves nothing.
- `cluster_report.txt` is written into the output folder (or leftovers, for
  `Drop outliers`) with the full per-cluster table. The status label is one
  elided line, so the report is where the detail lives.
- Output and leftovers folders must not overlap the input folder.
- `MAX_IMAGES = 15000` — `pdist` is O(n²); 15k images is already ~1.7 GB of
  float64 distances.

## Embedding cache

`{tempdir}/label-to-dataset/cluster_{md5(input_path)[:12]}/embeddings.npz`,
keyed per image by `path|size|mtime_ns`. Retuning tolerance on the same folder
is instant — only the clustering re-runs. Embedding 504 images takes ~90 s
cold, ~0 warm.

`_load_cache()` drops any row that isn't finite, so a cache poisoned by an
older buggy run heals itself instead of failing the clustering.
