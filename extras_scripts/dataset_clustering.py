"""Cluster a dataset by image embedding similarity to flatten its distribution.

Embeddings come from the same tagger stack the Caption tab uses
(`animetimm/convnextv2_huge.dbv4-full` via timm + safetensors), taking the
2816-d pre-logits vector instead of the 12476-d tag head. The tag head is read
from the *same* forward pass, for free, and only used to give each cluster a
human-readable name.

Purpose: shrink an overrepresented dataset so a style LoRA sees a balanced
spread of scenes rather than 200 near-identical frames of one, without
throwing away the rare material that keeps the LoRA flexible.
"""

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STRATEGY_CLUSTER = 'Cluster by folder'
STRATEGY_OUTLIERS = 'Drop outliers'

SCRIPT_INFO = {
    'name': 'Cluster Dataset by Similarity',
    'type': 'dataset',
    'description': (
        'Embed every image with convnextv2_huge.dbv4-full, cluster by cosine '
        'similarity, then either split into balanced {repeats}_{name} folders '
        'or move outliers to a leftovers folder. Matching .txt captions and '
        '-masklabel.png masks travel with their image.'
    ),
    'project_url': 'https://huggingface.co/animetimm/convnextv2_huge.dbv4-full',
    'parameters': [
        {'name': 'input_folder', 'type': 'folder', 'label': 'Input Folder', 'default': ''},
        {'name': 'output_folder', 'type': 'folder', 'label': 'Output Folder', 'default': '',
         'placeholder': 'Cluster folders are created here (Cluster by folder only)'},
        {'name': 'leftovers_folder', 'type': 'folder', 'label': 'Leftovers Folder', 'default': '',
         'placeholder': 'Outliers and over-cap images land here'},
        {'name': 'strategy', 'type': 'combo', 'label': 'Strategy',
         'options': [STRATEGY_CLUSTER, STRATEGY_OUTLIERS], 'default': STRATEGY_CLUSTER},
        {'name': 'tolerance', 'type': 'str', 'label': 'Cluster Tolerance (cosine distance)',
         'default': '0.60', 'placeholder': '0.60 - usable range ~0.45-0.80, higher merges more'},
        {'name': 'min_cluster_size', 'type': 'str', 'label': 'Min Cluster Size (below = outlier)',
         'default': '3', 'placeholder': '3'},
        {'name': 'cap_percent', 'type': 'str', 'label': 'Size Cap (% of median cluster)',
         'default': '150', 'placeholder': '150'},
        {'name': 'max_repeats', 'type': 'str', 'label': 'Max Repeats', 'default': '5',
         'placeholder': '5'},
        {'name': 'batch_size', 'type': 'str', 'label': 'Embedding Batch Size', 'default': '8',
         'placeholder': '8 - lower if you run out of VRAM'},
        {'name': 'dry_run', 'type': 'bool', 'label': 'Dry Run (report only, move nothing)',
         'default': True},
    ],
}

MODEL_REPO = 'animetimm/convnextv2_huge.dbv4-full'
MODELS_BASE = Path('./models/caption')
INPUT_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
MASK_SUFFIX = '-masklabel.png'
CAPTION_EXT = '.txt'

# pdist is O(n^2) in memory; 15k images is already ~1.7GB of float64 distances.
MAX_IMAGES = 15000

# Tags kept per image for cluster naming (sparse - the full 12476-d prob
# vector per image would be hundreds of MB on a large dataset).
_NAME_TAG_TOPK = 30
_NAME_TAG_FLOOR = 0.35


def check_available() -> tuple[bool, str]:
    for mod, hint in (('torch', 'torch'), ('timm', 'timm'),
                      ('safetensors', 'safetensors'), ('scipy', 'scipy'),
                      ('PIL', 'Pillow'), ('huggingface_hub', 'huggingface-hub')):
        try:
            __import__(mod)
        except ImportError:
            return False, f'{hint} not installed'
    return True, ''


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_images(root: Path) -> list[Path]:
    """All dataset images under root, recursively. Mask sidecars are not images."""
    found = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name.endswith(MASK_SUFFIX):
            continue
        found.append(path)
    return found


def _sidecars(path: Path) -> list[Path]:
    return [p for p in (path.with_suffix(CAPTION_EXT),
                        path.parent / f'{path.stem}{MASK_SUFFIX}') if p.exists()]


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_path(root: Path) -> Path:
    """Temp dir keyed by input folder, mirroring the app's labels_*/generated_* scheme."""
    digest = hashlib.md5(str(root.resolve()).encode()).hexdigest()[:12]
    cache_dir = Path(tempfile.gettempdir()) / 'label-to-dataset' / f'cluster_{digest}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'embeddings.npz'


def _file_key(path: Path) -> str:
    st = path.stat()
    return f'{path}|{st.st_size}|{st.st_mtime_ns}'


def _load_cache(cache_file: Path) -> dict:
    if not cache_file.exists():
        return {}
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            keys = [str(k) for k in data['keys']]
            embeds = data['embeds']
            tags = json.loads(str(data['tags_json']))
        # Drop poisoned rows so a cache written by an older, buggier run heals
        # itself on the next pass instead of failing the clustering.
        return {k: (embeds[i], tags[i]) for i, k in enumerate(keys)
                if np.isfinite(embeds[i]).all()}
    except Exception as e:
        logger.warning('Ignoring unreadable embedding cache %s: %s', cache_file, e)
        return {}


def _save_cache(cache_file: Path, cache: dict) -> None:
    if not cache:
        return
    keys = list(cache.keys())
    try:
        np.savez_compressed(
            cache_file,
            keys=np.array(keys),
            embeds=np.stack([cache[k][0] for k in keys]),
            tags_json=np.array(json.dumps([cache[k][1] for k in keys])),
        )
    except Exception as e:
        logger.warning('Could not write embedding cache: %s', e)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model() -> tuple:
    """Load the tagger and its tag list. Returns (model, device, tag_names).

    The model is deliberately untyped: `nn.Module.__getattr__` makes Pyright
    resolve `forward_features` / `forward_head` as Tensor attributes.
    """
    import timm
    import torch
    import huggingface_hub
    from safetensors.torch import load_file

    model_dir = MODELS_BASE / MODEL_REPO
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'model.safetensors'
    tags_path = model_dir / 'selected_tags.csv'
    config_path = model_dir / 'config.json'

    for fname, fpath in (('model.safetensors', model_path),
                         ('selected_tags.csv', tags_path),
                         ('config.json', config_path)):
        if not fpath.exists():
            try:
                huggingface_hub.hf_hub_download(
                    MODEL_REPO, filename=fname, local_dir=str(model_dir))
            except Exception as e:
                raise RuntimeError(
                    f'Could not download {fname} from {MODEL_REPO}: {e}. '
                    'This repo is gated - log in with `huggingface-cli login` '
                    'or place the files in models/caption/ manually.') from e

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model = timm.create_model(
        config['architecture'], pretrained=False,
        num_classes=config['num_classes'], **config.get('model_args', {}))
    model.load_state_dict(load_file(str(model_path)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    tag_names = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        for line in csv.DictReader(f):
            # Category 9 is the rating group - useless for naming a cluster.
            tag_names.append(None if line['category'] == '9'
                             else line['name'].replace('_', ' '))

    return model, device, tag_names


def _preprocess(path: Path) -> np.ndarray:
    """White-padded square, 512px, RGB, ImageNet-normalized CHW - same as the Caption tab."""
    from PIL import Image as PilImage

    img = PilImage.open(path).convert('RGBA')
    canvas = PilImage.new('RGBA', img.size, (255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert('RGB')

    max_dim = max(img.size)
    square = PilImage.new('RGB', (max_dim, max_dim), (255, 255, 255))
    square.paste(img, ((max_dim - img.width) // 2, (max_dim - img.height) // 2))
    if max_dim != INPUT_SIZE:
        square = square.resize((INPUT_SIZE, INPUT_SIZE),
                               resample=PilImage.Resampling.BICUBIC)

    arr = (np.asarray(square, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def _embed_all(root: Path, images: list[Path], batch_size: int,
               progress) -> tuple[np.ndarray, list[list[str]]]:
    """L2-normalized embeddings + per-image top tags, reusing a disk cache."""
    import torch

    cache_file = _cache_path(root)
    cache = _load_cache(cache_file)

    keys = [_file_key(p) for p in images]
    todo = [i for i, k in enumerate(keys) if k not in cache]

    if todo:
        progress(0, len(todo), f'Loading {MODEL_REPO}...')
        model, device, tag_names = _load_model()

        # fp16 makes this model emit all-NaN embeddings - ConvNeXtV2's GRN
        # takes a spatial L2 norm that overflows half range at 512px. bf16 has
        # the same exponent range as fp32, stays finite (cosine similarity
        # 0.9996 vs fp32), and runs ~2x faster. Anything else: plain fp32.
        use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()

        try:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                batch = np.stack([_preprocess(images[i]) for i in chunk])
                tensor = torch.from_numpy(batch).to(device)

                with torch.no_grad():
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        feats = model.forward_features(tensor)
                        embeds = model.forward_head(feats, pre_logits=True)
                        probs = torch.sigmoid(model.forward_head(feats))

                embeds = embeds.float().cpu().numpy()
                probs = probs.float().cpu().numpy()

                if not np.isfinite(embeds).all():
                    raise RuntimeError(
                        'Model produced non-finite embeddings - refusing to cache '
                        'them. Try Embedding Batch Size 1, or report this.')

                for row, idx in enumerate(chunk):
                    top = np.argsort(probs[row])[::-1][:_NAME_TAG_TOPK]
                    tags = [tag_names[t] for t in top
                            if tag_names[t] and probs[row][t] >= _NAME_TAG_FLOOR]
                    cache[keys[idx]] = (embeds[row].astype(np.float32), tags)

                done = min(start + batch_size, len(todo))
                if progress(done, len(todo), f'Embedding {done}/{len(todo)} images...'):
                    raise RuntimeError('Cancelled')
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _save_cache(cache_file, cache)
    else:
        progress(len(images), len(images), 'All embeddings served from cache')

    matrix = np.stack([cache[k][0] for k in keys]).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix, [cache[k][1] for k in keys]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster(embeds: np.ndarray, tolerance: float) -> np.ndarray:
    """Complete-linkage agglomerative clustering cut at a cosine-distance threshold.

    Complete linkage, not average: measured on a 504-image dataset, average
    linkage chains badly - between tolerance 0.55 and 0.60 its largest cluster
    jumped 152 -> 339 of 504 images while the median cluster stayed at 5, so
    there was no usable setting. Complete linkage bounds every cluster's
    diameter, so cluster size grows gradually and the knob is tunable. Ward was
    the other candidate and balances even better, but it forces every image
    into a cluster and so can never surface an outlier.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(embeds) == 1:
        return np.array([1])

    condensed = pdist(embeds, metric='cosine')
    linkage_matrix = linkage(condensed, method='complete')
    return fcluster(linkage_matrix, t=tolerance, criterion='distance')


def _farthest_point_keep(embeds: np.ndarray, members: list[int], keep: int) -> list[int]:
    """Keep `keep` members spread as widely as possible, starting from the medoid."""
    if keep >= len(members):
        return list(members)

    sub = embeds[members]
    # Cosine distance on L2-normalized vectors is 1 - dot.
    dists = 1.0 - (sub @ sub.T)

    selected = [int(np.argmin(dists.sum(axis=1)))]
    min_dist = dists[selected[0]].copy()
    while len(selected) < keep:
        min_dist[selected] = -1.0
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dists[nxt])

    return [members[i] for i in selected]


def _medoid(embeds: np.ndarray, members: list[int]) -> int:
    sub = embeds[members]
    dists = 1.0 - (sub @ sub.T)
    return members[int(np.argmin(dists.sum(axis=1)))]


def _sanitize(name: str) -> str:
    name = re.sub(r'[^\w\-]+', '_', name.strip().lower())
    return re.sub(r'_+', '_', name).strip('_')[:60]


def _cluster_name(members: list[int], tags: list[list[str]],
                  global_freq: dict, total: int, fallback: str) -> str:
    """Name a cluster by the tags distinctive to it, not merely the common ones."""
    local_freq: dict[str, int] = {}
    for idx in members:
        for tag in tags[idx]:
            local_freq[tag] = local_freq.get(tag, 0) + 1

    scored = []
    for tag, count in local_freq.items():
        local_share = count / len(members)
        if local_share < 0.5:
            continue
        scored.append((local_share - global_freq.get(tag, 0) / total, tag))

    scored.sort(reverse=True)
    picked = [_sanitize(tag) for _, tag in scored[:3] if _sanitize(tag)]
    return '_'.join(picked) if picked else _sanitize(fallback) or 'cluster'


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def _unique_dest(dest_dir: Path, name: str) -> Path:
    """Input subfolders get flattened, so equal stems from different folders can collide."""
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while (dest_dir / f'{stem}_{n}{suffix}').exists():
        n += 1
    return dest_dir / f'{stem}_{n}{suffix}'


def _move_with_sidecars(path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    sidecars = _sidecars(path)
    dest = _unique_dest(dest_dir, path.name)
    shutil.move(str(path), str(dest))
    for sidecar in sidecars:
        if sidecar.name.endswith(MASK_SUFFIX):
            target = dest_dir / f'{dest.stem}{MASK_SUFFIX}'
        else:
            target = dest_dir / f'{dest.stem}{sidecar.suffix}'
        shutil.move(str(sidecar), str(target))


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class _Plan:
    """Shared state for turning clusters into {repeats}_{name} folder moves."""

    images: list[Path]
    embeds: np.ndarray
    tags: list[list[str]]
    clusters: dict[int, list[int]]
    global_freq: dict[str, int]
    median: float
    cap: int
    max_repeats: int
    leftovers: Path

    def build(self, out_root: Path,
              kept_ids: list[int]) -> tuple[list[tuple[Path, Path]], list[str]]:
        moves: list[tuple[Path, Path]] = []
        report = [f'Output:         {out_root}', '',
                  f'{"#":>4}  {"size":>5}  {"kept":>5}  {"rep":>3}  folder']
        used_names: set[str] = set()

        for order, cluster_id in enumerate(
                sorted(kept_ids, key=lambda c: -len(self.clusters[c])), start=1):
            members = self.clusters[cluster_id]
            keep = _farthest_point_keep(self.embeds, members, self.cap)
            kept_set = set(keep)

            # Trim the big clusters down to the cap, repeat the small ones up.
            repeats = min(self.max_repeats,
                          max(1, int(round(self.median / len(keep)))))
            name = _cluster_name(members, self.tags, self.global_freq,
                                 len(self.images),
                                 self.images[_medoid(self.embeds, members)].stem)

            # Two clusters can score the same top tags. Without this they would
            # share a folder, silently merging groups the clustering separated.
            folder = f'{repeats}_{name}'
            if folder in used_names:
                n = 2
                while f'{folder}_{n}' in used_names:
                    n += 1
                folder = f'{folder}_{n}'
            used_names.add(folder)
            dest = out_root / folder

            moves.extend((self.images[i], dest) for i in keep)
            moves.extend((self.images[i], self.leftovers)
                         for i in members if i not in kept_set)

            report.append(f'{order:>4}  {len(members):>5}  {len(keep):>5}  '
                          f'{repeats:>3}  {dest.name}')

        return moves, report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_number(params: dict, key: str, default, cast, low, high):
    raw = str(params.get(key, '')).strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f'{key} must be a number, got {raw!r}')
    if not low <= value <= high:
        raise ValueError(f'{key} must be between {low} and {high}, got {value}')
    return value


def _validate_folders(root: Path, other: Path, label: str) -> None:
    if other == root or root in other.parents or other in root.parents:
        raise ValueError(f'{label} must not overlap the input folder')


def run(params: dict, progress_callback) -> None:
    input_folder = params.get('input_folder', '').strip()
    output_folder = params.get('output_folder', '').strip()
    leftovers_folder = params.get('leftovers_folder', '').strip()
    strategy = params.get('strategy', STRATEGY_CLUSTER)
    dry_run = bool(params.get('dry_run', True))

    tolerance = _parse_number(params, 'tolerance', 0.60, float, 0.01, 2.0)
    min_cluster_size = _parse_number(params, 'min_cluster_size', 3, int, 1, 1000)
    cap_percent = _parse_number(params, 'cap_percent', 150, float, 10, 1000)
    max_repeats = _parse_number(params, 'max_repeats', 5, int, 1, 100)
    batch_size = _parse_number(params, 'batch_size', 8, int, 1, 128)

    if not input_folder or not os.path.isdir(input_folder):
        raise ValueError(f'Input folder does not exist: {input_folder}')
    if not leftovers_folder:
        raise ValueError('Leftovers folder is required')
    if strategy == STRATEGY_CLUSTER and not output_folder:
        raise ValueError('Output folder is required for "Cluster by folder"')

    root = Path(input_folder).resolve()
    leftovers = Path(leftovers_folder).resolve()
    _validate_folders(root, leftovers, 'Leftovers folder')

    report_dir = leftovers
    if strategy == STRATEGY_CLUSTER:
        report_dir = Path(output_folder).resolve()
        _validate_folders(root, report_dir, 'Output folder')

    progress_callback(0, 1, 'Scanning for images...')
    images = _iter_images(root)
    if not images:
        progress_callback(0, 0, 'No images found')
        return
    if len(images) > MAX_IMAGES:
        raise ValueError(
            f'{len(images)} images exceeds the {MAX_IMAGES} limit - the pairwise '
            'distance matrix would not fit in memory. Split the dataset and run per part.')

    embeds, tags = _embed_all(root, images, batch_size, progress_callback)

    progress_callback(0, 1, f'Clustering {len(images)} images...')
    labels = _cluster(embeds, tolerance)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    outlier_ids = [c for c, m in clusters.items() if len(m) < min_cluster_size]
    kept_ids = [c for c in clusters if c not in set(outlier_ids)]
    outlier_indices = [i for c in outlier_ids for i in clusters[c]]

    if not kept_ids:
        raise ValueError(
            f'Every cluster is smaller than {min_cluster_size} images - raise the '
            'tolerance or lower Min Cluster Size.')

    median = float(np.median([len(clusters[c]) for c in kept_ids]))
    cap = max(1, int(round(median * cap_percent / 100.0)))

    global_freq: dict[str, int] = {}
    for tag_list in tags:
        for tag in tag_list:
            global_freq[tag] = global_freq.get(tag, 0) + 1

    report = [
        f'Input:          {root}',
        f'Strategy:       {strategy}',
        f'Images found:   {len(images)}',
        f'Tolerance:      {tolerance:g}   Min cluster size: {min_cluster_size}',
        f'Clusters:       {len(kept_ids)} kept, {len(outlier_ids)} below min size',
        f'Median size:    {median:g}   Cap: {cap} ({cap_percent:g}% of median)',
        f'Outlier images: {len(outlier_indices)}',
        '',
    ]

    moves: list[tuple[Path, Path]] = [(images[i], leftovers) for i in outlier_indices]

    if strategy == STRATEGY_CLUSTER:
        plan = _Plan(images=images, embeds=embeds, tags=tags, clusters=clusters,
                     global_freq=global_freq, median=median, cap=cap,
                     max_repeats=max_repeats, leftovers=leftovers)
        cluster_moves, cluster_report = plan.build(report_dir, kept_ids)
        moves.extend(cluster_moves)
        report.extend(cluster_report)
    else:
        report.append('Non-outlier images stay in place (cap not applied).')

    total_moves = len(moves)
    to_leftovers = sum(1 for _, dest in moves if dest == leftovers)
    report.extend([
        '',
        f'Images moved to leftovers: {to_leftovers}',
        f'Images moved in total:     {total_moves}',
        f'Images left in place:      {len(images) - total_moves}',
    ])

    if not dry_run:
        for i, (src, dest) in enumerate(moves):
            _move_with_sidecars(src, dest)
            if i % 25 == 0 and progress_callback(
                    i, total_moves, f'Moving {i}/{total_moves}...'):
                raise RuntimeError(f'Cancelled after {i} of {total_moves} moves')

    report_text = '\n'.join(report)
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / 'cluster_report.txt').write_text(report_text, encoding='utf-8')
    except OSError as e:
        logger.warning('Could not write cluster report: %s', e)
    logger.info('Dataset clustering report:\n%s', report_text)

    prefix = 'DRY RUN - ' if dry_run else ''
    progress_callback(
        total_moves, total_moves,
        f'{prefix}{len(kept_ids)} clusters, {to_leftovers} to leftovers, '
        f'{total_moves} moves - see cluster_report.txt')
