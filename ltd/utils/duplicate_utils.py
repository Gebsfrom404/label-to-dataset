"""Duplicate-image detection: perceptual hashing and local feature descriptors.

Two comparison families, plus a hybrid:

* **Perceptual hash (pHash)** — 64-bit DCT hash of a 32x32 grayscale copy.
  One integer per image, compared with a Hamming distance, so an all-pairs
  sweep over thousands of images is vectorised and fast. Robust to rescaling,
  re-encoding and mild colour edits; blind to crops and rotations.
* **Local feature descriptors (ORB)** — up to ``ORB_FEATURES`` keypoints per
  image, matched with a Lowe ratio test and verified with a RANSAC homography.
  Finds crops, rotations and framing changes, but every pair needs a real
  match, so it is O(n^2) in *matching* work, not just in comparisons.
* **Hybrid** — pHash builds a loose shortlist, ORB verifies only those pairs.
  Cheap, but inherits pHash's blind spot: rotated copies never reach the
  verification stage.

Scores are normalised to ``0.0..1.0`` for both families so the UI can show one
number: pHash uses ``1 - distance / 64``, ORB uses the share of keypoints that
survive matching + geometric verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ltd.utils.file_utils import cv_imread

# --- algorithms -------------------------------------------------------------

ALGO_PHASH = 'phash'
ALGO_ORB = 'orb'
ALGO_HYBRID = 'hybrid'

ALGORITHM_LABELS = {
    ALGO_PHASH: 'Perceptual hash (fast)',
    ALGO_ORB: 'Local feature descriptors (crops / rotations)',
    ALGO_HYBRID: 'Perceptual hash, then descriptor verify',
}
ALGORITHM_BY_LABEL = {label: key for key, label in ALGORITHM_LABELS.items()}

# Label masks live next to their image; they are never search candidates.
MASK_SUFFIX = '-masklabel.png'

# --- tuning -----------------------------------------------------------------

HASH_SIZE = 8
HASH_BITS = HASH_SIZE * HASH_SIZE  # 64
_HIGHFREQ_FACTOR = 4

ORB_FEATURES = 500
ORB_MAX_DIM = 640
ORB_MIN_KEYPOINTS = 12
_LOWE_RATIO = 0.75
_RANSAC_REPROJ = 5.0
_MIN_MATCHES_FOR_HOMOGRAPHY = 8

# tolerance (0 = strict, 100 = loose) → per-algorithm threshold
_PHASH_MAX_BITS = 25
_ORB_STRICT = 0.45
_ORB_LOOSE = 0.05
_HYBRID_SHORTLIST_BITS = 24

# Byte → set-bit-count lookup; avoids depending on np.bitwise_count.
_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint16)


# --- data -------------------------------------------------------------------

@dataclass
class OrbSignature:
    """Keypoint coordinates + binary descriptors for one image."""
    keypoints: np.ndarray   # (n, 2) float32
    descriptors: np.ndarray  # (n, 32) uint8


@dataclass
class ImageSignature:
    path: Path
    is_original: bool = False
    width: int = 0
    height: int = 0
    phash: np.ndarray | None = None   # (8,) uint8, packed 64-bit hash
    orb: OrbSignature | None = None


@dataclass
class DuplicateMember:
    path: Path
    score: float
    is_original: bool = False
    width: int = 0
    height: int = 0


@dataclass
class DuplicateGroup:
    """One set of images considered the same picture. Original first."""
    members: list[DuplicateMember] = field(default_factory=list)


# --- thresholds -------------------------------------------------------------

def phash_max_distance(tolerance: int) -> int:
    """Largest accepted Hamming distance for a tolerance of 0..100."""
    return round(min(max(tolerance, 0), 100) / 100 * _PHASH_MAX_BITS)


def orb_min_score(tolerance: int) -> float:
    """Smallest accepted keypoint-agreement ratio for a tolerance of 0..100."""
    t = min(max(tolerance, 0), 100) / 100
    return _ORB_STRICT - t * (_ORB_STRICT - _ORB_LOOSE)


def hybrid_shortlist_distance(tolerance: int) -> int:
    return max(phash_max_distance(tolerance), _HYBRID_SHORTLIST_BITS)


def threshold_description(algorithm: str, tolerance: int) -> str:
    """One-line explanation of what the current tolerance actually means."""
    if algorithm == ALGO_PHASH:
        return (f'match when at most {phash_max_distance(tolerance)} of '
                f'{HASH_BITS} hash bits differ')
    if algorithm == ALGO_ORB:
        return (f'match when at least {orb_min_score(tolerance):.0%} of '
                f'keypoints align')
    return (f'shortlist within {hybrid_shortlist_distance(tolerance)} bits, '
            f'then at least {orb_min_score(tolerance):.0%} of keypoints align')


# --- perceptual hash --------------------------------------------------------

def compute_phash(path: Path) -> tuple[np.ndarray, int, int] | None:
    """Return ``(packed_hash, width, height)``, or None if unreadable."""
    image = cv_imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    size = HASH_SIZE * _HIGHFREQ_FACTOR
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    low_freq = dct[:HASH_SIZE, :HASH_SIZE].flatten()
    # Median over everything but the DC term, which is an outlier by design.
    median = float(np.median(low_freq[1:]))
    return np.packbits(low_freq > median), width, height


def hamming_distances(row: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Bit distance of one packed hash (8,) against many (n, 8)."""
    if others.size == 0:
        return np.empty(0, dtype=np.uint16)
    return _POPCOUNT[np.bitwise_xor(row, others)].sum(axis=1)


def phash_score(distance: int) -> float:
    return 1.0 - distance / HASH_BITS


# --- local feature descriptors ---------------------------------------------

def create_orb_detector() -> cv2.ORB:
    """ORB detectors are stateful — one per worker thread, reused per image."""
    return cv2.ORB.create(nfeatures=ORB_FEATURES)


def compute_orb(path: Path,
                detector: cv2.ORB | None = None
                ) -> tuple[OrbSignature, int, int] | None:
    """Return ``(signature, width, height)`` at original scale, or None."""
    image = cv_imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > ORB_MAX_DIM:
        scale = ORB_MAX_DIM / longest
        image = cv2.resize(image, (max(1, round(width * scale)),
                                   max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    detector = detector or create_orb_detector()
    keypoints, descriptors = detector.detectAndCompute(image, None)
    if descriptors is None or len(keypoints) < ORB_MIN_KEYPOINTS:
        return None
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return OrbSignature(points, descriptors), width, height


def create_orb_matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def orb_similarity(a: OrbSignature | None, b: OrbSignature | None,
                   matcher: cv2.BFMatcher | None = None) -> float:
    """Share of keypoints that match *and* fit one homography (0.0..1.0).

    Falls back to the plain ratio-test share when there are too few matches
    for a homography, or when RANSAC cannot estimate one (near-degenerate
    keypoint layouts, e.g. flat or heavily repetitive images).
    """
    if a is None or b is None:
        return 0.0
    denominator = min(len(a.descriptors), len(b.descriptors))
    if denominator == 0:
        return 0.0
    matcher = matcher or create_orb_matcher()
    try:
        raw = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    except cv2.error:
        return 0.0

    good = [pair[0] for pair in raw
            if len(pair) == 2 and pair[0].distance < _LOWE_RATIO * pair[1].distance]
    if len(good) < _MIN_MATCHES_FOR_HOMOGRAPHY:
        return len(good) / denominator

    src = a.keypoints[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = b.keypoints[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    try:
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_REPROJ)
    except cv2.error:
        matrix, mask = None, None
    if matrix is None or mask is None:
        return len(good) / denominator
    return int(mask.sum()) / denominator


# --- grouping ---------------------------------------------------------------

def group_pairs(count: int, pairs: list[tuple[int, int, float]]
                ) -> list[list[int]]:
    """Union-find the matched pairs into groups of 2+ indices, index-ordered."""
    parent = list(range(count))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, _ in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [members for _, members in sorted(groups.items())
            if len(members) > 1]
