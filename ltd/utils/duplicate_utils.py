"""Duplicate-image detection: perceptual hashing and local feature descriptors.

Two comparison families, plus a hybrid:

* **Perceptual hash (pHash)** — 256-bit DCT hash of a 64x64 grayscale copy.
  32 bytes per image, compared with a Hamming distance, so an all-pairs sweep
  over thousands of images is vectorised and fast. Robust to rescaling,
  re-encoding and mild colour edits; blind to crops and rotations.
* **Local feature descriptors (ORB)** — up to ``ORB_FEATURES`` keypoints per
  image, matched with a Lowe ratio test and verified with a RANSAC homography.
  Finds crops, rotations and framing changes, but every pair needs a real
  match, so it is O(n^2) in *matching* work, not just in comparisons.
* **Hybrid** — pHash builds a loose shortlist, ORB verifies only those pairs.
  Cheap, but inherits pHash's blind spot: rotated copies never reach the
  verification stage.

Scores are normalised to ``0.0..1.0`` for both families so the UI can show one
number: pHash uses ``1 - distance / HASH_BITS``, ORB uses the share of
keypoints that survive matching + geometric verification.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
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

# 16x16 = 256 bits, not the more common 8x8/64. Measured on a real 397-image
# folder, a 64-bit hash put five *different* pairs at distance 0 — variants of
# one artwork differing by an added speech bubble hash identically, because a
# change over ~15% of the frame moves none of the 64 coarsest DCT
# coefficients. At 256 bits nothing in that folder collides at 0, while every
# genuine-duplicate transform (q20 JPEG, eighth-size, blur, small watermark)
# stays within 4 bits — a 5x margin against the 20 bits that separated the
# misreported pair. See gotchas-decisions.md.
HASH_SIZE = 16
HASH_BITS = HASH_SIZE * HASH_SIZE  # 256
_HIGHFREQ_FACTOR = 4

ORB_FEATURES = 500
ORB_MAX_DIM = 640
ORB_MIN_KEYPOINTS = 12
_LOWE_RATIO = 0.75
_RANSAC_REPROJ = 5.0
_MIN_MATCHES_FOR_HOMOGRAPHY = 8

# tolerance (0 = strict, 100 = loose) → per-algorithm threshold.
# 64/256 = 25% of the hash. Unrelated images sit around 50% by construction,
# so a quarter of the bits is already generous: on the sample folder it
# accepted 120 of 78606 pairs, where the old 39% ceiling accepted 5293.
_PHASH_MAX_BITS = 64
_ORB_STRICT = 0.45
_ORB_LOOSE = 0.05
_HYBRID_SHORTLIST_BITS = 96  # 37.5% — deliberately loose, ORB rejects

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
    """Everything computed from one file, plus the stamp it was computed from.

    ``is_original`` is per *run* (it comes from the source list, not the
    file), so a cached signature must never be trusted for it — the worker
    re-stamps it on every search.
    """
    path: Path
    is_original: bool = False
    width: int = 0
    height: int = 0
    phash: np.ndarray | None = None   # (8,) uint8, packed 64-bit hash
    orb: OrbSignature | None = None
    stamp: tuple[float, int] = (0.0, 0)  # (mtime, size)

    def nbytes(self) -> int:
        if self.orb is None:
            return 64
        return 64 + self.orb.keypoints.nbytes + self.orb.descriptors.nbytes


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


@dataclass
class PairScore:
    """One compared pair, holding the raw measurements rather than a verdict.

    Keeping the distance and the descriptor score (instead of "matched: yes")
    is what lets a tolerance change re-filter an existing result set without
    touching the images again.
    """
    a: str  # cache key (normalised absolute path)
    b: str
    hash_distance: int | None = None   # differing pHash bits, 0..64
    orb_score: float | None = None     # keypoint agreement, 0.0..1.0


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


# --- thresholds applied to a computed pair ---------------------------------

def pair_accepted(pair: PairScore, algorithm: str, tolerance: int) -> bool:
    """Does this already-measured pair match at the given tolerance?

    Every threshold is monotonic in tolerance, so the set accepted at ``t`` is
    a subset of the set accepted at any ``T >= t``. That is what makes a
    cached result set re-filterable instead of recomputable.
    """
    if algorithm == ALGO_PHASH:
        return (pair.hash_distance is not None
                and pair.hash_distance <= phash_max_distance(tolerance))
    if algorithm == ALGO_ORB:
        return (pair.orb_score is not None
                and pair.orb_score >= orb_min_score(tolerance))
    return (pair.hash_distance is not None
            and pair.hash_distance <= hybrid_shortlist_distance(tolerance)
            and pair.orb_score is not None
            and pair.orb_score >= orb_min_score(tolerance))


def pair_display_score(pair: PairScore, algorithm: str) -> float:
    """The 0..1 number shown next to a duplicate in the list."""
    if algorithm == ALGO_PHASH:
        return phash_score(pair.hash_distance or 0)
    return pair.orb_score or 0.0


# --- grouping ---------------------------------------------------------------

def group_pairs(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find the matched pairs into groups of 2+ indices, index-ordered."""
    parent = list(range(count))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [members for _, members in sorted(groups.items())
            if len(members) > 1]


def build_groups(signatures: list[ImageSignature], pairs: list[PairScore],
                 algorithm: str, tolerance: int) -> list[DuplicateGroup]:
    """Threshold the measured pairs and assemble the display groups.

    Pure and cheap — no image is touched — so a tolerance-only re-run goes
    straight here with the cached pairs.
    """
    index_of = {signature_key(s.path): i for i, s in enumerate(signatures)}
    matched: list[tuple[int, int, float]] = []
    for pair in pairs:
        if not pair_accepted(pair, algorithm, tolerance):
            continue
        a, b = index_of.get(pair.a), index_of.get(pair.b)
        if a is None or b is None:
            continue  # file vanished between runs
        matched.append((a, b, pair_display_score(pair, algorithm)))

    if any(s.is_original for s in signatures):
        return _build_original_groups(signatures, matched)
    return _build_symmetric_groups(signatures, matched)


def _build_original_groups(signatures: list[ImageSignature],
                           matched: list[tuple[int, int, float]]
                           ) -> list[DuplicateGroup]:
    # An image can match several originals — keep only its best match.
    best: dict[int, tuple[int, float]] = {}
    for a, b, score in matched:
        original, other = (a, b) if signatures[a].is_original else (b, a)
        current = best.get(other)
        if current is None or score > current[1]:
            best[other] = (original, score)

    by_original: dict[int, list[tuple[int, float]]] = {}
    for other, (original, score) in best.items():
        by_original.setdefault(original, []).append((other, score))

    groups = []
    for original in sorted(by_original,
                           key=lambda i: str(signatures[i].path).lower()):
        members = [_member(signatures[original], 1.0)]
        duplicates = sorted(
            by_original[original],
            key=lambda item: (-item[1], str(signatures[item[0]].path).lower()))
        members.extend(_member(signatures[index], score)
                       for index, score in duplicates)
        groups.append(DuplicateGroup(members))
    return groups


def _build_symmetric_groups(signatures: list[ImageSignature],
                            matched: list[tuple[int, int, float]]
                            ) -> list[DuplicateGroup]:
    best_score: dict[int, float] = {}
    for a, b, score in matched:
        best_score[a] = max(best_score.get(a, 0.0), score)
        best_score[b] = max(best_score.get(b, 0.0), score)

    groups = []
    for indices in group_pairs(len(signatures),
                               [(a, b) for a, b, _ in matched]):
        ordered = sorted(indices, key=lambda i: str(signatures[i].path).lower())
        groups.append(DuplicateGroup(
            [_member(signatures[i], best_score.get(i, 0.0)) for i in ordered]))
    return groups


def _member(signature: ImageSignature, score: float) -> DuplicateMember:
    return DuplicateMember(path=signature.path, score=score,
                           is_original=signature.is_original,
                           width=signature.width, height=signature.height)


# --- cache ------------------------------------------------------------------

def signature_key(path: Path) -> str:
    """Stable per-file cache key (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(str(path)))


def file_stamp(path: Path) -> tuple[float, int]:
    """(mtime, size) — an edited file gets a new stamp and is recomputed."""
    try:
        info = path.stat()
    except OSError:
        return (0.0, 0)
    return (info.st_mtime, info.st_size)


class SearchCache:
    """Survives between searches so a re-run only redoes what changed.

    Two layers:

    * **Signatures** — per file, keyed by path and validated against its
      ``(mtime, size)`` stamp. pHash rows are tiny and always kept; ORB
      descriptors are large, so they are kept only while the store stays
      under ``ORB_BUDGET_BYTES`` (oldest dropped first) and are recomputed if
      they were evicted.
    * **Pairs** — the measured pair list from the last comparison, tagged
      with the algorithm, the file set it covers, and the tolerance it was
      computed at. A re-run at the same or a tighter tolerance re-filters it
      instead of comparing again.
    """

    ORB_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self):
        self._signatures: OrderedDict[str, ImageSignature] = OrderedDict()
        self.pairs: list[PairScore] = []
        self.pairs_algorithm = ''
        self.pairs_fingerprint = ''
        self.pairs_tolerance = -1

    # -- signatures --

    def get(self, path: Path,
            stamp: tuple[float, int]) -> ImageSignature | None:
        """A cached signature for this exact file version, or None."""
        key = signature_key(path)
        cached = self._signatures.get(key)
        if cached is None or cached.stamp != stamp:
            return None
        self._signatures.move_to_end(key)
        return cached

    def put(self, signature: ImageSignature):
        self._signatures[signature_key(signature.path)] = signature
        self._enforce_budget()

    def _enforce_budget(self):
        """Drop ORB descriptors, oldest first, until back under budget.

        The signature itself stays (its pHash and dimensions cost nothing);
        only the heavy descriptor payload is released, and it is recomputed
        on demand if a later search needs it again.
        """
        total = sum(s.nbytes() for s in self._signatures.values())
        if total <= self.ORB_BUDGET_BYTES:
            return
        for signature in self._signatures.values():
            if total <= self.ORB_BUDGET_BYTES:
                break
            if signature.orb is not None:
                total -= signature.nbytes() - 64
                signature.orb = None

    # -- pairs --

    def pairs_usable(self, algorithm: str, fingerprint: str,
                     tolerance: int) -> bool:
        """True when the cached pairs already cover this request."""
        return (bool(self.pairs_algorithm)
                and algorithm == self.pairs_algorithm
                and fingerprint == self.pairs_fingerprint
                and tolerance <= self.pairs_tolerance)

    def store_pairs(self, pairs: list[PairScore], algorithm: str,
                    fingerprint: str, tolerance: int):
        self.pairs = pairs
        self.pairs_algorithm = algorithm
        self.pairs_fingerprint = fingerprint
        self.pairs_tolerance = tolerance

    def invalidate_pairs(self):
        self.pairs = []
        self.pairs_algorithm = ''
        self.pairs_fingerprint = ''
        self.pairs_tolerance = -1

    def clear(self):
        self._signatures.clear()
        self.invalidate_pairs()


def fingerprint_entries(entries: list[tuple[Path, bool]]) -> str:
    """Identify the exact file set a comparison covers.

    Includes each file's stamp and its original flag, so an edited image, a
    new file, or a changed *original* checkbox all force a fresh comparison.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    for path, is_original in entries:
        mtime, size = file_stamp(path)
        line = f'{signature_key(path)}|{mtime}|{size}|{int(is_original)}\n'
        digest.update(line.encode())
    return digest.hexdigest()
