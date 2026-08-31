"""Background duplicate search over one or more source folders.

Phases (each reports through ``status`` / ``progress``):

1. **Scan** — recurse every source folder, de-duplicating paths so nested
   sources are not compared with themselves.
2. **Signatures** — one pHash and/or ORB signature per image, taken from the
   ``SearchCache`` when the file is unchanged since the last search.
3. **Compare** — pairwise, recording the raw distance / keypoint score rather
   than a match verdict. When a source is flagged *original*, only
   original-vs-other pairs are tested; otherwise every pair is.
4. **Group** — threshold the measured pairs and assemble display groups.

Phases 2 and 3 are skipped entirely when the cache already holds a comparison
for this algorithm and file set at an equal or looser tolerance — changing
only the tolerance re-thresholds the existing measurements instead of reading
any image again.

Unreadable images and images with too few keypoints are dropped after the
signature phase and reported through ``skipped``.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal

from ltd.data.image_list_model import IMAGE_EXTENSIONS
from ltd.utils.duplicate_utils import (ALGO_HYBRID, ALGO_ORB, ALGO_PHASH,
                                       MASK_SUFFIX, ImageSignature,
                                       PairScore, SearchCache,
                                       build_groups, compute_orb,
                                       compute_phash, create_orb_detector,
                                       create_orb_matcher, file_stamp,
                                       fingerprint_entries, hamming_distances,
                                       hybrid_shortlist_distance,
                                       orb_min_score, orb_similarity,
                                       phash_max_distance, signature_key)
from ltd.workers.base_worker import BaseWorker


class DuplicateWorker(BaseWorker):
    """Finds duplicate images across the given sources."""

    groups_found = Signal(list)  # list[DuplicateGroup]
    scanned = Signal(int)        # images considered
    skipped = Signal(int)        # images without a usable signature
    reused = Signal(str)         # what this run was able to skip ('' = nothing)

    _PROGRESS_EVERY = 25

    def __init__(self, sources: list[tuple[Path, bool]], algorithm: str,
                 tolerance: int, cache: SearchCache | None = None,
                 parent=None):
        """``sources`` is a list of ``(folder, is_original)``.

        ``cache`` is owned by the caller and outlives the worker — that is
        what makes a second search cheap.
        """
        super().__init__(parent)
        self.sources = sources
        self.algorithm = algorithm
        self.tolerance = tolerance
        self.cache = cache if cache is not None else SearchCache()
        self._reuse_note = ''

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def do_work(self):
        self.status.emit('Scanning source folders...')
        entries = self._scan_sources()
        if self.is_cancelled or not entries:
            self.groups_found.emit([])
            return
        self.scanned.emit(len(entries))

        fingerprint = fingerprint_entries(entries)
        if self.cache.pairs_usable(self.algorithm, fingerprint,
                                   self.tolerance):
            signatures = self._cached_signatures(entries)
            if signatures is not None:
                self._finish(signatures, self.cache.pairs, len(entries),
                             'tolerance only — reused the cached comparison')
                return

        signatures = self._compute_signatures(entries)
        if self.is_cancelled:
            self.groups_found.emit([])
            return
        if len(signatures) < 2:
            self.skipped.emit(len(entries) - len(signatures))
            self.groups_found.emit([])
            return

        pairs = self._compare(signatures)
        if self.is_cancelled:
            self.groups_found.emit([])
            return
        self.cache.store_pairs(pairs, self.algorithm, fingerprint,
                               self.tolerance)
        self._finish(signatures, pairs, len(entries), self._reuse_note)

    def _finish(self, signatures: list[ImageSignature], pairs: list[PairScore],
                total: int, note: str):
        self.skipped.emit(total - len(signatures))
        self.reused.emit(note)
        self.status.emit('Grouping matches...')
        self.groups_found.emit(
            build_groups(signatures, pairs, self.algorithm, self.tolerance))

    # ------------------------------------------------------------------
    # 1. Scan
    # ------------------------------------------------------------------

    def _scan_sources(self) -> list[tuple[Path, bool]]:
        entries: list[list] = []      # [path, is_original]
        seen: dict[str, int] = {}     # normalised path → index in entries
        for folder, is_original in self.sources:
            if self.is_cancelled:
                break
            if not folder.is_dir():
                continue
            for file in sorted(folder.rglob('*')):
                if self.is_cancelled:
                    break
                if (not file.is_file()
                        or file.suffix.lower() not in IMAGE_EXTENSIONS
                        or file.name.lower().endswith(MASK_SUFFIX)):
                    continue
                key = signature_key(file)
                index = seen.get(key)
                if index is None:
                    seen[key] = len(entries)
                    entries.append([file, is_original])
                elif is_original:
                    # Overlapping sources: being an original wins.
                    entries[index][1] = True
        return [(path, flag) for path, flag in entries]

    # ------------------------------------------------------------------
    # 2. Signatures
    # ------------------------------------------------------------------

    def _wants(self) -> tuple[bool, bool]:
        """(pHash, ORB) — hybrid extracts ORB lazily, for shortlisted files."""
        return (self.algorithm in (ALGO_PHASH, ALGO_HYBRID),
                self.algorithm == ALGO_ORB)

    def _cached_signatures(self, entries: list[tuple[Path, bool]]
                           ) -> list[ImageSignature] | None:
        """Signatures for a fingerprint-matched run, without touching images.

        Grouping only needs the path, dimensions and original flag, all of
        which survive an ORB eviction. Returns None if anything is missing,
        so the caller falls back to a full run.
        """
        want_phash, want_orb = self._wants()
        signatures = []
        for path, is_original in entries:
            cached = self.cache.get(path, file_stamp(path))
            if cached is None:
                continue  # dropped last run (unreadable / too few keypoints)
            if (want_phash and cached.phash is None) or (
                    want_orb and cached.orb is None):
                return None
            cached.is_original = is_original
            signatures.append(cached)
        return signatures if len(signatures) >= 2 else None

    def _compute_signatures(self, entries: list[tuple[Path, bool]]
                            ) -> list[ImageSignature]:
        want_phash, want_orb = self._wants()
        total = len(entries)
        detector = create_orb_detector() if want_orb else None
        signatures: list[ImageSignature] = []
        computed = 0
        for index, (path, is_original) in enumerate(entries):
            if self.is_cancelled:
                break
            stamp = file_stamp(path)
            signature = self.cache.get(path, stamp)
            if signature is None:
                signature = ImageSignature(path=path, stamp=stamp)
            # The original flag comes from the source list, not the file, so
            # it is re-stamped on every run rather than trusted from cache.
            signature.is_original = is_original

            fresh = False
            if want_phash and signature.phash is None:
                result = compute_phash(path)
                fresh = True
                if result is not None:
                    signature.phash, signature.width, signature.height = result
            if want_orb and signature.orb is None:
                result = compute_orb(path, detector)
                fresh = True
                if result is not None:
                    signature.orb, signature.width, signature.height = result
            if fresh:
                computed += 1
                self.cache.put(signature)

            if not ((want_phash and signature.phash is None)
                    or (want_orb and signature.orb is None)):
                signatures.append(signature)
            if index % self._PROGRESS_EVERY == 0:
                self._report(index + 1, total, 'Reading images')
        self._report(total, total, 'Reading images')

        cached = total - computed
        self._reuse_note = (f'reused {cached} cached signature(s)'
                            if cached else '')
        return signatures

    # ------------------------------------------------------------------
    # 3. Compare
    # ------------------------------------------------------------------

    def _compare(self, signatures: list[ImageSignature]) -> list[PairScore]:
        if self.algorithm == ALGO_PHASH:
            return self._match_phash(signatures,
                                     phash_max_distance(self.tolerance))
        if self.algorithm == ALGO_ORB:
            return self._match_orb(signatures, orb_min_score(self.tolerance))
        shortlist = self._match_phash(
            signatures, hybrid_shortlist_distance(self.tolerance))
        return self._verify_shortlist(signatures, shortlist,
                                      orb_min_score(self.tolerance))

    # -- 3a. Perceptual hash comparison (vectorised per row) --

    def _match_phash(self, signatures: list[ImageSignature],
                     max_distance: int) -> list[PairScore]:
        # The signature phase drops anything without a hash, so the filter
        # here keeps every row — it just states that invariant.
        hashes = np.stack([s.phash for s in signatures if s.phash is not None])
        keys = [signature_key(s.path) for s in signatures]
        originals, others = self._split(signatures)
        pairs: list[PairScore] = []

        if originals:
            targets = np.array(others, dtype=int)
            if targets.size == 0:
                return pairs
            target_hashes = hashes[targets]
            for step, index in enumerate(originals):
                if self.is_cancelled:
                    break
                distances = hamming_distances(hashes[index], target_hashes)
                for hit in np.nonzero(distances <= max_distance)[0]:
                    pairs.append(PairScore(
                        keys[index], keys[int(targets[hit])],
                        hash_distance=int(distances[hit])))
                self._report(step + 1, len(originals), 'Comparing hashes')
            return pairs

        count = len(signatures)
        for index in range(count - 1):
            if self.is_cancelled:
                break
            distances = hamming_distances(hashes[index], hashes[index + 1:])
            for hit in np.nonzero(distances <= max_distance)[0]:
                pairs.append(PairScore(
                    keys[index], keys[index + 1 + int(hit)],
                    hash_distance=int(distances[hit])))
            if index % self._PROGRESS_EVERY == 0:
                self._report(index + 1, count, 'Comparing hashes')
        self._report(count, count, 'Comparing hashes')
        return pairs

    # -- 3b. Descriptor comparison (every pair is real work) --

    def _match_orb(self, signatures: list[ImageSignature],
                   min_score: float) -> list[PairScore]:
        matcher = create_orb_matcher()
        total = self._pair_count(signatures)
        pairs: list[PairScore] = []
        for done, (a, b) in enumerate(self._iter_pairs(signatures), start=1):
            if self.is_cancelled:
                break
            score = orb_similarity(signatures[a].orb, signatures[b].orb,
                                   matcher)
            if score >= min_score:
                pairs.append(PairScore(signature_key(signatures[a].path),
                                       signature_key(signatures[b].path),
                                       orb_score=score))
            if done % self._PROGRESS_EVERY == 0:
                self._report(done, total, 'Matching features')
        self._report(total, total, 'Matching features')
        return pairs

    def _verify_shortlist(self, signatures: list[ImageSignature],
                          shortlist: list[PairScore],
                          min_score: float) -> list[PairScore]:
        """Re-score pHash candidates with descriptors, dropping the rest."""
        if not shortlist or self.is_cancelled:
            return []
        by_key = {signature_key(s.path): s for s in signatures}
        involved = sorted({key for pair in shortlist
                           for key in (pair.a, pair.b)})
        detector = create_orb_detector()
        for done, key in enumerate(involved, start=1):
            if self.is_cancelled:
                return []
            signature = by_key.get(key)
            if signature is None or signature.orb is not None:
                continue  # already extracted, this run or a previous one
            result = compute_orb(signature.path, detector)
            if result is not None:
                signature.orb, signature.width, signature.height = result
                self.cache.put(signature)
            if done % self._PROGRESS_EVERY == 0:
                self._report(done, len(involved), 'Extracting features')
        self._report(len(involved), len(involved), 'Extracting features')

        matcher = create_orb_matcher()
        pairs: list[PairScore] = []
        for done, pair in enumerate(shortlist, start=1):
            if self.is_cancelled:
                break
            a, b = by_key.get(pair.a), by_key.get(pair.b)
            score = orb_similarity(a.orb if a else None,
                                   b.orb if b else None, matcher)
            if score >= min_score:
                pairs.append(PairScore(pair.a, pair.b,
                                       hash_distance=pair.hash_distance,
                                       orb_score=score))
            if done % self._PROGRESS_EVERY == 0:
                self._report(done, len(shortlist), 'Verifying candidates')
        self._report(len(shortlist), len(shortlist), 'Verifying candidates')
        return pairs

    # ------------------------------------------------------------------
    # Pair enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def _split(signatures: list[ImageSignature]) -> tuple[list[int], list[int]]:
        originals = [i for i, s in enumerate(signatures) if s.is_original]
        others = [i for i, s in enumerate(signatures) if not s.is_original]
        return originals, others

    def _iter_pairs(self, signatures: list[ImageSignature]
                    ) -> Iterator[tuple[int, int]]:
        originals, others = self._split(signatures)
        if originals:
            for index in originals:
                for other in others:
                    yield index, other
            return
        count = len(signatures)
        for a in range(count - 1):
            for b in range(a + 1, count):
                yield a, b

    def _pair_count(self, signatures: list[ImageSignature]) -> int:
        originals, others = self._split(signatures)
        if originals:
            return len(originals) * len(others)
        count = len(signatures)
        return count * (count - 1) // 2

    # ------------------------------------------------------------------

    def _report(self, current: int, total: int, phase: str):
        self.progress.emit(current, max(total, 1))
        self.status.emit(f'{phase} {current}/{max(total, 1)}')
