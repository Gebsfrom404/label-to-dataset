"""Background duplicate search over one or more source folders.

Phases (each reports through ``status`` / ``progress``):

1. **Scan** — recurse every source folder, de-duplicating paths so nested
   sources are not compared with themselves.
2. **Signatures** — one pHash and/or ORB signature per image.
3. **Compare** — pairwise. When a source is flagged *original*, only
   original-vs-other pairs are tested; otherwise every pair is.
4. **Group** — union-find (symmetric mode) or best-original assignment.

Unreadable images and images with too few keypoints are dropped after the
signature phase and reported through ``skipped``.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal

from ltd.data.image_list_model import IMAGE_EXTENSIONS
from ltd.utils.duplicate_utils import (ALGO_HYBRID, ALGO_ORB, ALGO_PHASH,
                                       MASK_SUFFIX, DuplicateGroup,
                                       DuplicateMember,
                                       ImageSignature, compute_orb,
                                       compute_phash, create_orb_detector,
                                       create_orb_matcher, group_pairs,
                                       hamming_distances,
                                       hybrid_shortlist_distance,
                                       orb_min_score, orb_similarity,
                                       phash_max_distance, phash_score)
from ltd.workers.base_worker import BaseWorker


class DuplicateWorker(BaseWorker):
    """Finds duplicate images across the given sources."""

    groups_found = Signal(list)  # list[DuplicateGroup]
    scanned = Signal(int)        # images considered
    skipped = Signal(int)        # images without a usable signature

    _PROGRESS_EVERY = 25

    def __init__(self, sources: list[tuple[Path, bool]], algorithm: str,
                 tolerance: int, parent=None):
        """``sources`` is a list of ``(folder, is_original)``."""
        super().__init__(parent)
        self.sources = sources
        self.algorithm = algorithm
        self.tolerance = tolerance

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

        signatures = self._compute_signatures(
            entries,
            want_phash=self.algorithm in (ALGO_PHASH, ALGO_HYBRID),
            want_orb=self.algorithm == ALGO_ORB)
        if self.is_cancelled:
            self.groups_found.emit([])
            return
        self.skipped.emit(len(entries) - len(signatures))
        if len(signatures) < 2:
            self.groups_found.emit([])
            return

        if self.algorithm == ALGO_PHASH:
            pairs = self._match_phash(
                signatures, phash_max_distance(self.tolerance))
        elif self.algorithm == ALGO_ORB:
            pairs = self._match_orb(signatures, orb_min_score(self.tolerance))
        else:
            shortlist = self._match_phash(
                signatures, hybrid_shortlist_distance(self.tolerance))
            pairs = self._verify_shortlist(
                signatures, shortlist, orb_min_score(self.tolerance))

        if self.is_cancelled:
            self.groups_found.emit([])
            return
        self.status.emit('Grouping matches...')
        self.groups_found.emit(self._build_groups(signatures, pairs))

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
                key = os.path.normcase(os.path.abspath(str(file)))
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

    def _compute_signatures(self, entries: list[tuple[Path, bool]],
                            want_phash: bool,
                            want_orb: bool) -> list[ImageSignature]:
        total = len(entries)
        detector = create_orb_detector() if want_orb else None
        signatures: list[ImageSignature] = []
        for index, (path, is_original) in enumerate(entries):
            if self.is_cancelled:
                break
            signature = ImageSignature(path=path, is_original=is_original)
            if want_phash:
                result = compute_phash(path)
                if result is not None:
                    signature.phash, signature.width, signature.height = result
            if want_orb:
                result = compute_orb(path, detector)
                if result is not None:
                    signature.orb, signature.width, signature.height = result
            if self._is_usable(signature, want_phash, want_orb):
                signatures.append(signature)
            if index % self._PROGRESS_EVERY == 0:
                self._report(index + 1, total, 'Reading images')
        self._report(total, total, 'Reading images')
        return signatures

    @staticmethod
    def _is_usable(signature: ImageSignature, want_phash: bool,
                   want_orb: bool) -> bool:
        if want_phash and signature.phash is None:
            return False
        if want_orb and signature.orb is None:
            return False
        return True

    # ------------------------------------------------------------------
    # 3a. Perceptual hash comparison (vectorised per row)
    # ------------------------------------------------------------------

    def _match_phash(self, signatures: list[ImageSignature],
                     max_distance: int) -> list[tuple[int, int, float]]:
        # The signature phase drops anything without a hash, so the filter
        # here keeps every row — it just states that invariant.
        hashes = np.stack([s.phash for s in signatures if s.phash is not None])
        originals, others = self._split(signatures)
        pairs: list[tuple[int, int, float]] = []

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
                    distance = int(distances[hit])
                    pairs.append((index, int(targets[hit]),
                                  phash_score(distance)))
                self._report(step + 1, len(originals), 'Comparing hashes')
            return pairs

        count = len(signatures)
        for index in range(count - 1):
            if self.is_cancelled:
                break
            distances = hamming_distances(hashes[index], hashes[index + 1:])
            for hit in np.nonzero(distances <= max_distance)[0]:
                distance = int(distances[hit])
                pairs.append((index, index + 1 + int(hit),
                              phash_score(distance)))
            if index % self._PROGRESS_EVERY == 0:
                self._report(index + 1, count, 'Comparing hashes')
        self._report(count, count, 'Comparing hashes')
        return pairs

    # ------------------------------------------------------------------
    # 3b. Descriptor comparison (every pair is real work)
    # ------------------------------------------------------------------

    def _match_orb(self, signatures: list[ImageSignature],
                   min_score: float) -> list[tuple[int, int, float]]:
        matcher = create_orb_matcher()
        total = self._pair_count(signatures)
        pairs: list[tuple[int, int, float]] = []
        for done, (a, b) in enumerate(self._iter_pairs(signatures), start=1):
            if self.is_cancelled:
                break
            score = orb_similarity(signatures[a].orb, signatures[b].orb,
                                   matcher)
            if score >= min_score:
                pairs.append((a, b, score))
            if done % self._PROGRESS_EVERY == 0:
                self._report(done, total, 'Matching features')
        self._report(total, total, 'Matching features')
        return pairs

    def _verify_shortlist(self, signatures: list[ImageSignature],
                          shortlist: list[tuple[int, int, float]],
                          min_score: float) -> list[tuple[int, int, float]]:
        """Re-score pHash candidates with descriptors, dropping the rest."""
        if not shortlist or self.is_cancelled:
            return []
        involved = sorted({index for a, b, _ in shortlist for index in (a, b)})
        detector = create_orb_detector()
        for done, index in enumerate(involved, start=1):
            if self.is_cancelled:
                return []
            signature = signatures[index]
            result = compute_orb(signature.path, detector)
            if result is not None:
                signature.orb, signature.width, signature.height = result
            if done % self._PROGRESS_EVERY == 0:
                self._report(done, len(involved), 'Extracting features')
        self._report(len(involved), len(involved), 'Extracting features')

        matcher = create_orb_matcher()
        pairs: list[tuple[int, int, float]] = []
        for done, (a, b, _) in enumerate(shortlist, start=1):
            if self.is_cancelled:
                break
            score = orb_similarity(signatures[a].orb, signatures[b].orb,
                                   matcher)
            if score >= min_score:
                pairs.append((a, b, score))
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
    # 4. Grouping
    # ------------------------------------------------------------------

    def _build_groups(self, signatures: list[ImageSignature],
                      pairs: list[tuple[int, int, float]]
                      ) -> list[DuplicateGroup]:
        if any(s.is_original for s in signatures):
            return self._build_original_groups(signatures, pairs)
        return self._build_symmetric_groups(signatures, pairs)

    def _build_original_groups(self, signatures: list[ImageSignature],
                               pairs: list[tuple[int, int, float]]
                               ) -> list[DuplicateGroup]:
        # An image can match several originals — keep only its best match.
        best: dict[int, tuple[int, float]] = {}
        for a, b, score in pairs:
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
            members = [self._member(signatures[original], 1.0)]
            duplicates = sorted(
                by_original[original],
                key=lambda item: (-item[1], str(signatures[item[0]].path).lower()))
            members.extend(self._member(signatures[index], score)
                           for index, score in duplicates)
            groups.append(DuplicateGroup(members))
        return groups

    def _build_symmetric_groups(self, signatures: list[ImageSignature],
                                pairs: list[tuple[int, int, float]]
                                ) -> list[DuplicateGroup]:
        best_score: dict[int, float] = {}
        for a, b, score in pairs:
            best_score[a] = max(best_score.get(a, 0.0), score)
            best_score[b] = max(best_score.get(b, 0.0), score)

        groups = []
        for indices in group_pairs(len(signatures), pairs):
            ordered = sorted(indices,
                             key=lambda i: str(signatures[i].path).lower())
            groups.append(DuplicateGroup(
                [self._member(signatures[i], best_score.get(i, 0.0))
                 for i in ordered]))
        return groups

    @staticmethod
    def _member(signature: ImageSignature, score: float) -> DuplicateMember:
        return DuplicateMember(path=signature.path, score=score,
                               is_original=signature.is_original,
                               width=signature.width, height=signature.height)

    # ------------------------------------------------------------------

    def _report(self, current: int, total: int, phase: str):
        self.progress.emit(current, max(total, 1))
        self.status.emit(f'{phase} {current}/{max(total, 1)}')
