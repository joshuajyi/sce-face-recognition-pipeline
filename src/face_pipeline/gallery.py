"""Local storage and nearest-profile matching for face embeddings."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from face_pipeline.matching import normalize_embedding


@dataclass(frozen=True)
class Match:
    """The best gallery match for one query embedding."""

    name: str | None
    score: float
    is_known: bool

    @property
    def label(self) -> str:
        return self.name if self.is_known and self.name else "Unknown"


class Gallery:
    """A small local gallery containing one normalized vector per person."""

    def __init__(self, profiles: dict[str, ArrayLike] | None = None) -> None:
        self._profiles: dict[str, NDArray[np.float32]] = {}
        for name, embedding in (profiles or {}).items():
            self._profiles[self._clean_name(name)] = normalize_embedding(embedding)

    @staticmethod
    def _clean_name(name: str) -> str:
        clean = " ".join(name.strip().split())
        if not clean:
            raise ValueError("Profile name cannot be empty")
        if len(clean) > 80:
            raise ValueError("Profile name must be 80 characters or fewer")
        return clean

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def __len__(self) -> int:
        return len(self._profiles)

    def enroll(self, name: str, embeddings: Iterable[ArrayLike]) -> NDArray[np.float32]:
        """Average several reference embeddings into one normalized profile."""

        clean_name = self._clean_name(name)
        normalized = [normalize_embedding(item) for item in embeddings]
        if not normalized:
            raise ValueError("At least one embedding is required for enrollment")

        expected_shape = normalized[0].shape
        if any(item.shape != expected_shape for item in normalized):
            raise ValueError("All enrollment embeddings must have the same size")

        profile = normalize_embedding(np.mean(np.stack(normalized), axis=0))
        self._profiles[clean_name] = profile
        return profile.copy()

    def delete(self, name: str) -> bool:
        return self._profiles.pop(name, None) is not None

    def clear(self) -> None:
        self._profiles.clear()

    def match(self, embedding: ArrayLike, threshold: float) -> Match:
        """Return the nearest profile, or Unknown when it misses the threshold."""

        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Cosine threshold must be between -1 and 1")
        if not self._profiles:
            return Match(name=None, score=-1.0, is_known=False)

        query = normalize_embedding(embedding)
        names = sorted(self._profiles)
        matrix = np.stack([self._profiles[name] for name in names])
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("Query embedding size does not match the gallery")

        scores = matrix @ query
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        return Match(
            name=names[best_index],
            score=best_score,
            is_known=best_score >= threshold,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = np.asarray(sorted(self._profiles), dtype=np.str_)
        embeddings = (
            np.stack([self._profiles[name] for name in names])
            if names.size
            else np.empty((0, 0), dtype=np.float32)
        )
        np.savez_compressed(path, names=names, embeddings=embeddings)

    @classmethod
    def load(cls, path: Path) -> Gallery:
        if not path.exists():
            return cls()

        with np.load(path, allow_pickle=False) as archive:
            if "names" not in archive or "embeddings" not in archive:
                raise ValueError(f"Invalid gallery file: {path}")
            names = archive["names"]
            embeddings = archive["embeddings"]

        if names.ndim != 1 or embeddings.ndim != 2 or len(names) != len(embeddings):
            raise ValueError(f"Invalid gallery shape in: {path}")
        return cls(
            {str(name): embedding for name, embedding in zip(names, embeddings, strict=True)}
        )
