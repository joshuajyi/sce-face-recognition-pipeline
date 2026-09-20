"""Small, testable vector operations used by face matching."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_embedding(embedding: ArrayLike) -> NDArray[np.float32]:
    """Return a flattened L2-normalized embedding.

    Normalizing once makes cosine similarity equal to a dot product. A zero
    vector is rejected because it does not contain a usable direction.
    """

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError("Embedding cannot be empty")

    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("Embedding cannot be a zero vector")
    return vector / norm


def cosine_similarity(left: ArrayLike, right: ArrayLike) -> float:
    """Compute cosine similarity between two embeddings."""

    left_normalized = normalize_embedding(left)
    right_normalized = normalize_embedding(right)
    if left_normalized.shape != right_normalized.shape:
        raise ValueError("Embeddings must have the same size")
    return float(np.dot(left_normalized, right_normalized))

