import numpy as np
import pytest

from face_pipeline.matching import cosine_similarity, normalize_embedding


def test_normalize_embedding_has_unit_length() -> None:
    result = normalize_embedding([3.0, 4.0])
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_cosine_similarity_known_cases() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        normalize_embedding([0, 0, 0])


def test_different_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same size"):
        cosine_similarity([1, 0], [1, 0, 0])
