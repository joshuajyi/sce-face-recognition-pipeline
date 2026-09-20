from pathlib import Path

import numpy as np
import pytest

from face_pipeline.gallery import Gallery


def test_gallery_accepts_known_and_rejects_unknown() -> None:
    gallery = Gallery()
    gallery.enroll("Alice", [[1.0, 0.0], [0.98, 0.02]])

    known = gallery.match([1.0, 0.0], threshold=0.8)
    unknown = gallery.match([0.0, 1.0], threshold=0.8)

    assert known.label == "Alice"
    assert known.is_known
    assert known.score > 0.99
    assert unknown.label == "Unknown"
    assert not unknown.is_known


def test_nearest_profile_is_selected() -> None:
    gallery = Gallery({"Alice": [1.0, 0.0], "Bob": [0.0, 1.0]})
    result = gallery.match([0.1, 0.9], threshold=0.5)
    assert result.label == "Bob"


def test_gallery_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gallery.npz"
    original = Gallery()
    original.enroll("Alice Example", [[1.0, 2.0, 3.0]])
    original.save(path)

    loaded = Gallery.load(path)
    assert loaded.names == ("Alice Example",)
    result = loaded.match([1.0, 2.0, 3.0], threshold=0.9)
    assert result.label == "Alice Example"


def test_gallery_file_does_not_require_pickle(tmp_path: Path) -> None:
    path = tmp_path / "gallery.npz"
    Gallery({"Alice": [1.0, 0.0]}).save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["names"].dtype.kind == "U"


def test_empty_gallery_returns_unknown() -> None:
    result = Gallery().match([1.0, 0.0], threshold=0.5)
    assert result.label == "Unknown"
    assert not result.is_known


def test_invalid_threshold_is_rejected() -> None:
    gallery = Gallery({"Alice": [1.0, 0.0]})
    with pytest.raises(ValueError, match="between -1 and 1"):
        gallery.match([1.0, 0.0], threshold=2.0)
