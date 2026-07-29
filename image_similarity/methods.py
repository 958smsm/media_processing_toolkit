"""Feature extraction and comparison strategies for image similarity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

FeatureExtractor = Callable[[Path], Any]
SimilarityFunction = Callable[[Any, Any], float]


@dataclass(frozen=True)
class SimilarityMethod:
    """Configuration for one image similarity strategy."""

    name: str
    output_suffix: str
    default_threshold: float
    extract: FeatureExtractor
    compare: SimilarityFunction


def _extract_average_hash(image_path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=float)
    return pixels > pixels.mean()


def _compare_average_hash(left: Any, right: Any) -> float:
    import numpy as np

    bit_count = int(left.size)
    if bit_count == 0:
        return 1.0
    hamming_distance = int(np.count_nonzero(left != right))
    return 1.0 - (hamming_distance / bit_count)


def _extract_mean_color(image_path: Path) -> Any:
    import numpy as np
    from PIL import Image, ImageStat

    with Image.open(image_path) as image:
        return np.asarray(ImageStat.Stat(image.convert("RGB")).mean, dtype=float)


def _compare_mean_color(left: Any, right: Any) -> float:
    import numpy as np

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left_norm == right_norm else 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _extract_hsv_histogram(image_path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        hsv_image = image.convert("RGB").resize((256, 256)).convert("HSV")
        hsv_pixels = np.asarray(hsv_image)

    histogram, _, _ = np.histogram2d(
        hsv_pixels[..., 0].ravel(),
        hsv_pixels[..., 1].ravel(),
        bins=(50, 60),
        range=((0, 256), (0, 256)),
    )
    flattened = histogram.ravel()
    norm = float(np.linalg.norm(flattened))
    return flattened if norm == 0.0 else flattened / norm


def _compare_histograms(left: Any, right: Any) -> float:
    import numpy as np

    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


METHODS = {
    "average-hash": SimilarityMethod(
        name="average-hash",
        output_suffix="_averagehash",
        default_threshold=0.95,
        extract=_extract_average_hash,
        compare=_compare_average_hash,
    ),
    "histogram": SimilarityMethod(
        name="histogram",
        output_suffix="_histogram",
        default_threshold=0.985,
        extract=_extract_hsv_histogram,
        compare=_compare_histograms,
    ),
    "mean-color": SimilarityMethod(
        name="mean-color",
        output_suffix="_meanstat",
        default_threshold=0.95,
        extract=_extract_mean_color,
        compare=_compare_mean_color,
    ),
}

DEFAULT_METHOD = "average-hash"


def get_method(name: str) -> SimilarityMethod:
    """Return a configured similarity method by name."""

    try:
        return METHODS[name]
    except KeyError as error:
        supported = ", ".join(sorted(METHODS))
        raise ValueError(
            f"Unknown similarity method {name!r}. Choose one of: {supported}."
        ) from error
