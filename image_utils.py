"""Image, configuration, and file-system utility functions."""

from __future__ import annotations

import json
import os
import platform
import re
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from glob import glob, has_magic
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
import yaml
from matplotlib import colors as matplotlib_colors


VIDEO_EXTENSIONS: tuple[str, ...] = ("mp4", "avi", "mov", "mkv", "flv", "lrf")
IMAGE_EXTENSIONS: tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "heic")

PathLike: TypeAlias = str | os.PathLike[str]
BoundingBox: TypeAlias = Sequence[float]
TextPosition: TypeAlias = str | Sequence[int]
Rectangle: TypeAlias = tuple[int, int, int, int]
OccupiedAreas: TypeAlias = MutableMapping[Hashable, list[Rectangle]]
Color: TypeAlias = str | Sequence[int]

__all__ = [
    "VIDEO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "named_color_to_rgb",
    "named_color_to_bgr",
    "hex_color_to_bgr",
    "calculate_bounding_box_iou",
    "calculate_text_style",
    "draw_text_without_overlap",
    "annotate_images_with_labels",
    "extract_first_number",
    "resize_images_to_height",
    "collect_input_files",
    "load_configuration",
    "create_unique_directory",
    "is_display_available",
    "resize_image",
    "get_file_size_megabytes",
]


def named_color_to_rgb(color: Any) -> Any:
    """Convert a named color to an RGB tuple in the range 0-255.

    Non-string values are returned unchanged.
    """
    if not isinstance(color, str):
        return color
    return tuple(round(channel * 255) for channel in matplotlib_colors.to_rgb(color))


def named_color_to_bgr(color: Any) -> Any:
    """Convert a named color to an OpenCV BGR tuple.

    Non-string values are returned unchanged.
    """
    if not isinstance(color, str):
        return color
    return tuple(reversed(named_color_to_rgb(color)))


def hex_color_to_bgr(hex_color: Any) -> Any:
    """Convert ``#RRGGBB`` or ``RRGGBB`` to an OpenCV BGR tuple."""
    if not isinstance(hex_color, str):
        return hex_color

    normalized_color = hex_color.removeprefix("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", normalized_color):
        raise ValueError(
            f"Expected a six-digit hexadecimal color, received {hex_color!r}."
        )

    red = int(normalized_color[0:2], 16)
    green = int(normalized_color[2:4], 16)
    blue = int(normalized_color[4:6], 16)
    return blue, green, red


def calculate_bounding_box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate intersection over union for two inclusive bounding boxes.

    Each box must contain ``(x_min, y_min, x_max, y_max)``.
    """
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Each bounding box must contain exactly four coordinates.")

    intersection_left = max(box_a[0], box_b[0])
    intersection_top = max(box_a[1], box_b[1])
    intersection_right = min(box_a[2], box_b[2])
    intersection_bottom = min(box_a[3], box_b[3])

    intersection_width = max(0.0, intersection_right - intersection_left + 1)
    intersection_height = max(0.0, intersection_bottom - intersection_top + 1)
    intersection_area = intersection_width * intersection_height

    box_a_area = max(0.0, box_a[2] - box_a[0] + 1) * max(
        0.0, box_a[3] - box_a[1] + 1
    )
    box_b_area = max(0.0, box_b[2] - box_b[0] + 1) * max(
        0.0, box_b[3] - box_b[1] + 1
    )

    union_area = box_a_area + box_b_area - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def calculate_text_style(
    image: np.ndarray,
    base_height: int = 100,
    base_width: int = 100,
    base_font_scale: float = 0.5,
    scaling_mode: str = "average",
) -> tuple[float, int, int]:
    """Calculate font scale, text thickness, and outline thickness for an image."""
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions.")
    if base_height <= 0 or base_width <= 0 or base_font_scale <= 0:
        raise ValueError("Base dimensions and font scale must be positive.")

    image_height, image_width = image.shape[:2]
    height_ratio = image_height / base_height
    width_ratio = image_width / base_width
    mode = scaling_mode.lower()

    if mode == "height":
        font_scale = base_font_scale * height_ratio
    elif mode == "width":
        font_scale = base_font_scale * width_ratio
    elif mode == "average":
        average_ratio = (height_ratio + width_ratio) / 2
        font_scale = max(
            0.3 * base_font_scale * average_ratio,
            base_font_scale * 0.6,
        )
    elif mode == "diagonal":
        image_diagonal = float(np.hypot(image_height, image_width))
        base_diagonal = float(np.hypot(base_height, base_width))
        font_scale = base_font_scale * image_diagonal / base_diagonal
    elif mode == "adaptive":
        image_diagonal = float(np.hypot(image_height, image_width))
        base_diagonal = float(np.hypot(base_height, base_width))
        font_scale = max(
            base_font_scale * (image_diagonal / base_diagonal) ** 0.5,
            base_font_scale * 0.6,
        )
    elif mode in {"fixed", "none"}:
        font_scale = base_font_scale
    else:
        raise ValueError(
            "scaling_mode must be height, width, average, diagonal, adaptive, "
            "or fixed."
        )

    text_thickness = max(1, round(font_scale * 2))
    outline_thickness = max(2, round(font_scale * 8))
    return font_scale, text_thickness, outline_thickness


def _resolve_text_position(
    image: np.ndarray,
    position: TextPosition,
    text_width: int,
    text_height: int,
    margin: int,
) -> tuple[int, int]:
    """Convert a named or numeric position to an OpenCV text origin."""
    if not isinstance(position, str):
        if len(position) != 2:
            raise ValueError("A numeric text position must contain two values.")
        return int(position[0]), int(position[1])

    image_height, image_width = image.shape[:2]
    normalized_position = position.lower().replace("_", "-").replace(" ", "-")

    if "top" in normalized_position:
        y_coordinate = margin + text_height
    elif "bottom" in normalized_position:
        y_coordinate = image_height - margin
    else:
        y_coordinate = (image_height + text_height) // 2

    if "left" in normalized_position:
        x_coordinate = margin
    elif "right" in normalized_position:
        x_coordinate = image_width - text_width - margin
    else:
        x_coordinate = (image_width - text_width) // 2

    return x_coordinate, y_coordinate


def _create_text_rectangle(
    origin: tuple[int, int],
    text_width: int,
    text_height: int,
    baseline: int,
    margin: int,
) -> Rectangle:
    x_coordinate, y_coordinate = origin
    return (
        x_coordinate - margin,
        y_coordinate - text_height - margin,
        x_coordinate + text_width + margin,
        y_coordinate + baseline + margin,
    )


def _rectangles_overlap(first: Rectangle, second: Rectangle) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _find_free_text_position(
    image: np.ndarray,
    preferred_origin: tuple[int, int],
    text_width: int,
    text_height: int,
    baseline: int,
    margin: int,
    occupied_rectangles: Sequence[Rectangle],
) -> tuple[tuple[int, int], Rectangle]:
    image_height, image_width = image.shape[:2]
    maximum_x = max(margin, image_width - text_width - margin)
    x_coordinate = min(max(preferred_origin[0], margin), maximum_x)
    preferred_y = min(
        max(preferred_origin[1], text_height + margin),
        max(text_height + margin, image_height - baseline - margin),
    )

    vertical_step = max(1, text_height + baseline + margin)
    candidate_y_values = [preferred_y]
    for step_index in range(1, image_height // vertical_step + 2):
        candidate_y_values.extend(
            (
                preferred_y + step_index * vertical_step,
                preferred_y - step_index * vertical_step,
            )
        )

    for y_coordinate in candidate_y_values:
        if y_coordinate - text_height < margin:
            continue
        if y_coordinate + baseline > image_height - margin:
            continue

        origin = x_coordinate, y_coordinate
        rectangle = _create_text_rectangle(
            origin,
            text_width,
            text_height,
            baseline,
            margin,
        )
        if not any(
            _rectangles_overlap(rectangle, occupied)
            for occupied in occupied_rectangles
        ):
            return origin, rectangle

    fallback_origin = x_coordinate, preferred_y
    return fallback_origin, _create_text_rectangle(
        fallback_origin,
        text_width,
        text_height,
        baseline,
        margin,
    )


def draw_text_without_overlap(
    image: np.ndarray,
    text: str,
    position: TextPosition,
    text_color: Color = "white",
    occupied_areas: OccupiedAreas | None = None,
    margin: int = 5,
    base_font_scale: float = 0.5,
    image_key: Hashable | None = None,
) -> OccupiedAreas:
    """Draw outlined text and track occupied rectangles to reduce overlap.

    ``position`` may be a numeric ``(x, y)`` origin or a label such as
    ``"top-left"``, ``"center"``, or ``"bottom-right"``.
    """
    if margin < 0:
        raise ValueError("margin cannot be negative.")

    tracked_areas = occupied_areas if occupied_areas is not None else {}
    tracking_key = image_key if image_key is not None else id(image)
    occupied_rectangles = tracked_areas.setdefault(tracking_key, [])

    font_scale, text_thickness, outline_thickness = calculate_text_style(
        image,
        base_font_scale=base_font_scale,
    )
    (text_width, text_height), baseline = cv2.getTextSize(
        str(text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness,
    )

    preferred_origin = _resolve_text_position(
        image,
        position,
        text_width,
        text_height,
        margin,
    )
    final_origin, occupied_rectangle = _find_free_text_position(
        image,
        preferred_origin,
        text_width,
        text_height,
        baseline,
        margin,
        occupied_rectangles,
    )
    occupied_rectangles.append(occupied_rectangle)

    cv2.putText(
        image,
        str(text),
        final_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(text),
        final_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        named_color_to_bgr(text_color),
        text_thickness,
        lineType=cv2.LINE_AA,
    )
    return tracked_areas


def annotate_images_with_labels(
    images: Iterable[np.ndarray],
    labels: Sequence[Any] | None = None,
) -> list[np.ndarray]:
    """Add a top border and an index or custom label to each image."""
    image_list = list(images)
    if labels is not None and len(labels) != len(image_list):
        raise ValueError("labels must contain one value for each image.")

    annotated_images: list[np.ndarray] = []
    for image_index, image in enumerate(image_list):
        border_height = max(round(image.shape[0] * 0.15), 15)
        annotated_image = cv2.copyMakeBorder(
            image,
            border_height,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        label = image_index if labels is None else labels[image_index]
        font_scale, text_thickness, _ = calculate_text_style(annotated_image)

        cv2.putText(
            annotated_image,
            str(label),
            (1, round(border_height * 0.95)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 200, 200),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        annotated_images.append(annotated_image)

    return annotated_images


def extract_first_number(path: PathLike, default: int | None = None) -> int | None:
    """Extract the first integer from a file name, or return ``default``."""
    match = re.search(r"\d+", Path(path).name)
    return int(match.group()) if match else default


def resize_images_to_height(
    images: Iterable[np.ndarray],
    target_height: int,
) -> list[np.ndarray]:
    """Resize images to one height while preserving their aspect ratios."""
    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    resized_images: list[np.ndarray] = []
    for image in images:
        current_height, current_width = image.shape[:2]
        if current_height == target_height:
            resized_images.append(image)
            continue

        scale = target_height / current_height
        target_width = max(1, round(current_width * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized_images.append(
            cv2.resize(image, (target_width, target_height), interpolation=interpolation)
        )

    return resized_images


def collect_input_files(
    input_paths: PathLike | Iterable[PathLike],
    *,
    recursive: bool = True,
) -> list[str]:
    """Resolve files, directories, and glob patterns into a sorted file list."""
    if isinstance(input_paths, (str, os.PathLike)):
        path_values: list[PathLike] = [input_paths]
    else:
        path_values = list(input_paths)

    collected_files: list[str] = []
    for path_value in path_values:
        path_text = os.fspath(path_value)
        path = Path(path_text)

        if has_magic(path_text):
            matches = glob(path_text, recursive=recursive)
            collected_files.extend(match for match in matches if Path(match).is_file())
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            collected_files.extend(str(item) for item in iterator if item.is_file())
        else:
            collected_files.append(path_text)

    return sorted(dict.fromkeys(collected_files))


def _normalize_configuration_keys(value: Any) -> Any:
    """Recursively replace hyphens with underscores in mapping keys."""
    if isinstance(value, Mapping):
        return {
            str(key).replace("-", "_"): _normalize_configuration_keys(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_configuration_keys(item) for item in value]
    return value


def load_configuration(path: PathLike) -> dict[str, Any]:
    """Load a JSON or YAML configuration and normalize keys to snake_case."""
    configuration_path = Path(path)
    suffix = configuration_path.suffix.lower()

    with configuration_path.open("r", encoding="utf-8") as configuration_file:
        if suffix in {".yaml", ".yml"}:
            loaded_data = yaml.safe_load(configuration_file)
        elif suffix == ".json":
            loaded_data = json.load(configuration_file)
        else:
            raise ValueError("Configuration files must use .json, .yaml, or .yml.")

    if loaded_data is None:
        return {}
    if not isinstance(loaded_data, Mapping):
        raise ValueError("The configuration root must be a mapping/object.")

    return _normalize_configuration_keys(loaded_data)


def create_unique_directory(base_directory: PathLike) -> Path:
    """Create an empty directory, adding ``_1``, ``_2``, and so on if needed."""
    base_path = Path(base_directory)
    base_path.mkdir(parents=True, exist_ok=True)

    if not any(base_path.iterdir()):
        return base_path

    suffix_number = 1
    while True:
        candidate_path = base_path.with_name(f"{base_path.name}_{suffix_number}")
        try:
            candidate_path.mkdir(parents=True, exist_ok=False)
            return candidate_path
        except FileExistsError:
            if candidate_path.is_dir() and not any(candidate_path.iterdir()):
                return candidate_path
            suffix_number += 1


def is_display_available() -> bool:
    """Return whether a graphical display is likely available."""
    if platform.system() == "Windows":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _positive_dimension(value: int | float, name: str) -> int:
    dimension = int(value)
    if dimension <= 0:
        raise ValueError(f"{name} must be positive.")
    return dimension


def _make_even(value: int) -> int:
    return value if value <= 1 else value - value % 2


def _create_canvas(
    image: np.ndarray,
    height: int,
    width: int,
    padding_color: int | Sequence[int],
) -> np.ndarray:
    if image.ndim == 2:
        if isinstance(padding_color, Sequence) and not isinstance(padding_color, str):
            padding_color = int(padding_color[0])
        return np.full((height, width), padding_color, dtype=image.dtype)

    channel_count = image.shape[2]
    if isinstance(padding_color, Sequence) and not isinstance(padding_color, str):
        if len(padding_color) != channel_count:
            raise ValueError(
                f"padding_color must contain {channel_count} values for this image."
            )
    return np.full((height, width, channel_count), padding_color, dtype=image.dtype)


def resize_image(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    padding_color: int | Sequence[int] = (0, 0, 0),
    *,
    force_even_dimensions: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Resize an image while preserving aspect ratio.

    Supplying one dimension performs a proportional resize. Supplying both creates
    an exact letterboxed canvas. The return value is ``(image, y_scale, x_scale)``.
    """
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions.")

    original_height, original_width = image.shape[:2]
    if width is None and height is None:
        return image, 1.0, 1.0

    target_width = _positive_dimension(width, "width") if width is not None else None
    target_height = _positive_dimension(height, "height") if height is not None else None

    if target_width is None or target_height is None:
        if target_width is None:
            resized_height = target_height
            resized_width = max(
                1,
                round(original_width * resized_height / original_height),
            )
        else:
            resized_width = target_width
            resized_height = max(
                1,
                round(original_height * resized_width / original_width),
            )

        if force_even_dimensions:
            resized_width = _make_even(resized_width)
            resized_height = _make_even(resized_height)

        scale_x = resized_width / original_width
        scale_y = resized_height / original_height
        interpolation = cv2.INTER_AREA if max(scale_x, scale_y) < 1 else cv2.INTER_LINEAR
        resized_image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        return resized_image, scale_y, scale_x

    content_scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * content_scale))
    resized_height = max(1, round(original_height * content_scale))

    if force_even_dimensions:
        resized_width = min(target_width, _make_even(resized_width))
        resized_height = min(target_height, _make_even(resized_height))

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    interpolation = cv2.INTER_AREA if content_scale < 1 else cv2.INTER_LINEAR
    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    output_image = _create_canvas(
        image,
        target_height,
        target_width,
        padding_color,
    )
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    output_image[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_image
    return output_image, scale_y, scale_x


def get_file_size_megabytes(path: PathLike, decimal_places: int = 2) -> float:
    """Return the file size in MiB, rounded to ``decimal_places``."""
    if decimal_places < 0:
        raise ValueError("decimal_places cannot be negative.")
    return round(Path(path).stat().st_size / (1024 * 1024), decimal_places)
