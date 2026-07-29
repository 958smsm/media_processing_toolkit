"""Shared discovery, clustering, and export logic."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .methods import DEFAULT_METHOD, SimilarityFunction, get_method

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif"})


@dataclass(frozen=True)
class ExtractionError:
    """An image that could not be decoded or processed."""

    image_path: Path
    message: str


@dataclass(frozen=True)
class ClusterResult:
    """Summary returned by :func:`cluster_images`."""

    output_dir: Path
    discovered_count: int
    processed_count: int
    clusters: tuple[tuple[Path, ...], ...]
    errors: tuple[ExtractionError, ...]
    exported_cluster_count: int
    exported_image_count: int

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)


def discover_images(source_dir: Path | str, recursive: bool = True) -> list[Path]:
    """Find supported images in a directory in deterministic order."""

    source = Path(source_dir).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source}")

    candidates = source.rglob("*") if recursive else source.glob("*")
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def extract_features(
    image_paths: Iterable[Path],
    extractor: Any,
    *,
    show_progress: bool = True,
    description: str = "Extracting image features",
) -> tuple[dict[Path, Any], list[ExtractionError]]:
    """Extract features while collecting per-file errors."""

    paths = list(image_paths)
    iterator: Iterable[Path] = paths
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(paths, desc=description)

    features: dict[Path, Any] = {}
    errors: list[ExtractionError] = []
    for image_path in iterator:
        try:
            features[image_path] = extractor(image_path)
        except ModuleNotFoundError:
            raise
        except Exception as error:  # One bad image should not abort a batch.
            errors.append(ExtractionError(image_path, str(error)))
    return features, errors


def cluster_features(
    features: Mapping[Path, Any],
    compare: SimilarityFunction,
    threshold: float,
) -> list[list[Path]]:
    """Greedily group each unvisited image with matches to its seed image.

    This intentionally retains the behavior of the original scripts: the first
    unvisited image becomes a cluster seed, and each later image is compared
    with that seed. Results are deterministic when ``features`` is ordered.
    """

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be between -1.0 and 1.0.")

    visited: set[Path] = set()
    clusters: list[list[Path]] = []
    image_paths = list(features)

    for seed_path in image_paths:
        if seed_path in visited:
            continue

        cluster = [seed_path]
        visited.add(seed_path)
        seed_feature = features[seed_path]

        for candidate_path in image_paths:
            if candidate_path in visited:
                continue
            similarity = compare(seed_feature, features[candidate_path])
            if similarity >= threshold:
                cluster.append(candidate_path)
                visited.add(candidate_path)

        clusters.append(cluster)

    return clusters


def export_clusters(
    clusters: Sequence[Sequence[Path]],
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    move: bool = False,
    include_singletons: bool = True,
) -> int:
    """Copy or move clustered images while preserving source subdirectories."""

    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    operation = shutil.move if move else shutil.copy2
    exported_count = 0

    output.mkdir(parents=True, exist_ok=True)
    exported_cluster_index = 0
    for cluster in clusters:
        if len(cluster) == 1 and not include_singletons:
            continue

        exported_cluster_index += 1
        cluster_dir = output / f"cluster_{exported_cluster_index:04d}"
        for image_path in cluster:
            resolved_image = Path(image_path).resolve()
            try:
                relative_path = resolved_image.relative_to(source)
            except ValueError as error:
                raise ValueError(
                    f"Image is outside the source directory: {resolved_image}"
                ) from error

            destination = cluster_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            operation(resolved_image, destination)
            exported_count += 1

    return exported_count


def cluster_images(
    source_dir: Path | str,
    *,
    method_name: str = DEFAULT_METHOD,
    threshold: float | None = None,
    output_dir: Path | str | None = None,
    recursive: bool = True,
    move: bool = False,
    include_singletons: bool = True,
    show_progress: bool = True,
) -> ClusterResult:
    """Discover, describe, cluster, and export images in one operation."""

    source = Path(source_dir).expanduser().resolve()
    method = get_method(method_name)
    selected_threshold = (
        method.default_threshold if threshold is None else threshold
    )
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source.with_name(f"{source.name}{method.output_suffix}")
    )
    if output == source or source in output.parents:
        raise ValueError("Output directory must be outside the source directory.")

    image_paths = discover_images(source, recursive=recursive)
    features, errors = extract_features(
        image_paths,
        method.extract,
        show_progress=show_progress,
        description=f"Extracting {method.name} features",
    )
    clusters = cluster_features(features, method.compare, selected_threshold)
    exported_image_count = export_clusters(
        clusters,
        source,
        output,
        move=move,
        include_singletons=include_singletons,
    )
    exported_cluster_count = sum(
        1
        for cluster in clusters
        if include_singletons or len(cluster) > 1
    )

    return ClusterResult(
        output_dir=output,
        discovered_count=len(image_paths),
        processed_count=len(features),
        clusters=tuple(tuple(cluster) for cluster in clusters),
        errors=tuple(errors),
        exported_cluster_count=exported_cluster_count,
        exported_image_count=exported_image_count,
    )
