"""Command-line interface for image similarity clustering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .core import cluster_images
from .methods import DEFAULT_METHOD, METHODS


def build_parser(
    *,
    default_method: str = DEFAULT_METHOD,
    default_threshold: float | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group visually similar images into cluster directories."
    )
    parser.add_argument("source", type=Path, help="Directory containing images.")
    parser.add_argument(
        "-m",
        "--method",
        choices=sorted(METHODS),
        default=default_method,
        help=f"Similarity method (default: {default_method}).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=default_threshold,
        help="Similarity threshold; method-specific default when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory; defaults to a sibling of the source directory.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move source files instead of copying them.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect images directly inside the source directory.",
    )
    parser.add_argument(
        "--exclude-singletons",
        action="store_true",
        help="Do not export clusters containing only one image.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_method: str = DEFAULT_METHOD,
    default_threshold: float | None = None,
) -> int:
    args = build_parser(
        default_method=default_method,
        default_threshold=default_threshold,
    ).parse_args(argv)

    try:
        result = cluster_images(
            args.source,
            method_name=args.method,
            threshold=args.threshold,
            output_dir=args.output,
            recursive=not args.no_recursive,
            move=args.move,
            include_singletons=not args.exclude_singletons,
            show_progress=not args.quiet,
        )
    except ModuleNotFoundError as error:
        dependency = error.name or str(error)
        print(
            f"error: missing dependency {dependency!r}; "
            "install packages from requirements.txt",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for error in result.errors:
        print(
            f"warning: could not process {error.image_path}: {error.message}",
            file=sys.stderr,
        )

    print(
        f"Found {result.cluster_count} clusters from "
        f"{result.processed_count}/{result.discovered_count} images; exported "
        f"{result.exported_cluster_count} clusters "
        f"({result.exported_image_count} images) to {result.output_dir}"
    )
    return 0
