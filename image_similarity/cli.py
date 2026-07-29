"""Command-line interface for image similarity clustering."""

from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit_runtime import configure_logging, parse_yaml_args

from .core import cluster_images
from .methods import DEFAULT_METHOD, METHODS

HERE = Path(__file__).resolve()
FEATURE_NAME = "image_similarity"


def build_parser(
    *,
    default_method: str = DEFAULT_METHOD,
    default_threshold: float | None = None,
) -> argparse.ArgumentParser:
    """Build the image-clustering CLI parser."""

    parser = argparse.ArgumentParser(
        description="Group visually similar images into cluster directories."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="Directory containing images.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_source",
        type=Path,
        help="Directory containing images.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=sorted(METHODS),
        default=default_method,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=default_threshold,
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "-M",
        "--move",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-S",
        "--singletons",
        action=argparse.BooleanOptionalAction,
        dest="include_singletons",
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-P",
        "--progress",
        action=argparse.BooleanOptionalAction,
        dest="show_progress",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_method: str = DEFAULT_METHOD,
    default_threshold: float | None = None,
) -> int:
    """Run the image-clustering CLI."""

    try:
        args, unknown_args, yaml_path = parse_yaml_args(
            build_parser(
                default_method=default_method,
                default_threshold=default_threshold,
            ),
            HERE,
            FEATURE_NAME,
            argv,
        )
        log = configure_logging(HERE, FEATURE_NAME, verbose=bool(args.verbose))
        log.debug("Loaded configuration from %s", yaml_path)
        if unknown_args:
            log.debug("Ignoring unknown arguments: %s", unknown_args)

        source = args.input_source or args.source
        if source is None:
            raise ValueError(
                "No source configured; use -i/--input or edit args.yaml."
            )
        result = cluster_images(
            source,
            method_name=args.method,
            threshold=args.threshold,
            output_dir=args.output,
            recursive=bool(args.recursive),
            move=bool(args.move),
            include_singletons=bool(args.include_singletons),
            overwrite=bool(args.overwrite),
            show_progress=bool(args.show_progress),
        )
    except ModuleNotFoundError as error:
        dependency = error.name or str(error)
        print(
            f"error: missing dependency {dependency!r}; "
            "install packages from requirements.txt",
            file=sys.stderr,
        )
        return 2
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for extraction_error in result.errors:
        log.warning(
            "Could not process %s: %s",
            extraction_error.image_path,
            extraction_error.message,
        )
    log.info(
        "Found %d clusters from %d/%d images; exported %d clusters "
        "(%d images) to %s",
        result.cluster_count,
        result.processed_count,
        result.discovered_count,
        result.exported_cluster_count,
        result.exported_image_count,
        result.output_dir,
    )
    return 0
