"""YAML- and CLI-driven entry point for yt-dlp downloads."""

from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
from typing import Sequence

import _youtube_download_core as _core
from toolkit_runtime import configure_logging, load_yaml_defaults, progress_iter

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem

DownloadFailure = _core.DownloadFailure
DownloadOptions = _core.DownloadOptions
DownloadSummary = _core.DownloadSummary
default_output_dir = _core.default_output_dir
download_youtube = _core.download_youtube
is_url = _core.is_url
load_download_groups = _core.load_download_groups

__all__ = [
    "DownloadFailure",
    "DownloadOptions",
    "DownloadSummary",
    "default_output_dir",
    "download_youtube",
    "is_url",
    "load_download_groups",
]


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download video URLs directly or from TXT/YAML lists."
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="One or more URLs, TXT files, or YAML files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory; defaults to a nearby yt_videos directory.",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="format_selector",
        help="yt-dlp format selector.",
    )
    parser.add_argument("-m", "--merge-format")
    parser.add_argument(
        "-a",
        "--archive",
        type=Path,
        help="Archive file used to skip completed downloads.",
    )
    playlist_group = parser.add_mutually_exclusive_group()
    playlist_group.add_argument(
        "-n",
        "--no-playlist",
        action="store_true",
        dest="no_playlist",
    )
    playlist_group.add_argument(
        "--playlist",
        action="store_false",
        dest="no_playlist",
    )
    parser.add_argument(
        "-x",
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-P",
        "--progress",
        action=argparse.BooleanOptionalAction,
        dest="show_progress",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(**(defaults or {}))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        defaults, parse_arguments, yaml_path = load_yaml_defaults(
            HERE,
            FEATURE_NAME,
            argv,
        )
        args, unknown_args = build_parser(defaults).parse_known_args(
            parse_arguments
        )
        log = configure_logging(
            HERE,
            FEATURE_NAME,
            verbose=bool(args.verbose),
        )
        log.debug("Loaded configuration from %s", yaml_path)
        if unknown_args:
            log.debug("Ignoring unknown arguments: %s", unknown_args)

        raw_inputs = args.inputs or []
        if isinstance(raw_inputs, str):
            raw_inputs = [raw_inputs]
        if not raw_inputs:
            raise ValueError(
                "No inputs configured; use -i/--inputs or edit args.yaml."
            )

        options = DownloadOptions(
            output_dir=Path(args.output) if args.output else None,
            format_selector=args.format_selector,
            merge_output_format=args.merge_format,
            archive_path=Path(args.archive) if args.archive else None,
            no_playlist=bool(args.no_playlist),
            continue_on_error=not bool(args.fail_fast),
            quiet=bool(args.quiet),
        )
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        ValueError,
    ) as error:
        logging.getLogger(FEATURE_NAME).error("%s", error)
        print(f"error: {error}", file=sys.stderr)
        return 2

    attempted = 0
    succeeded = 0
    failures: list[DownloadFailure] = []
    try:
        for input_value in progress_iter(
            raw_inputs,
            "Processing download inputs",
            enabled=bool(args.show_progress),
        ):
            log.info("Processing download input: %s", input_value)
            summary = download_youtube([input_value], options)
            attempted += summary.attempted
            succeeded += summary.succeeded
            failures.extend(summary.failures)
            for failure in summary.failures:
                log.error("%s: %s", failure.url, failure.message)
            if failures and args.fail_fast:
                break
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        ValueError,
    ) as error:
        log.error("%s", error)
        print(f"error: {error}", file=sys.stderr)
        return 2

    log.info(
        "Downloaded %d/%d URLs with %d failures.",
        succeeded,
        attempted,
        len(failures),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
