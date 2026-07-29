"""YAML- and CLI-driven entry point for advanced video compression."""

from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
from typing import Sequence

import _video_compressor_core as _core
from toolkit_runtime import configure_logging, load_yaml_defaults, progress_iter

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem

CompressionOptions = _core.CompressionOptions
CompressionResult = _core.CompressionResult
VideoInfo = _core.VideoInfo
build_scale_filter = _core.build_scale_filter
compress_video = _core.compress_video
ffprobe_video_info = _core.ffprobe_video_info
make_even = _core.make_even
output_path_for = _core.output_path_for
resolve_input_videos = _core.resolve_input_videos
run_ffmpeg_with_progress = _core.run_ffmpeg_with_progress

__all__ = [
    "CompressionOptions",
    "CompressionResult",
    "VideoInfo",
    "build_scale_filter",
    "compress_video",
    "ffprobe_video_info",
    "make_even",
    "output_path_for",
    "resolve_input_videos",
    "run_ffmpeg_with_progress",
]


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compress videos with an automatically estimated target bitrate."
        )
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="Video files, directories, globs, or TXT file lists.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to each input directory.",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        help="Filename suffix inserted before the extension.",
    )
    parser.add_argument(
        "-c",
        "--codec",
        choices=sorted(_core.CODEC_ENCODERS),
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=["low", "medium", "high", "very-high"],
    )
    parser.add_argument("-H", "--max-height", type=int)
    parser.add_argument("-a", "--audio-kbps", type=int)
    parser.add_argument("-p", "--preset")
    parser.add_argument(
        "-2",
        "--two-pass",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-g",
        "--hw",
        choices=["auto", "cpu", "cuda"],
        help="Hardware-accelerated input decoding mode.",
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
    parser.add_argument(
        "-f",
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("-F", "--ffmpeg-binary")
    parser.add_argument("-R", "--ffprobe-binary")
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

        input_paths = resolve_input_videos(raw_inputs)
        if not input_paths:
            raise ValueError("No input videos found.")
        output_dir = Path(args.output_dir) if args.output_dir else None
        options = CompressionOptions(
            codec=args.codec,
            quality=args.quality,
            max_height=args.max_height,
            audio_kbps=args.audio_kbps,
            preset=args.preset,
            two_pass=bool(args.two_pass),
            hardware=args.hw,
            overwrite=bool(args.overwrite),
            show_progress=bool(args.show_progress),
            ffmpeg_binary=args.ffmpeg_binary,
            ffprobe_binary=args.ffprobe_binary,
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

    failures = 0
    for input_path in progress_iter(
        input_paths,
        "Processing videos",
        enabled=bool(args.show_progress),
    ):
        output_path = output_path_for(
            input_path,
            output_dir=output_dir,
            suffix=args.suffix,
        )
        log.info("Compressing %s -> %s", input_path, output_path)
        try:
            result = compress_video(input_path, output_path, options)
        except Exception as error:
            failures += 1
            log.error("Failed %s: %s", input_path, error)
            if args.fail_fast:
                break
            continue

        log.info(
            "Completed %s at %dx%d, %d kbps in %.1f seconds%s",
            result.output_path,
            result.output_width,
            result.output_height,
            result.video_kbps,
            result.elapsed_seconds,
            " using CUDA decoding" if result.used_cuda else "",
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
