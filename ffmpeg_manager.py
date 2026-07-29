"""Public FFmpeg helpers, raw-frame writers, and bitrate CLI."""

from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
from typing import Sequence

import _ffmpeg_manager_core as _core
from toolkit_runtime import configure_logging, load_yaml_defaults, progress_iter

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem

BPP_PRESETS = _core.BPP_PRESETS
CODEC_ENCODERS = _core.CODEC_ENCODERS
FFmpegError = _core.FFmpegError
RawVideoWriter = _core.RawVideoWriter
FFmpegPipeWriter = _core.FFmpegPipeWriter
FFmpegVideoWriter = _core.FFmpegVideoWriter
auto_video_kbps = _core.auto_video_kbps
cuda_works_for_file = _core.cuda_works_for_file
ffmpeg_supports_hwaccel = _core.ffmpeg_supports_hwaccel
hardware_acceleration_args = _core.hardware_acceleration_args
normalize_codec = _core.normalize_codec
normalize_quality = _core.normalize_quality
nvidia_gpu_present = _core.nvidia_gpu_present
require_executable = _core.require_executable
run_capture = _core.run_capture
try_run_capture = _core.try_run_capture

__all__ = [
    "BPP_PRESETS",
    "CODEC_ENCODERS",
    "FFmpegError",
    "FFmpegPipeWriter",
    "FFmpegVideoWriter",
    "RawVideoWriter",
    "auto_video_kbps",
    "cuda_works_for_file",
    "ffmpeg_supports_hwaccel",
    "hardware_acceleration_args",
    "normalize_codec",
    "normalize_quality",
    "nvidia_gpu_present",
    "require_executable",
    "run_capture",
    "try_run_capture",
]


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate a video bitrate from dimensions and FPS."
    )
    parser.add_argument("-W", "--width", type=int, help="Video width.")
    parser.add_argument("-H", "--height", type=int, help="Video height.")
    parser.add_argument("-r", "--fps", type=float, help="Frames per second.")
    parser.add_argument(
        "-c",
        "--codec",
        choices=sorted(CODEC_ENCODERS),
        help="Target codec family.",
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=["low", "medium", "high", "very-high"],
        help="Bitrate quality preset.",
    )
    parser.add_argument(
        "-b",
        "--bitrate-kbps",
        type=int,
        help="Optional maximum bitrate.",
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

        if args.width is None or args.height is None or args.fps is None:
            raise ValueError("Width, height, and FPS are required.")

        estimates = progress_iter(
            [(args.width, args.height, args.fps)],
            "Calculating bitrate",
        )
        for width, height, fps in estimates:
            bitrate = auto_video_kbps(
                width,
                height,
                fps,
                args.codec,
                args.quality,
                args.bitrate_kbps,
            )
        log.info(
            "Estimated %s/%s bitrate: %d kbps",
            args.codec,
            args.quality,
            bitrate,
        )
        print(bitrate)
        return 0
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        ValueError,
    ) as error:
        logging.getLogger(FEATURE_NAME).error("%s", error)
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
