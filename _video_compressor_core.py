"""Compress videos with FFmpeg using resolution- and FPS-aware bitrates."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from ffmpeg_manager import (
    CODEC_ENCODERS,
    FFmpegError,
    auto_video_kbps,
    hardware_acceleration_args,
    normalize_codec,
    normalize_quality,
    require_executable,
    run_capture,
)

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".webm",
        ".wmv",
    }
)


@dataclass(frozen=True)
class VideoInfo:
    """Metadata required for bitrate estimation and progress reporting."""

    width: int
    height: int
    fps: float
    duration: float
    codec: str


@dataclass(frozen=True)
class CompressionOptions:
    """Settings for one compression operation."""

    codec: str = "h264"
    quality: str = "medium"
    max_height: int | None = None
    audio_kbps: int = 128
    preset: str = "medium"
    two_pass: bool = False
    hardware: str = "auto"
    overwrite: bool = False
    show_progress: bool = True
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"


@dataclass(frozen=True)
class CompressionResult:
    """Summary of a completed compression."""

    input_path: Path
    output_path: Path
    source_info: VideoInfo
    output_width: int
    output_height: int
    video_kbps: int
    used_cuda: bool
    elapsed_seconds: float


def ffprobe_video_info(
    input_path: Path | str,
    *,
    ffprobe_binary: str = "ffprobe",
) -> VideoInfo:
    """Read the first video stream with ``ffprobe``."""

    path = Path(input_path).expanduser().resolve()
    output = run_capture(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
    )
    data = json.loads(output)
    video_stream = next(
        (
            stream
            for stream in data.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}.")

    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid video dimensions reported for {path}.") from error

    frame_rate = (
        video_stream.get("avg_frame_rate")
        or video_stream.get("r_frame_rate")
        or "0/0"
    )
    try:
        fraction = Fraction(frame_rate)
        fps = (
            float(fraction)
            if fraction.numerator and fraction.denominator
            else 0.0
        )
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    try:
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    if fps <= 0 and duration > 0 and video_stream.get("nb_frames"):
        try:
            fps = float(video_stream["nb_frames"]) / duration
        except (TypeError, ValueError, ZeroDivisionError):
            fps = 0.0
    if fps <= 0:
        raise ValueError(f"Could not determine the frame rate for {path}.")

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        codec=str(video_stream.get("codec_name") or ""),
    )


def make_even(value: int) -> int:
    """Return the nearest lower positive even integer."""

    return max(2, int(value) - (int(value) % 2))


def build_scale_filter(
    width: int,
    height: int,
    max_height: int | None,
) -> tuple[str | None, int, int]:
    """Build a scale filter that preserves aspect ratio and even dimensions."""

    if not max_height or height <= max_height:
        return None, width, height
    if max_height < 2:
        raise ValueError("Maximum height must be at least 2 pixels.")

    output_height = make_even(max_height)
    output_width = make_even(round(width * output_height / height))
    return (
        f"scale={output_width}:{output_height}",
        output_width,
        output_height,
    )


def resolve_input_videos(inputs: Iterable[Path | str]) -> list[Path]:
    """Expand files, directories, globs, and text file lists."""

    discovered: list[Path] = []

    def add_input(value: Path | str) -> None:
        raw = str(value).strip()
        if not raw or raw.startswith("#"):
            return

        if glob.has_magic(raw):
            matches = glob.glob(raw, recursive=True)
            if not matches:
                raise FileNotFoundError(f"Input pattern matched no files: {raw}")
            for match in matches:
                add_input(match)
            return

        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_dir():
            discovered.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in VIDEO_EXTENSIONS
            )
            return
        if path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                listed = Path(line)
                if not listed.is_absolute():
                    listed = path.parent / listed
                add_input(listed)
            return
        discovered.append(path.resolve())

    for item in inputs:
        add_input(item)

    return sorted(set(discovered), key=lambda path: str(path).casefold())


class _ProgressBar:
    def __init__(self, total: float, description: str, enabled: bool) -> None:
        self.total = max(total, 1.0)
        self.current = 0.0
        self.description = description
        self._bar = None
        if enabled:
            try:
                from tqdm import tqdm

                self._bar = tqdm(
                    total=self.total,
                    unit="s",
                    dynamic_ncols=True,
                    desc=description,
                )
            except ModuleNotFoundError:
                self._bar = None

    def update_to(self, seconds: float) -> None:
        value = min(max(seconds, 0.0), self.total)
        delta = max(value - self.current, 0.0)
        if delta <= 0:
            return
        self.current = value
        if self._bar:
            self._bar.update(delta)

    def close(self) -> None:
        if self._bar:
            self._bar.close()


def _read_stderr(
    stream: Iterable[str],
    tail: deque[str],
    *,
    echo: bool,
) -> None:
    for line in stream:
        cleaned = line.rstrip()
        if cleaned:
            tail.append(cleaned)
            if echo:
                print(cleaned, file=sys.stderr, flush=True)


def run_ffmpeg_with_progress(
    command: Sequence[str],
    duration_seconds: float,
    description: str,
    *,
    show_progress: bool = True,
    echo_stderr: bool = False,
) -> None:
    """Run FFmpeg while consuming its machine-readable progress stream."""

    if not command:
        raise ValueError("FFmpeg command cannot be empty.")
    full_command = [
        str(command[0]),
        "-progress",
        "pipe:1",
        "-nostats",
        *[str(part) for part in command[1:]],
    ]
    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stderr_tail: deque[str] = deque(maxlen=100)
    stderr_thread = threading.Thread(
        target=_read_stderr,
        args=(process.stderr, stderr_tail),
        kwargs={"echo": echo_stderr},
        daemon=True,
    )
    stderr_thread.start()
    progress = _ProgressBar(
        duration_seconds,
        description,
        enabled=show_progress,
    )

    try:
        for line in process.stdout:
            key, separator, value = line.strip().partition("=")
            if not separator:
                continue
            if key in {"out_time_us", "out_time_ms"}:
                try:
                    progress.update_to(int(value) / 1_000_000.0)
                except ValueError:
                    pass
            elif key == "out_time":
                match = re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", value)
                if match:
                    hours, minutes, seconds = match.groups()
                    progress.update_to(
                        int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    )

        return_code = process.wait()
        stderr_thread.join(timeout=2)
        if return_code != 0:
            rendered = subprocess.list2cmdline(full_command)
            diagnostics = "\n".join(stderr_tail)
            raise FFmpegError(
                f"FFmpeg failed with exit code {return_code}: {rendered}\n"
                f"{diagnostics}"
            )
    finally:
        progress.close()
        process.stdout.close()
        process.stderr.close()


def _video_encoding_args(
    codec: str,
    preset: str,
    video_kbps: int,
) -> list[str]:
    encoder, codec_args = CODEC_ENCODERS[codec]
    maximum_rate = math.ceil(video_kbps * 1.20)
    buffer_size = math.ceil(video_kbps * 2.00)
    return [
        "-c:v",
        encoder,
        "-preset",
        preset,
        *codec_args,
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{maximum_rate}k",
        "-bufsize",
        f"{buffer_size}k",
    ]


def compress_video(
    input_path: Path | str,
    output_path: Path | str,
    options: CompressionOptions = CompressionOptions(),
) -> CompressionResult:
    """Compress one video and atomically publish the completed output."""

    started_at = time.monotonic()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Input and output paths must be different.")
    if output.exists() and not options.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if options.audio_kbps <= 0:
        raise ValueError("Audio bitrate must be positive.")

    ffmpeg_binary = require_executable(options.ffmpeg_binary)
    ffprobe_binary = require_executable(options.ffprobe_binary)
    codec = normalize_codec(options.codec)
    quality = normalize_quality(options.quality)
    source_info = ffprobe_video_info(
        source,
        ffprobe_binary=ffprobe_binary,
    )
    scale_filter, output_width, output_height = build_scale_filter(
        source_info.width,
        source_info.height,
        options.max_height,
    )
    video_kbps = auto_video_kbps(
        output_width,
        output_height,
        source_info.fps,
        codec,
        quality,
    )
    hardware_args = hardware_acceleration_args(
        source,
        options.hardware,
        ffmpeg_binary=ffmpeg_binary,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_name(
        f"{output.stem}.partial{output.suffix}"
    )
    input_args = [
        ffmpeg_binary,
        "-y",
        *hardware_args,
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if scale_filter:
        input_args.extend(["-vf", scale_filter])
    video_args = _video_encoding_args(codec, options.preset, video_kbps)
    audio_args = ["-c:a", "aac", "-b:a", f"{options.audio_kbps}k"]
    container_args = (
        ["-movflags", "+faststart"]
        if output.suffix.lower() in {".m4v", ".mov", ".mp4"}
        else []
    )

    try:
        if options.two_pass:
            with tempfile.TemporaryDirectory(
                prefix="video_compress_",
                dir=output.parent,
            ) as temporary_directory:
                pass_log = str(Path(temporary_directory) / "ffmpeg2pass")
                first_pass = [
                    *input_args,
                    *video_args,
                    "-pass",
                    "1",
                    "-passlogfile",
                    pass_log,
                    "-an",
                    "-f",
                    "null",
                    os.devnull,
                ]
                run_ffmpeg_with_progress(
                    first_pass,
                    source_info.duration,
                    "Encoding pass 1/2",
                    show_progress=options.show_progress,
                )

                second_pass = [
                    *input_args,
                    *video_args,
                    "-pass",
                    "2",
                    "-passlogfile",
                    pass_log,
                    *audio_args,
                    *container_args,
                    str(partial_output),
                ]
                run_ffmpeg_with_progress(
                    second_pass,
                    source_info.duration,
                    "Encoding pass 2/2",
                    show_progress=options.show_progress,
                )
        else:
            command = [
                *input_args,
                *video_args,
                *audio_args,
                *container_args,
                str(partial_output),
            ]
            run_ffmpeg_with_progress(
                command,
                source_info.duration,
                "Encoding",
                show_progress=options.show_progress,
            )

        partial_output.replace(output)
    except Exception:
        partial_output.unlink(missing_ok=True)
        raise

    return CompressionResult(
        input_path=source,
        output_path=output,
        source_info=source_info,
        output_width=output_width,
        output_height=output_height,
        video_kbps=video_kbps,
        used_cuda=bool(hardware_args),
        elapsed_seconds=time.monotonic() - started_at,
    )


def output_path_for(
    input_path: Path,
    *,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    """Build the destination path for one input."""

    destination_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else input_path.parent
    )
    return destination_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compress videos with an automatically estimated target bitrate."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video files, directories, glob patterns, or text file lists.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to each input directory.",
    )
    parser.add_argument(
        "--suffix",
        default="_compressed",
        help="Filename suffix inserted before the extension.",
    )
    parser.add_argument(
        "--codec",
        choices=sorted(CODEC_ENCODERS),
        default="h264",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "very-high"],
        default="medium",
    )
    parser.add_argument("--max-height", type=int)
    parser.add_argument("--audio-kbps", type=int, default=128)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--two-pass", action="store_true")
    parser.add_argument(
        "--hw",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Hardware-accelerated input decoding mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch after the first failed input.",
    )
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--ffprobe-binary", default="ffprobe")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        input_paths = resolve_input_videos(args.inputs)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not input_paths:
        print("error: no input videos found", file=sys.stderr)
        return 2

    options = CompressionOptions(
        codec=args.codec,
        quality=args.quality,
        max_height=args.max_height,
        audio_kbps=args.audio_kbps,
        preset=args.preset,
        two_pass=args.two_pass,
        hardware=args.hw,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        ffmpeg_binary=args.ffmpeg_binary,
        ffprobe_binary=args.ffprobe_binary,
    )

    failures = 0
    for input_path in input_paths:
        output_path = output_path_for(
            input_path,
            output_dir=args.output_dir,
            suffix=args.suffix,
        )
        LOGGER.info("Compressing %s -> %s", input_path, output_path)
        try:
            result = compress_video(input_path, output_path, options)
        except (
            FFmpegError,
            FileExistsError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as error:
            failures += 1
            LOGGER.error("%s", error)
            if args.fail_fast:
                break
            continue

        LOGGER.info(
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
