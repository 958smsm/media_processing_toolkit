"""Convert image sequences to video and extract video frames to images."""

from __future__ import annotations

import argparse, logging, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from toolkit_runtime import (
    ProgressBar, configure_logging, increment_path, natural_sort_key,
    parse_yaml_args, partial_output_path, progress_iter, resolve_files,
)
from video_compressor import resolve_input_videos

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem
LOGGER = logging.getLogger(FEATURE_NAME)
IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
VIDEO_CODEC_MAP = {
    ".avi": "XVID",
    ".mkv": "X264",
    ".mov": "mp4v",
    ".mp4": "mp4v",
    ".wmv": "WMV2",
}


@dataclass(frozen=True)
class ImagesToVideoResult:
    """Summary of an image-sequence encode."""

    input_count: int
    output_path: Path
    width: int
    height: int
    fps: float
    elapsed_seconds: float


@dataclass(frozen=True)
class VideoToImagesResult:
    """Summary of a frame-extraction operation."""

    input_path: Path
    output_dir: Path
    decoded_frames: int
    saved_frames: int
    source_fps: float
    frame_step: int
    elapsed_seconds: float


def discover_images(input_value: Path | str) -> list[Path]:
    """Expand an image directory, file, or glob in natural order."""

    images = resolve_files(
        [input_value],
        extensions=IMAGE_EXTENSIONS,
        allow_text_lists=False,
    )
    if not images:
        raise ValueError(f"No supported images found in {input_value}.")
    return images


def frame_to_timecode(frame_number: int, fps: float) -> str:
    """Convert a frame index to a filename-safe HH_MM_SS_mmm timecode."""

    if frame_number < 0 or fps <= 0:
        raise ValueError("Frame number cannot be negative and FPS must be positive.")
    total_milliseconds = round(frame_number / fps * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"


def sampling_frame_step(
    source_fps: float,
    *,
    interval_seconds: float | None = None,
    target_fps: float | None = None,
) -> int:
    """Calculate the number of decoded frames between saved images."""

    if source_fps <= 0:
        raise ValueError("Source FPS must be positive.")
    if interval_seconds is not None and target_fps is not None:
        raise ValueError("Choose interval_seconds or target_fps, not both.")
    if interval_seconds is not None:
        if interval_seconds <= 0:
            raise ValueError("Interval seconds must be positive.")
        return max(1, round(source_fps * interval_seconds))
    if target_fps is not None:
        if target_fps <= 0:
            raise ValueError("Target FPS must be positive.")
        return max(1, round(source_fps / target_fps))
    return 1


def images_to_video(
    image_input: Path | str,
    output_path: Path | str,
    *,
    fps: float = 20.0,
    codec: str | None = None,
    resize_frames: bool = False,
    overwrite: bool = False,
    show_progress: bool = True,
) -> ImagesToVideoResult:
    """Encode a naturally ordered image sequence with OpenCV."""

    import cv2

    if fps <= 0:
        raise ValueError("Output FPS must be positive.")
    images = discover_images(image_input)
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output video already exists: {output}")

    selected_codec = codec or VIDEO_CODEC_MAP.get(output.suffix.lower())
    if not selected_codec:
        supported = ", ".join(sorted(VIDEO_CODEC_MAP))
        raise ValueError(
            f"Unsupported output extension {output.suffix!r}; choose {supported}."
        )
    if len(selected_codec) != 4:
        raise ValueError("OpenCV codec must contain exactly four characters.")

    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise ValueError(f"Could not decode image: {images[0]}")
    height, width = first_frame.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = partial_output_path(output)
    writer = cv2.VideoWriter(
        str(partial_output),
        cv2.VideoWriter_fourcc(*selected_codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video writer for {partial_output}."
        )

    started_at = time.monotonic()
    try:
        for image_path in progress_iter(
            images,
            "Encoding images",
            enabled=show_progress,
        ):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Could not decode image: {image_path}")
            frame_height, frame_width = frame.shape[:2]
            if (frame_width, frame_height) != (width, height):
                if not resize_frames:
                    raise ValueError(
                        f"Image size mismatch for {image_path}; expected "
                        f"{width}x{height}, got {frame_width}x{frame_height}."
                    )
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    except Exception:
        writer.release()
        partial_output.unlink(missing_ok=True)
        raise
    else:
        writer.release()

    if not partial_output.is_file() or partial_output.stat().st_size == 0:
        partial_output.unlink(missing_ok=True)
        raise RuntimeError("OpenCV produced an empty video output.")
    partial_output.replace(output)
    return ImagesToVideoResult(
        input_count=len(images),
        output_path=output,
        width=width,
        height=height,
        fps=float(fps),
        elapsed_seconds=time.monotonic() - started_at,
    )


def _image_write_parameters(cv2: Any, extension: str, quality: int) -> list[int]:
    if not 0 <= quality <= 100:
        raise ValueError("Image quality must be between 0 and 100.")
    if extension in {".jpg", ".jpeg"}:
        return [cv2.IMWRITE_JPEG_QUALITY, quality]
    if extension == ".png":
        compression = round((100 - quality) * 9 / 100)
        return [cv2.IMWRITE_PNG_COMPRESSION, compression]
    if extension == ".webp":
        return [cv2.IMWRITE_WEBP_QUALITY, quality]
    return []


def video_to_images(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    interval_seconds: float | None = None,
    target_fps: float | None = None,
    image_extension: str = "jpg",
    image_quality: int = 95,
    overwrite: bool = False,
    show_progress: bool = True,
) -> VideoToImagesResult:
    """Decode a video sequentially and save frames at a selected interval."""

    import cv2

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")

    extension = f".{image_extension.lower().lstrip('.')}"
    if extension not in IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported image extension {extension!r}; choose {supported}."
        )
    write_parameters = _image_write_parameters(
        cv2,
        extension,
        image_quality,
    )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise ValueError(f"Could not determine FPS for {source}.")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = sampling_frame_step(
        source_fps,
        interval_seconds=interval_seconds,
        target_fps=target_fps,
    )

    requested_output = Path(output_dir).expanduser().resolve() / source.stem
    destination = (
        requested_output if overwrite else increment_path(requested_output)
    )
    destination.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    decoded_frames = 0
    saved_frames = 0

    with ProgressBar(
        "Extracting frames",
        total=float(total_frames) if total_frames > 0 else None,
        unit="frame",
        enabled=show_progress,
    ) as progress:
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_number = decoded_frames
                decoded_frames += 1
                if frame_number % frame_step == 0:
                    timecode = frame_to_timecode(frame_number, source_fps)
                    filename = (
                        f"{source.stem}_{frame_number:09d}_"
                        f"{timecode}{extension}"
                    )
                    image_path = destination / filename
                    if image_path.exists() and not overwrite:
                        raise FileExistsError(
                            f"Image already exists: {image_path}"
                        )
                    if not cv2.imwrite(
                        str(image_path),
                        frame,
                        write_parameters,
                    ):
                        raise RuntimeError(
                            f"Could not write image: {image_path}"
                        )
                    saved_frames += 1
                progress.update()
        finally:
            capture.release()

    return VideoToImagesResult(
        input_path=source,
        output_dir=destination,
        decoded_frames=decoded_frames,
        saved_frames=saved_frames,
        source_fps=source_fps,
        frame_step=frame_step,
        elapsed_seconds=time.monotonic() - started_at,
    )


def _default_video_output(image_input: Path | str) -> Path:
    path = Path(str(image_input)).expanduser().resolve()
    if path.is_dir():
        return path.parent / f"{path.name}.mp4"
    return path.parent / "images.mp4"


def build_parser() -> argparse.ArgumentParser:
    """Build the bidirectional-converter CLI parser."""

    parser = argparse.ArgumentParser(
        description="Convert image sequences to video or video to images."
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["images-to-video", "video-to-images"],
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="Image input or video files/directories/globs/TXT lists.",
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "-r",
        "--fps",
        type=float,
        help="Output video FPS or extracted target FPS.",
    )
    parser.add_argument("-s", "--interval-seconds", type=float)
    parser.add_argument("-e", "--image-extension")
    parser.add_argument("-q", "--image-quality", type=int)
    parser.add_argument("-c", "--codec")
    parser.add_argument(
        "-R",
        "--resize-frames",
        action=argparse.BooleanOptionalAction,
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
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bidirectional-converter CLI."""

    try:
        args, unknown_args, yaml_path = parse_yaml_args(
            build_parser(),
            HERE,
            FEATURE_NAME,
            argv,
        )
        log = configure_logging(HERE, FEATURE_NAME, verbose=bool(args.verbose))
        log.debug("Loaded configuration from %s", yaml_path)
        if unknown_args:
            log.debug("Ignoring unknown arguments: %s", unknown_args)

        raw_inputs: Iterable[str] = args.inputs or []
        if isinstance(raw_inputs, str):
            raw_inputs = [raw_inputs]
        raw_inputs = list(raw_inputs)
        if not raw_inputs:
            raise ValueError(
                "No inputs configured; use -i/--inputs or edit args.yaml."
            )
        if args.mode not in {"images-to-video", "video-to-images"}:
            raise ValueError("Select images-to-video or video-to-images mode.")
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.mode == "images-to-video":
        if len(raw_inputs) != 1:
            log.error("images-to-video mode accepts exactly one image input.")
            return 2
        output = args.output or _default_video_output(raw_inputs[0])
        try:
            result = images_to_video(
                raw_inputs[0],
                output,
                fps=args.fps,
                codec=args.codec,
                resize_frames=bool(args.resize_frames),
                overwrite=bool(args.overwrite),
                show_progress=bool(args.show_progress),
            )
        except Exception as error:
            log.error("Image-to-video conversion failed: %s", error)
            return 1
        log.info(
            "Created %s from %d images at %dx%d and %.3f FPS in %.1f seconds.",
            result.output_path,
            result.input_count,
            result.width,
            result.height,
            result.fps,
            result.elapsed_seconds,
        )
        return 0

    try:
        video_inputs = resolve_input_videos(raw_inputs)
        if not video_inputs:
            raise ValueError("No input videos found.")
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return 2

    failures = 0
    for video_path in progress_iter(
        video_inputs,
        "Processing videos",
        enabled=bool(args.show_progress),
    ):
        output_root = args.output or video_path.parent / "frames"
        log.info("Extracting frames from %s", video_path)
        try:
            result = video_to_images(
                video_path,
                output_root,
                interval_seconds=args.interval_seconds,
                target_fps=args.fps,
                image_extension=args.image_extension,
                image_quality=args.image_quality,
                overwrite=bool(args.overwrite),
                show_progress=bool(args.show_progress),
            )
        except Exception as error:
            failures += 1
            log.error("Failed %s: %s", video_path, error)
            if args.fail_fast:
                break
            continue
        log.info(
            "Saved %d/%d decoded frames from %s to %s in %.1f seconds.",
            result.saved_frames,
            result.decoded_frames,
            result.input_path,
            result.output_dir,
            result.elapsed_seconds,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
