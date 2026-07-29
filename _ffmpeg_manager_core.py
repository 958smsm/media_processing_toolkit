"""Reusable FFmpeg helpers and raw-frame video writers."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Sequence

BPP_PRESETS = {
    "h264": {"low": 0.07, "medium": 0.10, "high": 0.14, "very-high": 0.20},
    "hevc": {"low": 0.05, "medium": 0.07, "high": 0.10, "very-high": 0.14},
    "av1": {"low": 0.04, "medium": 0.06, "high": 0.09, "very-high": 0.13},
}

CODEC_ENCODERS = {
    "h264": ("libx264", ()),
    "hevc": ("libx265", ("-tag:v", "hvc1")),
    "av1": ("libaom-av1", ("-tag:v", "av01")),
}


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg command or process fails."""


def normalize_codec(codec_family: str) -> str:
    """Normalize common codec aliases."""

    codec = (codec_family or "h264").strip().lower()
    aliases = {"h265": "hevc", "x264": "h264", "x265": "hevc"}
    codec = aliases.get(codec, codec)
    if codec not in CODEC_ENCODERS:
        supported = ", ".join(CODEC_ENCODERS)
        raise ValueError(f"Unsupported codec {codec_family!r}; choose {supported}.")
    return codec


def normalize_quality(quality: str) -> str:
    """Normalize and validate a bitrate quality preset."""

    normalized = (quality or "medium").strip().lower().replace(" ", "-")
    if normalized not in {"low", "medium", "high", "very-high"}:
        raise ValueError(
            "Unsupported quality "
            f"{quality!r}; choose low, medium, high, or very-high."
        )
    return normalized


def auto_video_kbps(
    width: int,
    height: int,
    fps: float,
    codec_family: str = "h264",
    quality: str = "medium",
    bitrate_kbps: int | None = None,
) -> int:
    """Estimate a video bitrate using bits per pixel per frame."""

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("Width, height, and FPS must be positive.")

    codec = normalize_codec(codec_family)
    preset = normalize_quality(quality)
    estimated = width * height * fps * BPP_PRESETS[codec][preset] / 1000.0
    maximum = max(int(bitrate_kbps), 250) if bitrate_kbps else 80_000
    return int(round(min(max(estimated, 250.0), float(maximum))))


def require_executable(binary: str) -> str:
    """Resolve a required executable or raise a useful error."""

    candidate = Path(binary).expanduser()
    if candidate.parent != Path(".") and candidate.exists():
        return str(candidate.resolve())

    resolved = shutil.which(binary)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"Required executable {binary!r} was not found on PATH."
    )


def run_capture(command: Sequence[str]) -> str:
    """Run a command and return stdout, raising :class:`FFmpegError` on failure."""

    result = subprocess.run(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        rendered = subprocess.list2cmdline([str(part) for part in command])
        raise FFmpegError(
            f"Command failed with exit code {result.returncode}: {rendered}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def try_run_capture(command: Sequence[str]) -> str | None:
    """Return combined command output on success, otherwise ``None``."""

    try:
        result = subprocess.run(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    return (result.stdout or "") + (result.stderr or "")


def ffmpeg_supports_hwaccel(
    name: str,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> bool:
    """Return whether this FFmpeg build advertises a hardware accelerator."""

    output = try_run_capture([ffmpeg_binary, "-hide_banner", "-hwaccels"])
    if not output:
        return False

    requested = name.strip().lower()
    return any(line.strip().lower() == requested for line in output.splitlines())


def nvidia_gpu_present(*, nvidia_smi_binary: str = "nvidia-smi") -> bool:
    """Detect an NVIDIA GPU using ``nvidia-smi`` and a Windows fallback."""

    output = try_run_capture([nvidia_smi_binary, "-L"])
    if output and "gpu" in output.lower():
        return True

    if os.name != "nt":
        return False

    candidates = (
        Path(r"C:\Windows\System32\nvidia-smi.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    )
    for executable in candidates:
        if executable.exists():
            output = try_run_capture([str(executable), "-L"])
            if output and "gpu" in output.lower():
                return True

    adapters = try_run_capture(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty Name",
        ]
    )
    return bool(adapters and "nvidia" in adapters.lower())


_CUDA_PROBE_CACHE: dict[tuple[str, str], bool] = {}


def cuda_works_for_file(
    input_path: Path | str,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> bool:
    """Probe whether CUDA decoding initializes for a specific video."""

    path = str(Path(input_path).expanduser().resolve())
    cache_key = (ffmpeg_binary, path)
    if cache_key in _CUDA_PROBE_CACHE:
        return _CUDA_PROBE_CACHE[cache_key]

    result = subprocess.run(
        [
            ffmpeg_binary,
            "-hide_banner",
            "-v",
            "error",
            "-hwaccel",
            "cuda",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stderr = result.stderr or ""
    failure_markers = (
        "Failed setup for format cuda",
        "hwaccel initialisation returned error",
        "Hardware is lacking required capabilities",
        "not supported with this chroma format",
    )
    supported = result.returncode == 0 and not any(
        marker in stderr for marker in failure_markers
    )
    _CUDA_PROBE_CACHE[cache_key] = supported
    return supported


def hardware_acceleration_args(
    input_path: Path | str,
    mode: str = "auto",
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build FFmpeg input arguments for CPU or CUDA decoding."""

    selected = (mode or "auto").strip().lower()
    if selected in {"cpu", "off"}:
        return []
    if selected not in {"auto", "cuda"}:
        raise ValueError("Hardware mode must be auto, cpu, off, or cuda.")

    has_cuda = ffmpeg_supports_hwaccel(
        "cuda",
        ffmpeg_binary=ffmpeg_binary,
    )
    has_nvidia_gpu = nvidia_gpu_present()

    if selected == "cuda":
        if not has_cuda:
            raise FFmpegError(
                "CUDA was requested but this FFmpeg build does not support it."
            )
        if not has_nvidia_gpu:
            raise FFmpegError(
                "CUDA was requested but no NVIDIA GPU was detected."
            )
        return ["-hwaccel", "cuda"]

    if not has_cuda or not has_nvidia_gpu:
        return []
    if not cuda_works_for_file(input_path, ffmpeg_binary=ffmpeg_binary):
        return []
    return ["-hwaccel", "cuda"]


class RawVideoWriter:
    """Encode raw BGR frames through an FFmpeg subprocess."""

    def __init__(
        self,
        output_path: Path | str,
        width: int,
        height: int,
        fps: float,
        bitrate_kbps: int,
        *,
        codec_family: str = "h264",
        preset: str = "medium",
        crf: int | None = None,
        threads: int = 0,
        scale_filter: str | None = None,
        input_pixel_format: str = "bgr24",
        resize_frames: bool = False,
        overwrite: bool = False,
        ffmpeg_binary: str = "ffmpeg",
        logger: Any = None,
        stderr_tail_lines: int = 300,
        echo_stderr: bool = False,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0 or bitrate_kbps <= 0:
            raise ValueError(
                "Width, height, FPS, and bitrate must all be positive."
            )

        self.output_path = Path(output_path).expanduser()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.bitrate_kbps = int(bitrate_kbps)
        self.codec_family = normalize_codec(codec_family)
        self.preset = preset
        self.crf = crf
        self.threads = int(threads)
        self.scale_filter = scale_filter
        self.input_pixel_format = input_pixel_format
        self.resize_frames = resize_frames
        self.overwrite = overwrite
        self.ffmpeg_binary = ffmpeg_binary
        self.logger = logger
        self.echo_stderr = echo_stderr

        self.process: subprocess.Popen[bytes] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=stderr_tail_lines)
        self._stderr_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None

    @property
    def proc(self) -> subprocess.Popen[bytes] | None:
        """Backward-compatible alias for the active process."""

        return self.process

    def build_command(self) -> list[str]:
        """Build the FFmpeg command without starting a process."""

        encoder, codec_args = CODEC_ENCODERS[self.codec_family]
        overwrite_flag = "-y" if self.overwrite else "-n"
        command = [
            self.ffmpeg_binary,
            overwrite_flag,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.input_pixel_format,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
        ]
        if self.scale_filter:
            command.extend(["-vf", self.scale_filter])

        command.extend(
            [
                "-an",
                "-c:v",
                encoder,
                "-preset",
                self.preset,
                *codec_args,
            ]
        )
        if self.crf is not None:
            command.extend(["-crf", str(int(self.crf))])
        else:
            max_rate = math.ceil(self.bitrate_kbps * 1.20)
            buffer_size = math.ceil(self.bitrate_kbps * 2.00)
            command.extend(
                [
                    "-b:v",
                    f"{self.bitrate_kbps}k",
                    "-maxrate",
                    f"{max_rate}k",
                    "-bufsize",
                    f"{buffer_size}k",
                ]
            )
        if self.threads > 0:
            command.extend(["-threads", str(self.threads)])

        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ]
        )
        return command

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stderr is not None
        try:
            for raw_line in iter(process.stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                with self._stderr_lock:
                    self._stderr_tail.append(line)
                if self.echo_stderr:
                    print(line, file=sys.stderr, flush=True)
        finally:
            process.stderr.close()

    def stderr_tail(self) -> str:
        """Return the recent FFmpeg diagnostic output."""

        with self._stderr_lock:
            return "\n".join(self._stderr_tail)

    def open(self) -> RawVideoWriter:
        """Start the FFmpeg encoder."""

        if self.process and self.process.poll() is None:
            return self
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {self.output_path}")

        self.ffmpeg_binary = require_executable(self.ffmpeg_binary)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self.process,),
            daemon=True,
        )
        self._stderr_thread.start()

        return_code = self.process.poll()
        if return_code is not None:
            self._finish_stderr_thread()
            message = self._failure_message(
                f"FFmpeg failed to start with exit code {return_code}"
            )
            self.process = None
            raise FFmpegError(message)
        return self

    def _failure_message(self, summary: str) -> str:
        tail = self.stderr_tail()
        message = f"{summary} for {self.output_path}"
        if tail:
            message += f"\nFFmpeg stderr:\n{tail}"
        if self.logger:
            self.logger.error(message)
        return message

    def _finish_stderr_thread(self) -> None:
        if self._stderr_thread:
            self._stderr_thread.join(timeout=2)
            self._stderr_thread = None

    def write(self, frame: Any) -> None:
        """Write one uint8 BGR frame."""

        if frame is None:
            raise ValueError("Frame cannot be None.")
        if not self.process or not self.process.stdin:
            raise RuntimeError("FFmpeg writer is not open.")

        return_code = self.process.poll()
        if return_code is not None:
            raise FFmpegError(
                self._failure_message(
                    f"FFmpeg exited early with exit code {return_code}"
                )
            )

        if str(getattr(frame, "dtype", "")) != "uint8":
            raise ValueError("Expected a uint8 frame.")
        shape = getattr(frame, "shape", ())
        expected_shape = (self.height, self.width, 3)
        if tuple(shape) != expected_shape:
            if not self.resize_frames:
                raise ValueError(
                    f"Frame shape mismatch; expected {expected_shape}, got {shape}."
                )
            try:
                import cv2
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "OpenCV is required only when resize_frames=True."
                ) from error
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as error:
            raise FFmpegError(
                self._failure_message("FFmpeg closed its input pipe")
            ) from error

    def close(self, timeout: float | None = 120) -> None:
        """Finalize the output and surface encoder errors."""

        if not self.process:
            return

        process = self.process
        try:
            if process.stdin:
                process.stdin.close()
            try:
                return_code = (
                    process.wait(timeout=timeout)
                    if timeout is not None
                    else process.wait()
                )
            except subprocess.TimeoutExpired as error:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise FFmpegError(
                    self._failure_message("FFmpeg close timed out")
                ) from error

            self._finish_stderr_thread()
            if return_code != 0:
                raise FFmpegError(
                    self._failure_message(
                        f"FFmpeg failed with exit code {return_code}"
                    )
                )
        finally:
            self.process = None

    def abort(self, timeout: float = 5) -> None:
        """Stop an unfinished encode without treating it as successful."""

        if not self.process:
            return
        process = self.process
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self._finish_stderr_thread()
        finally:
            self.process = None

    def __enter__(self) -> RawVideoWriter:
        return self.open()

    def __exit__(self, exception_type: Any, *_args: Any) -> None:
        if exception_type is None:
            self.close()
        else:
            self.abort()


class FFmpegPipeWriter(RawVideoWriter):
    """Backward-compatible HEVC pipe writer with explicit ``open()``."""

    def __init__(
        self,
        out_path: Path | str,
        in_w: int,
        in_h: int,
        fps: float,
        v_kbps: int,
        preset: str = "medium",
        scale_filter: str | None = None,
        pix_fmt_in: str = "bgr24",
        logger: Any = None,
        stderr_tail_lines: int = 300,
        echo_stderr: bool = False,
        *,
        overwrite: bool = False,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        super().__init__(
            out_path,
            in_w,
            in_h,
            fps,
            v_kbps,
            codec_family="hevc",
            preset=preset,
            scale_filter=scale_filter,
            input_pixel_format=pix_fmt_in,
            overwrite=overwrite,
            ffmpeg_binary=ffmpeg_binary,
            logger=logger,
            stderr_tail_lines=stderr_tail_lines,
            echo_stderr=echo_stderr,
        )


class FFmpegVideoWriter(RawVideoWriter):
    """Backward-compatible writer that opens during construction."""

    def __init__(
        self,
        out_path: Path | str,
        width: int,
        height: int,
        fps: float,
        v_kbps: int,
        codec_family: str = "h264",
        preset: str = "veryfast",
        crf: int | None = None,
        threads: int = 0,
        *,
        resize_frames: bool = True,
        overwrite: bool = False,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        super().__init__(
            out_path,
            width,
            height,
            fps,
            v_kbps,
            codec_family=codec_family,
            preset=preset,
            crf=crf,
            threads=threads,
            resize_frames=resize_frames,
            overwrite=overwrite,
            ffmpeg_binary=ffmpeg_binary,
        )
        self.open()

    def release(self, timeout: float | None = 120) -> None:
        """Backward-compatible alias for :meth:`close`."""

        self.close(timeout=timeout)
