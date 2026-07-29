"""Download YouTube-compatible URLs with yt-dlp."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse


@dataclass(frozen=True)
class DownloadOptions:
    """Settings passed to yt-dlp."""

    output_dir: Path | None = None
    format_selector: str = "bestvideo+bestaudio/best"
    merge_output_format: str = "mp4"
    archive_path: Path | None = None
    no_playlist: bool = False
    continue_on_error: bool = True
    quiet: bool = False


@dataclass(frozen=True)
class DownloadFailure:
    """One URL that yt-dlp could not download."""

    url: str
    message: str


@dataclass(frozen=True)
class DownloadSummary:
    """Aggregate result for a download batch."""

    attempted: int
    succeeded: int
    failures: tuple[DownloadFailure, ...]


def is_url(value: str) -> bool:
    """Return whether a value is an HTTP(S) URL with a host."""

    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_urls(values: Iterable[object]) -> list[str]:
    raw_urls = [str(value or "").strip() for value in values]
    resume_positions = [
        index
        for index, value in enumerate(raw_urls)
        if value.casefold() == "resume"
    ]
    if resume_positions:
        raw_urls = raw_urls[resume_positions[-1] + 1 :]

    urls: list[str] = []
    for value in raw_urls:
        if (
            not value
            or value.startswith("#")
            or value.casefold() == "skip"
        ):
            continue
        if not is_url(value):
            raise ValueError(f"Invalid download URL: {value!r}")
        urls.append(value)
    return urls


def _safe_relative_folder(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return Path()

    folder = PurePath(raw)
    if folder.is_absolute() or ".." in folder.parts:
        raise ValueError(f"Unsafe output folder in input file: {raw!r}")
    return Path(*[part for part in folder.parts if part not in {"", "."}])


def load_download_groups(input_value: Path | str) -> dict[Path, list[str]]:
    """Load one URL, text list, or YAML mapping into output groups."""

    raw = str(input_value).strip()
    if is_url(raw):
        return {Path(): [raw]}

    input_path = Path(raw).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Download input does not exist: {input_path}")

    extension = input_path.suffix.lower()
    if extension == ".txt":
        values = input_path.read_text(encoding="utf-8").splitlines()
        return {Path(): _normalize_urls(values)}

    if extension not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported download input {input_path}; use URL, TXT, or YAML."
        )

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyYAML is required to read YAML download lists."
        ) from error

    with input_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if isinstance(loaded, list):
        return {Path(): _normalize_urls(loaded)}
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"YAML input must be a URL list or folder-to-URLs mapping: "
            f"{input_path}"
        )

    groups: dict[Path, list[str]] = {}
    for folder, values in loaded.items():
        if isinstance(values, str) or not isinstance(values, Iterable):
            raise ValueError(
                f"URL group {folder!r} must contain a list of URLs."
            )
        groups[_safe_relative_folder(folder)] = _normalize_urls(values)
    return groups


def default_output_dir(input_value: Path | str) -> Path:
    """Choose a predictable default output directory for one input."""

    raw = str(input_value).strip()
    if is_url(raw):
        return Path.cwd() / "yt_videos"
    return Path(raw).expanduser().resolve().parent / "yt_videos"


def download_youtube(
    inputs: Iterable[Path | str],
    options: DownloadOptions = DownloadOptions(),
) -> DownloadSummary:
    """Download every URL from direct, text, or YAML inputs."""

    try:
        import yt_dlp
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "yt-dlp is required; install packages from requirements.txt."
        ) from error

    attempted = 0
    succeeded = 0
    failures: list[DownloadFailure] = []

    for input_value in inputs:
        groups = load_download_groups(input_value)
        root_output = (
            options.output_dir.expanduser().resolve()
            if options.output_dir is not None
            else default_output_dir(input_value)
        )

        for relative_folder, urls in groups.items():
            destination = root_output / relative_folder
            destination.mkdir(parents=True, exist_ok=True)
            yt_dlp_options = {
                "outtmpl": str(
                    destination / "%(title).200B [%(id)s].%(ext)s"
                ),
                "format": options.format_selector,
                "merge_output_format": options.merge_output_format,
                "noplaylist": options.no_playlist,
                "quiet": options.quiet,
                "continuedl": True,
                "ignoreerrors": False,
            }
            if options.archive_path is not None:
                archive = options.archive_path.expanduser().resolve()
                archive.parent.mkdir(parents=True, exist_ok=True)
                yt_dlp_options["download_archive"] = str(archive)

            with yt_dlp.YoutubeDL(yt_dlp_options) as downloader:
                for url in urls:
                    attempted += 1
                    try:
                        return_code = downloader.download([url])
                        if return_code:
                            raise RuntimeError(
                                f"yt-dlp returned exit code {return_code}"
                            )
                    except Exception as error:
                        failures.append(DownloadFailure(url, str(error)))
                        if not options.continue_on_error:
                            return DownloadSummary(
                                attempted=attempted,
                                succeeded=succeeded,
                                failures=tuple(failures),
                            )
                    else:
                        succeeded += 1

    return DownloadSummary(
        attempted=attempted,
        succeeded=succeeded,
        failures=tuple(failures),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download video URLs directly or from TXT/YAML lists."
    )
    parser.add_argument(
        "inputs",
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
        default="bestvideo+bestaudio/best",
        help="yt-dlp format selector.",
    )
    parser.add_argument("--merge-format", default="mp4")
    parser.add_argument(
        "--archive",
        type=Path,
        help="yt-dlp archive file used to skip completed downloads.",
    )
    parser.add_argument("--no-playlist", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = DownloadOptions(
        output_dir=args.output,
        format_selector=args.format_selector,
        merge_output_format=args.merge_format,
        archive_path=args.archive,
        no_playlist=args.no_playlist,
        continue_on_error=not args.fail_fast,
        quiet=args.quiet,
    )

    try:
        summary = download_youtube(args.inputs, options)
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for failure in summary.failures:
        print(f"error: {failure.url}: {failure.message}", file=sys.stderr)
    print(
        f"Downloaded {summary.succeeded}/{summary.attempted} URLs "
        f"with {len(summary.failures)} failures."
    )
    return 1 if summary.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
