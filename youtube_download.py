"""Download video URLs directly or from TXT/YAML lists with yt-dlp."""

from __future__ import annotations

import argparse, logging, sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Sequence
from urllib.parse import urlparse

from toolkit_runtime import configure_logging, parse_yaml_args, progress_iter

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem
LOGGER = logging.getLogger(FEATURE_NAME)


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
    show_progress: bool = True


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
    if input_path.suffix.lower() == ".txt":
        values = input_path.read_text(encoding="utf-8").splitlines()
        return {Path(): _normalize_urls(values)}
    if input_path.suffix.lower() not in {".yaml", ".yml"}:
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
            downloader_options = {
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
                downloader_options["download_archive"] = str(archive)

            with yt_dlp.YoutubeDL(downloader_options) as downloader:
                for url in progress_iter(
                    urls,
                    "Downloading URLs",
                    enabled=options.show_progress and options.quiet,
                ):
                    attempted += 1
                    LOGGER.info("Downloading %s", url)
                    try:
                        return_code = downloader.download([url])
                        if return_code:
                            raise RuntimeError(
                                f"yt-dlp returned exit code {return_code}"
                            )
                    except Exception as error:
                        failures.append(DownloadFailure(url, str(error)))
                        LOGGER.error("Failed %s: %s", url, error)
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
    """Build the download CLI parser."""

    parser = argparse.ArgumentParser(
        description="Download video URLs directly or from TXT/YAML lists."
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="One or more URLs, TXT files, or YAML files.",
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "-f",
        "--format",
        dest="format_selector",
        help="yt-dlp format selector.",
    )
    parser.add_argument("-m", "--merge-format")
    parser.add_argument("-a", "--archive", type=Path)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the download CLI."""

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
            show_progress=bool(args.show_progress),
        )
        summary = download_youtube(raw_inputs, options)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    log.info(
        "Downloaded %d/%d URLs with %d failures.",
        summary.succeeded,
        summary.attempted,
        len(summary.failures),
    )
    return 1 if summary.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
