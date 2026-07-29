"""Shared YAML, logging, and progress helpers for command-line tools."""

from __future__ import annotations

import logging, sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class LineRotatingFileHandler(logging.Handler):
    """Rotate ``log_1.txt`` through ``log_5.txt`` by line count."""

    def __init__(
        self,
        log_directory: Path,
        *,
        max_lines: int = 3_000,
        backup_count: int = 5,
    ) -> None:
        super().__init__()
        if max_lines <= 0 or backup_count <= 0:
            raise ValueError("max_lines and backup_count must be positive.")

        self.log_directory = log_directory
        self.max_lines = max_lines
        self.backup_count = backup_count
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.current_path = self.log_directory / "log_1.txt"
        self._line_count = self._count_lines(self.current_path)

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return sum(1 for _line in stream)

    def _rotate(self) -> None:
        oldest = self.log_directory / f"log_{self.backup_count}.txt"
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.log_directory / f"log_{index}.txt"
            if source.exists():
                source.replace(
                    self.log_directory / f"log_{index + 1}.txt"
                )
        self._line_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._line_count >= self.max_lines:
                self._rotate()
            message = self.format(record)
            with self.current_path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
            self._line_count += message.count("\n") + 1
        except Exception:
            self.handleError(record)


def configure_logging(
    script_path: Path,
    feature_name: str,
    *,
    verbose: bool = False,
) -> logging.Logger:
    """Configure console and rotating text-file logging."""

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    marker = f"media_processing_toolkit:{feature_name}"
    if not any(
        getattr(handler, "_toolkit_marker", None) == marker
        for handler in root_logger.handlers
    ):
        log_directory = script_path.parent / "logs" / feature_name
        file_handler = LineRotatingFileHandler(log_directory)
        file_handler._toolkit_marker = marker
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler._toolkit_marker = marker
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s: %(message)s")
        )
        root_logger.addHandler(console_handler)

    logger = logging.getLogger(feature_name)
    logger.debug("Logging initialized in %s", script_path.parent / "logs")
    return logger


def _default_yaml_path(script_path: Path) -> Path:
    candidates = (
        script_path.parent / "args.yaml",
        script_path.parent.parent / "args.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_yaml_defaults(
    script_path: Path,
    feature_name: str,
    argv: Sequence[str] | None,
) -> tuple[dict[str, Any], list[str], Path]:
    """Load defaults and remove an explicit YAML path from CLI arguments."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    default_yaml_path = _default_yaml_path(script_path)
    if any(argument in {"-h", "--help"} for argument in arguments):
        return {}, arguments, default_yaml_path

    if (
        len(arguments) == 1
        and not arguments[0].startswith("-")
        and Path(arguments[0]).suffix.lower() in {".yaml", ".yml"}
    ):
        yaml_path = Path(arguments[0]).expanduser().resolve()
        parse_arguments: list[str] = []
    else:
        yaml_path = default_yaml_path
        parse_arguments = arguments

    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML configuration not found: {yaml_path}")

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyYAML is required; install packages from requirements.txt."
        ) from error

    with yaml_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")

    defaults = loaded.get(feature_name, {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise ValueError(
            f"YAML section {feature_name!r} must be a mapping: {yaml_path}"
        )
    return dict(defaults), parse_arguments, yaml_path


def progress_iter(
    values: Iterable[T],
    description: str,
    *,
    enabled: bool = True,
) -> Iterator[T]:
    """Iterate with tqdm, falling back to a simple count display."""

    items = list(values)
    if not enabled:
        yield from items
        return

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        total = len(items)
        for index, item in enumerate(items, start=1):
            print(
                f"\r{description}: {index}/{total}",
                end="",
                flush=True,
            )
            yield item
        if items:
            print()
        return

    yield from tqdm(items, desc=description, dynamic_ncols=True)
