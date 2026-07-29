"""YAML- and CLI-driven entry point for video frame deduplication."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_compressor as _video_compressor

sys.modules.setdefault("shrink_vid_advanced", _video_compressor)

from image_similarity import _dedupe_by_similarity_core as _core

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem
_core.HERE = HERE
_core.FEATURE_NAME = FEATURE_NAME

DedupeOptions = _core.DedupeOptions
DedupeResult = _core.DedupeResult
build_parser = _core.build_parser
deduplicate_video = _core.deduplicate_video
main = _core.main
output_path_for = _core.output_path_for
preprocess_for_compare = _core.preprocess_for_compare
safe_move_to_trash = _core.safe_move_to_trash
similarity_percent = _core.similarity_percent
validate_options = _core.validate_options

__all__ = [
    "DedupeOptions",
    "DedupeResult",
    "build_parser",
    "deduplicate_video",
    "main",
    "output_path_for",
    "preprocess_for_compare",
    "safe_move_to_trash",
    "similarity_percent",
    "validate_options",
]

if __name__ == "__main__":
    raise SystemExit(main())
