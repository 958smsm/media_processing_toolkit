"""Unified entry point for image similarity clustering.

Run this file directly or use ``python -m image_similarity``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_similarity.cli import main
from image_similarity.core import ClusterResult, cluster_images
from image_similarity.methods import DEFAULT_METHOD, METHODS

__all__ = [
    "ClusterResult",
    "DEFAULT_METHOD",
    "METHODS",
    "cluster_images",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
