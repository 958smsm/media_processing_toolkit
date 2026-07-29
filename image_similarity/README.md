# Image similarity clustering

Group images using one of three lightweight similarity strategies:

- `average-hash` detects images with similar low-frequency structure.
- `histogram` compares HSV color distributions.
- `mean-color` compares average RGB colors with cosine similarity.

## Installation

From the repository root:

```powershell
pip install -r requirements.txt
```

## Usage

Use the package entry point:

```powershell
python -m image_similarity D:\path\to\images --method average-hash
python -m image_similarity D:\path\to\images --method histogram --threshold 0.99
python -m image_similarity D:\path\to\images --method mean-color --output D:\output
```

Or run the unified script:

```powershell
python image_similarity\cluster.py D:\path\to\images --method histogram
```

Run `python -m image_similarity --help` for all options.

By default, images are copied into a sibling output directory. Their relative
source paths are preserved inside each cluster, preventing files with identical
names from overwriting each other. Use `--move` only when the originals should
be removed, and use `--exclude-singletons` to export only groups with matches.
