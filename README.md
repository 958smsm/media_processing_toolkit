# Media Processing Toolkit

Practical Python tools for image similarity, FFmpeg-based video processing, and
yt-dlp downloads.

## Installation

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Video tools also require `ffmpeg` and `ffprobe` on `PATH`.

## Configuration and execution

The executable tools support the same three modes:

```powershell
# Load the tool's section from args.yaml
python video_compressor.py

# Load the tool's section from another YAML file
python video_compressor.py D:\configs\video.yaml

# Load args.yaml defaults and override them with CLI arguments
python video_compressor.py -i video.mp4 -c hevc -q high
```

Configuration sections are named after their scripts:

- `ffmpeg_manager`
- `video_compressor`
- `youtube_download`
- `dedupe_by_similarity`

Every parser uses `parse_known_args()`, so host applications may pass additional
arguments. Run a script with `--help` to see its short and long options.

## Logging and progress

Each executable writes debug, info, and error messages under:

```text
logs/<script-name>/log_1.txt
```

Log files rotate through `log_5.txt`, with at most 3,000 lines in each file.
Batch and media operations display progress bars; they can be disabled with
`--no-progress` where appropriate.

## Compress videos

Compress individual videos, folders, globs, or TXT file lists:

```powershell
python video_compressor.py -i video.mp4 -c h264 -q medium
python video_compressor.py -i D:\videos -c hevc -H 1080
python video_compressor.py -i "*.mp4" -2 --hw auto
```

Completed encodes are published atomically, so failed jobs do not replace an
existing destination.

## Remove similar video frames

`image_similarity/dedupe_by_similarity.py` compares each frame with the last
kept frame and encodes retained frames as HEVC:

```powershell
python image_similarity\dedupe_by_similarity.py -i video.mp4 -t 0.95 -p 8
python image_similarity\dedupe_by_similarity.py -i D:\videos -H 1080 -q medium
```

Moving source files to trash is disabled by default. Enable it explicitly with
`--trash` and optionally select a destination with `--trash-dir`.

## Write raw frames with FFmpeg

`ffmpeg_manager.py` provides `RawVideoWriter`, `FFmpegPipeWriter`, and
`FFmpegVideoWriter`, along with bitrate and CUDA-detection helpers.

```python
from ffmpeg_manager import RawVideoWriter, auto_video_kbps

bitrate = auto_video_kbps(1920, 1080, 30, "hevc", "medium")
with RawVideoWriter(
    "output.mp4",
    1920,
    1080,
    30,
    bitrate,
    codec_family="hevc",
) as writer:
    writer.write(frame)
```

Run the module directly to estimate a bitrate:

```powershell
python ffmpeg_manager.py -W 1920 -H 1080 -r 30 -c hevc -q medium
```

## Download videos

Download direct URLs or load URLs from TXT/YAML files:

```powershell
python youtube_download.py -i "https://www.youtube.com/watch?v=..."
python youtube_download.py -i urls.txt -o D:\downloads
python youtube_download.py -i grouped_urls.yaml -a downloaded.txt
```

YAML download inputs may be a URL list or a mapping of relative output folders
to URL lists.

## Image similarity

See [image_similarity/README.md](image_similarity/README.md) for clustering
images by average hash, HSV histogram, mean color, or video-frame similarity.
