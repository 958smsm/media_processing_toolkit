# Image similarity

The package provides two workflows:

- cluster related images using average hash, HSV histograms, or mean color;
- shorten a video by dropping frames similar to the last retained frame.

## Cluster images

Run the package, the unified script, or the installed command:

```powershell
python -m image_similarity -i D:\images -m average-hash
python image_similarity\cluster.py D:\images -m histogram -t 0.99
image-cluster -i D:\images -m mean-color -o D:\clusters
```

With no arguments, the command reads the `image_similarity` section of the
repository `args.yaml`. A sole YAML path selects another configuration file;
CLI values otherwise override the default YAML.

Images are copied into a sibling output directory by default. Source-relative
paths are preserved inside each cluster, so repeated filenames do not collide.
The command refuses to reuse a nonempty output directory unless `--overwrite`
is explicit. Use `--move` only when source images should be removed, and
`--no-singletons` to export only groups containing matches.

Methods and default thresholds:

| Method | Purpose | Default |
| --- | --- | --- |
| `average-hash` | Similar low-frequency structure | `0.95` |
| `histogram` | Similar HSV color distribution | `0.985` |
| `mean-color` | Similar average RGB color | `0.95` |

Use `python -m image_similarity --help` for every option.

## Deduplicate video frames

```powershell
python image_similarity\dedupe_by_similarity.py -i video.mp4 -t 0.95 -p 8
video-dedupe -i D:\videos -o D:\deduped
```

The workflow compares each decoded frame with the last retained frame, writes
retained frames through FFmpeg, and publishes the completed file atomically.
Source files are never moved by default. `--trash` enables moving completed
inputs, and `--trash-dir` chooses the destination.

Both workflows display progress and write rotating logs below the repository
`logs` directory.
