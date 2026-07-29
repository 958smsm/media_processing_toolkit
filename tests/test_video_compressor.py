from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json, unittest

import video_compressor


class VideoInfoTests(unittest.TestCase):
    @patch.object(video_compressor._core, "run_capture")
    def test_parses_ffprobe_json(self, run_capture) -> None:
        run_capture.return_value = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30000/1001",
                    }
                ],
                "format": {"duration": "12.5"},
            }
        )

        info = video_compressor.ffprobe_video_info("video.mp4")

        self.assertEqual((info.width, info.height), (1920, 1080))
        self.assertAlmostEqual(info.fps, 29.970, places=3)
        self.assertEqual(info.duration, 12.5)


class ScalingTests(unittest.TestCase):
    def test_scales_to_even_dimensions(self) -> None:
        scale_filter, width, height = video_compressor.build_scale_filter(
            1920,
            1080,
            721,
        )

        self.assertEqual(height, 720)
        self.assertEqual(width % 2, 0)
        self.assertEqual(scale_filter, f"scale={width}:{height}")

    def test_does_not_upscale(self) -> None:
        result = video_compressor.build_scale_filter(640, 480, 1080)
        self.assertEqual(result, (None, 640, 480))


class InputResolutionTests(unittest.TestCase):
    def test_expands_directories_and_text_lists(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            first = nested / "first.mp4"
            second = root / "second.mkv"
            first.touch()
            second.touch()
            list_file = root / "inputs.txt"
            list_file.write_text("second.mkv\n", encoding="utf-8")

            directory_results = video_compressor.resolve_input_videos([root])
            list_results = video_compressor.resolve_input_videos([list_file])

            expected = sorted(
                [first.resolve(), second.resolve()],
                key=lambda path: str(path).casefold(),
            )
            self.assertEqual(directory_results, expected)
            self.assertEqual(list_results, [second.resolve()])


class CompressionTests(unittest.TestCase):
    def test_publishes_completed_output_atomically(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.mp4"
            output = root / "output.mp4"
            source.write_bytes(b"source")

            info = video_compressor.VideoInfo(
                width=1280,
                height=720,
                fps=30.0,
                duration=5.0,
                codec="h264",
            )

            def complete_command(command, *_args, **_kwargs) -> None:
                Path(command[-1]).write_bytes(b"encoded")

            options = video_compressor.CompressionOptions(
                hardware="cpu",
                show_progress=False,
            )
            with (
                patch.object(
                    video_compressor._core,
                    "require_executable",
                    side_effect=lambda binary: binary,
                ),
                patch.object(
                    video_compressor._core,
                    "ffprobe_video_info",
                    return_value=info,
                ),
                patch.object(
                    video_compressor._core,
                    "hardware_acceleration_args",
                    return_value=[],
                ),
                patch.object(
                    video_compressor._core,
                    "run_ffmpeg_with_progress",
                    side_effect=complete_command,
                ),
            ):
                result = video_compressor.compress_video(
                    source,
                    output,
                    options,
                )

            self.assertEqual(output.read_bytes(), b"encoded")
            self.assertEqual(result.output_path, output.resolve())
            self.assertFalse((root / "output.partial.mp4").exists())


if __name__ == "__main__":
    unittest.main()
