from pathlib import Path
from unittest.mock import patch
import unittest

import ffmpeg_manager


class BitrateTests(unittest.TestCase):
    def test_estimates_expected_h264_bitrate(self) -> None:
        bitrate = ffmpeg_manager.auto_video_kbps(
            1920,
            1080,
            30,
            "h264",
            "medium",
        )
        self.assertEqual(bitrate, 6221)

    def test_normalizes_codec_and_quality_aliases(self) -> None:
        self.assertEqual(ffmpeg_manager.normalize_codec("h265"), "hevc")
        self.assertEqual(
            ffmpeg_manager.normalize_quality("very high"),
            "very-high",
        )

    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            ffmpeg_manager.auto_video_kbps(0, 1080, 30)


class HardwareAccelerationTests(unittest.TestCase):
    @patch.object(ffmpeg_manager, "cuda_works_for_file", return_value=True)
    @patch.object(ffmpeg_manager, "nvidia_gpu_present", return_value=True)
    @patch.object(
        ffmpeg_manager,
        "ffmpeg_supports_hwaccel",
        return_value=True,
    )
    def test_auto_enables_working_cuda(
        self,
        _supports_cuda,
        _gpu_present,
        _cuda_works,
    ) -> None:
        arguments = ffmpeg_manager.hardware_acceleration_args(
            Path("video.mp4"),
            "auto",
        )
        self.assertEqual(arguments, ["-hwaccel", "cuda"])

    def test_cpu_mode_does_not_probe_hardware(self) -> None:
        with patch.object(
            ffmpeg_manager,
            "ffmpeg_supports_hwaccel",
        ) as probe:
            arguments = ffmpeg_manager.hardware_acceleration_args(
                Path("video.mp4"),
                "cpu",
            )
        self.assertEqual(arguments, [])
        probe.assert_not_called()


class RawVideoWriterTests(unittest.TestCase):
    def test_builds_hevc_command_without_starting_ffmpeg(self) -> None:
        writer = ffmpeg_manager.RawVideoWriter(
            "output.mp4",
            640,
            480,
            25,
            900,
            codec_family="hevc",
            overwrite=True,
        )
        command = writer.build_command()
        self.assertIn("libx265", command)
        self.assertIn("hvc1", command)
        self.assertIn("900k", command)
        self.assertEqual(command[-1], "output.mp4")

    def test_preserves_hevc_thread_and_low_memory_options(self) -> None:
        writer = ffmpeg_manager.FFmpegPipeWriter(
            "output.mp4",
            640,
            480,
            25,
            900,
            threads=4,
            low_memory=True,
            overwrite=True,
        )

        command = writer.build_command()
        parameters = command[command.index("-x265-params") + 1]

        self.assertIn("pools=4", parameters)
        self.assertIn("frame-threads=2", parameters)
        self.assertIn("rc-lookahead=5", parameters)
        self.assertIn("bframes=2", parameters)
        self.assertIn("ref=2", parameters)
        self.assertIn("lookahead-slices=0", parameters)
        self.assertIn("-threads", command)
        self.assertEqual(command[command.index("-threads") + 1], "4")


if __name__ == "__main__":
    unittest.main()
