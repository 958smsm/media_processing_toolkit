from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys, unittest

import image_video_converter


class DiscoveryTests(unittest.TestCase):
    def test_discovers_images_in_natural_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame_10 = root / "frame_10.jpg"
            frame_2 = root / "frame_2.jpg"
            frame_10.touch()
            frame_2.touch()
            (root / "notes.txt").touch()

            images = image_video_converter.discover_images(root)

            self.assertEqual(images, [frame_2.resolve(), frame_10.resolve()])


class SamplingTests(unittest.TestCase):
    def test_calculates_step_from_interval(self) -> None:
        self.assertEqual(
            image_video_converter.sampling_frame_step(
                30,
                interval_seconds=0.5,
            ),
            15,
        )

    def test_calculates_step_from_target_fps(self) -> None:
        self.assertEqual(
            image_video_converter.sampling_frame_step(30, target_fps=10),
            3,
        )

    def test_rejects_two_sampling_modes(self) -> None:
        with self.assertRaises(ValueError):
            image_video_converter.sampling_frame_step(
                30,
                interval_seconds=1,
                target_fps=5,
            )

    def test_formats_frame_timecode(self) -> None:
        self.assertEqual(
            image_video_converter.frame_to_timecode(45, 30),
            "00_00_01_500",
        )


class ImagesToVideoTests(unittest.TestCase):
    def test_encodes_images_and_publishes_output(self) -> None:
        read_paths: list[str] = []

        class FakeFrame:
            shape = (480, 640, 3)

        class FakeWriter:
            def __init__(self, output_path, *_args) -> None:
                self.output_path = Path(output_path)
                self.frames = []

            def isOpened(self) -> bool:
                return True

            def write(self, frame) -> None:
                self.frames.append(frame)

            def release(self) -> None:
                self.output_path.write_bytes(b"video")

        def read_image(path):
            read_paths.append(Path(path).name)
            return FakeFrame()

        fake_cv2 = SimpleNamespace(
            imread=read_image,
            resize=lambda frame, _size: frame,
            VideoWriter=FakeWriter,
            VideoWriter_fourcc=lambda *_codec: 1,
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            images.mkdir()
            (images / "frame_10.jpg").touch()
            (images / "frame_2.jpg").touch()
            output = root / "output.mp4"

            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                result = image_video_converter.images_to_video(
                    images,
                    output,
                    show_progress=False,
                )

            self.assertEqual(output.read_bytes(), b"video")
            self.assertEqual(result.input_count, 2)
            self.assertEqual(
                read_paths,
                ["frame_2.jpg", "frame_2.jpg", "frame_10.jpg"],
            )


class VideoToImagesTests(unittest.TestCase):
    def test_extracts_frames_at_target_fps(self) -> None:
        class FakeCapture:
            def __init__(self, _path) -> None:
                self.frames = [object() for _index in range(5)]

            def isOpened(self) -> bool:
                return True

            def get(self, property_id):
                if property_id == 1:
                    return 10.0
                if property_id == 2:
                    return 5
                return 0

            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop(0)

            def release(self) -> None:
                return None

        def write_image(path, _frame, _parameters) -> bool:
            Path(path).write_bytes(b"image")
            return True

        fake_cv2 = SimpleNamespace(
            CAP_PROP_FPS=1,
            CAP_PROP_FRAME_COUNT=2,
            IMWRITE_JPEG_QUALITY=3,
            IMWRITE_PNG_COMPRESSION=4,
            IMWRITE_WEBP_QUALITY=5,
            VideoCapture=FakeCapture,
            imwrite=write_image,
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.mp4"
            source.touch()

            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                result = image_video_converter.video_to_images(
                    source,
                    root / "frames",
                    target_fps=5,
                    show_progress=False,
                )

            self.assertEqual(result.decoded_frames, 5)
            self.assertEqual(result.saved_frames, 3)
            self.assertEqual(result.frame_step, 2)
            self.assertEqual(len(list(result.output_dir.glob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
