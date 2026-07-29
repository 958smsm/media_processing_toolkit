from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys, unittest

import youtube_download


class InputParsingTests(unittest.TestCase):
    def test_detects_http_urls(self) -> None:
        self.assertTrue(youtube_download.is_url("https://youtu.be/example"))
        self.assertFalse(youtube_download.is_url("not-a-url"))

    def test_text_resume_marker_skips_prior_urls(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "urls.txt"
            input_file.write_text(
                "https://example.com/old\n"
                "resume\n"
                "https://example.com/new\n",
                encoding="utf-8",
            )

            groups = youtube_download.load_download_groups(input_file)

            self.assertEqual(
                groups,
                {Path(): ["https://example.com/new"]},
            )

    def test_rejects_parent_folder_traversal(self) -> None:
        with self.assertRaises(ValueError):
            youtube_download._core._safe_relative_folder("../outside")


class DownloadTests(unittest.TestCase):
    def test_downloads_direct_url_with_injected_yt_dlp(self) -> None:
        downloaded: list[str] = []

        class FakeYoutubeDL:
            def __init__(self, options) -> None:
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def download(self, urls) -> int:
                downloaded.extend(urls)
                return 0

        fake_module = SimpleNamespace(YoutubeDL=FakeYoutubeDL)
        with TemporaryDirectory() as temporary_directory:
            options = youtube_download.DownloadOptions(
                output_dir=Path(temporary_directory),
            )
            with patch.dict(sys.modules, {"yt_dlp": fake_module}):
                summary = youtube_download.download_youtube(
                    ["https://example.com/video"],
                    options,
                )

        self.assertEqual(downloaded, ["https://example.com/video"])
        self.assertEqual(summary.attempted, 1)
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.failures, ())


if __name__ == "__main__":
    unittest.main()
