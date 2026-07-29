from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from image_similarity.core import (
    cluster_features, discover_images, export_clusters,
)


class DiscoverImagesTests(unittest.TestCase):
    def test_discovers_supported_extensions_case_insensitively(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            nested = source / "nested"
            nested.mkdir()
            first = source / "first.JPG"
            second = nested / "second.png"
            first.touch()
            second.touch()
            (nested / "notes.txt").touch()

            discovered = discover_images(source)

            self.assertEqual(discovered, [first.resolve(), second.resolve()])


class ClusterFeaturesTests(unittest.TestCase):
    def test_clusters_against_each_cluster_seed(self) -> None:
        paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        features = dict(zip(paths, [0.0, 0.1, 0.9]))
        clusters = cluster_features(
            features,
            compare=lambda left, right: 1.0 - abs(left - right),
            threshold=0.8,
        )
        self.assertEqual(clusters, [[paths[0], paths[1]], [paths[2]]])

    def test_rejects_out_of_range_threshold(self) -> None:
        with self.assertRaises(ValueError):
            cluster_features({}, lambda _left, _right: 1.0, threshold=1.1)


class ExportClustersTests(unittest.TestCase):
    def test_preserves_relative_paths_for_duplicate_names(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            first = source / "camera_a" / "image.jpg"
            second = source / "camera_b" / "image.jpg"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            exported_count = export_clusters(
                [[first, second]],
                source,
                output,
            )

            self.assertEqual(exported_count, 2)
            self.assertEqual(
                (output / "cluster_0001" / "camera_a" / "image.jpg").read_bytes(),
                b"first",
            )
            self.assertEqual(
                (output / "cluster_0001" / "camera_b" / "image.jpg").read_bytes(),
                b"second",
            )

    def test_rejects_nonempty_output_without_overwrite(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            image = source / "image.jpg"
            image.touch()
            (output / "old.txt").touch()

            with self.assertRaises(FileExistsError):
                export_clusters([[image]], source, output)


if __name__ == "__main__":
    unittest.main()
