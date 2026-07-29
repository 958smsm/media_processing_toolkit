from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import logging, sys, unittest

import toolkit_runtime


class YamlDefaultsTests(unittest.TestCase):
    def test_explicit_yaml_path_is_removed_from_parser_arguments(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script = root / "feature.py"
            config = root / "custom.yaml"
            config.write_text("placeholder", encoding="utf-8")
            fake_yaml = SimpleNamespace(
                safe_load=lambda _stream: {"feature": {"value": 7}}
            )

            with patch.dict(sys.modules, {"yaml": fake_yaml}):
                defaults, arguments, selected = (
                    toolkit_runtime.load_yaml_defaults(
                        script,
                        "feature",
                        [str(config)],
                    )
                )

        self.assertEqual(defaults, {"value": 7})
        self.assertEqual(arguments, [])
        self.assertEqual(selected, config.resolve())


class LoggingTests(unittest.TestCase):
    def test_rotates_logs_by_line_count(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            log_directory = Path(temporary_directory)
            handler = toolkit_runtime.LineRotatingFileHandler(
                log_directory,
                max_lines=2,
                backup_count=3,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger = logging.Logger("rotation-test")
            logger.addHandler(handler)

            for index in range(1, 6):
                logger.info("line %d", index)
            handler.close()

            self.assertEqual(
                (log_directory / "log_1.txt").read_text(encoding="utf-8"),
                "line 5\n",
            )
            self.assertEqual(
                (log_directory / "log_2.txt").read_text(encoding="utf-8"),
                "line 3\nline 4\n",
            )
            self.assertEqual(
                (log_directory / "log_3.txt").read_text(encoding="utf-8"),
                "line 1\nline 2\n",
            )


if __name__ == "__main__":
    unittest.main()
