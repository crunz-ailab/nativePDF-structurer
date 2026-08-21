"""Tests for the stable output-directory contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nativePDF_structurer.artifacts import prepare_output_layout


class OutputLayoutTests(unittest.TestCase):
    """Ensure minimal and debug modes create different optional directories."""

    def test_minimal_mode_skips_optional_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            reset_calls: list[Path] = []

            def reset(path: Path) -> None:
                reset_calls.append(path)
                path.mkdir(parents=True)

            layout = prepare_output_layout(
                root,
                reset=reset,
                formula_enabled=False,
                debug_enabled=False,
            )
            self.assertEqual(reset_calls, [root])
            self.assertTrue(layout.source_images.is_dir())
            self.assertFalse(layout.formula_images.exists())
            self.assertFalse(layout.reading_order_debug.exists())

    def test_enabled_optional_directories_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"

            def reset(path: Path) -> None:
                path.mkdir(parents=True)

            layout = prepare_output_layout(
                root,
                reset=reset,
                formula_enabled=True,
                debug_enabled=True,
            )
            self.assertTrue(layout.formula_images.is_dir())
            self.assertTrue(layout.reading_order_debug.is_dir())


if __name__ == "__main__":
    unittest.main()
