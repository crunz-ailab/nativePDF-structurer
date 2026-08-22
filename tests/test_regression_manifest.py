from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nativePDF_structurer.regression_manifest import (
    build_manifest,
    verify_manifest,
    write_manifest,
)


class RegressionManifestTests(unittest.TestCase):
    def test_build_manifest_uses_sorted_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "nested").mkdir()
            (root / "z.txt").write_bytes(b"z")
            (root / "nested" / "a.bin").write_bytes(b"abc")

            manifest = build_manifest(root, label="sample")

            self.assertEqual(manifest["label"], "sample")
            self.assertEqual(
                [record["path"] for record in manifest["files"]],
                ["nested/a.bin", "z.txt"],
            )

    def test_verify_manifest_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "output"
            root.mkdir()
            output_file = root / "document.md"
            output_file.write_bytes(b"before")
            manifest_path = Path(temporary_dir) / "manifest.json"
            write_manifest(root, manifest_path)

            self.assertEqual(verify_manifest(root, manifest_path), [])
            output_file.write_bytes(b"after!")
            self.assertEqual(
                verify_manifest(root, manifest_path),
                ["sha256 mismatch: document.md"],
            )

    def test_verify_manifest_detects_missing_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "output"
            root.mkdir()
            expected_file = root / "expected.txt"
            expected_file.write_bytes(b"expected")
            manifest_path = Path(temporary_dir) / "manifest.json"
            write_manifest(root, manifest_path)

            expected_file.unlink()
            (root / "unexpected.txt").write_bytes(b"unexpected")
            self.assertEqual(
                verify_manifest(root, manifest_path),
                ["missing: expected.txt", "unexpected: unexpected.txt"],
            )


if __name__ == "__main__":
    unittest.main()
