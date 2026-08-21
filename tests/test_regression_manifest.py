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
    #@GPT 测试同时覆盖嵌套相对路径和稳定排序，防止不同文件系统遍历顺序造成无意义的manifest变动。
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

    #@GPT 成功校验和内容变化使用同一份manifest，确保工具不是只比较文件名或大小。
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

    #@GPT 路径集合是输出契约的一部分；单独验证缺失和新增文件，避免图片重复或CSV消失时hash工具仍误报通过。
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
