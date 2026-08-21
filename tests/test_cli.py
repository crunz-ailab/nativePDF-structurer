"""Tests for the public command-line interface and its help contract."""

from __future__ import annotations

import unittest

from nativePDF_structurer.cli import build_parser


class CommandLineInterfaceTests(unittest.TestCase):
    """Keep user-selectable parser behavior discoverable from ``-h``."""

    #@GPT 项目将通过命令行而不是修改源码选择执行效果；直接检查 argparse 生成的帮助页，避免工程拆分时遗漏公开开关。
    def test_help_lists_public_execution_controls(self) -> None:
        """Expose page selection and optional processing modes in help."""

        help_text = build_parser().format_help()
        for option in (
            "-h, --help",
            "--pages",
            "--debug-output",
            "--heading-detection",
            "--font-heading-inference",
            "--table-header-detection",
            "--keep-header-text",
            "--formula-detection",
            "--pdf-outline-heading-assignment",
            "--force-pure-text-table-to-markdown",
        ):
            self.assertIn(option, help_text)


if __name__ == "__main__":
    unittest.main()
