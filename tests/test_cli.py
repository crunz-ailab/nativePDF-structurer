"""Tests for the public command-line interface and its help contract."""

from __future__ import annotations

import unittest

from nativePDF_structurer.cli import build_parser


class CommandLineInterfaceTests(unittest.TestCase):
    """Keep user-selectable parser behavior discoverable from ``-h``."""

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
