from __future__ import annotations

import unittest

from nativePDF_structurer.markdown_formatter import (
    format_sorted_lines_as_markdown,
)


class MarkdownFormatterTests(unittest.TestCase):
    def test_joins_text_in_same_virtual_row(self) -> None:
        markdown = format_sorted_lines_as_markdown(
            [
                {"virtual_row_index": 1, "kind": "text", "text": "左"},
                {"virtual_row_index": 1, "kind": "text", "text": "右"},
            ]
        )

        self.assertEqual(markdown, "左 右")

    def test_formats_caption_and_heading_rows(self) -> None:
        markdown = format_sorted_lines_as_markdown(
            [
                {
                    "virtual_row_index": 1,
                    "kind": "text",
                    "text": "图 1 设备",
                    "figure_role": "figure_caption",
                },
                {
                    "virtual_row_index": 2,
                    "kind": "text",
                    "text": "原始标题",
                    "heading_level": 2,
                    "heading_text_override": "章节 标题",
                },
            ]
        )

        self.assertEqual(markdown, "**图 1 设备**  \n## 章节 标题")

    def test_deduplicates_text_separators(self) -> None:
        markdown = format_sorted_lines_as_markdown(
            [
                {"virtual_row_index": 1, "kind": "text", "text": "前文"},
                {
                    "virtual_row_index": 2,
                    "kind": "text_separator",
                    "text": "---",
                },
                {
                    "virtual_row_index": 3,
                    "kind": "text_separator",
                    "text": "---",
                },
                {"virtual_row_index": 4, "kind": "text", "text": "后文"},
            ]
        )

        self.assertEqual(markdown, "前文\n\n---\n\n后文")


if __name__ == "__main__":
    unittest.main()
