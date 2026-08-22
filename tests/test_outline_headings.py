from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.outline_headings import (
    assign_pdf_outline_entries_to_lines,
    assign_pdf_outline_text_matches_to_lines,
    normalize_heading_match_text,
    outline_text_matches,
    strip_heading_number_prefix,
)


class OutlineHeadingTests(unittest.TestCase):
    def test_strict_normalization_and_prefix_removal(self) -> None:
        self.assertEqual(normalize_heading_match_text("１．２ 产品 概述"), "1.2产品概述")
        self.assertEqual(strip_heading_number_prefix("1.2产品概述"), "产品概述")
        entry = {"normalized_title": "1.2产品概述", "structural_title": "产品概述"}
        self.assertTrue(outline_text_matches("产品概述", entry))
        self.assertFalse(outline_text_matches("产品概况", entry))

    def test_assigns_multiline_outline_text_group(self) -> None:
        first = {"text": "下载固件", "bbox": fitz.Rect(10, 10, 60, 20), "block_index": 1, "block_line_index": 0}
        second = {"text": "和其他文件", "bbox": fitz.Rect(10, 21, 70, 31), "block_index": 1, "block_line_index": 1}
        entry = {
            "entry_index": 0,
            "level": 1,
            "normalized_title": normalize_heading_match_text("下载固件和其他文件"),
            "structural_title": normalize_heading_match_text("下载固件和其他文件"),
            "target_point": (10.0, 10.0),
        }
        assignments = assign_pdf_outline_text_matches_to_lines([first, second], [entry])
        match = assignments[id(first)]
        self.assertEqual(match["lines"], [first, second])
        self.assertEqual(match["combined_text"], "下载固件 和其他文件")

    def test_assigns_duplicate_title_nearest_target(self) -> None:
        upper = {"text": "设置", "bbox": fitz.Rect(10, 10, 40, 20), "block_index": 1, "block_line_index": 0}
        lower = {"text": "设置", "bbox": fitz.Rect(10, 100, 40, 110), "block_index": 2, "block_line_index": 0}
        entry = {
            "entry_index": 0,
            "level": 1,
            "normalized_title": "设置",
            "structural_title": "设置",
            "target_point": (10.0, 98.0),
        }
        assignments = assign_pdf_outline_entries_to_lines(
            [upper, lower],
            [entry],
            page_rect=fitz.Rect(0, 0, 100, 200),
            running_header_line_ids=set(),
            target_max_vertical_distance_points=72.0,
            target_max_x_distance_page_ratio=0.35,
            shared_target_tolerance_points=2.0,
        )
        self.assertIn(id(lower), assignments)
        self.assertNotIn(id(upper), assignments)


if __name__ == "__main__":
    unittest.main()
