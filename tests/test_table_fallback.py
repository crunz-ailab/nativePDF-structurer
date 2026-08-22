from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.table_detection import (
    detect_geometry_fallback_tables,
)
from nativePDF_structurer.table_structure import TableStructureDependencies


class TableFallbackTests(unittest.TestCase):
    def test_detects_two_row_visual_grid_path(self) -> None:
        def horizontal(y: float) -> dict:
            rule = {
                "orientation": "h",
                "position": y,
                "rect": fitz.Rect(0, y, 20, y),
            }
            return {"rect": fitz.Rect(0, y, 20, y), "rule": rule}

        def vertical(x: float) -> dict:
            rule = {
                "orientation": "v",
                "position": x,
                "rect": fitz.Rect(x, 0, x, 30),
                "segment_count": 1,
            }
            return {"rect": fitz.Rect(x, 0, x, 30), "rule": rule}

        drawings = [
            horizontal(0),
            horizontal(10),
            horizontal(20),
            vertical(5),
            vertical(15),
            {"rect": fitz.Rect(7, 12, 13, 18)},
        ]
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100))
        text_lines = [
            {
                "text": "a",
                "bbox": fitz.Rect(1, 1, 3, 3),
                "angle": 0,
            },
            {
                "text": "b",
                "bbox": fitz.Rect(6, 1, 8, 3),
                "angle": 0,
            },
            {
                "text": "c",
                "bbox": fitz.Rect(16, 1, 18, 3),
                "angle": 0,
            },
            {
                "text": "d",
                "bbox": fitz.Rect(1, 11, 3, 13),
                "angle": 0,
            },
            {
                "text": "e",
                "bbox": fitz.Rect(6, 11, 8, 13),
                "angle": 0,
            },
            {
                "text": "f",
                "bbox": fitz.Rect(16, 11, 18, 13),
                "angle": 0,
            },
            {
                "text": "g",
                "bbox": fitz.Rect(1, 21, 3, 23),
                "angle": 0,
            },
            {
                "text": "h",
                "bbox": fitz.Rect(6, 21, 8, 23),
                "angle": 0,
            },
            {
                "text": "i",
                "bbox": fitz.Rect(16, 21, 18, 23),
                "angle": 0,
            },
        ]

        result = detect_geometry_fallback_tables(
            page=page,
            text_lines=text_lines,
            excluded_header_footer_rects=[],
            drawings=drawings,
            axis_geometry_index=None,
            rule_alignment_tolerance_points=3.0,
            min_horizontal_rules=4,
            min_populated_rows=3,
            min_populated_columns=2,
            min_vertical_coverage_ratio=0.60,
            native_replacement_overlap_ratio=0.80,
            drawing_axis_rule=lambda drawing, _: drawing.get("rule"),
            merge_collinear_vertical_rules=lambda rules: rules,
            group_horizontal_rules_by_span=lambda rules: [rules],
            table_structure_dependencies=TableStructureDependencies(
                alignment_tolerance_points=3.0,
                skeleton_horizontal_gap_points=1.0,
                drawing_axis_rule=lambda drawing, _: drawing.get("rule"),
            ),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["standard_text_grid"], False)
        self.assertEqual(result[0].metadata["two_row_visual_grid"], True)
        self.assertEqual(result[0].metadata["populated_row_count"], 2)
        self.assertEqual(result[0].metadata["populated_column_count"], 3)

    def test_rejects_incomplete_horizontal_skeleton(self) -> None:
        rules = [
            {
                "orientation": "h",
                "position": 0,
                "rect": fitz.Rect(0, 0, 20, 0),
            },
            {
                "orientation": "h",
                "position": 10,
                "rect": fitz.Rect(0, 10, 20, 10),
            },
            {
                "orientation": "v",
                "position": 5,
                "rect": fitz.Rect(5, 0, 5, 10),
            },
            {
                "orientation": "v",
                "position": 15,
                "rect": fitz.Rect(15, 0, 15, 10),
            },
        ]
        drawings = [
            {"rect": fitz.Rect(rule["rect"]), "rule": rule}
            for rule in rules
        ]
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100))

        result = detect_geometry_fallback_tables(
            page=page,
            text_lines=[],
            excluded_header_footer_rects=[],
            drawings=drawings,
            axis_geometry_index=None,
            rule_alignment_tolerance_points=3.0,
            min_horizontal_rules=4,
            min_populated_rows=3,
            min_populated_columns=2,
            min_vertical_coverage_ratio=0.60,
            native_replacement_overlap_ratio=0.80,
            drawing_axis_rule=lambda drawing, _: drawing.get("rule"),
            merge_collinear_vertical_rules=lambda rules: rules,
            group_horizontal_rules_by_span=lambda rules: [rules],
            table_structure_dependencies=TableStructureDependencies(
                alignment_tolerance_points=3.0,
                skeleton_horizontal_gap_points=1.0,
                drawing_axis_rule=lambda drawing, _: drawing.get("rule"),
            ),
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
