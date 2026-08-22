from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.table_detection import (
    detect_simple_cluster_frame_tables,
)
from nativePDF_structurer.table_structure import TableStructureDependencies


class TableFrameDetectionTests(unittest.TestCase):
    def test_detects_local_frame_with_two_populated_columns(self) -> None:
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
        drawings = [{"rule": rule} for rule in rules]
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100))
        text_lines = [
            {
                "text": "left",
                "bbox": fitz.Rect(1, 2, 4, 8),
                "angle": 0,
            },
            {
                "text": "right",
                "bbox": fitz.Rect(7, 2, 13, 8),
                "angle": 0,
            },
        ]

        result = detect_simple_cluster_frame_tables(
            page=page,
            text_lines=text_lines,
            excluded_header_footer_rects=[],
            cluster_bbox=fitz.Rect(0, 0, 20, 10),
            drawings=drawings,
            axis_geometry_index=None,
            page_inventory=None,
            drawing_axis_rule=lambda drawing, _: drawing["rule"],
            group_horizontal_rules_by_span=lambda horizontal: [horizontal],
            rect_is_mostly_inside_any=lambda *_: False,
            alignment_tolerance=1.0,
            table_structure_dependencies=TableStructureDependencies(
                alignment_tolerance_points=1.0,
                skeleton_horizontal_gap_points=1.0,
                drawing_axis_rule=lambda drawing, _: drawing["rule"],
            ),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].detection_source, "cluster_frame_fallback")
        self.assertEqual(result[0].extract(), [["left", "right", None]])
        self.assertEqual(result[0].metadata["separator_count"], 2)


if __name__ == "__main__":
    unittest.main()
