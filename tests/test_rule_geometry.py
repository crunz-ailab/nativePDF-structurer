from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.rule_geometry import (
    drawing_item_axis_segments,
    make_normalized_axis_rect,
    merge_exact_collinear_axis_segments,
)


class RuleGeometryTests(unittest.TestCase):
    #@GPT 轴向矩形只做标量规范化，不改变端点范围。
    def test_normalizes_axis_rect(self) -> None:
        self.assertEqual(
            make_normalized_axis_rect(8, 7, 2, 1),
            (2, 1, 8, 7),
        )

    #@GPT 细水平线和细竖直线分别生成对应方向片段，厚度策略由调用方传入。
    def test_extracts_axis_segments(self) -> None:
        horizontal = ("l", fitz.Point(10, 20), fitz.Point(40, 20.5))
        vertical = ("l", fitz.Point(10, 20), fitz.Point(10.5, 50))
        self.assertEqual(
            drawing_item_axis_segments(
                horizontal,
                "drawing#1",
                maximum_rule_thickness_points=3.0,
            )[0]["orientation"],
            "h",
        )
        self.assertEqual(
            drawing_item_axis_segments(
                vertical,
                "drawing#2",
                maximum_rule_thickness_points=3.0,
            )[0]["orientation"],
            "v",
        )

    #@GPT 同Y且端点相接的片段合并为一条逻辑横线，真实空白则保持分开。
    def test_merges_exact_collinear_segments(self) -> None:
        segments = [
            {
                "orientation": "h",
                "rect": make_normalized_axis_rect(0, 10, 20, 10),
                "position": 10,
                "element_ids": ["drawing#1"],
            },
            {
                "orientation": "h",
                "rect": make_normalized_axis_rect(20, 10, 40, 10),
                "position": 10,
                "element_ids": ["drawing#2"],
            },
            {
                "orientation": "h",
                "rect": make_normalized_axis_rect(50, 10, 60, 10),
                "position": 10,
                "element_ids": ["drawing#3"],
            },
        ]
        rules = merge_exact_collinear_axis_segments(
            segments,
            "h",
            coordinate_decimal_places=3,
        )
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["rect"].x1, 40)
        self.assertEqual(
            rules[0]["element_ids"],
            ["drawing#1", "drawing#2"],
        )


if __name__ == "__main__":
    unittest.main()
