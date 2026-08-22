from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.rule_bands import (
    detect_rule_line_framework_bbox,
)


def one_group(rules):
    return [rules]


def positions(rules):
    return sorted({rule["position"] for rule in rules})


def ids(rule):
    return set(rule.get("element_ids", []))


class RuleBandFrameworkTests(unittest.TestCase):
    def test_two_rule_framework_requires_visual_and_external_text(self):
        rules = [
            {
                "rect": fitz.Rect(0, 10, 100, 10),
                "position": 10,
                "element_ids": ["drawing#1"],
            },
            {
                "rect": fitz.Rect(0, 50, 100, 50),
                "position": 50,
                "element_ids": ["drawing#2"],
            },
        ]
        bbox, metadata = detect_rule_line_framework_bbox(
            region_bbox=fitz.Rect(55, 15, 95, 45),
            drawings=[],
            page_rect=fitz.Rect(0, 0, 100, 100),
            logical_horizontal_rules=rules,
            text_lines=[{"bbox": fitz.Rect(5, 20, 45, 30)}],
            alignment_tolerance_points=3.0,
            group_horizontal_rules_by_span=one_group,
            unique_rule_positions=positions,
            rule_element_ids=ids,
        )
        self.assertEqual(bbox, fitz.Rect(0, 10, 100, 50))
        self.assertEqual(metadata["kind"], "two_rule_band_framework")
        self.assertEqual(metadata["outside_anchor_text_line_count"], 1)
        self.assertEqual(
            metadata["drawing_element_ids"],
            ["drawing#1", "drawing#2"],
        )

    def test_two_rule_framework_rejects_without_external_text(self):
        rules = [
            {"rect": fitz.Rect(0, 10, 100, 10), "position": 10},
            {"rect": fitz.Rect(0, 50, 100, 50), "position": 50},
        ]
        bbox, metadata = detect_rule_line_framework_bbox(
            region_bbox=fitz.Rect(55, 15, 95, 45),
            drawings=[],
            page_rect=fitz.Rect(0, 0, 100, 100),
            logical_horizontal_rules=rules,
            text_lines=[],
            alignment_tolerance_points=3.0,
            group_horizontal_rules_by_span=one_group,
            unique_rule_positions=positions,
            rule_element_ids=ids,
        )
        self.assertIsNone(bbox)
        self.assertIsNone(metadata)


if __name__ == "__main__":
    unittest.main()
