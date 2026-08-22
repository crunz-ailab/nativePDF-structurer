from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.table_skeleton import (
    find_axis_skeleton_components,
    select_budgeted_axis_rules_without_complete_skeleton,
    select_cluster_table_budget,
    select_cluster_table_skeleton_drawings,
)


class TableSkeletonTests(unittest.TestCase):
    def test_skeleton_selection_injects_helpers_and_preserves_source_ids(self) -> None:
        drawing = {
            "rect": fitz.Rect(0, 0, 20, 20),
            "items": [],
            "color": (0, 0, 0),
        }
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[],
            text_lines=[],
        )
        rule = {
            "element_ids": ["d:0"],
            "rect": fitz.Rect(0, 10, 20, 10),
            "orientation": "h",
            "position": 10,
        }

        result = select_cluster_table_skeleton_drawings(
            cluster_drawings=[drawing],
            cluster_bbox=fitz.Rect(0, 0, 20, 20),
            axis_geometry_index={
                "logical_horizontal_rules": [rule],
                "logical_vertical_rules": [],
            },
            page_inventory=inventory,
            drawing_budget=2000,
            alignment_tolerance=1.0,
            minimum_structural_span=1.0,
            coordinate_decimal_places=3,
        )

        self.assertEqual(result["mode"], "axis_rules_without_complete_skeleton")
        self.assertEqual(result["source_element_ids"], ["d:0"])
        self.assertEqual(len(result["drawings"]), 1)
        self.assertEqual(result["drawings"][0]["_source_element_ids"], ["d:0"])
        self.assertEqual(
            result["drawings"][0]["items"][0][0],
            "l",
        )

    def test_axis_rule_budget_is_direction_balanced(self) -> None:
        rules = [
            {
                "orientation": "h",
                "position": 0,
                "rect": fitz.Rect(0, 0, 100, 0),
            },
            {
                "orientation": "h",
                "position": 50,
                "rect": fitz.Rect(0, 50, 80, 50),
            },
            {
                "orientation": "v",
                "position": 0,
                "rect": fitz.Rect(0, 0, 0, 100),
            },
            {
                "orientation": "v",
                "position": 50,
                "rect": fitz.Rect(50, 0, 50, 80),
            },
        ]

        selected = select_budgeted_axis_rules_without_complete_skeleton(
            rules=rules,
            cluster_bbox=fitz.Rect(0, 0, 100, 100),
            budget=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {rule["orientation"] for rule in selected},
            {"h", "v"},
        )
        self.assertEqual(selected, [rules[0], rules[2]])

    def test_cluster_budget_uses_spatial_order(self) -> None:
        drawings = [
            {"rect": fitz.Rect(20, 20, 30, 30), "seqno": 2},
            {"rect": fitz.Rect(10, 10, 20, 20), "seqno": 3},
            {"rect": fitz.Rect(5, 10, 15, 20), "seqno": 1},
        ]

        selected = select_cluster_table_budget(
            drawings,
            drawing_budget=2,
        )

        self.assertEqual(selected, [drawings[2], drawings[1]])

    def test_axis_skeleton_component_detects_multi_cell_grid(self) -> None:
        horizontal_rules = [
            {
                "orientation": "h",
                "position": y,
                "rect": fitz.Rect(0, y, 20, y),
                "element_ids": ["d:0"],
            }
            for y in (0, 10, 20)
        ]
        vertical_rules = [
            {
                "orientation": "v",
                "position": x,
                "rect": fitz.Rect(x, 0, x, 20),
                "element_ids": ["d:0"],
            }
            for x in (0, 20)
        ]

        components = find_axis_skeleton_components(
            axis_geometry_index={
                "logical_horizontal_rules": horizontal_rules,
                "logical_vertical_rules": vertical_rules,
            },
            cluster_bbox=fitz.Rect(0, 0, 20, 20),
            allowed_element_ids={"d:0"},
            alignment_tolerance=1.0,
            minimum_structural_span=1.0,
            coordinate_decimal_places=3,
        )

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["horizontal_position_count"], 3)
        self.assertEqual(components[0]["vertical_position_count"], 2)
        self.assertTrue(components[0]["outer_envelope_supported"])
        self.assertEqual(components[0]["source_element_ids"], ["d:0"])


if __name__ == "__main__":
    unittest.main()
