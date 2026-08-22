from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.table_structure import (
    TableStructureDependencies,
    configure_table_container_claim,
    drawing_is_table_container_structure,
    merge_horizontal_rules_for_table_skeleton,
    refine_table_claim_confidence,
    table_cell_boundary_positions,
    unique_rule_positions,
)


class TableStructurePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dependencies = TableStructureDependencies(
            alignment_tolerance_points=3.0,
            skeleton_horizontal_gap_points=1.0,
            drawing_axis_rule=lambda drawing, _: drawing.get("rule"),
        )

    def test_unique_rule_positions_groups_with_tolerance(self) -> None:
        rules = [
            {"position": 0.0},
            {"position": 1.0},
            {"position": 10.0},
        ]

        self.assertEqual(
            unique_rule_positions(rules, tolerance=3.0),
            [0.5, 10.0],
        )

    def test_horizontal_skeleton_merge_respects_one_point_gap(self) -> None:
        rules = [
            {
                "orientation": "h",
                "position": 10.0,
                "rect": fitz.Rect(0, 10, 5, 10),
                "element_id": "d:0",
            },
            {
                "orientation": "h",
                "position": 10.0,
                "rect": fitz.Rect(6, 10, 10, 10),
                "element_id": "d:1",
            },
            {
                "orientation": "h",
                "position": 10.0,
                "rect": fitz.Rect(20, 10, 25, 10),
                "element_id": "d:2",
            },
        ]

        merged = merge_horizontal_rules_for_table_skeleton(
            rules,
            dependencies=self.dependencies,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["rect"], fitz.Rect(0, 10, 10, 10))
        self.assertEqual(merged[0]["element_ids"], ["d:0", "d:1"])
        self.assertEqual(merged[1]["element_ids"], ["d:2"])

    def test_container_structure_uses_cell_boundaries(self) -> None:
        table = types.SimpleNamespace(
            cells=[
                (0, 0, 10, 10),
                (10, 0, 20, 10),
            ]
        )
        self.assertEqual(
            table_cell_boundary_positions(table),
            ([0.0, 10.0, 20.0], [0.0, 10.0]),
        )
        boundary_line = {
            # "rect": fitz.Rect(0, 0, 20, 0),
            "rect": fitz.Rect(0, 0, 20, 0.5),
            "rule": {
                "orientation": "h",
                # "position": 0.0,
                # "rect": fitz.Rect(0, 0, 20, 0),
                "position": 0.25,
                "rect": fitz.Rect(0, 0, 20, 0.5),
            },
        }
        internal_box = {
            # "rect": fitz.Rect(3, 3, 7, 7),
            "rect": fitz.Rect(4, 4, 6, 6),
            "fill": (0.5, 0.5, 0.5),
            # "items": [("re", fitz.Rect(3, 3, 7, 7), 1)],
            "items": [("re", fitz.Rect(4, 4, 6, 6), 1)],
        }

        self.assertTrue(
            drawing_is_table_container_structure(
                drawing=boundary_line,
                table=table,
                table_bbox=fitz.Rect(0, 0, 20, 10),
                page_rect=fitz.Rect(0, 0, 100, 100),
                dependencies=self.dependencies,
            )
        )
        self.assertFalse(
            drawing_is_table_container_structure(
                drawing=internal_box,
                table=table,
                table_bbox=fitz.Rect(0, 0, 20, 10),
                page_rect=fitz.Rect(0, 0, 100, 100),
                dependencies=self.dependencies,
            )
        )

    def test_full_table_claim_owns_bbox_content(self) -> None:
        drawing = {
            "rect": fitz.Rect(0, 0, 20, 0),
            "rule": {
                "orientation": "h",
                "position": 0.0,
                "rect": fitz.Rect(0, 0, 20, 0),
            },
        }
        image = {"bbox": fitz.Rect(5, 2, 15, 8)}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[image],
            text_lines=[],
        )
        table = types.SimpleNamespace(cells=[(0, 0, 20, 10)])
        candidate = {
            "table_number": 1,
            "table": table,
            "bbox": fitz.Rect(0, 0, 20, 10),
            "raw_row_count": 2,
            "raw_column_count": 2,
            "nonempty_cell_count": 2,
            "table_owns_full_visual_bbox": True,
        }
        claim = {
            "claimed_element_ids": ["d:0"],
            "feature_element_ids": ["d:0"],
            "input_element_ids": ["d:0"],
            "metadata": {},
            "reason_codes": [],
        }

        result = configure_table_container_claim(
            claim=claim,
            candidate=candidate,
            page_inventory=inventory,
            page_rect=inventory.page_rect,
            dependencies=self.dependencies,
        )

        self.assertTrue(result["logical_table_container"])
        self.assertEqual(result["container_element_id"], "tc:0")
        self.assertEqual(
            claim["claimed_element_ids"],
            ["d:0", "i:0", "tc:0"],
        )
        self.assertEqual(result["internal_visual_drawing_ids"], [])

    def test_confidence_refinement_preserves_source_score(self) -> None:
        claim = {"confidence": 0.85, "metadata": {}}
        candidate = {
            "table_cell_count": 4,
            "nonempty_cell_count": 2,
            "table_visual_metadata": {
                "image_block_count": 1,
                "non_rule_drawing_count": 8,
            },
        }

        refine_table_claim_confidence(claim, candidate)

        self.assertEqual(claim["confidence"], 0.85)
        self.assertEqual(claim["metadata"]["semantic_fill_ratio"], 0.75)
        self.assertFalse(
            claim["metadata"]["confidence_used_for_ownership"]
        )


if __name__ == "__main__":
    unittest.main()
