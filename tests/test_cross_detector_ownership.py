from __future__ import annotations

# import types
import unittest

import fitz

from nativePDF_structurer.cross_detector_ownership import (
    CrossDetectorOwnershipDependencies,
    build_table_visual_composite_output_groups,
    build_visual_text_block_keys,
    evaluate_source_cluster_local_bridge,
    resolve_table_visual_candidate_ownership,
)
from nativePDF_structurer.inventory import PageInventory


class CrossDetectorOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dependencies = CrossDetectorOwnershipDependencies(
            ownership_overlap_threshold=0.65,
            source_cluster_explanation_ratio=0.80,
            object_peer_overlap_ratio=0.10,
            rect_is_mostly_inside_any=lambda **_: False,
        )

    def test_overlap_ownership_prefers_larger_visual_region(self) -> None:
        decisions, suppressed = resolve_table_visual_candidate_ownership(
            table_candidates=[
                {
                    "bbox": (10, 10, 30, 30),
                    "will_emit_if_unclaimed": True,
                }
            ],
            visual_region_rects=[fitz.Rect(0, 0, 100, 100)],
            dependencies=self.dependencies,
        )

        self.assertTrue(decisions[0]["claimed_by_visual_region"])
        self.assertEqual(decisions[0]["visual_region_index"], 0)
        self.assertEqual(suppressed, set())

    def test_source_cluster_bridge_rejects_page_placeholder(self) -> None:
        page_rect = fitz.Rect(0, 0, 100, 100)
        local = evaluate_source_cluster_local_bridge(
            source_cluster_rect=fitz.Rect(10, 10, 50, 50),
            table_rect=fitz.Rect(10, 10, 30, 30),
            visual_rect=fitz.Rect(25, 25, 70, 70),
            page_rect=page_rect,
            dependencies=self.dependencies,
        )
        placeholder = evaluate_source_cluster_local_bridge(
            source_cluster_rect=page_rect,
            table_rect=fitz.Rect(10, 10, 30, 30),
            visual_rect=fitz.Rect(25, 25, 70, 70),
            page_rect=page_rect,
            dependencies=self.dependencies,
        )

        self.assertTrue(local["source_cluster_is_specific"])
        self.assertFalse(placeholder["source_cluster_is_specific"])
        self.assertFalse(placeholder["source_cluster_is_local_bridge"])

    def test_visual_text_block_keys_use_nearest_body_lines(self) -> None:
        text_a = {"bbox": fitz.Rect(0, 10, 20, 15), "text": "before"}
        text_inside = {"bbox": fitz.Rect(10, 22, 20, 26), "text": "inside"}
        text_b = {"bbox": fitz.Rect(0, 45, 20, 50), "text": "after"}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[],
            image_blocks=[],
            text_lines=[text_a, text_inside, text_b],
        )
        keys = build_visual_text_block_keys(
            page_inventory=inventory,
            text_lines=[text_a, text_inside, text_b],
            table_candidates=[
                {
                    "table_number": 1,
                    "bbox": fitz.Rect(10, 20, 40, 40),
                    "structurally_valid_table": True,
                    "will_emit_if_unclaimed": True,
                }
            ],
            visual_records=[],
            excluded_header_footer_rects=[],
            dependencies=self.dependencies,
        )

        self.assertEqual(keys["page#1:table#1"], ("t:0", "t:2"))
        self.assertGreater(float(keys["__top_band_tolerance__"][0]), 0.0)

    def test_build_composite_group_keeps_claim_roles(self) -> None:
        table = {
            "table_number": 1,
            "bbox": fitz.Rect(10, 10, 30, 30),
            "structurally_valid_table": True,
            "will_emit_if_unclaimed": True,
            "table_has_complex_visual_content": True,
            "source_cluster_bbox": fitz.Rect(8, 8, 40, 40),
        }
        visual = {
            "claim_id": "page#1:visual_region#1",
            "content_bbox": fitz.Rect(0, 0, 50, 50),
            "bbox": fitz.Rect(0, 0, 50, 50),
        }
        table_claim = {
            "claim_id": "page#1:table#1",
            "metadata": {},
            "reason_codes": [],
        }
        visual_claim = {
            "claim_id": "page#1:visual_region#1",
            "metadata": {"text_element_ids": []},
            "reason_codes": [],
        }
        drawing = {"rect": fitz.Rect(10, 10, 30, 10)}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[],
            text_lines=[],
        )

        groups = build_table_visual_composite_output_groups(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            page_inventory=inventory,
            text_lines=[],
            excluded_header_footer_rects=[],
            table_candidates=[table],
            visual_records=[visual],
            visual_markdown_elements=[{"claim_id": visual["claim_id"]}],
            table_claims=[table_claim],
            visual_claims=[visual_claim],
            dependencies=self.dependencies,
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["renderer_claim_id"],
            "page#1:visual_region#1",
        )
        self.assertEqual(
            groups[0]["member_table_claim_ids"],
            ["page#1:table#1"],
        )
        self.assertEqual(table["output_group_role"], "member_table")
        self.assertEqual(visual["output_group_role"], "renderer")


if __name__ == "__main__":
    unittest.main()
