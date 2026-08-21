from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.text_ownership import (
    build_stage_f_text_ownership_preview,
)


class TextOwnershipTests(unittest.TestCase):
    def _inventory(self) -> PageInventory:
        return PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[],
            image_blocks=[],
            text_lines=[
                {"bbox": fitz.Rect(0, 0, 10, 10), "text": "line"}
            ],
        )

    #@GPT 文字框达到重叠门槛时应分配给存活候选，低于门槛且没有显式声明时应保留在正文。
    def test_overlap_threshold_controls_text_assignment(self) -> None:
        inventory = self._inventory()
        claims = [
            {"claim_id": "visual#1", "detector": "visual_region", "confidence": 0.8, "claimed_element_ids": []}
        ]
        resolution = {
            "candidates": [
                {
                    "claim_id": "visual#1",
                    "survives_preview": True,
                    "owned_feature_bbox": [0, 0, 6, 10],
                }
            ]
        }

        preview = build_stage_f_text_ownership_preview(
            inventory=inventory,
            claims=claims,
            resolution=resolution,
            emitted_claim_ids={"visual#1"},
            overlap_threshold=0.5,
        )

        self.assertEqual(preview["owner_by_text_element_id"], {"t:0": "visual#1"})

        resolution["candidates"][0]["owned_feature_bbox"] = [0, 0, 4, 10]
        below_threshold = build_stage_f_text_ownership_preview(
            inventory=inventory,
            claims=claims,
            resolution=resolution,
            emitted_claim_ids={"visual#1"},
            overlap_threshold=0.5,
        )
        self.assertEqual(below_threshold["owner_by_text_element_id"], {})

    #@GPT 显式Claim文字提议即使不落入结构bbox也应保留；多候选时仍按检测器优先级稳定裁决。
    def test_explicit_text_and_detector_priority(self) -> None:
        inventory = self._inventory()
        claims = [
            {
                "claim_id": "visual#1",
                "detector": "visual_region",
                "confidence": 1.0,
                "claimed_element_ids": ["t:0"],
                "metadata": {},
            },
            {
                "claim_id": "table#1",
                "detector": "table_detector",
                "confidence": 0.1,
                "claimed_element_ids": [],
                "metadata": {},
            },
        ]
        resolution = {
            "candidates": [
                {
                    "claim_id": "visual#1",
                    "survives_preview": True,
                    "owned_feature_bbox": [90, 90, 95, 95],
                },
                {
                    "claim_id": "table#1",
                    "survives_preview": True,
                    "owned_feature_bbox": [0, 0, 10, 10],
                },
            ]
        }

        preview = build_stage_f_text_ownership_preview(
            inventory=inventory,
            claims=claims,
            resolution=resolution,
            emitted_claim_ids={"visual#1", "table#1"},
            overlap_threshold=0.5,
        )

        self.assertEqual(preview["owner_by_text_element_id"], {"t:0": "table#1"})
        self.assertEqual(preview["conflict_text_count"], 1)


if __name__ == "__main__":
    unittest.main()
