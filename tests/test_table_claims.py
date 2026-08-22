from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.table_structure import build_table_claims


class TableClaimsTests(unittest.TestCase):
    def test_build_table_claims_normalizes_detected_tables(self) -> None:
        drawing = {"rect": fitz.Rect(0, 0, 20, 20), "items": []}
        inventory = PageInventory(
            page_number=3,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[],
            text_lines=[],
        )
        table = types.SimpleNamespace(
            bbox=(0, 0, 20, 20),
            claim_input_element_ids=["d:0"],
            detection_source="geometry_fallback",
            source_cluster_bbox=(0, 0, 30, 30),
        )

        claims = build_table_claims(
            page_number=3,
            tables=[table],
            page_inventory=inventory,
            detection_mode="cluster_budgeted",
            confidence_fn=lambda _: 0.8,
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim_id"], "page#3:table#1")
        self.assertEqual(claims[0]["input_element_ids"], ["d:0"])
        self.assertEqual(claims[0]["claimed_element_ids"], ["d:0"])
        self.assertEqual(
            claims[0]["metadata"]["source_cluster_bbox"],
            [0.0, 0.0, 30.0, 30.0],
        )
        self.assertIn("detected_by:geometry_fallback", claims[0]["reason_codes"])


if __name__ == "__main__":
    unittest.main()
