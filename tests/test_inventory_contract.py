from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory


class PageInventoryContractTests(unittest.TestCase):
    def _inventory(self) -> tuple[PageInventory, dict, dict, dict]:
        drawing = {
            "rect": fitz.Rect(0, 0, 20, 20),
            "items": [
                (
                    "l",
                    fitz.Point(0, 0),
                    fitz.Point(20, 20),
                )
            ],
        }
        image = {"bbox": fitz.Rect(30, 0, 40, 10)}
        text = {"bbox": fitz.Rect(0, 30, 40, 40), "text": "demo"}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[image],
            text_lines=[text],
        )
        return inventory, drawing, image, text

    def test_inventory_registers_stable_payload_ids(self) -> None:
        inventory, drawing, image, text = self._inventory()

        self.assertEqual(inventory.ids_of_kind("drawing"), ["d:0"])
        self.assertEqual(inventory.ids_of_kind("image"), ["i:0"])
        self.assertEqual(inventory.ids_of_kind("text"), ["t:0"])
        self.assertEqual(inventory.id_for_payload(drawing, "drawing"), "d:0")
        self.assertEqual(inventory.id_for_payload(image, "image"), "i:0")
        self.assertEqual(inventory.id_for_payload(text, "text"), "t:0")
        self.assertEqual(
            inventory.ids_for_payloads([drawing, drawing], "drawing"),
            ["d:0"],
        )

    def test_logical_containers_have_separate_kinds(self) -> None:
        inventory, *_ = self._inventory()

        table_id = inventory.register_table_container(
            table_number=1,
            bbox=fitz.Rect(0, 0, 20, 20),
            payload={"kind": "table"},
        )
        visual_id = inventory.register_visual_container(
            container_number=1,
            bbox=fitz.Rect(0, 0, 20, 20),
            payload={"kind": "visual"},
        )

        self.assertEqual(table_id, "tc:0")
        self.assertEqual(visual_id, "vc:0")
        self.assertTrue(inventory.entry(table_id)["labels"]["logical_table_container"])
        self.assertTrue(inventory.entry(visual_id)["labels"]["closed_visual_container"])

    def test_conflict_resolution_lazily_refines_drawing_items(self) -> None:
        inventory, *_ = self._inventory()
        claims = [
            {
                "claim_id": "page#1:table#1",
                "detector": "table_detector",
                "confidence": 0.9,
                "claimed_element_ids": ["d:0"],
                "feature_element_ids": ["d:0"],
                "bbox": [0, 0, 20, 20],
            },
            {
                "claim_id": "page#1:visual_region#1",
                "detector": "visual_region",
                "confidence": 1.0,
                "claimed_element_ids": ["d:0"],
                "feature_element_ids": ["d:0"],
                "bbox": [0, 0, 20, 20],
            },
        ]

        resolution = inventory.resolve_conflicts(claims)

        self.assertIn("di:0:0", inventory.entries)
        self.assertEqual(resolution["invalid_element_reference_count"], 0)
        self.assertGreaterEqual(resolution["conflict_element_count"], 1)
        table_result = next(
            item
            for item in resolution["candidates"]
            if item["claim_id"] == "page#1:table#1"
        )
        visual_result = next(
            item
            for item in resolution["candidates"]
            if item["claim_id"] == "page#1:visual_region#1"
        )
        self.assertEqual(table_result["won_feature_count"], 1)
        self.assertEqual(visual_result["won_feature_count"], 0)


if __name__ == "__main__":
    unittest.main()
