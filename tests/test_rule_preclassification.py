from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.rule_preclassification import (
    build_preclassified_text_separator_outputs,
    detect_preclassified_rule_bands,
)


class FakeInventory:
    def __init__(self, drawings, images, text_lines):
        self.page_number = 1
        self.entries = {}
        self._by_identity = {}
        for kind, items in (
            ("drawing", drawings),
            ("image", images),
            ("text", text_lines),
        ):
            for index, item in enumerate(items, start=1):
                element_id = f"{kind}#{index}"
                self.entries[element_id] = {
                    "kind": kind,
                    "payload": item,
                }
                self._by_identity[(kind, id(item))] = element_id

    def id_for_payload(self, payload, kind):
        return self._by_identity.get((kind, id(payload)))

    def payload_of(self, element_id):
        return self.entries[element_id]["payload"]

    def entry(self, element_id):
        return self.entries[element_id]

    def ids_for_payloads(self, payloads, kind):
        return [
            element_id
            for payload in payloads
            if (
                element_id := self.id_for_payload(payload, kind)
            ) is not None
        ]


class RulePreclassificationTests(unittest.TestCase):
    def test_routes_two_rule_visual_band_before_table_detection(self):
        top = {"rect": fitz.Rect(0, 10, 100, 10), "items": []}
        bottom = {"rect": fitz.Rect(0, 50, 100, 50), "items": []}
        image = {"bbox": fitz.Rect(60, 15, 90, 45)}
        text = {
            "kind": "text",
            "angle": 0,
            "bbox": fitz.Rect(5, 20, 45, 30),
        }
        drawings = [top, bottom]
        inventory = FakeInventory(drawings, [image], [text])
        logical_rules = [
            {
                "rect": top["rect"],
                "position": 10,
                "element_ids": ["drawing#1"],
            },
            {
                "rect": bottom["rect"],
                "position": 50,
                "element_ids": ["drawing#2"],
            },
        ]

        result = detect_preclassified_rule_bands(
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=drawings,
            text_lines=[text],
            source_images=[image],
            page_inventory=inventory,
            logical_horizontal_rules=logical_rules,
            axis_geometry_index={
                "legacy_axis_rule_by_element_id": {}
            },
            drawing_is_stroked_preclassified_horizontal_rule=(
                lambda *_: True
            ),
            rule_element_ids=lambda rule: set(
                rule.get("element_ids", [])
            ),
            merge_collinear_vertical_rules=lambda rules: rules,
            group_horizontal_rules_by_span=lambda rules: [rules],
            unique_rule_positions=lambda rules: sorted(
                {rule["position"] for rule in rules}
            ),
            horizontal_pair_repeats_in_another_span=(
                lambda **_: False
            ),
            image_is_large_enough=lambda **_: True,
            meaningful_overlap=lambda *_: True,
            vertical_rule_connects_horizontal_pair=lambda **_: False,
            horizontal_pair_has_boundary_grid_evidence=(
                lambda **_: {
                    "has_boundary_grid_evidence": False
                }
            ),
            minimum_x_overlap_ratio=0.8,
            alignment_tolerance_points=3.0,
            minimum_vertical_coverage_ratio=0.6,
            minimum_image_area_points_squared=160.0,
            vector_connection_tolerance_points=1.0,
        )

        self.assertEqual(len(result["bands"]), 1)
        self.assertEqual(result["bands"][0]["mode"], "visual_rule_band")
        self.assertEqual(result["table_detection_drawings"], [])

    def test_builds_preclassified_text_separator_claim(self):
        def claim(**kwargs):
            return {"claim_id": kwargs["output_element_id"], **kwargs}

        markdown, records, claims = (
            build_preclassified_text_separator_outputs(
                page_number=2,
                bands=[
                    {
                        "mode": "text_separator",
                        "bbox": fitz.Rect(0, 10, 100, 50),
                        "rule_drawing_ids": ["drawing#1", "drawing#2"],
                        "text_element_ids": ["text#1"],
                        "reason": "two_aligned_horizontal_rules_with_native_text",
                    }
                ],
                create_claim_manifest=claim,
            )
        )
        self.assertEqual(markdown[0]["text"], "---")
        self.assertEqual(records[0]["claim_id"], "page#2:text_separator#1")
        self.assertEqual(claims[0]["detector_name"], "text_separator")


if __name__ == "__main__":
    unittest.main()
