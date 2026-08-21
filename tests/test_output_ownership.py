from __future__ import annotations

import unittest

from nativePDF_structurer.output_ownership import (
    apply_claim_resolution_to_outputs,
)


class OutputOwnershipTests(unittest.TestCase):
    #@GPT 存活Claim应保留记录和Markdown，落选Claim应产生可审计suppression动作并写入真实输出路径。
    def test_survival_and_suppression_are_separated(self) -> None:
        records = [
            {"claim_id": "visual#1", "output_path": "/tmp/one.png"},
            {"claim_id": "visual#2", "output_path": "/tmp/two.png"},
        ]
        elements = [
            {"claim_id": "visual#1", "output_path": "/tmp/one.png"},
            {"claim_id": "visual#2", "output_path": "/tmp/two.png"},
        ]
        resolution = {
            "candidates": [
                {
                    "claim_id": "visual#1",
                    "survives_preview": True,
                    "won_count": 2,
                    "won_feature_count": 1,
                    "survival_reason": "survives",
                },
                {
                    "claim_id": "visual#2",
                    "survives_preview": False,
                    "won_count": 0,
                    "won_feature_count": 0,
                    "survival_reason": "lost_all_feature_elements",
                },
            ]
        }

        active_records, active_elements, actions = (
            apply_claim_resolution_to_outputs(
                records=records,
                markdown_elements=elements,
                resolution=resolution,
                output_kind="visual_region",
            )
        )

        self.assertEqual(
            [record["claim_id"] for record in active_records],
            ["visual#1"],
        )
        self.assertEqual(
            [element["claim_id"] for element in active_elements],
            ["visual#1"],
        )
        self.assertEqual(actions[0]["reason"], "lost_all_feature_elements")
        self.assertTrue(records[0]["claim_resolution"]["active"])
        self.assertFalse(records[1]["claim_resolution"]["active"])

    #@GPT 存活父表格/父视觉块时，子视觉原子应被抑制但记录保留；父Claim失败时不能误删子输出。
    def test_parent_visual_ownership_suppresses_only_active_child(self) -> None:
        records = [
            {
                "claim_id": "visual#parent",
                "output_path": "/tmp/parent.png",
            },
            {
                "claim_id": "visual#child",
                "output_path": "/tmp/child.png",
                "embedded_in_parent_visual": True,
                "parent_visual_claim_id": "visual#parent",
            },
        ]
        elements = [dict(record) for record in records]
        resolution = {
            "candidates": [
                {
                    "claim_id": "visual#parent",
                    "survives_preview": True,
                    "won_count": 1,
                    "won_feature_count": 1,
                    "survival_reason": "survives",
                },
                {
                    "claim_id": "visual#child",
                    "survives_preview": True,
                    "won_count": 1,
                    "won_feature_count": 1,
                    "survival_reason": "survives",
                },
            ]
        }

        active_records, active_elements, actions = (
            apply_claim_resolution_to_outputs(
                records=records,
                markdown_elements=elements,
                resolution=resolution,
                output_kind="visual_region",
            )
        )

        self.assertEqual(
            [record["claim_id"] for record in active_records],
            ["visual#parent"],
        )
        self.assertEqual(
            [element["claim_id"] for element in active_elements],
            ["visual#parent"],
        )
        self.assertEqual(actions[0]["reason"], "embedded_in_parent_table")


if __name__ == "__main__":
    unittest.main()
