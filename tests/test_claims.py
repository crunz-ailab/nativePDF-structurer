from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.claims import (
    CLAIM_DETECTOR_PRIORITY,
    claim_effective_priority,
    create_claim_manifest,
)


class ClaimPolicyTests(unittest.TestCase):
    #@GPT Claim格式测试固定集合归一化、feature子集、置信度夹取和bbox转正语义，避免模块迁移后清单字节内容发生变化。
    def test_create_claim_manifest_normalizes_contract_fields(self) -> None:
        claim = create_claim_manifest(
            detector_name="visual_region",
            output_element_id="page#1:visual#1",
            input_element_ids=["d:2", "d:2"],
            claimed_element_ids=["d:1", "d:2"],
            feature_element_ids=["d:2", "d:missing"],
            bbox=fitz.Rect(20, 30, 10, 5),
            confidence=1.5,
            reason_codes=["vector", "vector", "image"],
            output_artifacts=[{"type": "png"}],
            metadata={"source": "test"},
        )

        self.assertEqual(claim["input_element_ids"], ["d:1", "d:2"])
        self.assertEqual(claim["claimed_element_ids"], ["d:1", "d:2"])
        self.assertEqual(claim["feature_element_ids"], ["d:2"])
        self.assertEqual(claim["confidence"], 1.0)
        self.assertEqual(claim["bbox"], [10.0, 5.0, 20.0, 30.0])
        self.assertEqual(claim["reason_codes"], ["image", "vector"])

    #@GPT 优先级测试保证当前策略仍是表格高于分隔符、视觉区域和原图，未知检测器保持0而不是隐式继承其他类型。
    def test_effective_priority_uses_central_policy(self) -> None:
        self.assertEqual(
            claim_effective_priority(
                {"detector": "table_detector"},
                CLAIM_DETECTOR_PRIORITY,
            ),
            300,
        )
        self.assertEqual(
            claim_effective_priority(
                {"detector": "unknown"},
                CLAIM_DETECTOR_PRIORITY,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
