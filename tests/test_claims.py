from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.claims import (
    CLAIM_DETECTOR_PRIORITY,
    claim_effective_priority,
    create_claim_manifest,
)


class ClaimPolicyTests(unittest.TestCase):
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
