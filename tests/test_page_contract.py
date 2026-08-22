from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from nativePDF_structurer.page_contract import (
    EXPECTED_PAGE_STAGE_HISTORY,
    validate_page_context,
)
from nativePDF_structurer.page_context import PageRunContext


class PageContractTests(unittest.TestCase):
    def _context(self) -> PageRunContext:
        context = PageRunContext(
            page_index=0,
            page_number=1,
            page=object(),
            page_rect=(0, 0, 100, 100),
            page_dict={},
            text_lines=[],
            source_images=[],
            raw_drawings=[],
        )
        context.markdown_lines = ["## Page 1"]
        context.markdown_section = "## Page 1"
        context.page_record = {"page_number": 1}
        return context

    def test_valid_completed_page_has_no_warnings(self) -> None:
        self.assertEqual(validate_page_context(self._context()), [])

    def test_invalid_cross_stage_references_are_reported(self) -> None:
        context = self._context()
        context.table_markdown_elements = [{"claim_id": "table#missing"}]
        context.table_ownership_decisions = [{"table_index": 4}]

        warnings = validate_page_context(context)

        self.assertIn(
            "table.markdown_claim_missing:table#missing",
            warnings,
        )
        self.assertIn("table_ownership.table_index_out_of_range", warnings)

    def test_stage_history_mismatch_is_reported(self) -> None:
        context = self._context()
        context.stage_history = list(EXPECTED_PAGE_STAGE_HISTORY[:-1])

        self.assertIn(
            "stage_history.mismatch",
            validate_page_context(context),
        )

    def test_complete_stage_history_has_no_warning(self) -> None:
        context = self._context()
        context.stage_history = list(EXPECTED_PAGE_STAGE_HISTORY)

        self.assertEqual(validate_page_context(context), [])

    def test_claim_reference_integrity_is_reported(self) -> None:
        context = self._context()
        context.page_inventory = SimpleNamespace(
            entries={
                "d:1": {"owner": "table#1"},
            }
        )
        context.detector_claims = [
            {
                "claim_id": "table#1",
                "output_element_id": "table#1",
                "input_element_ids": ["d:1", "d:missing"],
                "claimed_element_ids": ["d:1"],
                "feature_element_ids": ["d:1", "d:feature-missing"],
            },
            {"claim_id": "table#1"},
        ]
        context.page_inventory.entries["d:1"]["owner"] = "claim#unknown"
        context.claim_resolution = {
            "owner_by_element_id": {
                "d:missing": "claim#unknown",
            }
        }

        warnings = validate_page_context(context)

        self.assertIn(
            "claim.input_element_ids_missing:table#1:d:missing",
            warnings,
        )
        self.assertIn(
            "claim.feature_not_claimed:table#1:d:feature-missing",
            warnings,
        )
        self.assertIn("claim.duplicate_claim_id:table#1", warnings)
        self.assertIn(
            "inventory.owner_unknown_claim:d:1:claim#unknown",
            warnings,
        )
        self.assertIn("resolution.element_missing:d:missing", warnings)
        self.assertIn(
            "resolution.owner_unknown_claim:d:missing:claim#unknown",
            warnings,
        )

    def test_output_boundary_contract_is_reported(self) -> None:
        context = self._context()
        context.table_markdown_elements = [
            {
                "bbox": [0, 0, 20, 20],
                "render_bbox": [0, 0, 120, 20],
                "sort_bbox": [20, 10, 10, 20],
            }
        ]
        context.text_separator_markdown_elements = [
            {
                "bbox": [0, 30, 100, 30],
                "render_bbox": [0, 30, 100, 30],
                "sort_bbox": [0, 30, 100, 30],
            }
        ]

        warnings = validate_page_context(context)

        self.assertIn(
            "table.element_bbox_outside_page:0:render_bbox",
            warnings,
        )
        self.assertIn(
            "table.element_bbox_reversed:0:sort_bbox",
            warnings,
        )

    def test_valid_output_boundary_contract_has_no_warning(self) -> None:
        context = self._context()
        context.visual_region_markdown_elements = [
            {
                "bbox": [0, 0, 100, 40],
                "render_bbox": [0, 0, 100, 40],
                "sort_bbox": [0, 0, 100, 2],
            }
        ]

        self.assertEqual(validate_page_context(context), [])

    def test_output_artifact_contract_is_reported(self) -> None:
        context = self._context()
        missing_path = Path("/tmp/nonexistent-page-contract-image.png")
        common_bbox = {
            "bbox": [0, 0, 20, 20],
            "render_bbox": [0, 0, 20, 20],
            "sort_bbox": [0, 0, 20, 2],
            "output_path": missing_path,
        }
        context.detector_claims = [
            {"claim_id": "visual#1"},
            {"claim_id": "visual#2"},
        ]
        context.visual_region_markdown_elements = [
            {**common_bbox, "claim_id": "visual#1"},
            {**common_bbox, "claim_id": "visual#2"},
        ]
        context.visual_region_records = [
            {
                "claim_id": "visual#1",
                "output_path": str(missing_path),
            }
        ]

        warnings = validate_page_context(context)

        self.assertIn(
            f"visual.artifact_file_missing:0:{missing_path}",
            warnings,
        )
        self.assertIn(
            "visual.artifact_path_duplicate:1:visual:0",
            warnings,
        )
        self.assertIn(
            "visual.markdown_record_claim_missing:visual#2",
            warnings,
        )

    def test_valid_output_artifact_contract_has_no_warning(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "image.png"
            output_path.write_bytes(b"png")
            context = self._context()
            context.detector_claims = [{"claim_id": "image#1"}]
            context.source_image_markdown_elements = [
                {
                    "bbox": [0, 0, 20, 20],
                    "render_bbox": [0, 0, 20, 20],
                    "sort_bbox": [0, 0, 20, 2],
                    "output_path": output_path,
                    "claim_id": "image#1",
                }
            ]
            context.source_image_records = [
                {
                    "output_path": str(output_path),
                    "claim_id": "image#1",
                }
            ]

            self.assertEqual(validate_page_context(context), [])


if __name__ == "__main__":
    unittest.main()
