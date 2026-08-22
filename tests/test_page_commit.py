from __future__ import annotations

import unittest

from nativePDF_structurer.page_commit import commit_page_context
from nativePDF_structurer.page_context import PageRunContext
from nativePDF_structurer.page_contract import EXPECTED_PAGE_STAGE_HISTORY
from nativePDF_structurer.result import ParseResult


class PageCommitTests(unittest.TestCase):
    def test_commit_transfers_completed_page_state(self) -> None:
        context = PageRunContext(
            page_index=2,
            page_number=3,
            page=object(),
            page_rect=object(),
            page_dict={},
            text_lines=[],
            source_images=[],
            raw_drawings=[],
        )
        context.inventory_audit_record = {"page_number": 3}
        context.header_footer_record = {"matched": True}
        context.record_timing("header_footer_detection", 0.125)
        context.suppressed_claim_actions = [{"claim_id": "table#1"}]
        context.logical_horizontal_rule_audit = [{"rule": 1}]
        context.figure_caption_records = [{"text": "Figure 1"}]
        context.table_records = [{"table": 1}]
        context.visual_region_records = [{"visual": 1}]
        context.source_image_records = [{"image": 1}]
        context.table_claims = [{"claim_id": "table#1"}]
        context.visual_region_claims = [{"claim_id": "visual#1"}]
        context.source_image_claims = [{"claim_id": "image#1"}]
        context.detector_claims = [
            *context.table_claims,
            *context.visual_region_claims,
            *context.source_image_claims,
        ]
        context.composite_output_groups = [{"group_id": "group#1"}]
        context.claim_resolution = {"owners": {}}
        context.formula_records = [{"formula": 1}]
        context.heading_records = [{"heading": 1}]
        context.page_record = {"page_number": 3, "text_reflow": {}}
        context.stage_history = ["initialization", "finalization"]
        context.markdown_section = "## Page 3"
        result = ParseResult()

        commit_page_context(result, context)

        self.assertEqual(result.inventory_audit, [{"page_number": 3}])
        self.assertEqual(
            result.header_footer_pages,
            [
                {
                    "page_number": 3,
                    "matched": True,
                    "detection_seconds": 0.125,
                }
            ],
        )
        self.assertEqual(result.tables, [{"table": 1}])
        self.assertEqual(result.visual_regions, [{"visual": 1}])
        self.assertEqual(result.source_images, [{"image": 1}])
        self.assertEqual(result.formula_regions, [{"formula": 1}])
        self.assertEqual(result.headings, [{"heading": 1}])
        self.assertEqual(
            result.figure_captions,
            [{"page_number": 3, "text": "Figure 1"}],
        )
        self.assertEqual(result.pages, [context.page_record])
        self.assertNotIn(
            "page_contract_stage_history",
            context.page_record,
            "stage history is debug-only and must not appear in minimal commits",
        )
        self.assertEqual(result.markdown_sections, ["## Page 3"])
        self.assertEqual(
            result.claim_resolution_actions[0]["suppressed_output_count"],
            1,
        )
        self.assertIs(
            result.detector_claims[0]["resolution_preview"],
            context.claim_resolution,
        )

    def test_debug_commit_persists_stage_contract_audit(self) -> None:
        context = PageRunContext(
            page_index=0,
            page_number=1,
            page=object(),
            page_rect=object(),
            page_dict={},
            text_lines=[],
            source_images=[],
            raw_drawings=[],
        )
        context.page_record = {"page_number": 1}
        context.markdown_lines = ["## Page 1"]
        context.markdown_section = "## Page 1"
        context.stage_history = list(EXPECTED_PAGE_STAGE_HISTORY)

        result = ParseResult()
        commit_page_context(result, context, validate=True)

        self.assertEqual(
            context.page_record["page_contract_warnings"],
            [],
        )
        self.assertEqual(
            context.page_record["page_contract_stage_history"],
            list(EXPECTED_PAGE_STAGE_HISTORY),
        )


if __name__ == "__main__":
    unittest.main()
