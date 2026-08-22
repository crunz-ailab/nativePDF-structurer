from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nativePDF_structurer.page_context import PageRunContext
from nativePDF_structurer.page_finalize import finalize_page_context


class PageFinalizeTests(unittest.TestCase):
    def test_finalize_builds_page_record_and_markdown_section(self) -> None:
        context = PageRunContext(
            page_index=0,
            page_number=1,
            page=SimpleNamespace(rotation=0),
            page_rect=(0, 0, 100, 100),
            page_dict={},
            text_lines=[],
            source_images=[],
            raw_drawings=[],
        )
        context.markdown_page_title = "## Page 1"
        context.markdown_lines = ["## Page 1", "正文"]
        context.header_footer_record = {"header": {}, "footer": {}}
        context.precluster_object_exclusions = {
            "protected_table_numbers": [],
            "protected_image_group_indices": [],
            "tables": [],
        }
        context.preclassified_rule_band_result = {
            "table_detection_drawings": [],
            "metadata": {},
        }
        context.table_detection_metadata = {}
        context.axis_geometry_index = {"metadata": {}}
        context.reading_order_metadata = {"direction_groups": []}

        result = finalize_page_context(
            context,
            page_started=time.perf_counter(),
        )

        self.assertEqual(result.markdown_section, "## Page 1\n正文")
        self.assertEqual(result.page_record["page_number"], 1)
        self.assertEqual(
            result.page_record["timing_seconds"]["page_total"],
            round(context.page_total_seconds, 6),
        )
