"""Tests for CLI-facing configuration validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nativePDF_structurer.config import ParseRunContext, RunConfig
from nativePDF_structurer.page_context import PageRunContext


class RunConfigTests(unittest.TestCase):
    """Keep page notation and validation independent from the parser core."""

    def test_page_ranges_are_expanded_sorted_and_deduplicated(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            config = RunConfig.from_page_spec(
                source_pdf=source.name,
                output_dir=Path(source.name).with_suffix(""),
                page_spec="3,1-2,2",
            )
        self.assertEqual(config.pages, (1, 2, 3))

    def test_descending_page_range_is_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            with self.assertRaisesRegex(ValueError, "invalid page range"):
                RunConfig.from_page_spec(
                    source_pdf=source.name,
                    output_dir=Path(source.name).with_suffix(""),
                    page_spec="4-2",
                )

    def test_parse_run_context_normalizes_resolved_pages(self) -> None:
        """Store one immutable, sorted page tuple beside the config."""

        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            config = RunConfig(
                source_pdf=source.name,
                output_dir=Path(source.name).with_suffix(""),
            )
            context = ParseRunContext(
                config=config,
                page_numbers=(4, 2, 4),
            )

        self.assertIs(context.config, config)
        self.assertEqual(context.page_numbers, (2, 4))
        self.assertIsInstance(context.page_numbers, tuple)

    def test_strategy_flags_preserve_defaults_and_accept_overrides(self) -> None:
        """Expose outline and pure-text-table policies in RunConfig."""

        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            default_config = RunConfig(
                source_pdf=source.name,
                output_dir=Path(source.name).with_suffix(""),
            )
            override_config = RunConfig(
                source_pdf=source.name,
                output_dir=Path(source.name).with_suffix(""),
                enable_pdf_outline_heading_assignment=False,
                force_pure_text_table_to_markdown=True,
            )

        self.assertTrue(default_config.enable_pdf_outline_heading_assignment)
        self.assertFalse(default_config.force_pure_text_table_to_markdown)
        self.assertFalse(
            override_config.enable_pdf_outline_heading_assignment
        )
        self.assertTrue(override_config.force_pure_text_table_to_markdown)

    def test_formula_output_mode_is_validated(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            config = RunConfig(
                source_pdf=source.name,
                output_dir=Path(source.name).with_suffix(""),
                formula_output_mode="hybrid",
            )
            self.assertEqual(config.formula_output_mode, "hybrid")
            with self.assertRaisesRegex(ValueError, "formula_output_mode"):
                RunConfig(
                    source_pdf=source.name,
                    output_dir=Path(source.name).with_suffix(""),
                    formula_output_mode="unknown",
                )

    def test_page_run_context_keeps_raw_and_working_drawings(self) -> None:
        """Initialize page state without copying the drawing list."""

        raw_drawings = [{"id": "drawing#1"}]
        page_context = PageRunContext(
            page_index=2,
            page_number=3,
            page=object(),
            page_rect=(0, 0, 100, 100),
            page_dict={"blocks": []},
            text_lines=[],
            source_images=[],
            raw_drawings=raw_drawings,
        )

        self.assertIs(page_context.drawings, raw_drawings)
        self.assertIs(page_context.raw_drawings, raw_drawings)
        page_context.drawings = []
        page_context.record_timing("inventory", 0.125)
        page_context.table_candidates.append({"table_number": 1})
        page_context.table_skeleton_drawing_ids.add("d:1")
        page_context.visual_region_records.append(
            {"claim_id": "page#3:visual_region#1"}
        )
        page_context.dense_vector_page = True
        page_context.table_detection_skipped = True
        page_context.table_detection_skip_reason = (
            "cluster_drawing_budget_applied"
        )
        page_context.table_ownership_decisions.append(
            {"table_index": 0}
        )
        page_context.suppressed_visual_indices.add(2)
        self.assertIs(page_context.raw_drawings, raw_drawings)
        self.assertEqual(page_context.timings["inventory"], 0.125)
        self.assertEqual(
            page_context.table_candidates,
            [{"table_number": 1}],
        )
        self.assertEqual(
            page_context.table_skeleton_drawing_ids,
            {"d:1"},
        )
        self.assertEqual(len(page_context.visual_region_records), 1)
        self.assertTrue(page_context.dense_vector_page)
        self.assertTrue(page_context.table_detection_skipped)
        self.assertEqual(
            page_context.table_detection_skip_reason,
            "cluster_drawing_budget_applied",
        )
        self.assertEqual(
            page_context.table_ownership_decisions,
            [{"table_index": 0}],
        )
        self.assertEqual(page_context.suppressed_visual_indices, {2})

        second_context = PageRunContext(
            page_index=3,
            page_number=4,
            page=object(),
            page_rect=(0, 0, 100, 100),
            page_dict={"blocks": []},
            text_lines=[],
            source_images=[],
            raw_drawings=[],
        )
        self.assertEqual(second_context.table_candidates, [])
        self.assertEqual(second_context.table_skeleton_drawing_ids, set())
        self.assertEqual(second_context.visual_region_records, [])
        self.assertFalse(second_context.dense_vector_page)
        self.assertEqual(second_context.table_ownership_decisions, [])
        self.assertEqual(second_context.suppressed_visual_indices, set())


if __name__ == "__main__":
    unittest.main()
