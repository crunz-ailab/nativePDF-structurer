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
        #@GPT 使用临时占位PDF只验证配置边界，不启动PyMuPDF解析流程。
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

    #@GPT 解析上下文在配置边界固定页号，避免可变列表或未排序页号进入核心；它不重新验证PDF存在性，文件校验仍由RunConfig负责。
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

    #@GPT 两个策略的默认值必须保持历史行为，同时允许配置对象显式覆盖；这里只验证边界存储，不启动解析流程。
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

    #@GPT 页面上下文必须区分原始drawing和过滤后working drawing；这项测试锁定第一阶段迁移所依赖的对象身份和可变阶段字段。
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

        #@GPT 新页面必须获得独立的表格集合，防止前一页的候选或骨架ID泄漏到下一页。
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
