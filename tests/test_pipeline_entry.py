"""Tests for the public pipeline entry point."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nativePDF_structurer.config import ParseRunContext, RunConfig
from nativePDF_structurer.pipeline import run
from nativePDF_structurer.result import ParseResult


class _OpenDocumentStub:
    """Provide the context-manager and page-count API used by run()."""

    page_count = 8

    def __enter__(self) -> "_OpenDocumentStub":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class PipelineEntryTests(unittest.TestCase):
    """Protect configuration transfer and result identity at the boundary."""

    #@GPT 公开run应返回pipeline创建的同一个ParseResult，并将全部用户开关放进不可变ParseRunContext。
    def test_run_returns_pipeline_parse_result_without_copying(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            config = RunConfig(
                source_pdf=Path(source.name),
                output_dir=Path(source.name).with_suffix(""),
                pages=(2, 5),
                debug_output=True,
                enable_heading_detection=False,
                enable_font_heading_inference=True,
                enable_table_header_detection=True,
                keep_header_text=True,
                enable_formula_detection=False,
                enable_pdf_outline_heading_assignment=False,
                force_pure_text_table_to_markdown=True,
            )
            expected = ParseResult(summary={"processed_pages": 2})
            pipeline_main = Mock(return_value=expected)
            with patch(
                "nativePDF_structurer.pipeline.main",
                pipeline_main,
            ), patch(
                "nativePDF_structurer.pipeline.fitz.open",
                return_value=_OpenDocumentStub(),
            ):
                actual = run(config)

        self.assertIs(actual, expected)
        pipeline_main.assert_called_once()
        context = pipeline_main.call_args.args[0]
        self.assertIsInstance(context, ParseRunContext)
        self.assertIs(context.config, config)
        self.assertEqual(context.page_numbers, (2, 5))
        self.assertFalse(
            context.config.enable_pdf_outline_heading_assignment
        )
        self.assertTrue(context.config.force_pure_text_table_to_markdown)


if __name__ == "__main__":
    unittest.main()
