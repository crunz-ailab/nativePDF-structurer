from __future__ import annotations

import types
import unittest

from nativePDF_structurer.page_contract import PageStage
from nativePDF_structurer.pipeline import (
    PagePipelineDependencies,
    process_page,
)


class FakePageContext:
    def __init__(self) -> None:
        self.stage_history: list[str] = []
        self.timings = {"inventory_build": 0.25}
        self.outline_page_name = None
        self.markdown_page_title = ""
        self.markdown_lines: list[str] = []

    # def record_stage(self, stage: PageStage) -> None:
    #     self.stage_history.append(stage.value)
    #@GPT PageStage是集中管理的字符串常量类而非Enum；测试假对象直接记录传入字符串，保持与PageRunContext契约一致。
    def record_stage(self, stage: str) -> None:
        self.stage_history.append(stage)


class PagePipelineTests(unittest.TestCase):
    #@GPT 纯内存测试固定完整页面阶段顺序和关键参数传递，不打开PDF或执行检测算法，避免编排变更只能靠长回归发现。
    def test_process_page_runs_complete_stage_order(self) -> None:
        context = FakePageContext()
        page = object()
        calls: list[str] = []

        def callback(name: str, result: object | None = None):
            def run(*args, **kwargs):
                calls.append(name)
                return result

            return run

        initialization = types.SimpleNamespace(
            page_number=3,
            page=page,
            page_started=1.0,
            page_context=context,
        )
        header_footer = types.SimpleNamespace(header_text="header")
        rule_bands = types.SimpleNamespace(bands=[{"mode": "text"}])
        boundary = types.SimpleNamespace(
            emitted_claim_ids={"claim:1"},
            active_composite_table_claim_ids={"table:1"},
        )
        document_state = types.SimpleNamespace(
            outline_page_name_index={3: {"title": "Section"}}
        )
        dependencies = PagePipelineDependencies(
            document_run_state=document_state,
            table_pipeline_dependencies="table-dependencies",
            initialize_page_context=lambda page_index: initialization,
            process_page_header_footer=callback(
                "header_footer", header_footer
            ),
            prepare_page_geometry=callback("geometry"),
            prepare_page_rule_bands=callback("rule_bands", rule_bands),
            detect_page_tables=callback("table_detection"),
            build_page_table_candidates=callback("table_candidates"),
            finalize_page_table_candidates=callback("table_finalization"),
            prepare_page_visual_protection=callback("visual_protection"),
            detect_page_visual_regions=callback("visual_detection"),
            propose_page_output_candidates=callback("output_proposal"),
            detect_page_source_images=callback("source_images"),
            resolve_page_claims=callback("claim_resolution"),
            apply_page_boundary_ownership=callback(
                "boundary_ownership", boundary
            ),
            process_page_semantics=callback("semantics"),
            process_page_reading_order=callback("reading_order"),
            finalize_page_context=callback("finalization"),
            format_markdown_pdf_page_title=lambda **_: "## Section",
            keep_header_text=True,
            formula_detection_enabled=True,
            debug_output_enabled=False,
            force_pure_text_table_to_markdown=False,
        )

        result = process_page(2, dependencies=dependencies)

        self.assertIs(result, context)
        self.assertEqual(
            context.stage_history,
            [
                PageStage.INITIALIZATION,
                PageStage.HEADER_FOOTER,
                PageStage.GEOMETRY,
                PageStage.RULE_BANDS,
                PageStage.TABLE_DETECTION,
                PageStage.TABLE_CANDIDATES,
                PageStage.TABLE_FINALIZATION,
                PageStage.VISUAL_PROTECTION,
                PageStage.VISUAL_DETECTION,
                PageStage.OUTPUT_PROPOSAL,
                PageStage.SOURCE_IMAGES,
                PageStage.CLAIM_RESOLUTION,
                PageStage.BOUNDARY_OWNERSHIP,
                PageStage.SEMANTICS,
                PageStage.READING_ORDER,
                PageStage.FINALIZATION,
            ],
        )
        self.assertEqual(context.markdown_lines, ["## Section", "", "页眉:header"])
        self.assertEqual(
            calls,
            [
                "header_footer",
                "geometry",
                "rule_bands",
                "table_detection",
                "table_candidates",
                "table_finalization",
                "visual_protection",
                "visual_detection",
                "output_proposal",
                "source_images",
                "claim_resolution",
                "boundary_ownership",
                "semantics",
                "reading_order",
                "finalization",
            ],
        )


if __name__ == "__main__":
    unittest.main()
