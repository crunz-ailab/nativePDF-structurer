from __future__ import annotations

import unittest

from nativePDF_structurer.run_state import (
    DocumentRunMetrics,
    DocumentRunState,
    PageInitialization,
    PageHeaderFooterResult,
    PageGeometryPreparation,
    PageRuleBandPreparation,
    PageTableDetectionResult,
    PageTableCandidateBuildResult,
    PageTableCandidateFinalization,
    PageVisualProtectionPreparation,
    PageVisualDetectionResult,
    PageOutputCandidateProposal,
    PageSourceImageDetectionResult,
    PageClaimResolutionResult,
    PageBoundaryOwnershipResult,
    PageSemanticProcessingResult,
    PageReadingOrderProcessingResult,
)


class DocumentRunStateTests(unittest.TestCase):
    #@GPT 两次文档运行不能共享drawing缓存或累计指标；测试default_factory隔离，避免服务进程连续解析文档时发生跨文档污染。
    def test_metrics_instances_are_independent(self) -> None:
        first = DocumentRunMetrics()
        second = DocumentRunMetrics()

        first.table_detection_seconds += 1.5

        self.assertEqual(first.table_detection_seconds, 1.5)
        self.assertEqual(second.table_detection_seconds, 0.0)

    def test_state_keeps_shared_document_resources(self) -> None:
        document = object()
        output_layout = object()
        drawings_cache: dict[int, list[dict]] = {}
        metrics = DocumentRunMetrics()
        state = DocumentRunState(
            document=document,
            output_layout=output_layout,
            source_page_count=10,
            page_indices=[0, 1],
            drawings_cache=drawings_cache,
            header_footer_templates=[],
            outline_heading_index={},
            outline_page_name_index={},
            typography_profile={"enabled": False},
            metrics=metrics,
        )

        self.assertIs(state.document, document)
        self.assertIs(state.output_layout, output_layout)
        self.assertIs(state.drawings_cache, drawings_cache)
        self.assertIs(state.metrics, metrics)

    def test_page_initialization_has_explicit_page_boundary(self) -> None:
        initialization = PageInitialization(
            page_number=2,
            page=object(),
            page_started=1.0,
            page_drawings=[],
            page_dict={},
            text_lines=[],
            source_images=[],
            page_context=object(),
            page_inventory=object(),
            page_drawing_seconds=0.1,
            page_inventory_seconds=0.2,
        )

        self.assertEqual(initialization.page_number, 2)
        self.assertEqual(initialization.page_inventory_seconds, 0.2)

    def test_header_footer_result_exposes_downstream_contract(self) -> None:
        result = PageHeaderFooterResult(
            page_drawings=[],
            header_footer_rects=[],
            header_footer_record={"header": {}, "footer": {}},
            vector_exclusion_rects=[],
            header_text="",
            header_text_lines=[],
            detection_seconds=0.1,
        )

        self.assertEqual(result.page_drawings, [])
        self.assertEqual(result.detection_seconds, 0.1)

    def test_geometry_preparation_exposes_both_rule_views(self) -> None:
        result = PageGeometryPreparation(
            axis_geometry_index={"metadata": {}},
            logical_horizontal_rules=[{"position": 10.0}],
            logical_horizontal_rule_audit=[{"source": "d:1"}],
            axis_seconds=0.1,
            logical_rule_seconds=0.2,
        )

        self.assertEqual(len(result.logical_horizontal_rules), 1)
        self.assertEqual(result.logical_rule_seconds, 0.2)

    def test_rule_band_preparation_exposes_consumed_drawing_ids(self) -> None:
        result = PageRuleBandPreparation(
            rule_band_result={"metadata": {}},
            bands=[{"mode": "text_separator"}],
            drawing_ids={"d:1", "d:2"},
            detection_seconds=0.1,
            output_count=1,
        )

        self.assertEqual(result.output_count, 1)
        self.assertEqual(result.drawing_ids, {"d:1", "d:2"})

    def test_table_detection_result_exposes_budget_status(self) -> None:
        result = PageTableDetectionResult(
            detected_tables=[],
            metadata={"budget_limited_cluster_count": 1},
            table_claims=[],
            gate_seconds=0.2,
            detection_seconds=0.3,
            dense_vector_page=True,
            detection_skipped=True,
            skip_reason="cluster_drawing_budget_applied",
        )

        self.assertTrue(result.detection_skipped)
        self.assertEqual(
            result.skip_reason,
            "cluster_drawing_budget_applied",
        )

    def test_candidate_build_result_keeps_candidate_order(self) -> None:
        candidates = [{"table_number": 1}, {"table_number": 2}]
        result = PageTableCandidateBuildResult(
            candidates=candidates,
            detected_table_count=2,
        )

        self.assertIs(result.candidates, candidates)
        self.assertEqual(result.detected_table_count, 2)

    def test_candidate_finalization_exposes_visual_protection_ids(self) -> None:
        result = PageTableCandidateFinalization(
            candidates=[{"table_number": 1}],
            claims_by_id={"table#1": {"claim_id": "table#1"}},
            skeleton_drawing_ids={"d:1"},
            full_drawing_ids={"d:1", "d:2"},
            adjacent_rule_band_attachment={"attached_count": 1},
        )

        self.assertEqual(result.skeleton_drawing_ids, {"d:1"})
        self.assertIn("d:2", result.full_drawing_ids)

    def test_visual_protection_separates_object_and_table_rects(self) -> None:
        result = PageVisualProtectionPreparation(
            object_exclusions={"protected_rects": [[0, 0, 1, 1]]},
            protected_rects=["object_rect"],
            table_visual_protection_rects=["table_rect"],
        )

        self.assertEqual(result.protected_rects, ["object_rect"])
        self.assertEqual(
            result.table_visual_protection_rects,
            ["table_rect"],
        )

    def test_visual_detection_preserves_two_timing_channels(self) -> None:
        result = PageVisualDetectionResult(
            records=[{"region_number": 1}],
            markdown_elements=[],
            rects=[],
            text_rects=[],
            claims=[],
            base_seconds=0.2,
            preclassified_seconds=0.1,
            embedded_child_visual_count=0,
        )

        self.assertEqual(result.base_seconds, 0.2)
        self.assertEqual(result.preclassified_seconds, 0.1)

    def test_output_proposal_exposes_stable_table_decision_index(self) -> None:
        decision = {"table_index": 0, "claimed_by_visual_region": False}
        result = PageOutputCandidateProposal(
            table_ownership_decisions=[decision],
            suppressed_visual_indices=set(),
            decisions_by_table_index={0: decision},
            composite_output_groups=[],
        )

        self.assertIs(result.decisions_by_table_index[0], decision)

    def test_source_image_detection_keeps_claims_with_records(self) -> None:
        result = PageSourceImageDetectionResult(
            records=[{"image_number": 1}],
            markdown_elements=[],
            claims=[{"claim_id": "image#1"}],
            detection_seconds=0.3,
        )

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.claims[0]["claim_id"], "image#1")

    def test_claim_resolution_result_keeps_restore_audit(self) -> None:
        result = PageClaimResolutionResult(
            resolution={"owners": {}},
            suppressed_actions=[{"claim_id": "image#1"}],
            output_group_finalization={"active_group_count": 1},
            restore_actions=[{"claim_id": "table#1"}],
            restored_table_screenshot_seconds=0.2,
        )

        self.assertEqual(len(result.suppressed_actions), 1)
        self.assertEqual(result.restored_table_screenshot_seconds, 0.2)

    def test_boundary_result_keeps_precise_text_rects(self) -> None:
        result = PageBoundaryOwnershipResult(
            excluded_table_rects=["table"],
            visual_region_rects=["visual"],
            visual_region_text_rects=["visual_text"],
            source_image_text_rects=["image_text"],
            emitted_claim_ids={"visual#1"},
            active_composite_table_claim_ids={"table#1"},
            visual_application={},
            source_application={},
        )

        self.assertEqual(result.visual_region_text_rects, ["visual_text"])
        self.assertEqual(result.source_image_text_rects, ["image_text"])

    def test_semantic_result_preserves_formula_and_heading_channels(self) -> None:
        result = PageSemanticProcessingResult(
            figure_caption_records=[],
            formula_records=[{"formula_number": 1}],
            formula_markdown_elements=[],
            formula_text_rects=["formula_text"],
            formula_seconds=0.2,
            text_separator_markdown_elements=[],
            text_separator_records=[],
            text_separator_seconds=0.1,
            heading_records=[{"heading_level": 2}],
            heading_seconds=0.05,
        )

        self.assertEqual(len(result.formula_records), 1)
        self.assertEqual(result.heading_records[0]["heading_level"], 2)

    def test_reading_order_result_keeps_grouped_markdown(self) -> None:
        result = PageReadingOrderProcessingResult(
            metadata={"direction_groups": []},
            grouped_markdown={"0": "正文"},
            normal_direction_metadata={"node_count": 1},
            debug_path=None,
            markdown_lines=["## Page 1", "正文"],
        )

        self.assertEqual(result.grouped_markdown["0"], "正文")
        self.assertEqual(result.normal_direction_metadata["node_count"], 1)


if __name__ == "__main__":
    unittest.main()
