from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
#@GPT 预聚簇依赖和检测编排已集中在table_detection，专项测试改用同一主题入口。
from nativePDF_structurer.table_detection import (
    TablePreclusterDependencies,
    detect_tables_by_drawing_clusters,
)
from nativePDF_structurer.table_structure import (
    TableStructureDependencies,
)


class TableDetectionOrchestrationTests(unittest.TestCase):
    #@GPT 空页面应走整页低预算路径并返回稳定的模式统计，不触发PyMuPDF表格候选或几何补检结果。
    def test_empty_page_returns_whole_page_metadata(self) -> None:
        page = types.SimpleNamespace(
            number=0,
            rect=fitz.Rect(0, 0, 100, 100),
            find_tables=lambda **_: types.SimpleNamespace(tables=[]),
        )
        inventory = PageInventory(
            page_number=1,
            page_rect=page.rect,
            drawings=[],
            image_blocks=[],
            text_lines=[],
        )
        tables, metadata, claims = detect_tables_by_drawing_clusters(
            page=page,
            text_lines=[],
            excluded_header_footer_rects=[],
            drawings=[],
            page_inventory=inventory,
            axis_geometry_index={
                "drawing_by_element_id": {},
                "logical_horizontal_rules": [],
                "legacy_axis_rule_by_element_id": {},
            },
            cluster_table_drawing_budget=2000,
            fallback_table_max_vertical_break_points=18.0,
            fallback_table_min_horizontal_rules=4,
            fallback_table_min_populated_columns=2,
            fallback_table_min_populated_rows=3,
            fallback_table_min_vertical_coverage_ratio=0.60,
            fallback_table_min_vertical_segment_points=8.0,
            fallback_table_native_expansion_area_ratio=1.05,
            fallback_table_native_replacement_overlap_ratio=0.80,
            fallback_table_rule_alignment_tolerance_points=3.0,
            logical_horizontal_rule_coordinate_decimal_places=3,
            table_precluster_dependencies=TablePreclusterDependencies(
                mask_pixels_per_point=0.5,
                close_gap_points=4.0,
                attach_margin_points=2.0,
                drawing_overlap_ratio=0.10,
                max_background_area_page_ratio=0.80,
                drawing_is_filled_horizontal_bar=lambda _: False,
                drawing_is_long_horizontal_single_line=lambda _: False,
            ),
            table_structure_dependencies=TableStructureDependencies(
                alignment_tolerance_points=3.0,
                skeleton_horizontal_gap_points=1.0,
                drawing_axis_rule=lambda *_: None,
            ),
            drawing_axis_rule=lambda *_: None,
            group_horizontal_rules_by_span=lambda _: [],
            merge_collinear_vertical_rules=lambda rules: rules,
            meaningful_overlap=lambda *_: False,
            rect_is_mostly_inside_any=lambda **_: False,
        )

        self.assertEqual(tables, [])
        self.assertEqual(claims, [])
        self.assertEqual(metadata["mode"], "whole_page_under_budget")
        self.assertEqual(metadata["tested_drawing_count"], 0)


if __name__ == "__main__":
    unittest.main()
