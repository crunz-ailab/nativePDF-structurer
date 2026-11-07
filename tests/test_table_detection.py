from __future__ import annotations

import types
import unittest
import ast
import inspect
from unittest.mock import Mock

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.table_detection import (
    TablePreclusterDependencies,
    detect_tables_by_drawing_clusters,
)
from nativePDF_structurer.table_structure import (
    TableStructureDependencies,
)


class TableDetectionOrchestrationTests(unittest.TestCase):
    def test_empty_page_returns_whole_page_metadata(self) -> None:
        page = types.SimpleNamespace(
            number=0,
            rect=fitz.Rect(0, 0, 100, 100),
            find_tables=Mock(return_value=types.SimpleNamespace(tables=[])),
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
        page.find_tables.assert_called_once_with(paths=[], use_layout=False)

    def test_all_native_detection_paths_disable_layout_inference(self) -> None:
        """整页与簇级调用都必须显式关闭布局推理。

        Both whole-page and cluster calls must explicitly disable inference.
        """
        tree = ast.parse(inspect.getsource(detect_tables_by_drawing_clusters))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "find_tables"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            with self.subTest(line=call.lineno):
                flags = [kw.value for kw in call.keywords if kw.arg == "use_layout"]
                self.assertEqual(len(flags), 1)
                self.assertIs(ast.literal_eval(flags[0]), False)


if __name__ == "__main__":
    unittest.main()
