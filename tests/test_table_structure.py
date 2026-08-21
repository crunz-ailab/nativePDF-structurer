from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.table_structure import (
    TableStructureDependencies,
    configure_table_container_claim,
    drawing_is_table_container_structure,
    merge_horizontal_rules_for_table_skeleton,
    refine_table_claim_confidence,
    table_cell_boundary_positions,
    unique_rule_positions,
)


class TableStructurePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        #@GPT 专项测试使用与主流程相同的3pt边界容差和1pt横向骨架拼接间隙，防止测试建立另一套判据。
        self.dependencies = TableStructureDependencies(
            alignment_tolerance_points=3.0,
            skeleton_horizontal_gap_points=1.0,
            drawing_axis_rule=lambda drawing, _: drawing.get("rule"),
        )

    #@GPT 近邻规则位置必须用中位数折叠，超过容差的位置仍保持独立边界。
    def test_unique_rule_positions_groups_with_tolerance(self) -> None:
        rules = [
            {"position": 0.0},
            {"position": 1.0},
            {"position": 10.0},
        ]

        self.assertEqual(
            unique_rule_positions(rules, tolerance=3.0),
            [0.5, 10.0],
        )

    #@GPT 同Y且端点相距不超过1pt的横线应恢复为一条逻辑线；较远片段必须保持独立。
    def test_horizontal_skeleton_merge_respects_one_point_gap(self) -> None:
        rules = [
            {
                "orientation": "h",
                "position": 10.0,
                "rect": fitz.Rect(0, 10, 5, 10),
                "element_id": "d:0",
            },
            {
                "orientation": "h",
                "position": 10.0,
                "rect": fitz.Rect(6, 10, 10, 10),
                "element_id": "d:1",
            },
            {
                "orientation": "h",
                "position": 10.0,
                "rect": fitz.Rect(20, 10, 25, 10),
                "element_id": "d:2",
            },
        ]

        merged = merge_horizontal_rules_for_table_skeleton(
            rules,
            dependencies=self.dependencies,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["rect"], fitz.Rect(0, 10, 10, 10))
        self.assertEqual(merged[0]["element_ids"], ["d:0", "d:1"])
        self.assertEqual(merged[1]["element_ids"], ["d:2"])

    #@GPT 原生cells边界既能识别对齐规则线，也会拒绝位于单元格内部、未对齐边界的设备图矩形。
    def test_container_structure_uses_cell_boundaries(self) -> None:
        table = types.SimpleNamespace(
            cells=[
                (0, 0, 10, 10),
                (10, 0, 20, 10),
            ]
        )
        self.assertEqual(
            table_cell_boundary_positions(table),
            ([0.0, 10.0, 20.0], [0.0, 10.0]),
        )
        boundary_line = {
            #@GPT 严格零高度Rect与表格Rect的PyMuPDF面积交集会判空；使用真实描边常见的0.5pt bbox验证容器规则。
            # "rect": fitz.Rect(0, 0, 20, 0),
            "rect": fitz.Rect(0, 0, 20, 0.5),
            "rule": {
                "orientation": "h",
                # "position": 0.0,
                # "rect": fitz.Rect(0, 0, 20, 0),
                "position": 0.25,
                "rect": fitz.Rect(0, 0, 20, 0.5),
            },
        }
        internal_box = {
            #@GPT 3pt正好等于表格边界容差，不能作为“明确未对齐”样例；改用距四边均4pt的内部矩形。
            # "rect": fitz.Rect(3, 3, 7, 7),
            "rect": fitz.Rect(4, 4, 6, 6),
            "fill": (0.5, 0.5, 0.5),
            # "items": [("re", fitz.Rect(3, 3, 7, 7), 1)],
            "items": [("re", fitz.Rect(4, 4, 6, 6), 1)],
        }

        self.assertTrue(
            drawing_is_table_container_structure(
                drawing=boundary_line,
                table=table,
                table_bbox=fitz.Rect(0, 0, 20, 10),
                page_rect=fitz.Rect(0, 0, 100, 100),
                dependencies=self.dependencies,
            )
        )
        self.assertFalse(
            drawing_is_table_container_structure(
                drawing=internal_box,
                table=table,
                table_bbox=fitz.Rect(0, 0, 20, 10),
                page_rect=fitz.Rect(0, 0, 100, 100),
                dependencies=self.dependencies,
            )
        )

    #@GPT 完整截图表格必须统一认领bbox内的drawing和image并登记tc容器，避免视觉/原图检测重复输出同一内容。
    def test_full_table_claim_owns_bbox_content(self) -> None:
        drawing = {
            "rect": fitz.Rect(0, 0, 20, 0),
            "rule": {
                "orientation": "h",
                "position": 0.0,
                "rect": fitz.Rect(0, 0, 20, 0),
            },
        }
        image = {"bbox": fitz.Rect(5, 2, 15, 8)}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[image],
            text_lines=[],
        )
        table = types.SimpleNamespace(cells=[(0, 0, 20, 10)])
        candidate = {
            "table_number": 1,
            "table": table,
            "bbox": fitz.Rect(0, 0, 20, 10),
            "raw_row_count": 2,
            "raw_column_count": 2,
            "nonempty_cell_count": 2,
            "table_owns_full_visual_bbox": True,
        }
        claim = {
            "claimed_element_ids": ["d:0"],
            "feature_element_ids": ["d:0"],
            "input_element_ids": ["d:0"],
            "metadata": {},
            "reason_codes": [],
        }

        result = configure_table_container_claim(
            claim=claim,
            candidate=candidate,
            page_inventory=inventory,
            page_rect=inventory.page_rect,
            dependencies=self.dependencies,
        )

        self.assertTrue(result["logical_table_container"])
        self.assertEqual(result["container_element_id"], "tc:0")
        self.assertEqual(
            claim["claimed_element_ids"],
            ["d:0", "i:0", "tc:0"],
        )
        self.assertEqual(result["internal_visual_drawing_ids"], [])

    #@GPT 当前置信度策略只记录局部密度审计而不改变来源分数，确保复杂矢量表格不会再次因内部图形被扣分。
    def test_confidence_refinement_preserves_source_score(self) -> None:
        claim = {"confidence": 0.85, "metadata": {}}
        candidate = {
            "table_cell_count": 4,
            "nonempty_cell_count": 2,
            "table_visual_metadata": {
                "image_block_count": 1,
                "non_rule_drawing_count": 8,
            },
        }

        refine_table_claim_confidence(claim, candidate)

        self.assertEqual(claim["confidence"], 0.85)
        self.assertEqual(claim["metadata"]["semantic_fill_ratio"], 0.75)
        self.assertFalse(
            claim["metadata"]["confidence_used_for_ownership"]
        )


if __name__ == "__main__":
    unittest.main()
