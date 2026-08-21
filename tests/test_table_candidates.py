from __future__ import annotations

import types
import unittest

import fitz

#@GPT 候选策略已经收敛到table_structure；测试继续覆盖原行为而不保留已删除模块的兼容入口。
from nativePDF_structurer.table_structure import (
    classify_table_candidate_structure,
    count_nonempty_table_cells,
    deduplicate_overlapping_table_candidates,
    table_claim_confidence,
    table_contains_complex_visual_content,
)


class TableCandidatePolicyTests(unittest.TestCase):
    #@GPT 重叠候选应保留面积更大的表格，低于0.80覆盖率的相邻候选仍应保留。
    def test_deduplicate_keeps_larger_candidate(self) -> None:
        large = types.SimpleNamespace(bbox=(0, 0, 100, 100))
        nested = types.SimpleNamespace(bbox=(10, 10, 90, 90))
        separate = types.SimpleNamespace(bbox=(90, 0, 130, 40))

        result = deduplicate_overlapping_table_candidates(
            [nested, separate, large]
        )

        self.assertEqual(result, [large, separate])

    #@GPT 表格初始置信度映射固定现有三类检测来源及未知来源回退值。
    def test_table_claim_confidence_uses_detection_source(self) -> None:
        self.assertEqual(
            table_claim_confidence(types.SimpleNamespace()),
            0.95,
        )
        self.assertEqual(
            table_claim_confidence(
                types.SimpleNamespace(detection_source="geometry_fallback")
            ),
            0.85,
        )
        self.assertEqual(
            table_claim_confidence(
                types.SimpleNamespace(detection_source="unknown")
            ),
            0.70,
        )

    #@GPT 单元格计数必须忽略None、空串和纯空白，使单值异常候选的判定不受PyMuPDF空单元格表达差异影响。
    def test_count_nonempty_table_cells_ignores_blank_values(self) -> None:
        self.assertEqual(
            count_nonempty_table_cells(
                [[None, "", "  "], ["value", " second "]]
            ),
            2,
        )

    #@GPT 三种结构覆盖单列重复内容、正常多列表格和稀疏候选，锁定表格结构门语义。
    def test_classify_table_candidate_structure_variants(self) -> None:
        single_column_valid, single_audit = (
            classify_table_candidate_structure(
                extracted_raw=[["first"], ["second"]],
                table_visual_metadata={},
                raw_row_count=2,
                raw_column_count=1,
                nonempty_cell_count=2,
            )
        )
        multi_column_valid, multi_audit = (
            classify_table_candidate_structure(
                extracted_raw=[["left", "right"], ["", "tail"]],
                table_visual_metadata={},
                raw_row_count=2,
                raw_column_count=2,
                nonempty_cell_count=3,
            )
        )
        sparse_valid, sparse_audit = classify_table_candidate_structure(
            extracted_raw=[["dimension", None], [None, "42"]],
            table_visual_metadata={},
            raw_row_count=2,
            raw_column_count=2,
            nonempty_cell_count=2,
        )

        self.assertTrue(single_column_valid)
        self.assertEqual(single_audit["occupied_row_count"], 2)
        self.assertTrue(multi_column_valid)
        self.assertEqual(multi_audit["multi_value_row_count"], 1)
        self.assertFalse(sparse_valid)
        self.assertEqual(
            sparse_audit["rejection_reason"],
            "sparse_cells_without_multi_column_row",
        )

    #@GPT 规则线自身不应迫使表格截图，而图片块、非规则矢量和数学字体文字任一命中都必须保留视觉表达。
    def test_complex_visual_content_uses_injected_local_evidence(self) -> None:
        table_bbox = fitz.Rect(0, 0, 100, 100)
        page_rect = fitz.Rect(0, 0, 200, 200)

        def axis_rule(drawing: dict, _page_rect: fitz.Rect) -> object | None:
            return object() if drawing.get("is_rule") else None

        is_complex, metadata = table_contains_complex_visual_content(
            table_bbox=table_bbox,
            page_dict={"blocks": []},
            drawings=[{"rect": (0, 10, 100, 11), "is_rule": True}],
            page_rect=page_rect,
            drawing_axis_rule=axis_rule,
            math_font_hints=("CMSY",),
        )
        self.assertFalse(is_complex)
        self.assertEqual(metadata["non_rule_drawing_count"], 0)

        is_complex, metadata = table_contains_complex_visual_content(
            table_bbox=table_bbox,
            page_dict={
                "blocks": [
                    {"type": 1, "bbox": (10, 10, 30, 30)},
                    {
                        "type": 0,
                        "lines": [
                            {
                                "bbox": (40, 40, 80, 55),
                                "spans": [{"font": "Subset-CMSY10"}],
                            }
                        ],
                    },
                ]
            },
            drawings=[{"rect": (20, 60, 50, 90), "is_rule": False}],
            page_rect=page_rect,
            drawing_axis_rule=axis_rule,
            math_font_hints=("CMSY",),
        )
        self.assertTrue(is_complex)
        self.assertEqual(
            metadata,
            {
                "image_block_count": 1,
                "non_rule_drawing_count": 1,
                "math_font_line_count": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
