from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.page_reflow import (
    append_reflowed_text_to_markdown,
    build_reflowed_page_text,
)


class PageReflowTests(unittest.TestCase):
    @staticmethod
    def _inside(*, line_rect, excluded_table_rects):
        return any(
            fitz.Rect(rect).contains(fitz.Rect(line_rect))
            for rect in excluded_table_rects
        )

    @staticmethod
    def _sort(*, elements, source_angle, page_rect):
        ordered = sorted(
            elements,
            key=lambda item: (
                item["bbox"].y0,
                item["bbox"].x0,
            ),
        )
        result = []
        for row_index, element in enumerate(ordered, start=1):
            result.append(
                {
                    **element,
                    "virtual_row_index": row_index,
                    "virtual_x": element["bbox"].x0,
                }
            )
        return result, {
            "row_count": len(result),
            "cut_count": 0,
            "node_count": len(result),
            "cuts": [],
            "nodes": [],
        }

    #@GPT 页面重排模块只负责排除统计、方向分组和调用排序/格式化回调，不应丢失正文或已声明分隔线。
    def test_build_reflowed_page_text_preserves_metadata_and_sections(self) -> None:
        lines = [
            {
                "kind": "text",
                "text": "页眉",
                "angle": 0,
                "bbox": fitz.Rect(10, 10, 30, 18),
            },
            {
                "kind": "text",
                "text": "正文",
                "angle": 0,
                "bbox": fitz.Rect(10, 40, 30, 48),
            },
            {
                "kind": "text_separator",
                "text": "---",
                "angle": 0,
                "bbox": fitz.Rect(10, 50, 30, 51),
            },
            {
                "kind": "text",
                "text": "旋转说明",
                "angle": 90,
                "bbox": fitz.Rect(50, 40, 58, 80),
            },
        ]

        metadata, grouped = build_reflowed_page_text(
            text_lines=lines,
            page_rect=fitz.Rect(0, 0, 100, 100),
            excluded_header_footer_rects=[fitz.Rect(0, 0, 100, 20)],
            excluded_table_rects=[],
            excluded_visual_region_rects=[],
            excluded_formula_rects=[],
            line_is_inside_any_table=self._inside,
            sort_elements_with_gap_tree=self._sort,
        )

        self.assertEqual(metadata["skipped_header_footer_text_line_count"], 1)
        self.assertEqual(metadata["direction_groups"][0]["angle"], 0)
        self.assertEqual(metadata["direction_groups"][1]["angle"], 90)
        self.assertIn("正文", grouped["0"])
        self.assertIn("---", grouped["0"])
        self.assertIn("旋转说明", grouped["90"])

    #@GPT 方向小节追加顺序必须固定，不能按字典插入顺序漂移，保证Markdown字节回归稳定。
    def test_append_reflowed_text_uses_stable_direction_order(self) -> None:
        page_markdown: list[str] = []

        append_reflowed_text_to_markdown(
            page_markdown,
            {
                "90": "逆时针",
                "0": "正常",
                "180": "倒置",
                "-90": "顺时针",
            },
        )

        self.assertEqual(
            page_markdown,
            [
                "",
                "### 页面文本",
                "",
                "正常",
                "",
                "### 旋转说明文本（顺时针转正后）",
                "",
                "顺时针",
                "",
                "### 旋转说明文本（逆时针转正后）",
                "",
                "逆时针",
                "",
                "### 倒置文本（旋转 180°后）",
                "",
                "倒置",
            ],
        )


if __name__ == "__main__":
    unittest.main()
