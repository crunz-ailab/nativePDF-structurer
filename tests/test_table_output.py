from __future__ import annotations

import types
import unittest
from pathlib import Path

import fitz

from nativePDF_structurer.table_output import (
    emit_table_candidate,
    infer_table_rotation,
    normalize_table_shape,
    rotate_table_data,
    table_to_markdown,
    #@GPT Pyflakes确认以下两个输出helper未在本测试文件中使用；保留名称注释便于恢复专项覆盖。
    # save_rotated_table_image,
    # write_table_csv,
)


def candidate_fixture(**overrides):
    table = types.SimpleNamespace(
        bbox=(10, 10, 40, 30),
        detection_source="pymupdf_native",
    )
    candidate = {
        "table": table,
        "table_number": 1,
        "display_table_number": 1,
        "extracted_raw": [["a"]],
        "extracted_reflowed": [["a"]],
        "rotation_info": {"degrees": 0},
        "rotation_degrees": 0,
        "csv_name": "page_001_table_01.csv",
        "csv_path": Path("tables/page_001_table_01.csv"),
        "raw_row_count": 1,
        "raw_column_count": 1,
        "reflowed_row_count": 1,
        "reflowed_column_count": 1,
        "table_cell_count": 1,
        "exceeds_threshold": False,
        "force_markdown_table": False,
        "should_render_table_image": False,
        "table_has_complex_visual_content": False,
        "table_visual_metadata": {},
        "claimed_by_visual_region": False,
        "will_emit_if_unclaimed": False,
        "is_anomalous_single_value_table": False,
        "nonempty_cell_count": 1,
    }
    candidate.update(overrides)
    return candidate


class TableOutputTests(unittest.TestCase):
    #@GPT 序列化辅助函数迁移后仍保留UTF-8 BOM和Markdown网格语义。
    def test_table_serializers_are_owned_by_table_output(self):
        self.assertEqual(
            table_to_markdown([["a|b", "line\nnext"]]),
            "| a\\|b | line<br>next |\n| --- | --- |",
        )

    #@GPT 不等长行先补齐；旋转辅助函数迁移后仍保持原有矩阵形状契约。
    def test_table_shape_and_rotation_helpers(self):
        self.assertEqual(
            normalize_table_shape([["a"], ["b", "c"]]),
            [["a", None], ["b", "c"]],
        )
        self.assertEqual(
            rotate_table_data([["a", "b"], ["c", "d"]], -90),
            [["c", "a"], ["d", "b"]],
        )
        self.assertEqual(
            rotate_table_data([["a", "b"], ["c", "d"]], 90),
            [["b", "d"], ["a", "c"]],
        )

    #@GPT 主方向统计按字符数加权；显式策略参数替代核心隐式常量。
    def test_infers_dominant_table_rotation(self):
        result = infer_table_rotation(
            [
                {"angle": 90, "text": "abcdef", "bbox": fitz.Rect(0, 0, 20, 20)},
                {"angle": 0, "text": "a", "bbox": fitz.Rect(0, 0, 20, 20)},
            ],
            fitz.Rect(0, 0, 20, 20),
            minimum_direction_text_chars=5,
            rotation_dominance_threshold=0.70,
        )
        self.assertEqual(result["suggested_rotation_degrees"], 90)
        self.assertEqual(result["status"], "rotation_detected")

    def common_kwargs(self):
        return {
            "page": types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100)),
            "page_number": 1,
            "tables_dir": Path("tables"),
            "table_images_dir": Path("table_images_auto_rotated"),
            "table_image_dpi": 200,
            "table_to_markdown": lambda table: "| a |",
            "padded_table_candidate_clip": lambda **kwargs: fitz.Rect(kwargs["bbox"]),
            "save_rotated_table_image": lambda **_: (100, 50),
        }

    #@GPT 被视觉区域认领的表格只生成审计记录，不写CSV、Markdown或截图。
    def test_claimed_table_emits_record_only(self):
        csv_calls = []
        result = emit_table_candidate(
            candidate=candidate_fixture(claimed_by_visual_region=True),
            write_table_csv=lambda **kwargs: csv_calls.append(kwargs),
            **self.common_kwargs(),
        )
        self.assertEqual(csv_calls, [])
        self.assertIsNone(result["markdown_element"])
        self.assertFalse(result["embedded"])
        self.assertTrue(result["record"]["claimed_by_visual_region"])

    #@GPT 小型未输出表格仍保留CSV导出，但不生成独立Markdown原子。
    def test_small_table_keeps_csv_without_markdown(self):
        csv_calls = []
        result = emit_table_candidate(
            candidate=candidate_fixture(),
            write_table_csv=lambda **kwargs: csv_calls.append(kwargs),
            **self.common_kwargs(),
        )
        self.assertEqual(len(csv_calls), 1)
        self.assertIsNone(result["markdown_element"])
        self.assertFalse(result["record"]["emitted"])

    #@GPT 截图表格使用实际clip渲染PNG，同时保持检测bbox作为排序锚点。
    def test_rendered_table_builds_png_markdown(self):
        class Page:
            rect = fitz.Rect(0, 0, 100, 100)

            def __init__(self):
                self.calls = []

            def get_pixmap(self, **kwargs):
                self.calls.append(kwargs)
                return object()

        page = Page()
        kwargs = self.common_kwargs()
        kwargs["page"] = page
        result = emit_table_candidate(
            candidate=candidate_fixture(
                should_render_table_image=True,
                will_emit_if_unclaimed=True,
                exceeds_threshold=True,
            ),
            write_table_csv=lambda **_: None,
            **kwargs,
        )
        self.assertEqual(page.calls[0]["dpi"], 200)
        self.assertTrue(result["embedded"])
        self.assertEqual(result["markdown_element"]["mode"], "rendered_table")
        self.assertIsNotNone(result["record"]["image_path"])

    #@GPT 强制纯文本表格输出CSV和Markdown，但不生成PNG。
    def test_forced_markdown_table(self):
        result = emit_table_candidate(
            candidate=candidate_fixture(
                force_markdown_table=True,
                will_emit_if_unclaimed=True,
            ),
            write_table_csv=lambda **_: None,
            **self.common_kwargs(),
        )
        self.assertEqual(
            result["markdown_element"]["mode"],
            "pure_text_markdown_table",
        )
        self.assertFalse(result["embedded"])


if __name__ == "__main__":
    unittest.main()
