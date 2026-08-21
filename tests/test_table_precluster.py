from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
#@GPT 表格预聚簇已并入table_detection；测试仍单独覆盖聚簇算法行为。
from nativePDF_structurer.table_detection import (
    TablePreclusterDependencies,
    build_pre_table_drawing_clusters,
)


class TablePreclusterTests(unittest.TestCase):
    def setUp(self) -> None:
        #@GPT 测试依赖使用小间距和单位掩码比例，便于精确验证分离簇；局部线型回调不引入核心模块。
        self.dependencies = TablePreclusterDependencies(
            mask_pixels_per_point=1.0,
            close_gap_points=1.0,
            attach_margin_points=2.0,
            drawing_overlap_ratio=0.10,
            max_background_area_page_ratio=0.80,
            drawing_is_filled_horizontal_bar=lambda _: False,
            drawing_is_long_horizontal_single_line=lambda _: False,
        )

    #@GPT 空drawing页保留整页回退簇，使上层检测器继续得到稳定的单簇元数据而不是特殊None分支。
    def test_empty_page_returns_full_page_cluster(self) -> None:
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100))
        inventory = PageInventory(
            page_number=1,
            page_rect=page.rect,
            drawings=[],
            image_blocks=[],
            text_lines=[],
        )

        clusters = build_pre_table_drawing_clusters(
            page=page,
            excluded_header_footer_rects=[],
            drawings=[],
            page_inventory=inventory,
            dependencies=self.dependencies,
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["bbox"], page.rect)
        self.assertEqual(clusters[0]["drawing_count"], 0)

    #@GPT 两个面积足够且间距远大于闭运算核的矩形必须形成两个表格检测簇，锁定预聚簇不把独立对象桥接的语义。
    def test_separated_rectangles_form_two_clusters(self) -> None:
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 120, 100))
        drawings = [
            {"rect": fitz.Rect(10, 10, 30, 30), "items": []},
            {"rect": fitz.Rect(80, 50, 105, 75), "items": []},
        ]
        inventory = PageInventory(
            page_number=1,
            page_rect=page.rect,
            drawings=drawings,
            image_blocks=[],
            text_lines=[],
        )

        clusters = build_pre_table_drawing_clusters(
            page=page,
            excluded_header_footer_rects=[],
            drawings=drawings,
            page_inventory=inventory,
            dependencies=self.dependencies,
        )

        self.assertEqual(len(clusters), 2)
        self.assertEqual(
            [cluster["drawing_count"] for cluster in clusters],
            [1, 1],
        )


if __name__ == "__main__":
    unittest.main()
