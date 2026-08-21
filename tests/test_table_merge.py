from __future__ import annotations

import types
import unittest

import fitz

#@GPT 几何兼容表格和候选合并已收敛到table_structure，测试不再依赖旧小模块。
from nativePDF_structurer.table_structure import (
    GeometryFallbackTable,
    combine_native_and_fallback_tables,
)


class TableMergeTests(unittest.TestCase):
    #@GPT 相邻表头和开放网格应按相同X跨度合并，并删除共享边界行。
    def test_adjacent_native_and_fallback_fragments_merge(self) -> None:
        native = types.SimpleNamespace(
            bbox=(0, 0, 100, 10),
            extract=lambda: [["header"]],
        )
        fallback = GeometryFallbackTable(
            bbox=fitz.Rect(0, 10, 100, 30),
            data=[["header"], ["value"]],
        )

        result = combine_native_and_fallback_tables(
            [native],
            [fallback],
            max_vertical_break_points=18.0,
            native_replacement_overlap_ratio=0.8,
            native_expansion_area_ratio=1.05,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].detection_source, "merged_native_fragment")
        self.assertEqual(result[0].extract(), [["header"], ["value"]])
        self.assertEqual(result[0].bbox, (0.0, 0.0, 100.0, 30.0))

    #@GPT fallback完整覆盖原生半表且面积达到扩张比例时，应替换而不是重复保留。
    def test_larger_fallback_replaces_native_candidate(self) -> None:
        native = types.SimpleNamespace(
            bbox=(0, 0, 100, 100),
            extract=lambda: [["a"], ["b"], ["c"], ["d"]],
        )
        fallback = GeometryFallbackTable(
            bbox=fitz.Rect(0, 0, 100, 120),
            data=[["a"], ["b"], ["c"], ["d"]],
        )

        result = combine_native_and_fallback_tables(
            [native],
            [fallback],
            max_vertical_break_points=18.0,
            native_replacement_overlap_ratio=0.8,
            native_expansion_area_ratio=1.05,
        )

        self.assertEqual(result, [fallback])


if __name__ == "__main__":
    unittest.main()
