from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.table_structure import (
    GeometryFallbackTable,
    combine_native_and_fallback_tables,
)


class TableMergeTests(unittest.TestCase):
    def test_adjacent_native_and_fallback_fragments_merge(self) -> None:
        native = types.SimpleNamespace(
            bbox=(0, 0, 100, 10),
            extract=lambda: [["header"]],
            # 相邻 bbox 本身不是拼接证明，测试还需提供真实单元格边界。
            cells=[(0, 0, 100, 10)],
        )
        fallback = GeometryFallbackTable(
            bbox=fitz.Rect(0, 10, 100, 30),
            data=[["header"], ["value"]],
            # 两行位于不同空间区间，即使文字相同也都应保留。
            metadata={"column_positions": [0, 100], "row_positions": [10, 20, 30]},
        )

        result = combine_native_and_fallback_tables(
            [native],
            [fallback],
            max_vertical_break_points=18.0,
            native_replacement_overlap_ratio=0.8,
            native_expansion_area_ratio=1.05,
            alignment_tolerance_points=3.0,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].detection_source, "merged_native_fragment")
        self.assertEqual(result[0].extract(), [["header"], ["header"], ["value"]])
        self.assertEqual(result[0].bbox, (0.0, 0.0, 100.0, 30.0))

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
            alignment_tolerance_points=3.0,
        )

        self.assertEqual(result, [fallback])

    # 使用合成网格验证几何关系，不依赖文档内容或特定页码。
    @staticmethod
    def combine(native, fallback):
        return combine_native_and_fallback_tables(
            native, fallback, max_vertical_break_points=18.0,
            native_replacement_overlap_ratio=0.8,
            native_expansion_area_ratio=1.05,
            alignment_tolerance_points=3.0,
        )

    @staticmethod
    def native(rect, data, columns=None, rows=None):
        cells = []
        if columns is not None and rows is not None:
            cells = [(x0, y0, x1, y1) for y0, y1 in zip(rows, rows[1:])
                     for x0, x1 in zip(columns, columns[1:])]
        return types.SimpleNamespace(bbox=tuple(rect), extract=lambda: data, cells=cells)

    @staticmethod
    def fallback(rect, data, columns=None, rows=None):
        metadata = {} if columns is None else {
            "column_positions": columns, "row_positions": rows,
        }
        return GeometryFallbackTable(fitz.Rect(rect), data, metadata)

    def test_same_region_keeps_native_grid_instead_of_adding_rows(self):
        data = [[str(i), str(i)] for i in range(5)]
        native = self.native((0, 0, 100, 100), data)
        fallback = self.fallback((0, 0, 100, 100), [["other", "grid"]] * 3)
        self.assertEqual(self.combine([native], [fallback]), [native])
        self.assertEqual(native.extract(), data)
        self.assertEqual(native.candidate_merge_audit[-1]["decision"],
                         "keep_existing_same_or_covering_region")

    def test_subpoint_border_differences_do_not_create_fragments(self):
        native = self.native((0, 0.1, 100, 100.2), [[str(i)] for i in range(5)])
        fallback = self.fallback((0, 0, 100, 100), [[str(i)] for i in range(3)])
        self.assertEqual(self.combine([native], [fallback]), [native])

    def test_complete_fallback_replaces_all_covered_native_fragments(self):
        data = [[str(i), str(i)] for i in range(9)]
        upper = self.native((0, 0.2, 100, 20), data[:2])
        lower = self.native((0, 30, 100, 90), data[3:])
        outside = self.native((200, 0, 300, 90), [["unrelated"]])
        for native_tables in ([upper, lower, outside], [outside, lower, upper]):
            fallback = self.fallback((0, 0, 100, 90), data)
            result = self.combine(native_tables, [fallback])
            self.assertEqual(result, [fallback, outside])
            self.assertEqual(result[0].extract(), data)
            audit = fallback.candidate_merge_audit[-1]
            self.assertEqual(audit["decision"], "replace_all_covered_fragments")
            self.assertEqual(sorted(audit["alternative_row_counts"]), [2, 6])

    def test_full_native_keeps_subset_fallback_out(self):
        native = self.native((0, 0, 100, 100), [[str(i)] for i in range(9)])
        subset = self.fallback((0, 0, 100, 20), [["a"], ["b"]])
        self.assertEqual(self.combine([native], [subset]), [native])

    def test_containment_selection_is_atomic(self):
        fragment = self.native((0, 0, 100, 10), [["part"]])
        full = self.native((0, 0, 100, 100), [["full"]] * 5)
        fallback = self.fallback((0, 0, 100, 80), [["subset"]] * 3)
        result = self.combine([fragment, full], [fallback])
        self.assertEqual(result, [fragment, full])

    def test_partial_overlap_does_not_concatenate_full_data_sets(self):
        native = self.native((0, 0, 100, 60), [["a"], ["b"]])
        fallback = self.fallback((0, 40, 100, 100), [["b"], ["c"]])
        result = self.combine([native], [fallback])
        self.assertEqual(result, [native, fallback])
        self.assertEqual(fallback.candidate_merge_audit[-1]["decision"],
                         "partial_overlap_not_concatenated")

    def test_adjacent_large_fragments_join_by_grid_not_row_count(self):
        upper = [[str(i), "same"] for i in range(4)]
        lower = [[str(i), "same"] for i in range(4, 8)]
        native = self.native((0, 0, 100, 40), upper, [0, 50, 100], [0, 10, 20, 30, 40])
        fallback = self.fallback((0, 40, 100, 80), lower, [0, 50, 100], [40, 50, 60, 70, 80])
        result = self.combine([native], [fallback])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].extract(), upper + lower)

    def test_close_independent_tables_do_not_join_across_empty_gap(self):
        native = self.native((0, 0, 100, 10), [["a"]], [0, 100], [0, 10])
        fallback = self.fallback((0, 20, 100, 40), [["b"], ["c"]], [0, 100], [20, 30, 40])
        self.assertEqual(self.combine([native], [fallback]), [native, fallback])

    def test_adjacent_boxes_without_cell_geometry_do_not_join(self):
        native = self.native((0, 0, 100, 10), [["a"]])
        fallback = self.fallback((0, 10, 100, 30), [["b"], ["c"]])
        self.assertEqual(self.combine([native], [fallback]), [native, fallback])

    def test_mismatched_columns_do_not_join(self):
        native = self.native((0, 0, 100, 10), [["a", "b"]], [0, 50, 100], [0, 10])
        fallback = self.fallback((0, 10, 100, 30), [["c", "d"], ["e", "f"]],
                                 [0, 70, 100], [10, 20, 30])
        self.assertEqual(self.combine([native], [fallback]), [native, fallback])

    def test_even_small_row_overlap_is_not_concatenated(self):
        native = self.native((0, 0, 100, 10), [["a"]], [0, 100], [0, 10])
        fallback = self.fallback((0, 9.8, 100, 30), [["b"], ["c"]], [0, 100], [9.8, 20, 30])
        self.assertEqual(self.combine([native], [fallback]), [native, fallback])

    def test_reversed_vertical_input_still_preserves_spatial_order(self):
        native = self.native((0, 20, 100, 30), [["lower"]], [0, 100], [20, 30])
        fallback = self.fallback((0, 0, 100, 20), [["upper"], ["middle"]], [0, 100], [0, 10, 20])
        result = self.combine([native], [fallback])
        self.assertEqual(result[0].extract(), [["upper"], ["middle"], ["lower"]])

    def test_fallback_order_does_not_change_covering_result(self):
        native = self.native((0, 0, 100, 10), [["a"]])
        small = self.fallback((0, 0, 100, 20), [["a"], ["b"]])
        full = self.fallback((0, 0, 100, 50), [[str(i)] for i in range(5)])
        self.assertEqual(self.combine([native], [small, full]), [full])
        self.assertEqual(self.combine([native], [full, small]), [full])

    def test_bad_grid_metadata_cannot_authorize_a_join(self):
        native = self.native((0, 0, 100, 10), [["a"]], [0, 100], [0, 10])
        fallback = self.fallback((0, 10, 100, 30), [["b"], ["c"]], [0, 100], [10, 30])
        self.assertEqual(self.combine([native], [fallback]), [native, fallback])

    def test_empty_inputs(self):
        self.assertEqual(self.combine([], []), [])


if __name__ == "__main__":
    unittest.main()
