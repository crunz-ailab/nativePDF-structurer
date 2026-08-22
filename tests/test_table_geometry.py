from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.table_structure import extract_geometry_table_data


class TableGeometryTests(unittest.TestCase):
    def test_builds_cells_and_counts_populated_rows_and_columns(self) -> None:
        data, populated_rows, populated_columns = extract_geometry_table_data(
            bbox=fitz.Rect(0, 0, 100, 40),
            horizontal_positions=[0, 20, 40],
            separator_positions=[50],
            text_lines=[
                {
                    "text": "second",
                    "bbox": fitz.Rect(10, 8, 20, 12),
                    "angle": 0,
                },
                {
                    "text": "first",
                    "bbox": fitz.Rect(10, 2, 20, 6),
                    "angle": 0,
                },
                {
                    "text": "right",
                    "bbox": fitz.Rect(60, 22, 75, 26),
                    "angle": 0,
                },
                {
                    "text": "rotated",
                    "bbox": fitz.Rect(60, 2, 75, 6),
                    "angle": 90,
                },
                {
                    "text": "outside",
                    "bbox": fitz.Rect(110, 2, 120, 6),
                    "angle": 0,
                },
            ],
        )

        self.assertEqual(data, [["first\nsecond", None], [None, "right"]])
        self.assertEqual(populated_rows, 2)
        self.assertEqual(populated_columns, 2)

    def test_empty_geometry_returns_empty_cells(self) -> None:
        data, populated_rows, populated_columns = extract_geometry_table_data(
            bbox=fitz.Rect(0, 0, 10, 10),
            horizontal_positions=[0, 10],
            separator_positions=[],
            text_lines=[],
        )

        self.assertEqual(data, [[None]])
        self.assertEqual(populated_rows, 0)
        self.assertEqual(populated_columns, 0)


if __name__ == "__main__":
    unittest.main()
