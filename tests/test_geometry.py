from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.geometry import (
    AxisRect,
    as_fitz_rect,
    drawing_item_bbox,
    rect_area,
    rect_center,
    union_rects,
)


class GeometryPrimitiveTests(unittest.TestCase):
    def test_rect_primitives(self) -> None:
        rect = fitz.Rect(10, 20, 30, 50)

        self.assertEqual(rect_center(rect), fitz.Point(20, 35))
        self.assertEqual(rect_area(rect), 600.0)
        self.assertEqual(
            union_rects([rect, fitz.Rect(0, 40, 40, 60)]),
            fitz.Rect(0, 20, 40, 60),
        )

    def test_union_requires_one_rectangle(self) -> None:
        with self.assertRaises(ValueError):
            union_rects([])

    def test_axis_rect_conversion(self) -> None:
        rect = AxisRect(1, 2, 5, 8)

        self.assertEqual(rect.width, 4)
        self.assertEqual(rect.height, 6)
        self.assertEqual(as_fitz_rect(rect), fitz.Rect(1, 2, 5, 8))
        fitz_rect = fitz.Rect(2, 3, 4, 5)
        self.assertIs(as_fitz_rect(fitz_rect), fitz_rect)

    def test_drawing_item_bbox_supports_pymupdf_primitives(self) -> None:
        item = (
            "l",
            fitz.Point(10, 20),
            fitz.Point(30, 40),
        )
        self.assertEqual(
            drawing_item_bbox(item),
            fitz.Rect(10, 20, 30, 40),
        )
        self.assertEqual(
            drawing_item_bbox(("re", fitz.Rect(5, 6, 15, 16))),
            fitz.Rect(5, 6, 15, 16),
        )
        self.assertEqual(
            drawing_item_bbox(
                (
                    "qu",
                    fitz.Quad(
                        fitz.Point(1, 2),
                        fitz.Point(4, 2),
                        fitz.Point(1, 5),
                        fitz.Point(4, 5),
                    ),
                )
            ),
            fitz.Rect(1, 2, 4, 5),
        )
        self.assertIsNone(drawing_item_bbox(("noop",)))


if __name__ == "__main__":
    unittest.main()
