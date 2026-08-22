from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.table_detection import (
    TablePreclusterDependencies,
    build_pre_table_drawing_clusters,
)


class TablePreclusterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dependencies = TablePreclusterDependencies(
            mask_pixels_per_point=1.0,
            close_gap_points=1.0,
            attach_margin_points=2.0,
            drawing_overlap_ratio=0.10,
            max_background_area_page_ratio=0.80,
            drawing_is_filled_horizontal_bar=lambda _: False,
            drawing_is_long_horizontal_single_line=lambda _: False,
        )

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
