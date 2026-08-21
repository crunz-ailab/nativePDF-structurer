from __future__ import annotations

import unittest
from pathlib import Path

import fitz

from nativePDF_structurer.formula_regions import (
    attach_equation_numbers_to_clusters,
    cluster_formula_candidates,
    extract_formula_line_records,
    formula_column_key,
    process_page_formula_regions,
)


class FormulaRegionTests(unittest.TestCase):
    #@GPT 双栏公式按文本框中心分栏，跨越中缝的公式才进入full，避免左右栏同高度串联。
    def test_formula_column_key(self) -> None:
        page = fitz.Rect(0, 0, 200, 300)
        self.assertEqual(formula_column_key(fitz.Rect(10, 10, 60, 30), page), "left")
        self.assertEqual(formula_column_key(fitz.Rect(140, 10, 190, 30), page), "right")
        self.assertEqual(formula_column_key(fitz.Rect(80, 10, 120, 30), page), "full")

    #@GPT 公式候选只依赖原生文本层字体/符号结构，并单独保留公式编号，不调用OCR或模型。
    def test_extracts_formula_and_equation_number(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "bbox": (10, 10, 70, 24),
                            "spans": [
                                {
                                    "text": r"\frac{a}{b}",
                                    "font": "CambriaMath",
                                }
                            ],
                        },
                        {
                            "bbox": (72, 12, 82, 22),
                            "spans": [
                                {"text": "(1)", "font": "Times-Roman"}
                            ],
                        },
                        {
                            "bbox": (10, 40, 80, 52),
                            "spans": [
                                {"text": "This is prose", "font": "Times-Roman"}
                            ],
                        },
                    ],
                }
            ]
        }
        records = extract_formula_line_records(
            page_dict=page_dict,
            page_rect=fitz.Rect(0, 0, 200, 300),
            excluded_rects=[],
            line_is_inside_any_table=lambda **_: False,
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["is_formula_candidate"])
        self.assertTrue(records[1]["is_equation_number"])

    #@GPT 公式碎片只在同一栏且垂直间距不超过策略门限时合并，编号再作为独立锚点附着。
    def test_clusters_fragments_and_attaches_number(self) -> None:
        candidates = [
            {
                "column": "left",
                "bbox": fitz.Rect(10, 10, 50, 16),
                "text": "a",
            },
            {
                "column": "left",
                "bbox": fitz.Rect(10, 17, 50, 23),
                "text": "b",
            },
        ]
        clusters = cluster_formula_candidates(candidates)
        self.assertEqual(len(clusters), 1)
        number = {
            "column": "left",
            "bbox": fitz.Rect(55, 16, 65, 22),
            "text": "(1)",
        }
        attach_equation_numbers_to_clusters(clusters, [number])
        self.assertEqual(clusters[0]["equation_numbers"], [number])
        self.assertEqual(clusters[0]["bbox"].x1, 65.0)

    #@GPT 无公式页面应返回空记录和空Markdown，不调用截图接口。
    def test_process_empty_formula_page(self) -> None:
        page = type("Page", (), {"rect": fitz.Rect(0, 0, 100, 100)})()
        records, markdown, owned, _seconds = process_page_formula_regions(
            page=page,
            page_dict={"blocks": []},
            excluded_table_rects=[],
            excluded_visual_text_rects=[],
            excluded_header_footer_rects=[],
            page_number=1,
            formula_dir=Path("formula_images"),
            formula_image_dpi=200,
            formula_min_unnumbered_width_page_ratio=0.28,
            formula_padding_points=3.0,
            line_is_inside_any_table=lambda **_: False,
        )
        self.assertEqual(records, [])
        self.assertEqual(markdown, [])
        self.assertEqual(owned, [])


if __name__ == "__main__":
    unittest.main()
