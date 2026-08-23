from __future__ import annotations

import unittest
from pathlib import Path

import fitz

from nativePDF_structurer.formula_regions import (
    attach_equation_numbers_to_clusters,
    build_number_seeded_formula_clusters,
    cluster_formula_candidates,
    extract_formula_line_records,
    formula_column_key,
    process_page_formula_regions,
    try_build_formula_latex,
)


class FormulaRegionTests(unittest.TestCase):
    def test_formula_column_key(self) -> None:
        page = fitz.Rect(0, 0, 200, 300)
        self.assertEqual(formula_column_key(fitz.Rect(10, 10, 60, 30), page), "left")
        self.assertEqual(formula_column_key(fitz.Rect(140, 10, 190, 30), page), "right")
        self.assertEqual(formula_column_key(fitz.Rect(80, 10, 120, 30), page), "full")

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

    def test_number_seed_collects_mixed_math_font_fragments(self) -> None:
        """Use an equation label to recover native lines split by font runs."""
        page = fitz.Rect(0, 0, 200, 300)
        features = [
            {
                "bbox": fitz.Rect(10, 100, 60, 112),
                "text": "Ip = (2.69 x 105) n",
                "column": "left",
                "is_equation_number": False,
                "math_font_character_count": 3,
                "prose_word_count": 0,
                "signal_score": 3,
                "has_inline_equation_number": False,
            },
            {
                "bbox": fitz.Rect(70, 102, 80, 112),
                "text": "(1)",
                "column": "left",
                "is_equation_number": True,
                "math_font_character_count": 0,
                "prose_word_count": 0,
                "signal_score": 0,
                "has_inline_equation_number": False,
            },
        ]
        clusters = build_number_seeded_formula_clusters(features, page)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["equation_numbers"][0]["text"], "(1)")

    def test_simple_formula_can_be_emitted_as_latex(self) -> None:
        cluster = {
            "lines": [
                {
                    "bbox": fitz.Rect(10, 10, 80, 22),
                    "text": "Ip = a/b",
                    "math_font_character_count": 3,
                    "latex_command_count": 0,
                    "prose_word_count": 0,
                }
            ],
            "equation_numbers": [{"text": "(1)"}],
        }
        self.assertEqual(try_build_formula_latex(cluster), r"Ip = a/b \tag{1}")


if __name__ == "__main__":
    unittest.main()
