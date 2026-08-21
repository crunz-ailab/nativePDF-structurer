from __future__ import annotations
import re
import unittest
import fitz
from nativePDF_structurer.figure_captions import classify_figure_caption_roles

class FigureCaptionTests(unittest.TestCase):
    #@GPT 图号结构且紧邻视觉区域、两者间无正文时应建立唯一图题关系。
    def test_assigns_adjacent_caption(self) -> None:
        line = {"text": "图 1 - 示例", "bbox": fitz.Rect(10, 32, 60, 40), "angle": 0, "block_line_count": 1}
        records = classify_figure_caption_roles(
            text_lines=[line],
            visual_records=[{"bbox": [10, 10, 60, 30], "claim_id": "v:1", "region_number": 1}],
            excluded_rects=[],
            page_rect=fitz.Rect(0, 0, 100, 100),
            label_pattern=re.compile(r"^图\s*\d+"),
            max_text_characters=120,
            max_vertical_gap_multiplier=4.0,
            max_vertical_gap_points=24.0,
            rect_is_mostly_inside_any=lambda **_: False,
        )
        self.assertEqual(records[0]["target_claim_id"], "v:1")
        self.assertEqual(line["figure_role"], "figure_caption")

if __name__ == "__main__": unittest.main()
