from __future__ import annotations

import types
import unittest

import fitz

from nativePDF_structurer.header_footer import (
    extract_header_footer_candidates,
    extract_header_text_for_markdown,
    detect_page_header_footer_bands,
    filter_confirmed_header_footer_drawings,
    header_footer_candidates_match,
    merge_footer_drawing_candidates,
)


class HeaderFooterCandidateTests(unittest.TestCase):
    #@GPT 同基线、相近高度且水平相接的页尾drawing应合并为一个候选，并保留全部原始drawing索引。
    def test_merges_adjacent_footer_drawings(self) -> None:
        candidates = [
            {
                "kind": "drawing",
                "bbox": fitz.Rect(5, 90, 10, 95),
                "content_signature": "a",
                "source_drawing_index": 1,
            },
            {
                "kind": "drawing",
                "bbox": fitz.Rect(11, 90, 16, 95),
                "content_signature": "b",
                "source_drawing_index": 2,
            },
        ]
        result = merge_footer_drawing_candidates(
            candidates=candidates,
            page_rect=fitz.Rect(0, 0, 100, 100),
            height_tolerance_points=1.0,
            x_tolerance_points=2.0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_drawing_indices"], [1, 2])
        self.assertEqual(result[0]["composite_member_count"], 2)

    #@GPT 页尾镜像对象按最近外侧页边距离匹配，而不是强制奇偶页具有相同绝对x0。
    def test_matches_mirrored_footer_candidate(self) -> None:
        first = {
            "kind": "drawing",
            "height": 5.0,
            "width": 20.0,
            "x0": 5.0,
            "edge_offset": 3.0,
            "horizontal_position_mode": "outer_edge",
            "outer_offset": 5.0,
            "content_signature": "same",
        }
        second = {
            **first,
            "x0": 75.0,
            "outer_side": "right",
        }
        self.assertTrue(
            header_footer_candidates_match(
                first,
                second,
                height_tolerance_points=1.0,
                x_tolerance_points=2.0,
                edge_tolerance_points=3.0,
            )
        )

    #@GPT 页尾候选只收集底部带内drawing，正文对象即使内容摘要可生成也不应进入候选列表。
    def test_extracts_only_footer_band_drawings(self) -> None:
        footer = {"rect": fitz.Rect(10, 94, 40, 96)}
        body = {"rect": fitz.Rect(10, 60, 40, 62)}
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100))
        result = extract_header_footer_candidates(
            page=page,
            page_dict={"blocks": []},
            edge="footer",
            drawings=[footer, body],
            edge_page_ratio_by_edge={"header": 0.18, "footer": 0.08},
            edge_tolerance_points=3.0,
            candidate_count=3,
            max_background_area_page_ratio=0.80,
            image_content_signature=lambda _: "image",
            drawing_content_signature=lambda drawing: (
                "footer" if drawing is footer else "body"
            ),
            height_tolerance_points=1.0,
            x_tolerance_points=2.0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content_signature"], "footer")
        self.assertEqual(result[0]["source_drawing_indices"], [0])

    #@GPT 模板学习按样本页支持率和同页对象数独立启用页尾模板，不应因为页眉没有候选而关闭页尾。
    def test_builds_footer_template_independently(self) -> None:
        class SamplePage:
            def __init__(self, number: int) -> None:
                self.number = number
                self.rect = fitz.Rect(0, 0, 100, 100)

            def get_text(self, *_args, **_kwargs):
                return {"blocks": []}

            def get_drawings(self):
                return [{"rect": fitz.Rect(10, 94, 40, 96)}]

        document = types.SimpleNamespace(
            page_count=2,
            load_page=lambda index: SamplePage(index),
        )
        from nativePDF_structurer.header_footer import build_header_footer_templates

        templates = build_header_footer_templates(
            document=document,
            drawings_cache={},
            sample_start_page=1,
            sample_max_page=2,
            candidate_count=3,
            detect_footer=True,
            minimum_matched_objects_by_edge={"header": 1, "footer": 1},
            minimum_support_ratio_by_edge={"header": 0.5, "footer": 0.5},
            edge_page_ratio_by_edge={"header": 0.18, "footer": 0.08},
            edge_tolerance_points=3.0,
            height_tolerance_points=1.0,
            x_tolerance_points=2.0,
            max_background_area_page_ratio=0.80,
            image_content_signature=lambda _: "image",
            drawing_content_signature=lambda _: "footer",
        )
        self.assertFalse(templates["header"]["enabled"])
        self.assertTrue(templates["footer"]["enabled"])
        self.assertEqual(len(templates["footer"]["templates"]), 1)

    #@GPT 逐页模板命中达到独立页尾门槛后，应生成页尾排除带并精确记录当前页drawing索引。
    def test_detects_footer_band_and_filters_confirmed_drawing(self) -> None:
        footer = {"rect": fitz.Rect(10, 94, 40, 96)}
        body = {"rect": fitz.Rect(10, 60, 40, 62)}
        page = types.SimpleNamespace(rect=fitz.Rect(0, 0, 100, 100))
        common = {
            "edge_page_ratio_by_edge": {"header": 0.18, "footer": 0.08},
            "edge_tolerance_points": 3.0,
            "candidate_count": 3,
            "max_background_area_page_ratio": 0.80,
            "image_content_signature": lambda _: "image",
            "drawing_content_signature": lambda drawing: (
                "footer" if drawing is footer else "body"
            ),
            "height_tolerance_points": 1.0,
            "x_tolerance_points": 2.0,
        }
        footer_template = extract_header_footer_candidates(
            page=page,
            page_dict={"blocks": []},
            edge="footer",
            drawings=[footer, body],
            **common,
        )[0]
        bands, record = detect_page_header_footer_bands(
            page=page,
            page_dict={"blocks": []},
            templates={
                "header": {"enabled": False, "templates": []},
                "footer": {
                    "enabled": True,
                    "minimum_matched_objects": 1,
                    "templates": [footer_template],
                },
            },
            drawings=[footer, body],
            detect_footer=True,
            minimum_matched_objects_by_edge={"header": 2, "footer": 1},
            band_padding_points=2.0,
            **common,
        )
        self.assertEqual(len(bands), 1)
        self.assertEqual(record["footer"]["matched_drawing_indices"], [0])

        filtered, audit = filter_confirmed_header_footer_drawings(
            drawings=[footer, body],
            header_footer_record=record,
        )
        self.assertEqual(filtered, [body])
        self.assertEqual(
            audit["vector_filter"]["removed_drawing_count"],
            1,
        )

    #@GPT 可选页眉文字提取应按重叠率选择、规范化空白并去除同bbox重复行；关闭开关时调用方可完全跳过该函数。
    def test_extracts_and_normalizes_header_text(self) -> None:
        duplicate = {
            "text": " Chapter   One ",
            "bbox": fitz.Rect(5, 2, 40, 8),
            "angle": 0,
        }
        text, lines = extract_header_text_for_markdown(
            text_lines=[
                duplicate,
                dict(duplicate),
                {
                    "text": "body",
                    "bbox": fitz.Rect(5, 20, 40, 28),
                    "angle": 0,
                },
            ],
            header_band=fitz.Rect(0, 0, 100, 12),
            overlap_threshold=0.50,
        )
        self.assertEqual(text, "Chapter One")
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
