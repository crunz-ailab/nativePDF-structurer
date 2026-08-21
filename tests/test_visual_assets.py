from __future__ import annotations

import unittest
from pathlib import Path

import fitz
import numpy as np

from nativePDF_structurer.inventory import PageInventory
from nativePDF_structurer.claims import create_claim_manifest
from nativePDF_structurer.visual_assets import (
    clear_rect_on_visual_mask,
    collect_visual_drawing_candidates,
    build_visual_regions_from_candidates,
    build_visual_region_output_record,
    build_visual_region_claims,
    detect_closed_visual_rule_containers,
    detect_coarse_visual_regions,
    draw_rect_on_visual_mask,
    extract_source_image_blocks,
    format_visual_region_native_text,
    image_has_complex_transform,
    image_is_large_enough,
    image_rects_overlap,
    merge_pure_vector_same_layer_once,
    merge_visual_region_records,
    merge_visual_regions_sharing_images,
    padded_image_clip,
    promote_right_attached_crop_text_lines,
    promote_text_lines_at_region_external_connectors,
    render_visual_regions,
    segment_bridges_opposite_visual_sides,
    segment_rect_intersection_interval,
    text_bbox_overlaps_visual_crop,
    text_is_owned_by_visual_structure,
    pure_vector_same_layer_should_merge,
    visual_regions_cross_distinct_closed_containers,
)


class VisualAssetPrimitiveTests(unittest.TestCase):
    #@GPT 图片合并只接受有实质面积交集的对象；边界接触和极小交集不能把两张图片组成复合区域。
    def test_image_overlap_uses_minimum_area_ratio(self) -> None:
        self.assertTrue(
            image_rects_overlap(
                fitz.Rect(0, 0, 10, 10),
                fitz.Rect(5, 0, 15, 10),
            )
        )
        self.assertFalse(
            image_rects_overlap(
                fitz.Rect(0, 0, 10, 10),
                fitz.Rect(10, 0, 20, 10),
            )
        )

    #@GPT 图片尺寸规则保留细条和极小面积双重过滤；页面大小不应改变绝对pt策略。
    def test_image_size_filter(self) -> None:
        page = fitz.Rect(0, 0, 100, 100)
        self.assertFalse(image_is_large_enough(fitz.Rect(0, 0, 40, 5), page))
        self.assertFalse(image_is_large_enough(fitz.Rect(0, 0, 10, 10), page))
        self.assertTrue(image_is_large_enough(fitz.Rect(0, 0, 20, 20), page))

    #@GPT 只有旋转/倾斜矩阵进入渲染分支；普通缩放和位移矩阵仍允许直接复制原图。
    def test_transform_and_padding_helpers(self) -> None:
        self.assertFalse(image_has_complex_transform((1, 0, 0, 1, 0, 0)))
        self.assertTrue(image_has_complex_transform((1, 0.2, 0, 1, 0, 0)))
        self.assertEqual(
            tuple(padded_image_clip(fitz.Rect(1, 1, 9, 9), fitz.Rect(0, 0, 10, 10))),
            (0.0, 0.0, 10.0, 10.0),
        )

    #@GPT 图片块提取只负责保留PDF原始字段，不在清点阶段提前过滤或改变source_index。
    def test_extracts_native_image_blocks(self) -> None:
        blocks = [
            {"type": 0, "bbox": (0, 0, 1, 1)},
            {
                "type": 1,
                "bbox": (2, 3, 12, 13),
                "ext": "jpeg",
                "image": b"bytes",
                "mask": None,
                "width": 40,
                "height": 50,
                "transform": (1, 0, 0, 1, 0, 0),
            },
        ]
        result = extract_source_image_blocks({"blocks": blocks})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_index"], 2)
        self.assertEqual(result[0]["extension"], "jpeg")
        self.assertEqual(result[0]["bbox"], fitz.Rect(2, 3, 12, 13))

    #@GPT 普通区域合并必须合并稳定drawing身份并保留结构排序框；不同闭合容器即使距离很近也不能被空间规则桥接。
    def test_visual_region_merge_preserves_ids_and_container_isolation(self) -> None:
        first = {
            "bbox": fitz.Rect(0, 0, 10, 10),
            "sort_source_bbox": fitz.Rect(0, 0, 10, 4),
            "drawing_element_ids": ["d:1"],
            "closed_visual_container_element_ids": ["c:1"],
        }
        second = {
            "bbox": fitz.Rect(11, 2, 20, 12),
            "sort_source_bbox": fitz.Rect(11, 2, 20, 5),
            "drawing_element_ids": ["d:2"],
            "closed_visual_container_element_ids": ["c:1"],
        }
        merged = merge_visual_region_records(
            [first, second],
            fitz.Rect(0, 0, 100, 100),
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["drawing_element_ids"], ["d:1", "d:2"])
        self.assertEqual(merged[0]["closed_visual_container_element_ids"], ["c:1"])
        self.assertEqual(tuple(merged[0]["sort_source_bbox"]), (0.0, 0.0, 20.0, 5.0))
        self.assertTrue(
            visual_regions_cross_distinct_closed_containers(
                {"closed_visual_container_element_ids": ["c:1"]},
                {"closed_visual_container_element_ids": ["c:2"]},
            )
        )

    #@GPT 纯矢量同层合并只执行一轮，并依据两个已聚合bbox的投影关系；不递归扩大同层判定范围。
    def test_pure_vector_same_layer_merge_is_one_pass(self) -> None:
        page = fitz.Rect(0, 0, 100, 100)
        self.assertTrue(
            pure_vector_same_layer_should_merge(
                fitz.Rect(0, 0, 10, 10),
                fitz.Rect(12, 1, 22, 11),
                page,
            )
        )
        result = merge_pure_vector_same_layer_once(
            [
                {
                    "bbox": fitz.Rect(0, 0, 10, 10),
                    "drawing_element_ids": ["d:1"],
                },
                {
                    "bbox": fitz.Rect(12, 1, 22, 11),
                    "drawing_element_ids": ["d:2"],
                },
            ],
            page,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["drawing_element_ids"], ["d:1", "d:2"])

    #@GPT 同一PDF原始图片被多个矢量簇引用时只合并同一结构域，避免图片重复渲染；不同闭合容器仍保持隔离。
    def test_shared_source_image_merge(self) -> None:
        regions = [
            {
                "bbox": fitz.Rect(0, 0, 10, 10),
                "structural_rects": [fitz.Rect(0, 0, 10, 10)],
                "attached_image_rects": [fitz.Rect(2, 2, 8, 8)],
                "attached_source_image_indices": [1],
                "drawing_element_ids": ["d:1"],
                "closed_visual_container_element_ids": ["c:1"],
            },
            {
                "bbox": fitz.Rect(12, 0, 22, 10),
                "structural_rects": [fitz.Rect(12, 0, 22, 10)],
                "attached_image_rects": [fitz.Rect(2, 2, 8, 8)],
                "attached_source_image_indices": [1],
                "drawing_element_ids": ["d:2"],
                "closed_visual_container_element_ids": ["c:1"],
            },
        ]
        result = merge_visual_regions_sharing_images(regions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["attached_source_image_indices"], [1])
        self.assertEqual(result[0]["drawing_element_ids"], ["d:1", "d:2"])

    #@GPT 掩码绘制和清除必须使用同一PDF到像素坐标映射，否则前置保护区域会在闭运算后残留桥接像素。
    def test_visual_mask_draw_and_clear_are_inverse_on_same_rect(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        page_rect = fitz.Rect(0, 0, 20, 20)
        target = fitz.Rect(4, 5, 10, 12)
        draw_rect_on_visual_mask(mask, target, page_rect, 1.0)
        self.assertGreater(int(mask.sum()), 0)
        clear_rect_on_visual_mask(mask, target, page_rect, 1.0)
        self.assertEqual(int(mask[5:13, 4:11].sum()), 0)

    #@GPT 候选生成优先复用台账标签，并在端点复核后过滤无标记水平分隔线；普通矢量保留原稳定ID。
    def test_collect_visual_candidates_uses_inventory_labels(self) -> None:
        normal = {"rect": fitz.Rect(10, 10, 20, 20), "type": "s"}
        separator = {"rect": fitz.Rect(10, 30, 50, 32), "type": "f"}
        background = {"rect": fitz.Rect(0, 0, 100, 100), "type": "f"}
        drawings = [normal, separator, background]
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=drawings,
            image_blocks=[],
            text_lines=[],
        )
        inventory.entry("d:0")["labels"] = {
            "degenerate": False,
            "background": False,
            "page_enclosing_stroke_frame": False,
            "filled_horizontal_bar": False,
            "long_horizontal_stroke": False,
        }
        inventory.entry("d:1")["labels"] = {
            "degenerate": False,
            "background": False,
            "page_enclosing_stroke_frame": False,
            "filled_horizontal_bar": True,
            "long_horizontal_stroke": False,
        }
        inventory.entry("d:2")["labels"] = {
            "degenerate": False,
            "background": True,
            "page_enclosing_stroke_frame": False,
            "filled_horizontal_bar": False,
            "long_horizontal_stroke": False,
        }

        candidates = collect_visual_drawing_candidates(
            page=object(),
            all_table_rects=[],
            excluded_header_footer_rects=[],
            drawings=drawings,
            page_inventory=inventory,
            max_background_area_page_ratio=0.80,
            build_endpoint_marker_grid=lambda _: {},
            drawing_has_endpoint_marker=lambda **_: False,
            drawing_is_filled_horizontal_bar=lambda _: False,
            drawing_is_long_horizontal_single_line=lambda _: False,
            drawing_is_page_enclosing_stroke_frame=lambda **_: False,
            rect_is_mostly_inside_any=lambda **_: False,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["element_id"], "d:0")
        self.assertIs(candidates[0]["drawing_payload"], normal)

    #@GPT 闭合容器检测必须由两条上下横线、两条贯穿竖线和至少一个内部候选共同触发，并把容器登记到页面台账。
    def test_detects_closed_visual_container_from_axis_rules(self) -> None:
        drawings = [
            {"rect": fitz.Rect(10, 10, 30, 10)},
            {"rect": fitz.Rect(10, 30, 30, 30)},
            {"rect": fitz.Rect(10, 10, 10, 30)},
            {"rect": fitz.Rect(30, 10, 30, 30)},
            {"rect": fitz.Rect(15, 15, 20, 20)},
        ]
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=drawings,
            image_blocks=[],
            text_lines=[],
        )
        for element_id in inventory.ids_of_kind("drawing"):
            inventory.entry(element_id)["labels"] = {
                "visible_semantic_stroke": True,
            }

        def rule(orientation: str, rect: fitz.Rect, position: float, index: int) -> dict:
            return {
                "orientation": orientation,
                "rect": rect,
                "position": position,
                "element_ids": [f"d:{index}"],
            }

        axis_index = {
            "logical_horizontal_rules": [
                rule("h", fitz.Rect(10, 10, 30, 10), 10, 0),
                rule("h", fitz.Rect(10, 30, 30, 30), 30, 1),
            ],
            "logical_vertical_rules": [
                rule("v", fitz.Rect(10, 10, 10, 30), 10, 2),
                rule("v", fitz.Rect(30, 10, 30, 30), 30, 3),
            ],
        }
        result = detect_closed_visual_rule_containers(
            page_rect=fitz.Rect(0, 0, 100, 100),
            axis_geometry_index=axis_index,
            page_inventory=inventory,
            candidate_element_ids={"d:0", "d:1", "d:2", "d:3", "d:4"},
            excluded_drawing_element_ids=set(),
            all_table_rects=[],
            container_rule_tolerance_points=1.0,
            max_background_area_page_ratio=0.80,
            drawing_has_visible_semantic_stroke=lambda _: True,
            rule_element_ids=lambda item: set(item.get("element_ids", [])),
            rect_is_mostly_inside_any=lambda **_: False,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0]["bbox"]), (10.0, 10.0, 30.0, 30.0))
        self.assertEqual(result[0]["interior_candidate_element_ids"], ["d:4"])
        self.assertEqual(result[0]["container_element_id"], "vc:0")

    #@GPT 连通域阶段应把空间上分离的合格候选分别输出，并保留每个候选的稳定drawing ID。
    def test_builds_initial_regions_from_mask_components(self) -> None:
        regions = build_visual_regions_from_candidates(
            page_rect=fitz.Rect(0, 0, 100, 100),
            candidates=[
                {"bbox": fitz.Rect(10, 10, 30, 30), "element_id": "d:1"},
                {"bbox": fitz.Rect(60, 60, 80, 80), "element_id": "d:2"},
            ],
            closed_visual_containers=[],
            all_table_rects=[],
            mask_pixels_per_point=1.0,
            close_gap_page_ratio=0.001,
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            sorted(region["drawing_element_ids"] for region in regions),
            [["d:1"], ["d:2"]],
        )

    #@GPT crop候选允许靠近左右侧且具有垂直投影的文字，但拒绝完全位于视觉区域上方的表头文字。
    def test_visual_text_crop_and_ownership_geometry(self) -> None:
        visual = fitz.Rect(10, 10, 30, 30)
        expanded = fitz.Rect(5, 5, 40, 35)
        self.assertTrue(
            text_bbox_overlaps_visual_crop(
                fitz.Rect(31, 15, 38, 22),
                visual,
                expanded,
                maximum_gap=4.0,
            )
        )
        self.assertFalse(
            text_bbox_overlaps_visual_crop(
                fitz.Rect(10, 2, 30, 8),
                visual,
                expanded,
                maximum_gap=4.0,
            )
        )
        self.assertTrue(
            text_is_owned_by_visual_structure(
                fitz.Rect(12, 12, 20, 18),
                [visual],
            )
        )

    #@GPT Liang-Barsky相交负责证明路径真实穿过区域；跨满相对边的结构线必须与单端外连说明线区分。
    def test_segment_intersection_and_structural_bridge(self) -> None:
        region = fitz.Rect(0, 0, 10, 10)
        self.assertEqual(
            segment_rect_intersection_interval(
                fitz.Point(-5, 5),
                fitz.Point(15, 5),
                region,
            ),
            (0.25, 0.75),
        )
        self.assertTrue(
            segment_bridges_opposite_visual_sides(
                fitz.Point(0, 5),
                fitz.Point(10, 5),
                region,
                boundary_tolerance=1.0,
            )
        )

    #@GPT 外连矢量只进行一跳文字附着，并返回真正连接该文字的原始drawing供截图扩张。
    def test_promotes_text_at_external_connector_endpoint(self) -> None:
        drawing = {
            "rect": fitz.Rect(5, 5, 20, 5),
            "items": [
                ("l", fitz.Point(5, 5), fitz.Point(20, 5)),
            ],
        }
        line = {
            "text": "label",
            "bbox": fitz.Rect(20, 3, 26, 7),
        }
        promoted, extension_drawings = (
            promote_text_lines_at_region_external_connectors(
                candidate_text_lines=[line],
                attached_text_lines=[],
                region_rect=fitz.Rect(0, 0, 10, 10),
                drawings=[drawing],
                maximum_gap=8.0,
            )
        )
        self.assertEqual(promoted, [line])
        self.assertEqual(extension_drawings, [drawing])

    #@GPT 小图标右侧且与视觉簇有Y投影的crop文字可提升为图内文字，大尺寸附图不触发该规则。
    def test_promotes_right_attached_crop_text_for_small_image(self) -> None:
        line = {"text": "warning", "bbox": fitz.Rect(21, 5, 40, 12)}
        result = promote_right_attached_crop_text_lines(
            crop_text_lines=[line],
            attached_text_lines=[],
            visual_rect=fitz.Rect(0, 0, 20, 20),
            attached_image_rects=[fitz.Rect(2, 2, 10, 10)],
        )
        self.assertEqual(result, [line])

    #@GPT 原生图内文字格式化只负责Markdown转义和引用块结构，排序策略由调用方提供且不改变文字内容。
    def test_formats_native_visual_text_with_injected_sorter(self) -> None:
        lines = [
            {"text": "a|b", "angle": 0, "bbox": fitz.Rect(0, 0, 10, 5)},
            {"text": "line\nnext", "angle": 0, "bbox": fitz.Rect(0, 6, 10, 12)},
        ]
        markdown, ordered = format_visual_region_native_text(
            text_lines=lines,
            page_rect=fitz.Rect(0, 0, 20, 20),
            sort_lines_in_reading_order=lambda **kwargs: kwargs["lines"],
        )
        self.assertEqual(ordered, lines)
        self.assertIn("> **图内文字（PDF原生文本）**", markdown)
        self.assertIn("a\\|b", markdown)
        self.assertIn("line next", markdown)

    #@GPT 视觉输出构造器必须区分完整截图render_bbox与文字扩张前sort_source_bbox，并让Markdown元素与审计记录共享claim_id。
    def test_builds_visual_output_record_with_two_bboxes(self) -> None:
        markdown_element, record, render_bbox, owned_text_rects = (
            build_visual_region_output_record(
                page_number=1,
                region_number=2,
                region={
                    "bbox": fitz.Rect(10, 10, 30, 30),
                    "sort_source_bbox": fitz.Rect(12, 12, 28, 28),
                    "content_bbox": fitz.Rect(12, 12, 28, 28),
                    "drawing_count": 2,
                    "component_count": 1,
                    "included_text_line_count": 1,
                    "included_image_count": 0,
                    "included_text_lines": [
                        {
                            "text": "caption",
                            "angle": 0,
                            "bbox": fitz.Rect(13, 13, 20, 17),
                        }
                    ],
                },
                output_bbox=fitz.Rect(8, 8, 32, 32),
                output_path=Path("page_001_visual_region_02.png"),
                output_width=200,
                output_height=200,
                render_seconds=0.125678,
                page_rect=fitz.Rect(0, 0, 100, 100),
                sort_lines_in_reading_order=lambda **kwargs: kwargs["lines"],
            )
        )
        self.assertEqual(render_bbox, fitz.Rect(8, 8, 32, 32))
        self.assertEqual(markdown_element["claim_id"], "page#1:visual_region#2")
        self.assertEqual(record["claim_id"], "page#1:visual_region#2")
        self.assertEqual(record["sort_source_bbox"], [12.0, 12.0, 28.0, 28.0])
        self.assertEqual(record["sort_bbox"], [12.0, 12.0, 28.0, 14.0])
        self.assertEqual(len(owned_text_rects), 1)

    #@GPT 视觉Claim应直接使用候选保存的drawing/image/container ID，文字通过台账对象身份映射，不允许从扩张后的bbox重新猜测。
    def test_builds_visual_claim_from_stable_ids(self) -> None:
        drawing = {"rect": fitz.Rect(0, 0, 10, 10)}
        image = {"bbox": fitz.Rect(1, 1, 9, 9)}
        line = {"text": "inside", "angle": 0, "bbox": fitz.Rect(2, 2, 8, 4)}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[image],
            text_lines=[line],
        )
        container_id = inventory.register_visual_container(
            container_number=1,
            bbox=fitz.Rect(0, 0, 10, 10),
            payload={},
        )
        claims = build_visual_region_claims(
            page_number=1,
            regions=[
                {
                    "bbox": fitz.Rect(0, 0, 10, 10),
                    "drawing_count": 1,
                    "included_image_count": 1,
                    "included_text_line_count": 1,
                    "drawing_element_ids": ["d:0"],
                    "attached_image_element_ids": ["i:0"],
                    "closed_visual_container_element_ids": [container_id],
                    "included_text_lines": [line],
                }
            ],
            records=[{"output_path": "visual_regions/example.png"}],
            page_inventory=inventory,
            create_claim_manifest=create_claim_manifest,
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["claimed_element_ids"],
            ["d:0", "i:0", "t:0", "vc:0"],
        )
        self.assertEqual(
            claims[0]["feature_element_ids"],
            ["d:0", "i:0", "vc:0"],
        )
        self.assertIn("owned_native_text", claims[0]["reason_codes"])

    #@GPT 页面渲染器应调用一次PyMuPDF截图接口，并把输出记录、Markdown原子和Claim作为同一候选结果返回。
    def test_renders_visual_regions_and_builds_outputs(self) -> None:
        class FakePixmap:
            width = 120
            height = 80

            def __init__(self) -> None:
                self.saved_path = None

            def save(self, path: Path) -> None:
                self.saved_path = path

        class FakePage:
            rect = fitz.Rect(0, 0, 100, 100)

            def __init__(self) -> None:
                self.calls = []
                self.pixmap = FakePixmap()

            def get_pixmap(self, **kwargs):
                self.calls.append(kwargs)
                return self.pixmap

        drawing = {"rect": fitz.Rect(10, 10, 30, 30)}
        inventory = PageInventory(
            page_number=1,
            page_rect=fitz.Rect(0, 0, 100, 100),
            drawings=[drawing],
            image_blocks=[],
            text_lines=[],
        )
        page = FakePage()
        records, markdown, output_rects, owned_text, claims = (
            render_visual_regions(
                page=page,
                page_number=1,
                regions=[
                    {
                        "bbox": fitz.Rect(10, 10, 30, 30),
                        "sort_source_bbox": fitz.Rect(10, 10, 30, 30),
                        "content_bbox": fitz.Rect(10, 10, 30, 30),
                        "drawing_count": 1,
                        "component_count": 1,
                        "included_text_line_count": 0,
                        "included_image_count": 0,
                        "drawing_element_ids": ["d:0"],
                        "included_text_lines": [],
                    }
                ],
                regions_dir=Path("visual_regions"),
                page_inventory=inventory,
                render_dpi=200,
                sort_lines_in_reading_order=lambda **kwargs: kwargs["lines"],
                create_claim_manifest=create_claim_manifest,
            )
        )
        self.assertEqual(len(page.calls), 1)
        self.assertEqual(page.calls[0]["dpi"], 200)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(markdown), 1)
        self.assertEqual(len(output_rects), 1)
        self.assertEqual(owned_text, [])
        self.assertEqual(claims[0]["feature_element_ids"], ["d:0"])

    #@GPT 粗视觉总编排在没有候选drawing时应安全返回空列表，不触发掩码、容器或文字附着阶段。
    def test_coarse_visual_regions_empty_input(self) -> None:
        page = type("Page", (), {"rect": fitz.Rect(0, 0, 100, 100)})()
        inventory = PageInventory(
            page_number=1,
            page_rect=page.rect,
            drawings=[],
            image_blocks=[],
            text_lines=[],
        )
        result = detect_coarse_visual_regions(
            page=page,
            page_dict={},
            text_lines=[],
            all_table_rects=[],
            excluded_header_footer_rects=[],
            logical_horizontal_rules=[],
            drawings=[],
            source_images=[],
            page_inventory=inventory,
            axis_geometry_index={},
            max_background_area_page_ratio=0.80,
            build_endpoint_marker_grid=lambda _: {},
            drawing_has_endpoint_marker=lambda **_: False,
            drawing_has_visible_semantic_stroke=lambda _: True,
            drawing_is_filled_horizontal_bar=lambda _: False,
            drawing_is_long_horizontal_single_line=lambda _: False,
            drawing_is_page_enclosing_stroke_frame=lambda **_: False,
            detect_rule_line_framework_bbox=lambda **_: (None, {}),
            group_horizontal_rules_by_span=lambda _: [],
            unique_rule_positions=lambda _: [],
            rule_element_ids=lambda rule: set(rule.get("element_ids", [])),
            rect_is_contained_with_tolerance=lambda **_: False,
            rect_is_mostly_inside_any=lambda **_: False,
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
