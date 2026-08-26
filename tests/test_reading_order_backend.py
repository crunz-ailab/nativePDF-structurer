from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.reading_order_backend import (
    GapTreeSorter,
    apply_attached_separator_heading_precedence,
    attach_narrow_text_domains_by_rows,
    coalesce_horizontal_text_fragments,
    create_reading_order_backend,
    pair_repeated_same_row_columns,
    reparent_cross_column_leading_row,
    split_gap_tree_nodes_by_horizontal_gaps,
    sort_lines_in_reading_order,
    transformed_reading_position,
)


class FakeSorter:
    def __init__(self, *, get_bbox):
        self.get_bbox = get_bbox
        self.current_rows = ["row"]
        self.current_cuts = [(1, 2, 3, 4)]
        self.sorted = False

    def sort(self, elements):
        self.sorted = True
        elements.reverse()

    def get_nodes_text_blocks(self):
        return [[{"id": "node"}]]


class ReadingOrderBackendTests(unittest.TestCase):
    def test_sort_lines_in_reading_order_with_explicit_tolerance(self) -> None:
        lines = [
            {"text": "右", "bbox": fitz.Rect(40, 10, 50, 18)},
            {"text": "左", "bbox": fitz.Rect(10, 11, 20, 19)},
            {"text": "下", "bbox": fitz.Rect(10, 30, 20, 38)},
        ]

        sorted_lines = sort_lines_in_reading_order(
            lines=lines,
            source_angle=0,
            page_rect=fitz.Rect(0, 0, 100, 100),
            row_tolerance_points=3.0,
        )

        self.assertEqual([line["text"] for line in sorted_lines], ["左", "右", "下"])
        self.assertEqual(
            [line["virtual_row_index"] for line in sorted_lines],
            [1, 1, 2],
        )

    def test_attached_separator_moves_before_heading(self) -> None:
        heading = self._text("合并后的章节标题", 10, 20, 120, 32)
        separator = {
            "kind": "text_separator",
            "text": "---",
            "bbox": (10, 17, 120, 18),
            "gaptree_bbox": (10, 17, 120, 18),
            "sort_before_heading": True,
            "attached_heading_bbox": (10, 20, 80, 32),
            "attached_heading_text": "旧的碎片标题",
            "source_rule_bbox": (10, 17, 120, 17),
        }

        groups, metadata = apply_attached_separator_heading_precedence(
            [[heading], [separator]],
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["attachment_count"], 1)
        self.assertEqual(
            [element["kind"] for element in groups[0]],
            ["text_separator", "text"],
        )
        self.assertEqual(groups[1], [])

    def test_attached_separator_without_heading_stays_in_place(self) -> None:
        heading = self._text("其他标题", 10, 20, 100, 32)
        separator = {
            "kind": "text_separator",
            "text": "---",
            "bbox": (10, 17, 100, 18),
            "gaptree_bbox": (10, 17, 100, 18),
            "sort_before_heading": True,
            "attached_heading_bbox": (150, 20, 190, 32),
        }
        original = [[heading], [separator]]

        groups, metadata = apply_attached_separator_heading_precedence(
            original,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["unresolved_count"], 1)
        self.assertEqual(groups, original)

    def test_horizontal_partition_splits_visual_then_short_text(self) -> None:
        visual = {
            "kind": "visual_region",
            "text": "![visual](visual.png)",
            "bbox": (10, 10, 100, 40),
            "render_bbox": (10, 10, 100, 40),
            "gaptree_bbox": (10, 10, 100, 12),
        }
        caption = self._text("产品号", 10, 60, 50, 70)

        groups, metadata = split_gap_tree_nodes_by_horizontal_gaps(
            [[visual, caption]],
            source_angle=0,
            page_rect=fitz.Rect(0, 0, 200, 200),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0]["kind"], "visual_region")
        self.assertEqual(groups[1][0]["text"], "产品号")

    def test_horizontal_partition_uses_render_bbox_not_sort_anchor(self) -> None:
        visual = {
            "kind": "visual_region",
            "text": "![visual](visual.png)",
            "bbox": (10, 10, 100, 12),
            "render_bbox": (10, 10, 100, 40),
            "sort_bbox": (10, 10, 100, 12),
            "gaptree_bbox": (10, 10, 100, 12),
        }
        nearby_text = self._text("紧邻文字", 10, 45, 60, 55)

        groups, metadata = split_gap_tree_nodes_by_horizontal_gaps(
            [[visual, nearby_text]],
            source_angle=0,
            page_rect=fitz.Rect(0, 0, 200, 200),
            row_tolerance_points=3.0,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_horizontal_partition_keeps_multiple_visuals_together(self) -> None:
        first = {
            "kind": "visual_region",
            "text": "first",
            "bbox": (10, 10, 80, 30),
            "gaptree_bbox": (10, 10, 80, 12),
        }
        second = {
            "kind": "source_image",
            "text": "second",
            "bbox": (10, 60, 80, 80),
            "gaptree_bbox": (10, 60, 80, 62),
        }

        groups, metadata = split_gap_tree_nodes_by_horizontal_gaps(
            [[first, second]],
            source_angle=0,
            page_rect=fitz.Rect(0, 0, 200, 200),
            row_tolerance_points=3.0,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(len(groups), 1)
        self.assertEqual([item["text"] for item in groups[0]], ["first", "second"])

    def test_horizontal_partition_preserves_node_order_without_split(self) -> None:
        node_groups = [
            [self._text("左栏上部", 10, 10, 50, 20)],
            [self._text("左栏续行", 10, 40, 50, 50)],
            [self._text("右栏上部", 80, 12, 120, 22)],
        ]

        groups, metadata = split_gap_tree_nodes_by_horizontal_gaps(
            node_groups,
            source_angle=0,
            page_rect=fitz.Rect(0, 0, 200, 200),
            row_tolerance_points=3.0,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(
            [group[0]["text"] for group in groups],
            ["左栏上部", "左栏续行", "右栏上部"],
        )
        self.assertEqual(
            metadata["reason"],
            "no_accepted_horizontal_partition",
        )

    def test_reparents_cross_column_leading_row(self) -> None:
        def column(anchor_x):
            return [
                self._text(
                    f"列{anchor_x}-{index}",
                    anchor_x,
                    y,
                    anchor_x + 40,
                    y + 10,
                )
                for index, y in enumerate((100, 200, 300, 400), start=1)
            ]

        node_groups = [
            [
                self._text("左栏首行", 10, 70, 50, 80),
                self._text("右栏首行", 80, 70, 120, 80),
            ],
            column(10),
            column(80),
        ]

        corrected, metadata = reparent_cross_column_leading_row(
            node_groups,
            fitz.Rect(0, 0, 500, 1000),
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["moved_element_count"], 2)
        self.assertEqual(len(corrected), 2)
        self.assertEqual(
            corrected[0][0]["text"],
            "左栏首行",
        )
        self.assertEqual(
            corrected[1][0]["text"],
            "右栏首行",
        )

    def test_reparent_requires_multiple_target_columns(self) -> None:
        def column(anchor_x):
            return [
                self._text("正文", anchor_x, y, anchor_x + 40, y + 10)
                for y in (100, 200, 300, 400)
            ]

        node_groups = [
            [self._text("单个首行", 10, 70, 50, 80)],
            column(10),
            column(80),
        ]

        corrected, metadata = reparent_cross_column_leading_row(
            node_groups,
            fitz.Rect(0, 0, 500, 1000),
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["reason"], "no_cross_column_leading_row")
        self.assertEqual(
            [[element["text"] for element in group] for group in corrected],
            [
                ["单个首行"],
                ["正文", "正文", "正文", "正文"],
                ["正文", "正文", "正文", "正文"],
            ],
        )

    def test_transformed_reading_position_uses_virtual_upright_coordinates(self) -> None:
        rect = fitz.Rect(10, 20, 30, 40)
        page_rect = fitz.Rect(0, 0, 100, 200)

        self.assertEqual(
            transformed_reading_position(rect, 0, page_rect),
            (30.0, 20.0),
        )
        self.assertEqual(
            transformed_reading_position(rect, -90, page_rect),
            (20.0, 170.0),
        )
        self.assertEqual(
            transformed_reading_position(rect, 90, page_rect),
            (80.0, 30.0),
        )
        self.assertEqual(
            transformed_reading_position(rect, 180, page_rect),
            (170.0, 80.0),
        )

    @staticmethod
    def _text(text, x0, y0, x1, y1):
        bbox = (x0, y0, x1, y1)
        return {
            "kind": "text",
            "text": text,
            "bbox": bbox,
            "gaptree_bbox": bbox,
        }

    def test_attaches_repeated_narrow_domain_by_same_row(self) -> None:
        node_groups = [
            [
                self._text("1", 10, 10, 15, 18),
                self._text("2", 10, 30, 15, 38),
            ],
            [
                self._text("打开控制柜", 25, 10, 75, 18),
                self._text("松开螺丝", 25, 30, 75, 38),
            ],
        ]

        corrected, metadata = attach_narrow_text_domains_by_rows(
            node_groups,
            fitz.Rect(0, 0, 200, 100),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["moved_element_count"], 2)
        self.assertEqual(len(corrected), 1)
        self.assertEqual(
            [element["text"] for element in corrected[0]],
            ["1", "打开控制柜", "2", "松开螺丝"],
        )

    def test_marks_same_node_narrow_domain_rows(self) -> None:
        node_groups = [[
            self._text("1", 10, 10, 15, 18),
            self._text("打开控制柜", 25, 10, 75, 18),
            self._text("2", 10, 30, 15, 38),
            self._text("松开螺丝", 25, 30, 75, 38),
        ]]

        corrected, metadata = attach_narrow_text_domains_by_rows(
            node_groups,
            fitz.Rect(0, 0, 200, 100),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(len(corrected), 1)
        self.assertEqual(
            [element["text"] for element in corrected[0]],
            ["1", "打开控制柜", "2", "松开螺丝"],
        )
        self.assertTrue(
            all(
                element.get("narrow_domain_row_group")
                for element in corrected[0]
            )
        )

    def test_marks_single_same_node_narrow_pair(self) -> None:
        node_groups = [[
            self._text("6", 10, 10, 15, 18),
            self._text("按相反顺序安装", 25, 10, 100, 18),
            self._text("页尾", 10, 40, 50, 48),
        ]]

        corrected, metadata = attach_narrow_text_domains_by_rows(
            node_groups,
            fitz.Rect(0, 0, 200, 100),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(
            [element["text"] for element in corrected[0]],
            ["6", "按相反顺序安装", "页尾"],
        )
        self.assertEqual(metadata["moved_element_count"], 1)

    def test_does_not_attach_across_rows(self) -> None:
        node_groups = [
            [
                self._text("1", 10, 10, 15, 18),
                self._text("2", 10, 30, 15, 38),
            ],
            [
                self._text("只在另一行", 25, 60, 75, 68),
            ],
        ]

        corrected, metadata = attach_narrow_text_domains_by_rows(
            node_groups,
            fitz.Rect(0, 0, 200, 100),
            row_tolerance_points=3.0,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["reason"], "no_same_row_narrow_domain_pair")
        self.assertEqual(
            [[element["text"] for element in group] for group in corrected],
            [["1", "2"], ["只在另一行"]],
        )

    def test_preserves_groups_without_repeated_narrow_domain(self) -> None:
        node_groups = [
            [self._text("普通正文", 10, 10, 80, 18)],
            [self._text("另一段正文", 10, 30, 90, 38)],
        ]

        corrected, metadata = attach_narrow_text_domains_by_rows(
            node_groups,
            fitz.Rect(0, 0, 200, 100),
            row_tolerance_points=3.0,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["reason"], "no_repeated_narrow_text_domain")
        self.assertEqual(
            [[element["text"] for element in group] for group in corrected],
            [["普通正文"], ["另一段正文"]],
        )

    # 覆盖章节目录中“左侧章节编号 + 右侧标题”的逐行配对场景。
    def test_pairs_repeated_same_row_columns_for_chapter_list(self) -> None:
        left_column = [
            self._text(f"第{index}章", 157, y, 199, y + 10)
            for index, y in enumerate((100, 130, 160, 190), start=1)
        ]
        right_column = [
            self._text(text, 211, y, x1, y + 10)
            for text, y, x1 in (
                ("投标邀请", 100, 390),
                ("项目说明与要求", 130, 430),
                ("技术参数", 160, 350),
                ("合同条款", 190, 410),
            )
        ]
        groups, metadata = pair_repeated_same_row_columns(
            [
                [self._text("目录", 100, 60, 500, 70)],
                left_column,
                right_column,
                [self._text("正文开始", 100, 230, 500, 240)],
            ],
            fitz.Rect(0, 0, 600, 800),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["merged_pair_count"], 1)
        self.assertEqual(len(groups), 3)
        self.assertEqual(
            [element["text"] for element in groups[1]],
            [
                "第1章", "投标邀请",
                "第2章", "项目说明与要求",
                "第3章", "技术参数",
                "第4章", "合同条款",
            ],
        )

    # 覆盖多级序号列宽略窄但超过旧窄列阈值的步骤说明场景。
    def test_pairs_multilevel_numbered_rows(self) -> None:
        left_column = [
            self._text("11.2.1", 139.1, 100, 172.436, 110),
            self._text("11.2.2", 139.1, 130, 172.436, 140),
        ]
        right_column = [
            self._text("拆下前面板", 184.1, 100, 320, 110),
            self._text("断开风扇连接器", 184.1, 130, 360, 140),
        ]
        groups, metadata = pair_repeated_same_row_columns(
            [
                [self._text("步骤", 100, 70, 500, 80)],
                left_column,
                right_column,
                [self._text("后续操作", 100, 170, 500, 180)],
            ],
            fitz.Rect(0, 0, 600, 800),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(
            [element["text"] for element in groups[1]],
            ["11.2.1", "拆下前面板", "11.2.2", "断开风扇连接器"],
        )

    # 覆盖一条左侧序号对应右侧多行换行说明的真实页面结构。
    def test_pairs_numbered_rows_with_wrapped_right_text(self) -> None:
        left_column = [
            self._text("11.2.1", 139.1, 100, 172.436, 110),
            self._text("11.2.2", 139.1, 160, 172.436, 170),
        ]
        right_column = [
            self._text("出厂价及附件费用", 184.1, 100, 360, 110),
            self._text("和运至最终目的地的运输费", 184.1, 115, 360, 125),
            self._text("安装调试、检验和服务费", 184.1, 130, 360, 140),
            self._text("按照招标文件完成全部费用", 184.1, 160, 360, 170),
        ]
        groups, metadata = pair_repeated_same_row_columns(
            [
                [self._text("步骤说明", 100, 70, 500, 80)],
                left_column,
                right_column,
                [self._text("后续正文", 100, 210, 500, 220)],
            ],
            fitz.Rect(0, 0, 600, 800),
            row_tolerance_points=3.0,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["pairs"][0]["row_count"], 2)
        self.assertEqual(
            [element["text"] for element in groups[1]],
            [
                "11.2.1",
                "出厂价及附件费用",
                "和运至最终目的地的运输费",
                "安装调试、检验和服务费",
                "11.2.2",
                "按照招标文件完成全部费用",
            ],
        )
        self.assertTrue(
            all(
                element.get("narrow_domain_row_group")
                and element.get("same_row_column_pair")
                for element in groups[1]
            )
        )

    # 防止把真正从页面边缘开始的双栏正文误合并为逐行列表。
    def test_does_not_pair_outer_two_column_body_text(self) -> None:
        left_column = [
            self._text("左栏一行", 10, 100, 90, 110),
            self._text("左栏二行", 10, 130, 90, 140),
        ]
        right_column = [
            self._text("右栏一行", 110, 100, 190, 110),
            self._text("右栏二行", 110, 130, 190, 140),
        ]
        original = [
            [self._text("上方正文", 10, 70, 190, 80)],
            left_column,
            right_column,
            [self._text("下方正文", 10, 170, 190, 180)],
        ]
        groups, metadata = pair_repeated_same_row_columns(
            original,
            fitz.Rect(0, 0, 200, 300),
            row_tolerance_points=3.0,
        )

        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["reason"], "no_accepted_pair")
        self.assertEqual(
            [[element["text"] for element in group] for group in groups],
            [[element["text"] for element in group] for group in original],
        )

    def test_coalesces_stable_bimodal_fragments(self) -> None:
        elements = []
        for row in range(10):
            y = row * 20
            for x in (0, 10, 20, 100, 110, 120):
                elements.append({
                    "kind": "text",
                    "text": "x",
                    "bbox": (x, y, x + 5, y + 8),
                    "gaptree_bbox": (x, y, x + 5, y + 8),
                })
        result, metadata = coalesce_horizontal_text_fragments(elements)
        self.assertTrue(metadata["enabled"])
        self.assertGreater(metadata["merged_fragment_count"], 0)

    def test_gaptree_adapter_forwards_backend_contract(self) -> None:
        elements = [{"id": 1}, {"id": 2}]
        sorter = GapTreeSorter(FakeSorter, get_bbox=lambda item: item["id"])

        sorter.sort(elements)

        self.assertEqual(elements, [{"id": 2}, {"id": 1}])
        self.assertEqual(sorter.get_nodes_text_blocks(), [[{"id": "node"}]])
        self.assertEqual(sorter.current_rows, ["row"])
        self.assertEqual(sorter.current_cuts, [(1, 2, 3, 4)])
        self.assertEqual(sorter.backend_name, "gaptree")

    def test_factory_selects_and_rejects_backends(self) -> None:
        sorter = create_reading_order_backend(
            " GapTree ",
            implementation=FakeSorter,
            get_bbox=lambda item: item["id"],
        )
        self.assertEqual(sorter.backend_name, "gaptree")
        with self.assertRaises(ValueError):
            create_reading_order_backend(
                "internal",
                implementation=FakeSorter,
                get_bbox=lambda item: item["id"],
            )


if __name__ == "__main__":
    unittest.main()
