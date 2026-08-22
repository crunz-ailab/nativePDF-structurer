from __future__ import annotations

import unittest

import fitz

from nativePDF_structurer.reading_order_backend import (
    GapTreeSorter,
    apply_attached_separator_heading_precedence,
    attach_narrow_text_domains_by_rows,
    coalesce_horizontal_text_fragments,
    create_reading_order_backend,
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
    #@GPT 视觉区域内部文字行排序迁移后仍按虚拟Y分行，并且行容差由调用方显式传入。
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

    #@GPT 已声明关联的Markdown分隔符应从原节点移到目标标题之前，不重新依赖标题文本内容判断。
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

    #@GPT 关联bbox无法由任何文字框包含时保持原节点不变，并在元数据中记录未解析数量。
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

    #@GPT 视觉原子与其下方短文本带被显著水平空白分开时，二级XY-Cut应拆成上下两个区块。
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

    #@GPT 水平分区必须使用视觉原子的完整render_bbox；顶部sort锚点不能虚构一条可切分的大空白。
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

    #@GPT 两个视觉组件之间即使有空白也不能按“视觉+短文本带”规则拆分，保护复合图片整体性。
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

    #@GPT 无有效水平切分时必须保持GapTree的列顺序；该回归覆盖论文双栏中左栏续行被全局Y排序挪到右栏之后的故障。
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

    #@GPT 两列稳定正文的首行若同时被提前放入同一标题节点，应分别回挂到对应列节点。
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

    #@GPT 只有一个列首候选时不触发跨列回挂，避免标题或缩进文本单独改变阅读顺序。
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

    #@GPT 旋转页面的中心点排序键必须与完整bbox使用同一方向约定，防止文字行和视觉原子排序不一致。
    def test_transformed_reading_position_uses_virtual_upright_coordinates(self) -> None:
        rect = fitz.Rect(10, 20, 30, 40)
        page_rect = fitz.Rect(0, 0, 100, 200)

        self.assertEqual(
            transformed_reading_position(rect, 0, page_rect),
            (30.0, 20.0),
        )
        self.assertEqual(
            transformed_reading_position(rect, -90, page_rect),
            #@GPT 旋转后的virtual_x使用页面高度减去矩形中心Y；本例中心Y为30，所以结果是170而非矩形左侧坐标推导值。
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

    #@GPT 同一Y行的重复窄列元素应移动到右侧目标节点，并保留目标节点内的行内X排序。
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

    #@GPT 后端已把编号和说明放在同一节点时，右挂阶段仍应恢复同行标记，而不是依赖跨节点搬运。
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

    #@GPT 被视觉原子隔开的单个末项仍可按“窄左域+窄同行空白”在同节点恢复同行关系。
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

    #@GPT 目标行与窄列不在同一Y带时不能跨行搬运，避免目录标题或步骤说明错配。
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

    #@GPT 没有重复窄域时应返回浅复制节点，不改变调用方的元素归属和原始顺序。
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

    #@GPT 稳定重复左边界间距形成双峰时才合并同行文本碎片，保护目录和双栏正文的原子边界。
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

    #@GPT 适配器必须透传排序、节点和行元数据，不引入额外排序或坐标变换。
    def test_gaptree_adapter_forwards_backend_contract(self) -> None:
        elements = [{"id": 1}, {"id": 2}]
        sorter = GapTreeSorter(FakeSorter, get_bbox=lambda item: item["id"])

        sorter.sort(elements)

        self.assertEqual(elements, [{"id": 2}, {"id": 1}])
        self.assertEqual(sorter.get_nodes_text_blocks(), [[{"id": "node"}]])
        self.assertEqual(sorter.current_rows, ["row"])
        self.assertEqual(sorter.current_cuts, [(1, 2, 3, 4)])
        self.assertEqual(sorter.backend_name, "gaptree")

    #@GPT 后端工厂只注册发布版GapTree，并拒绝未知名称。
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
