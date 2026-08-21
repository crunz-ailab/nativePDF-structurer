"""构建页面级重排元数据和分方向Markdown小节。

模块先排除已被结构化输出拥有
的文字，再按量化方向分组，调用阅读顺序后端，并将稳定的方向小节追加到
页面Markdown。

Build page-level reflow metadata and directional Markdown sections.

Removes text already owned by structural outputs, groups remaining atoms by
quantized direction, invokes the reading-order backend, and appends stable
directional sections to page Markdown.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import fitz

from .markdown_formatter import format_sorted_lines_as_markdown

#@GPT 页面级文本重排只编排排除区域、方向分组、阅读顺序回调和Markdown格式化，不读取文档级状态。
#@GPT 几何排除判据与排序实现由核心显式注入，使本模块能够独立测试并避免反向依赖兼容单文件。

def build_reflowed_page_text(
    text_lines: list[dict],
    page_rect: fitz.Rect,
    #@GPT 页眉/页尾模板不匹配文本内容，但模板确认后排除带内原生文字，防止重复页眉仍进入Markdown。
    excluded_header_footer_rects: list[fitz.Rect],
    excluded_table_rects: list[fitz.Rect],
    #@GPT 视觉区域已归属文字现在与PNG同块输出，此独立列表仅从全局正文流去重，不会丢失语义文本。
    excluded_visual_region_rects: list[fitz.Rect],
    #@GPT 公式渲染图同样只排除已归属的原生公式行框，避免公式碎片再次插入Markdown正文。
    excluded_formula_rects: list[fitz.Rect],
    *,
    line_is_inside_any_table: Callable[..., bool],
    sort_elements_with_gap_tree: Callable[..., tuple[list[dict], dict]],
) -> tuple[dict, dict[str, str]]:
    """
    处理表格外文本：

    1. 排除图片/矢量模板已确认的页眉和页尾文字。
    2. 排除已截图嵌入的大表格内部文字。
    3. 排除已渲染公式图片所拥有的原生公式碎片行。
    4. 按 line['dir'] 分成 0 / -90 / 90 / 180 度四组。
    5. 每组映射进独立的转正坐标系。
    6. 按转正后的阅读顺序输出 Markdown。
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    #@GPT 单独统计页眉/页尾带内被省略的原生文字，方便确认模板是否过宽。
    skipped_header_footer_text_line_count = 0
    skipped_table_text_line_count = 0
    #@GPT 单独统计视觉区域内被去重的文字行，避免继续混入表格统计。
    skipped_visual_region_text_line_count = 0
    #@GPT 统计被公式渲染图接管的碎片行，便于确认“图片+正文”没有重复。
    skipped_formula_text_line_count = 0
    #@GPT 当前页首目录标题已由外层Markdown H2完整表达；统计并跳过其原始文本框，避免相同标题在页面正文再次出现。
    skipped_duplicate_page_title_text_line_count = 0
    #@GPT 多行正文标题的后续原生行已经合并到leader标题中；单独统计这些去重行，避免与页面H2去重混淆。
    skipped_outline_heading_continuation_text_line_count = 0

    for line in text_lines:
        if (
            line.get("kind", "text") == "text"
            and line.get(
                "suppress_as_outline_heading_continuation"
            )
        ):
            skipped_outline_heading_continuation_text_line_count += 1
            continue
        if (
            line.get("kind", "text") == "text"
            and line.get("suppress_as_duplicate_page_title")
        ):
            skipped_duplicate_page_title_text_line_count += 1
            continue
        if (
            line.get("kind", "text") == "text"
            and line_is_inside_any_table(
                line_rect=line["bbox"],
                excluded_table_rects=(
                    excluded_header_footer_rects
                ),
            )
        ):
            skipped_header_footer_text_line_count += 1
            continue

        #@GPT 仅对原生文字执行截图区域去重，Markdown 图片元素自身必须继续进入阅读顺序。
        if (
            line.get("kind", "text") == "text"
            and line_is_inside_any_table(
            line_rect=line["bbox"],
            excluded_table_rects=excluded_table_rects,
            )
        ):
            skipped_table_text_line_count += 1
            continue

        if (
            line.get("kind", "text") == "text"
            and line_is_inside_any_table(
                line_rect=line["bbox"],
                excluded_table_rects=excluded_visual_region_rects,
            )
        ):
            skipped_visual_region_text_line_count += 1
            continue

        if (
            line.get("kind", "text") == "text"
            and line_is_inside_any_table(
                line_rect=line["bbox"],
                excluded_table_rects=excluded_formula_rects,
            )
        ):
            skipped_formula_text_line_count += 1
            continue

        groups[line["angle"]].append(line)

    labels = {
        0: "正常方向文本",
        -90: "顺时针转正文本",
        90: "逆时针转正文本",
        180: "旋转 180°转正文本",
    }

    grouped_markdown: dict[str, str] = {}
    direction_groups: list[dict] = []

    for angle in [0, -90, 90, 180]:
        group_lines = groups.get(angle, [])

        if not group_lines:
            continue

        #@GPT 所有文本和视觉原子确定后统一交给阅读顺序后端，避免局部坐标排序破坏多栏结构。
        sorted_lines, gaptree_metadata = (
            sort_elements_with_gap_tree(
                elements=group_lines,
                source_angle=angle,
                page_rect=page_rect,
            )
        )

        markdown_text = format_sorted_lines_as_markdown(
            sorted_lines
        )

        if markdown_text:
            grouped_markdown[str(angle)] = markdown_text

        direction_groups.append(
            {
                "angle": angle,
                "label": labels[angle],
                #@GPT 图片现与文字共用坐标排序；统计时分别记录两者，避免把 Markdown 图片链接误算成正文字符。
                # "text_line_count": len(group_lines),
                "text_line_count": sum(
                    line.get("kind", "text") == "text"
                    for line in group_lines
                ),
                "image_element_count": sum(
                    line.get("kind") == "image"
                    for line in group_lines
                ),
                #@GPT character_count只统计文本原子，图片和表格Markdown长度不参与文字规模审计。
                "character_count": sum(
                    len(line["text"])
                    for line in group_lines
                    if line.get("kind", "text") == "text"
                ),
                #@GPT 标题分类在文本原子进入GapTree前完成；这里保存每个方向实际进入Markdown的标题数量和字号，便于阶段一回归审计。
                "heading_line_count": sum(
                    bool(line.get("heading_level"))
                    for line in group_lines
                ),
                "heading_lines": [
                    {
                        #@GPT 多行目录标题在GapTree中由leader代表；元数据同步记录合并后的标题文字而不是仅第一行。
                        "text": line.get(
                            "heading_text_override",
                            line["text"],
                        ),
                        "level": line["heading_level"],
                        "font_size": line.get(
                            "heading_effective_font_size"
                        ),
                    }
                    for line in group_lines
                    if line.get("heading_level")
                ],
                #@GPT 将布局树节点和竖切线摘要写入页面元数据，便于不看冗长JSON也能生成可读报告。
                "gaptree": gaptree_metadata,
            }
        )

    metadata = {
        "skipped_header_footer_text_line_count": (
            skipped_header_footer_text_line_count
        ),
        "skipped_table_text_line_count": (
            skipped_table_text_line_count
        ),
        "skipped_visual_region_text_line_count": (
            skipped_visual_region_text_line_count
        ),
        "skipped_formula_text_line_count": (
            skipped_formula_text_line_count
        ),
        #@GPT 页面标题去重数量独立记录；这不是表格/视觉区域删除，可直接核对目录强匹配是否只消费了一行。
        "skipped_duplicate_page_title_text_line_count": (
            skipped_duplicate_page_title_text_line_count
        ),
        "skipped_outline_heading_continuation_text_line_count": (
            skipped_outline_heading_continuation_text_line_count
        ),
        #@GPT 标题记录随文本重排返回，避免额外扫描最终Markdown字符串而丢失原始字号和bbox证据。
        "heading_line_count": sum(
            direction.get("heading_line_count", 0)
            for direction in direction_groups
        ),
        "heading_lines": [
            heading
            for direction in direction_groups
            for heading in direction.get("heading_lines", [])
        ],
        "direction_groups": direction_groups,
    }

    return metadata, grouped_markdown

#@GPT 各方向文本通过唯一输出入口追加到页面Markdown，确保标签和小节顺序稳定。
def append_reflowed_text_to_markdown(
    page_markdown: list[str],
    grouped_markdown: dict[str, str],
) -> None:
    """将各个方向的表格外文本以独立小节写入 Markdown。"""
    labels = {
        "0": "页面文本",
        "-90": "旋转说明文本（顺时针转正后）",
        "90": "旋转说明文本（逆时针转正后）",
        "180": "倒置文本（旋转 180°后）",
    }

    for direction_key in ["0", "-90", "90", "180"]:
        text = grouped_markdown.get(direction_key)

        if not text:
            continue

        page_markdown.extend(
            [
                "",
                f"### {labels[direction_key]}",
                "",
                text,
            ]
        )
