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


def build_reflowed_page_text(
    text_lines: list[dict],
    page_rect: fitz.Rect,
    excluded_header_footer_rects: list[fitz.Rect],
    excluded_table_rects: list[fitz.Rect],
    excluded_visual_region_rects: list[fitz.Rect],
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
    skipped_header_footer_text_line_count = 0
    skipped_table_text_line_count = 0
    skipped_visual_region_text_line_count = 0
    skipped_formula_text_line_count = 0
    skipped_duplicate_page_title_text_line_count = 0
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
                # "text_line_count": len(group_lines),
                "text_line_count": sum(
                    line.get("kind", "text") == "text"
                    for line in group_lines
                ),
                "image_element_count": sum(
                    line.get("kind") == "image"
                    for line in group_lines
                ),
                "character_count": sum(
                    len(line["text"])
                    for line in group_lines
                    if line.get("kind", "text") == "text"
                ),
                "heading_line_count": sum(
                    bool(line.get("heading_level"))
                    for line in group_lines
                ),
                "heading_lines": [
                    {
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
        "skipped_duplicate_page_title_text_line_count": (
            skipped_duplicate_page_title_text_line_count
        ),
        "skipped_outline_heading_continuation_text_line_count": (
            skipped_outline_heading_continuation_text_line_count
        ),
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
