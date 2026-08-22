"""将已经完成阅读顺序排序的页面原子格式化为稳定Markdown块。

模块输出标题、
图题、文字行、视觉原子和主题分隔线；它不读取PDF几何，也不调整元素顺序。

关键不变量:
    非文本原子独占段落；``---`` 前后使用块级空行；同一阅读行的文本按稳定
    硬换行规则输出。

Format ordered page atoms as stable Markdown blocks.

Consumes atoms whose reading order is already determined and emits headings,
captions, text rows, visual atoms, and thematic separators. This module does
not inspect PDF geometry or change element ordering.
"""

from __future__ import annotations

from collections import defaultdict

def format_sorted_lines_as_markdown(
    sorted_lines: list[dict],
) -> str:
    """
    将排序后的文本行生成 Markdown。

    同一虚拟阅读行里的多个块以空格拼接；
    每个逻辑阅读行使用 Markdown 硬换行。
    """
    if not sorted_lines:
        return ""

    rows: dict[int, list[dict]] = defaultdict(list)

    for line in sorted_lines:
        # rows[line["virtual_row_index"]].append(line["text"])
        rows[line["virtual_row_index"]].append(line)

    output_rows: list[str] = []
    text_separator_marker = "\x00GPT_TEXT_SEPARATOR\x00"

    for row_index in sorted(rows):
        row_elements = rows[row_index]
        figure_caption_elements = [
            element
            for element in row_elements
            if element.get("kind", "text") == "text"
            and element.get("figure_role") == "figure_caption"
        ]
        if (
            len(row_elements) == 1
            and len(figure_caption_elements) == 1
        ):
            caption_text = " ".join(
                figure_caption_elements[0].get("text", "").split()
            ).strip()
            if caption_text:
                output_rows.append(f"**{caption_text}**")
                continue
        row_heading_elements = [
            element
            for element in row_elements
            if element.get("kind", "text") == "text"
            and element.get("heading_level")
        ]
        if (
            len(row_elements) == 1
            and len(row_heading_elements) == 1
        ):
            heading_element = row_heading_elements[0]
            heading_level = int(
                heading_element["heading_level"]
            )
            heading_text = " ".join(
                str(
                    heading_element.get(
                        "heading_text_override",
                        heading_element["text"],
                    )
                ).split()
            ).strip()
            if heading_text:
                output_rows.append(
                    f"{'#' * heading_level} {heading_text}"
                )
                continue

        text_buffer: list[str] = []

        for element in rows[row_index]:
            if element.get("kind", "text") != "text":
                buffered_text = " ".join(text_buffer).strip()

                if buffered_text:
                    output_rows.append(buffered_text)

                text_buffer = []
                output_rows.append(
                    text_separator_marker
                    if element.get("kind")
                    == "text_separator"
                    else element["text"]
                )
                continue

            text_buffer.append(element["text"])

        buffered_text = " ".join(text_buffer).strip()
        if buffered_text:
            output_rows.append(buffered_text)

    markdown_blocks: list[str] = []
    current_text_rows: list[str] = []

    for row in output_rows:
        if row == text_separator_marker:
            if current_text_rows:
                markdown_blocks.append(
                    "  \n".join(current_text_rows)
                )
                current_text_rows = []

            if not markdown_blocks or markdown_blocks[-1] != "---":
                markdown_blocks.append("---")
            continue

        current_text_rows.append(row)

    if current_text_rows:
        markdown_blocks.append(
            "  \n".join(current_text_rows)
        )

    return "\n\n".join(markdown_blocks)
