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

#@GPT Markdown格式化与PDF检测、阅读顺序算法拆分：本模块只消费已排序原子，不读取页面对象或修改几何信息。
#@GPT 标题、图题、图像/表格原子和纯文本分隔线的输出规则集中在此处，避免核心重排流程混入字符串拼接细节。
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

    #@GPT 阅读行保留完整元素对象，使非文本原子能够独占Markdown段落并保持bbox顺序。
    rows: dict[int, list[dict]] = defaultdict(list)

    for line in sorted_lines:
        # rows[line["virtual_row_index"]].append(line["text"])
        rows[line["virtual_row_index"]].append(line)

    output_rows: list[str] = []
    #@GPT 纯文字水平线不能与普通文字行只用硬换行连接，否则 `---` 可能与前后文字被Markdown解析成setext标题或紧贴段落；先使用内部标记，最后按块输出空行。
    text_separator_marker = "\x00GPT_TEXT_SEPARATOR\x00"

    for row_index in sorted(rows):
        row_elements = rows[row_index]
        figure_caption_elements = [
            element
            for element in row_elements
            if element.get("kind", "text") == "text"
            and element.get("figure_role") == "figure_caption"
        ]
        #@GPT 图题使用独立粗体段落，提供类似表头的视觉区分但不制造章节层级；多列同行或混合元素时不强行格式化。
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
        #@GPT 只有独立阅读行才提升为Markdown标题；多列同行或包含图片/分隔符的行继续按普通行输出，避免目录列锚点被改写成标题。
        if (
            len(row_elements) == 1
            and len(row_heading_elements) == 1
        ):
            heading_element = row_heading_elements[0]
            heading_level = int(
                heading_element["heading_level"]
            )
            #@GPT 同一PDF文本块内换行的目录标题由首行携带合并文本；普通单行标题使用原text字段。
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

        #@GPT 图片可能与文字拥有相近基线；逐元素刷新文字缓冲，避免图片链接被拼进普通句子。
        text_buffer: list[str] = []

        for element in rows[row_index]:
            #@GPT GapTree输入包含表格、图片和矢量区域等原子块；所有非文本元素独占Markdown段落。
            if element.get("kind", "text") != "text":
                buffered_text = " ".join(text_buffer).strip()

                if buffered_text:
                    output_rows.append(buffered_text)

                text_buffer = []
                #@GPT 只有纯文字分隔线需要在最终文本块之间增加空行；图片/表格仍保持原来的原子行衏为。
                output_rows.append(
                    text_separator_marker
                    if element.get("kind")
                    == "text_separator"
                    else element["text"]
                )
                continue

            text_buffer.append(element["text"])

        #@GPT 每个GapTree阅读行都必须在该行结束时刷新文字缓冲；此前这段逻辑误缩进到外层循环之后，导致纯文字分隔带之间的正文被吞掉，只剩最后一行。
        buffered_text = " ".join(text_buffer).strip()
        if buffered_text:
            output_rows.append(buffered_text)

    #@GPT 块级空行确保Markdown `---` 被渲染为横向规则线，而不是相邻文字的下划线语法。
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
