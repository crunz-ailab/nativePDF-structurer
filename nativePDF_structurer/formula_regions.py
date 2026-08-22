"""根据PDF原生文字层检测并渲染行间公式区域。

模块使用字体族、数学符号、栏位
几何、纵向聚类和公式编号锚点定位公式；确认区域以图片输出，不重建LaTeX，
也不执行OCR。

Detect and render display-formula regions from native PDF text.

Uses font families, mathematical symbols, column geometry, vertical
clustering, and equation-number anchors to locate display formulas. Accepted
regions are rendered as images; the module does not reconstruct LaTeX or run
OCR.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from pathlib import Path

import fitz

from .geometry import union_rects
from .visual_assets import build_top_sort_bbox


FORMULA_CLUSTER_VERTICAL_GAP_POINTS = 10.0
FORMULA_NUMBER_ATTACH_MARGIN_POINTS = 14.0
MATH_FONT_HINTS = (
    "CambriaMath", "CMMI", "CMEX", "CMSY", "MSBM", "Math",
    "Symbol", "STIX", "MSAM", "CMMIB", "NewTXM", "NewTXSY",
    "txmia", "txsys", "PazoMath", "XCharterMath", "LMMath", "CMBSY",
)
MATH_SYMBOLS = set("=+-−±×÷*/^_{}[]()<>≤≥≈≠∫∑∏√∞∂∇·⋅→←↦∈∀∃|‖")
MATH_GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
LATEX_COMMAND_PATTERN = re.compile(r"\\\s*[A-Za-z]+")
EQUATION_NUMBER_PATTERN = re.compile(r"^\s*\((?:Eq\.\s*)?\d+[a-z]?\)\s*$", re.I)
EQUATION_NUMBER_AT_END_PATTERN = re.compile(r"\((?:Eq\.\s*)?\d+[a-z]?\)\s*$", re.I)

def formula_column_key(
    rect: fitz.Rect,
    page_rect: fitz.Rect,
) -> str:
    """将公式碎片粗分到左栏、右栏或跨栏，防止双栏页面横向串联。"""
    midpoint = (page_rect.x0 + page_rect.x1) / 2
    gutter_margin = page_rect.width * 0.025
    center_x = (rect.x0 + rect.x1) / 2

    if rect.x1 <= midpoint + gutter_margin and center_x < midpoint:
        return "left"

    if rect.x0 >= midpoint - gutter_margin and center_x > midpoint:
        return "right"

    return "full"

def extract_formula_line_records(
    page_dict: dict,
    page_rect: fitz.Rect,
    excluded_rects: list[fitz.Rect],
    *,
    line_is_inside_any_table: Callable[..., bool],
) -> list[dict]:
    """提取带公式结构特征的文本行和独立公式编号。"""
    records: list[dict] = []

    for block_index, block in enumerate(page_dict["blocks"]):
        if block["type"] != 0:
            continue

        for line_index, line in enumerate(block["lines"]):
            text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            ).strip()

            if not text:
                continue

            bbox = fitz.Rect(line["bbox"])

            if line_is_inside_any_table(
                line_rect=bbox,
                excluded_table_rects=excluded_rects,
            ):
                continue

            non_space_count = sum(
                not character.isspace() for character in text
            )
            math_font_character_count = 0

            for span in line.get("spans", []):
                span_text = span.get("text", "")
                font_name = span.get("font", "")

                if any(
                    hint.lower() in font_name.lower()
                    for hint in MATH_FONT_HINTS
                ):
                    math_font_character_count += sum(
                        not character.isspace()
                        for character in span_text
                    )

            math_font_ratio = (
                math_font_character_count / non_space_count
                if non_space_count > 0
                else 0.0
            )
            latex_command_count = len(
                LATEX_COMMAND_PATTERN.findall(text)
            )
            math_symbol_count = sum(
                character in MATH_SYMBOLS for character in text
            )
            greek_count = sum(
                character in MATH_GREEK for character in text
            )
            prose_word_count = len(
                re.findall(r"[A-Za-z]{3,}", text)
            )
            is_equation_number = bool(
                EQUATION_NUMBER_PATTERN.fullmatch(text)
            )
            has_inline_equation_number = bool(
                EQUATION_NUMBER_AT_END_PATTERN.search(text)
            ) and not is_equation_number
            structural_signal_count = (
                latex_command_count * 3
                + math_symbol_count
                + greek_count
            )
            is_formula_candidate = (
                not is_equation_number
                and (
                    latex_command_count > 0
                    or (
                        math_font_ratio >= 0.65
                        and structural_signal_count > 0
                        and prose_word_count <= 4
                    )
                    or (
                        math_font_ratio >= 0.85
                        and non_space_count <= 8
                        and prose_word_count <= 1
                    )
                )
            )

            if not (is_formula_candidate or is_equation_number):
                continue

            records.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "block_index": block_index,
                    "line_index": line_index,
                    "column": formula_column_key(bbox, page_rect),
                    "is_formula_candidate": is_formula_candidate,
                    "is_equation_number": is_equation_number,
                    "has_inline_equation_number": (
                        has_inline_equation_number
                    ),
                    "math_font_ratio": math_font_ratio,
                    "latex_command_count": latex_command_count,
                    "math_symbol_count": math_symbol_count,
                    "greek_count": greek_count,
                    "prose_word_count": prose_word_count,
                    "signal_score": structural_signal_count,
                }
            )

    return records

def cluster_formula_candidates(
    candidates: list[dict],
) -> list[dict]:
    """把被PDF拆散的公式行聚合为显示公式候选区域。"""
    clusters: list[dict] = []

    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["column"],
            item["bbox"].y0,
            item["bbox"].x0,
        ),
    ):
        matching_cluster: dict | None = None
        minimum_gap = math.inf

        for cluster in clusters:
            if cluster["column"] != candidate["column"]:
                continue

            cluster_bbox = cluster["bbox"]
            vertical_gap = max(
                0.0,
                candidate["bbox"].y0 - cluster_bbox.y1,
                cluster_bbox.y0 - candidate["bbox"].y1,
            )

            if (
                vertical_gap
                <= FORMULA_CLUSTER_VERTICAL_GAP_POINTS
                and vertical_gap < minimum_gap
            ):
                matching_cluster = cluster
                minimum_gap = vertical_gap

        if matching_cluster is None:
            clusters.append(
                {
                    "column": candidate["column"],
                    "bbox": fitz.Rect(candidate["bbox"]),
                    "lines": [candidate],
                }
            )
            continue

        matching_cluster["lines"].append(candidate)
        matching_cluster["bbox"] = union_rects(
            [matching_cluster["bbox"], candidate["bbox"]]
        )

    return clusters

def attach_equation_numbers_to_clusters(
    clusters: list[dict],
    equation_numbers: list[dict],
) -> None:
    """把独立的(1)、(2)等编号分配给纵向最近的同栏公式。"""
    for number in equation_numbers:
        best_cluster: dict | None = None
        best_distance = math.inf
        number_center_y = (
            number["bbox"].y0 + number["bbox"].y1
        ) / 2

        for cluster in clusters:
            if cluster["column"] != number["column"]:
                continue

            cluster_bbox = cluster["bbox"]

            if not (
                cluster_bbox.y0 - FORMULA_NUMBER_ATTACH_MARGIN_POINTS
                <= number_center_y
                <= cluster_bbox.y1 + FORMULA_NUMBER_ATTACH_MARGIN_POINTS
            ):
                continue

            distance = abs(
                number_center_y
                - (cluster_bbox.y0 + cluster_bbox.y1) / 2
            )

            if distance < best_distance:
                best_cluster = cluster
                best_distance = distance

        if best_cluster is None:
            continue

        best_cluster.setdefault("equation_numbers", []).append(
            number
        )
        best_cluster["bbox"] = union_rects(
            [best_cluster["bbox"], number["bbox"]]
        )

def process_page_formula_regions(
    page: fitz.Page,
    page_dict: dict,
    excluded_table_rects: list[fitz.Rect],
    excluded_visual_text_rects: list[fitz.Rect],
    excluded_header_footer_rects: list[fitz.Rect],
    page_number: int,
    formula_dir: Path,
    *,
    formula_image_dpi: int,
    formula_min_unnumbered_width_page_ratio: float,
    formula_padding_points: float,
    line_is_inside_any_table: Callable[..., bool],
) -> tuple[list[dict], list[dict], list[fitz.Rect], float]:
    """检测显示公式、渲染图片，并返回需要从正文精确去重的行框。"""
    started = time.perf_counter()
    line_records = extract_formula_line_records(
        page_dict=page_dict,
        page_rect=page.rect,
        excluded_rects=[
            *excluded_table_rects,
            *excluded_visual_text_rects,
            *excluded_header_footer_rects,
        ],
        line_is_inside_any_table=line_is_inside_any_table,
    )
    formula_candidates = [
        record
        for record in line_records
        if record["is_formula_candidate"]
    ]
    equation_numbers = [
        record
        for record in line_records
        if record["is_equation_number"]
    ]
    clusters = cluster_formula_candidates(formula_candidates)
    attach_equation_numbers_to_clusters(
        clusters=clusters,
        equation_numbers=equation_numbers,
    )
    accepted_clusters: list[dict] = []

    for cluster in clusters:
        lines = cluster["lines"]
        equation_number_lines = cluster.get(
            "equation_numbers", []
        )
        has_number_anchor = bool(equation_number_lines) or any(
            line["has_inline_equation_number"]
            for line in lines
        )
        total_signal_score = sum(
            line["signal_score"] for line in lines
        )
        prose_word_count = sum(
            line["prose_word_count"] for line in lines
        )
        unnumbered_high_confidence = (
            cluster["bbox"].width
            >= page.rect.width
            * formula_min_unnumbered_width_page_ratio
            and total_signal_score >= 6
            and prose_word_count <= 6
        )

        if not (has_number_anchor or unnumbered_high_confidence):
            continue

        cluster["equation_numbers"] = equation_number_lines
        cluster["has_number_anchor"] = has_number_anchor
        cluster["total_signal_score"] = total_signal_score
        accepted_clusters.append(cluster)

    records: list[dict] = []
    markdown_elements: list[dict] = []
    owned_text_rects: list[fitz.Rect] = []

    for formula_number, cluster in enumerate(
        sorted(
            accepted_clusters,
            key=lambda item: (
                item["bbox"].y0,
                item["bbox"].x0,
            ),
        ),
        start=1,
    ):
        owned_lines = [
            *cluster["lines"],
            *cluster["equation_numbers"],
        ]
        owned_rects = [
            fitz.Rect(line["bbox"]) for line in owned_lines
        ]
        raw_bbox = union_rects(owned_rects)
        output_bbox = fitz.Rect(
            max(
                page.rect.x0,
                raw_bbox.x0 - formula_padding_points,
            ),
            max(
                page.rect.y0,
                raw_bbox.y0 - formula_padding_points,
            ),
            min(
                page.rect.x1,
                raw_bbox.x1 + formula_padding_points,
            ),
            min(
                page.rect.y1,
                raw_bbox.y1 + formula_padding_points,
            ),
        )
        output_name = (
            f"page_{page_number:03d}_formula_"
            f"{formula_number:02d}.png"
        )
        output_path = formula_dir / output_name
        render_started = time.perf_counter()
        pixmap = page.get_pixmap(
            clip=output_bbox,
            dpi=formula_image_dpi,
            alpha=False,
            annots=True,
        )
        pixmap.save(output_path)
        render_seconds = time.perf_counter() - render_started
        relative_path = f"formula_images/{output_name}"
        markdown_text = "\n\n".join(
            [
                f"![Page {page_number} Mathematical Formula "
                f"{formula_number}]({relative_path})",
                "> 数学公式（PDF页面渲染图）",
            ]
        )
        owned_text_rects.extend(owned_rects)
        markdown_elements.append(
            {
                "bbox": output_bbox,
                "render_bbox": output_bbox,
                "sort_bbox": build_top_sort_bbox(output_bbox),
                "text": markdown_text,
                "direction": (1.0, 0.0),
                "angle": 0,
                "kind": "formula",
                "markdown": markdown_text,
                "mode": "rendered_formula",
                "output_path": output_path,
            }
        )
        records.append(
            {
                "page_number": page_number,
                "formula_number": formula_number,
                "bbox": [
                    round(value, 3) for value in output_bbox
                ],
                "render_bbox": [
                    round(value, 3) for value in output_bbox
                ],
                "sort_bbox": [
                    round(value, 3)
                    for value in build_top_sort_bbox(output_bbox)
                ],
                "column": cluster["column"],
                "has_number_anchor": cluster[
                    "has_number_anchor"
                ],
                "owned_text_line_count": len(owned_lines),
                "equation_labels": [
                    line["text"]
                    for line in cluster["equation_numbers"]
                ],
                "output_path": str(output_path),
                "output_width": pixmap.width,
                "output_height": pixmap.height,
                "render_seconds": round(render_seconds, 6),
            }
        )

    return (
        records,
        markdown_elements,
        owned_text_rects,
        time.perf_counter() - started,
    )
