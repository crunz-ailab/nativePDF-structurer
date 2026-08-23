"""根据PDF原生文字层检测并渲染行间公式区域。

模块使用字体族、数学符号、栏位几何、纵向聚类和公式编号锚点定位公式。
默认将确认区域渲染为图片；在显式开启时，仅对能够安全线性化的简单公式
输出LaTeX，否则回退图片。模块不执行OCR。

Detect and render display-formula regions from native PDF text.

Uses font families, mathematical symbols, column geometry, vertical clustering,
and equation-number anchors to locate display formulas. Accepted regions are
rendered as images by default; an explicit LaTeX mode emits only safely
linearized simple formulas and falls back to images for ambiguous structures.
The module does not run OCR.
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
FORMULA_NUMBER_SEED_VERTICAL_MARGIN_POINTS = 18.0
FORMULA_NUMBER_SEED_MAX_PROSE_WORDS_PER_LINE = 1
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
LATEX_SAFE_TEXT_PATTERN = re.compile(
    r"^[A-Za-z0-9\\{}^_+=*/().,;:'\-\s×÷≤≥≈≠±]+$"
)

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

def extract_formula_line_feature_records(
    page_dict: dict,
    page_rect: fitz.Rect,
    excluded_rects: list[fitz.Rect],
    *,
    line_is_inside_any_table: Callable[..., bool],
) -> list[dict]:
    """提取页面文本行及其公式结构特征，不预先丢弃普通行。"""
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
            math_font_span_count = 0

            for span in line.get("spans", []):
                span_text = span.get("text", "")
                font_name = span.get("font", "")

                if any(
                    hint.lower() in font_name.lower()
                    for hint in MATH_FONT_HINTS
                ):
                    math_font_span_count += 1
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
                    "math_font_character_count": (
                        math_font_character_count
                    ),
                    "math_font_span_count": math_font_span_count,
                    "non_space_count": non_space_count,
                    "latex_command_count": latex_command_count,
                    "math_symbol_count": math_symbol_count,
                    "greek_count": greek_count,
                    "prose_word_count": prose_word_count,
                    "signal_score": structural_signal_count,
                    "spans": list(line.get("spans", [])),
                }
            )

    return records


def extract_formula_line_records(
    page_dict: dict,
    page_rect: fitz.Rect,
    excluded_rects: list[fitz.Rect],
    *,
    line_is_inside_any_table: Callable[..., bool],
) -> list[dict]:
    """提取带公式结构特征的文本行和独立公式编号。"""
    return [
        record
        for record in extract_formula_line_feature_records(
            page_dict=page_dict,
            page_rect=page_rect,
            excluded_rects=excluded_rects,
            line_is_inside_any_table=line_is_inside_any_table,
        )
        if record["is_formula_candidate"]
        or record["is_equation_number"]
    ]


def build_number_seeded_formula_clusters(
    line_features: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """使用独立公式编号反向收集同栏展示公式成员。"""
    equation_numbers = [
        record
        for record in line_features
        if record["is_equation_number"]
    ]
    clusters: list[dict] = []

    for number in equation_numbers:
        number_bbox = fitz.Rect(number["bbox"])
        number_center_y = (number_bbox.y0 + number_bbox.y1) / 2
        band_y0 = (
            number_center_y
            - FORMULA_NUMBER_SEED_VERTICAL_MARGIN_POINTS
        )
        band_y1 = (
            number_center_y
            + FORMULA_NUMBER_SEED_VERTICAL_MARGIN_POINTS
        )
        same_band = [
            record
            for record in line_features
            if not record["is_equation_number"]
            # A wide formula may cross the physical midpoint and be classified
            # as ``full``.  The equation number remains the local column anchor.
            and record["column"] in {number["column"], "full"}
            and record["bbox"].y1 >= band_y0
            and record["bbox"].y0 <= band_y1
            and record["bbox"].x0 < number_bbox.x0
            and record["bbox"].x1 <= number_bbox.x0 + 1.0
        ]
        if not same_band:
            continue

        prose_lines = [
            record
            for record in same_band
            if record["prose_word_count"]
            > FORMULA_NUMBER_SEED_MAX_PROSE_WORDS_PER_LINE
        ]
        if prose_lines:
            continue

        math_seed_lines = [
            record
            for record in same_band
            if record["math_font_character_count"] > 0
        ]
        if not math_seed_lines:
            continue

        has_relation_or_fraction_signal = any(
            any(symbol in record["text"] for symbol in ("=", "×", "÷", "/"))
            for record in same_band
        )
        if not has_relation_or_fraction_signal:
            continue

        member_bbox = union_rects(
            [record["bbox"] for record in same_band]
        )
        clusters.append(
            {
                "column": number["column"],
                "bbox": union_rects([member_bbox, number_bbox]),
                "lines": sorted(
                    same_band,
                    key=lambda record: (
                        record["bbox"].y0,
                        record["bbox"].x0,
                    ),
                ),
                "equation_numbers": [number],
                "has_number_anchor": True,
                "detection_reason": (
                    "equation_number_seed_with_math_font_band"
                ),
            }
        )

    return clusters


def try_build_formula_latex(cluster: dict) -> str | None:
    """尝试从一个公式簇生成保守的线性 LaTeX 表达式。"""
    lines = sorted(
        cluster.get("lines", []),
        key=lambda item: (item["bbox"].y0, item["bbox"].x0),
    )
    if not lines or len(lines) > 1:
        return None
    line = lines[0]
    text = str(line.get("text", "")).strip()
    if not text or line.get("prose_word_count", 0) > 2:
        return None
    if not line.get("math_font_character_count") and not line.get(
        "latex_command_count"
    ):
        return None
    if not any(symbol in text for symbol in ("=", "×", "÷", "/", "\\")):
        return None
    # PyMuPDF may expose a command as ``\\ f r a c``; collapse only spaces
    # between a backslash and command letters, retaining ordinary token gaps.
    latex = re.sub(r"\\\s+([A-Za-z])", r"\\\1", text)
    latex = latex.replace("×", r"\times ").replace("÷", r"\div ")
    latex = latex.replace("−", "-").replace("≤", r"\leq ")
    latex = latex.replace("≥", r"\geq ").replace("≈", r"\approx ")
    latex = latex.replace("≠", r"\ne ").replace("±", r"\pm ")
    latex = re.sub(r"\s+", " ", latex).strip()
    if len(latex) > 240 or not LATEX_SAFE_TEXT_PATTERN.fullmatch(latex):
        return None
    equation_numbers = cluster.get("equation_numbers", [])
    if equation_numbers:
        label = str(equation_numbers[0].get("text", "")).strip()
        match = re.fullmatch(r"\((\d+[a-z]?)\)", label, re.I)
        if match:
            latex = f"{latex} \\tag{{{match.group(1)}}}"
    return latex

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
    formula_output_mode: str = "image",
) -> tuple[list[dict], list[dict], list[fitz.Rect], float]:
    """检测显示公式、渲染图片，并返回需要从正文精确去重的行框。"""
    started = time.perf_counter()
    line_features = extract_formula_line_feature_records(
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
        for record in line_features
        if record["is_formula_candidate"]
    ]
    equation_numbers = [
        record
        for record in line_features
        if record["is_equation_number"]
    ]
    clusters = cluster_formula_candidates(formula_candidates)
    attach_equation_numbers_to_clusters(
        clusters=clusters,
        equation_numbers=equation_numbers,
    )
    seeded_clusters = build_number_seeded_formula_clusters(
        line_features=line_features,
        page_rect=page.rect,
    )
    # 编号种子路径只替换与其明显重叠的旧候选；页面上的无编号公式继续使用旧路径。
    for seeded in seeded_clusters:
        seeded_rect = fitz.Rect(seeded["bbox"])
        duplicate = False
        for existing in clusters:
            existing_rect = fitz.Rect(existing["bbox"])
            intersection = existing_rect & seeded_rect
            if intersection.is_empty:
                continue
            overlap = intersection.get_area() / max(
                1.0,
                min(existing_rect.get_area(), seeded_rect.get_area()),
            )
            if overlap >= 0.50:
                duplicate = True
                break
        if not duplicate:
            clusters.append(seeded)
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
        latex = try_build_formula_latex(cluster)
        latex_supported = latex is not None
        needs_image = (
            formula_output_mode == "image"
            or latex is None
            or formula_output_mode == "hybrid"
        )
        output_path = formula_dir / output_name if needs_image else None
        pixmap = None
        render_seconds = 0.0
        if needs_image:
            render_started = time.perf_counter()
            pixmap = page.get_pixmap(
                clip=output_bbox,
                dpi=formula_image_dpi,
                alpha=False,
                annots=True,
            )
            pixmap.save(output_path)
            render_seconds = time.perf_counter() - render_started
        image_markdown = ""
        if needs_image:
            relative_path = f"formula_images/{output_name}"
            image_markdown = "\n\n".join(
                [
                    f"![Page {page_number} Mathematical Formula "
                    f"{formula_number}]({relative_path})",
                    "> 数学公式（PDF页面渲染图）",
                ]
            )
        if formula_output_mode == "image" or latex is None:
            markdown_text = image_markdown
            emitted_mode = "rendered_formula"
        elif formula_output_mode == "latex":
            markdown_text = f"$$\n{latex}\n$$"
            emitted_mode = "latex"
        else:
            markdown_text = f"$$\n{latex}\n$$\n\n{image_markdown}"
            emitted_mode = "hybrid"
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
                "mode": emitted_mode,
                "output_path": output_path,
                "latex": latex,
                "latex_supported": latex_supported,
                "output_mode": emitted_mode,
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
                "output_path": str(output_path) if output_path else None,
                "latex": latex,
                "latex_supported": latex_supported,
                "output_mode": emitted_mode,
                "output_width": pixmap.width if pixmap else None,
                "output_height": pixmap.height if pixmap else None,
                "render_seconds": round(render_seconds, 6),
            }
        )

    return (
        records,
        markdown_elements,
        owned_text_rects,
        time.perf_counter() - started,
    )
