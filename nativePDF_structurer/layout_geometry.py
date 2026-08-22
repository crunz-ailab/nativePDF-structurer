"""提供文字、PDF目录、规则线和空间版面关系的共享算法。

模块提取原生文字行
元数据，解释PDF outline，分类轴向规则线，构建空间索引，并评估表格、
视觉区域、页眉页尾和阅读顺序共同使用的对象关系。

关键不变量:
    * 所有判定只使用PDF原生几何和元数据，不执行OCR。
    * 共享容差集中定义，并由多个检测阶段复用。
    * 空间索引只优化查询性能，不改变overlap语义。
    * 几何辅助函数只提供证据，不执行最终归属仲裁。

Provide shared text, outline, rule, and spatial layout algorithms.

Groups coordinate and rule-structure helpers used together by table, visual,
header/footer, heading, and reading-order stages. It extracts native text-line
metadata, interprets PDF outlines, classifies axis rules, builds spatial
indexes, and evaluates object relationships.

Key invariants:
    * All decisions use native PDF geometry and metadata; no OCR is performed.
    * Shared tolerances are defined once and reused across detector stages.
    * Spatial indexes accelerate queries without changing overlap semantics.
    * Helpers report geometric evidence and do not perform final ownership.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import statistics

import fitz

from .geometry import rect_area, rect_center, union_rects
from .heading_detection import (
    text_line_style_signature,
    weighted_median_font_size,
)
from .inventory import PageInventory
from .outline_headings import (
    normalize_heading_match_text,
    strip_heading_number_prefix,
)
from .rule_geometry import rule_element_ids
from .table_structure import unique_rule_positions
from .visual_assets import image_is_large_enough, image_rects_overlap


HEADING_MIN_PLAUSIBLE_FONT_SIZE_POINTS = 2.0
FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO = 0.15
FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS = 3.0
FALLBACK_TABLE_MIN_VERTICAL_SEGMENT_POINTS = 8.0
FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS = 3.0
FALLBACK_TABLE_MAX_VERTICAL_BREAK_POINTS = 18.0
FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO = 0.60
PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO = 0.80
PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS = 1.0
PRECLASSIFIED_RULE_BAND_MIN_BOUNDARY_VERTICAL_RULE_COUNT = 2
OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS = 1.5
OBJECT_PEER_OVERLAP_RATIO = 0.10
OBJECT_VECTOR_SPATIAL_GRID_CELL_POINTS = 64.0
OVERLAY_INTERSECTION_RATIO = 0.10


def quantize_line_direction(
    direction: tuple[float, float],
) -> int:
    """
    将 PyMuPDF line['dir'] 量化为：
    0 / 90 / -90 / 180。

    约定：
    0    = 不旋转
    -90  = 图片及 CSV 顺时针旋转 90°
    90   = 图片及 CSV 逆时针旋转 90°
    180  = 图片及 CSV 旋转 180°
    """
    dx, dy = direction
    angle = math.degrees(math.atan2(dy, dx))
    snapped = int(round(angle / 90.0) * 90)

    if snapped == 270:
        return -90

    if snapped == -270:
        return 90

    if snapped == -180:
        return 180

    return snapped



def extract_text_lines(page_dict: dict) -> list[dict]:
    """
    将 get_text('dict') 的嵌套 blocks / lines 打平成列表。

    每条记录包含：
    - bbox: 当前文本行的原页面坐标
    - text: 当前行所有 span 拼接后的内容
    - direction: 原始 line['dir']
    - angle: 量化后的文字方向
    - font_stats: 当前行各 span 的字号、字体、粗体标记和字符权重
    """
    text_lines: list[dict] = []

    for block_index, block in enumerate(page_dict["blocks"]):
        if block["type"] != 0:
            continue

        block_lines = block.get("lines", [])
        for line_index, line in enumerate(block_lines):
            spans = line.get("spans", [])
            text = "".join(
                span.get("text", "") for span in spans
            ).strip()

            if not text:
                continue

            direction = tuple(line["dir"])
            font_stats: list[dict] = []
            weighted_size_samples: list[tuple[float, int]] = []
            bold_character_count = 0
            visible_character_count = 0

            for span in spans:
                span_text = span.get("text", "")
                character_count = len(
                    "".join(span_text.split())
                )
                if character_count <= 0:
                    continue

                font_size = float(span.get("size", 0.0) or 0.0)
                font_name = str(span.get("font", "") or "")
                font_flags = int(span.get("flags", 0) or 0)
                is_bold = bool(
                    font_flags & fitz.TEXT_FONT_BOLD
                ) or any(
                    marker in font_name.lower()
                    for marker in ("bold", "black", "heavy")
                )
                font_stats.append(
                    {
                        "size": font_size,
                        "font": font_name,
                        "flags": font_flags,
                        "bold": is_bold,
                        "character_count": character_count,
                    }
                )
                if (
                    font_size
                    >= HEADING_MIN_PLAUSIBLE_FONT_SIZE_POINTS
                ):
                    weighted_size_samples.append(
                        (font_size, character_count)
                    )
                visible_character_count += character_count
                if is_bold:
                    bold_character_count += character_count

            effective_font_size = weighted_median_font_size(
                weighted_size_samples
            )

            text_lines.append(
                {
                    "bbox": fitz.Rect(line["bbox"]),
                    "text": text,
                    "direction": direction,
                    "angle": quantize_line_direction(direction),
                    "kind": "text",
                    "font_stats": font_stats,
                    "effective_font_size": effective_font_size,
                    "bold_character_ratio": (
                        bold_character_count
                        / visible_character_count
                        if visible_character_count
                        else 0.0
                    ),
                    "visible_character_count": visible_character_count,
                    "block_index": block_index,
                    "block_line_index": line_index,
                    "block_line_count": len(block_lines),
                    "block_bbox": fitz.Rect(
                        block.get("bbox", line["bbox"])
                    ),
                }
            )

    return text_lines



def build_pdf_outline_heading_index(
    document: fitz.Document,
    toc_entries: list[list] | None = None,
) -> dict[int, list[dict]]:
    """把PDF书签整理为按1-based页码索引的标题清单。"""
    outline_by_page: dict[int, list[dict]] = defaultdict(list)
    for entry_index, toc_entry in enumerate(
        toc_entries
        if toc_entries is not None
        else document.get_toc(simple=False)
    ):
        level, title, page_number, destination = toc_entry
        normalized_title = normalize_heading_match_text(title)
        if (
            page_number < 1
            or page_number > document.page_count
        ):
            continue
        page = document.load_page(page_number - 1)
        raw_target = destination.get("to")
        target_point = (
            fitz.Point(raw_target) * page.transformation_matrix
            if raw_target is not None
            else None
        )
        outline_by_page[page_number].append(
            {
                "entry_index": entry_index,
                "level": max(1, int(level)),
                "title": " ".join(str(title).split()),
                "normalized_title": normalized_title,
                "structural_title": strip_heading_number_prefix(
                    normalized_title
                ),
                "target_point": (
                    [float(target_point.x), float(target_point.y)]
                    if target_point is not None
                    else None
                ),
            }
        )
    return dict(outline_by_page)



def build_pdf_outline_page_name_index(
    page_count: int,
    toc_entries: list[list],
) -> dict[int, dict]:
    """把PDF书签目录映射为每页可显示的活动章节名称。"""
    entries_by_page: dict[int, list[dict]] = defaultdict(list)
    hierarchy_titles: list[str] = []

    for entry_index, toc_entry in enumerate(toc_entries):
        if len(toc_entry) < 3:
            continue
        level, title, target_page = toc_entry[:3]
        level = max(1, int(level))
        title = " ".join(str(title).split())
        target_page = int(target_page)
        if (
            not title
            or target_page < 1
            or target_page > page_count
        ):
            continue

        while len(hierarchy_titles) >= level:
            hierarchy_titles.pop()
        hierarchy_titles.append(title)
        entries_by_page[target_page].append(
            {
                "entry_index": entry_index,
                "level": level,
                "title": title,
                "outline_path": list(hierarchy_titles),
                "target_page": target_page,
            }
        )

    page_name_index: dict[int, dict] = {}
    active_entry: dict | None = None
    for page_number in range(1, page_count + 1):
        exact_entries = entries_by_page.get(page_number, [])
        if exact_entries:
            selected_entry = exact_entries[0]
            source = "exact_outline_target"
        elif active_entry is not None:
            selected_entry = active_entry
            source = "inherited_outline_section"
        else:
            continue

        page_name_index[page_number] = {
            **selected_entry,
            "page_number": page_number,
            "source": source,
            "exact_page_entry_count": len(exact_entries),
        }
        if exact_entries:
            active_entry = exact_entries[-1]

    return page_name_index



def format_markdown_pdf_page_title(
    page_number: int,
    outline_page_name: dict | None,
) -> str:
    """返回目录驱动的Markdown页面二级标题。"""
    if outline_page_name and outline_page_name.get("title"):
        return f"## {outline_page_name['title']}"
    return f"## Page {page_number}"



def line_matches_running_header_layout(
    line: dict,
    typography_profile: dict,
    page_rect: fitz.Rect,
) -> bool:
    """判断标题样式行是否命中重复顶部版式模板。"""
    if page_rect.height <= 0:
        return False
    line_rect = fitz.Rect(line["bbox"]).normalize()
    y_ratio = round(line_rect.y0 / page_rect.height, 2)
    style_key = text_line_style_signature(line)["key"]
    return any(
        template["style_key"] == style_key
        and abs(template["y_ratio"] - y_ratio) <= 0.01
        for template in typography_profile.get(
            "running_header_layout_templates",
            [],
        )
    )



def line_is_inside_any_table(
    line_rect: fitz.Rect,
    excluded_table_rects: list[fitz.Rect],
) -> bool:
    """
    用文本行中心点判断它是否属于某个已嵌入的大表格区域。

    属于大表格的文本不会进入页面正文 Markdown，
    避免 C/R、I、R、Rv 等密集单元格文字混入说明文本。
    """
    center = rect_center(line_rect)

    return any(
        table_rect.contains(center)
        for table_rect in excluded_table_rects
    )



def drawing_axis_rule(
    drawing: dict,
    page_rect: fitz.Rect,
) -> dict | None:
    """将单路径轴对齐细线归一为规则线记录。"""
    items = drawing.get("items", [])

    if len(items) != 1:
        return None

    item = items[0]
    item_type = item[0]

    if item_type == "re":
        rect = fitz.Rect(item[1]).normalize()
    elif item_type == "l":
        first_point, second_point = item[1], item[2]
        rect = fitz.Rect(
            min(first_point.x, second_point.x),
            min(first_point.y, second_point.y),
            max(first_point.x, second_point.x),
            max(first_point.y, second_point.y),
        )
    else:
        return None

    minimum_horizontal_length = (
        page_rect.width
        * FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO
    )
    horizontal = (
        rect.width >= minimum_horizontal_length
        and rect.height
        <= FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS
    )
    vertical = (
        rect.height >= FALLBACK_TABLE_MIN_VERTICAL_SEGMENT_POINTS
        and rect.width
        <= FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS
    )

    if horizontal == vertical:
        return None

    return {
        "orientation": "h" if horizontal else "v",
        "rect": rect,
        "position": (
            (rect.y0 + rect.y1) / 2
            if horizontal
            else (rect.x0 + rect.x1) / 2
        ),
        "seqno": drawing.get("seqno"),
    }



def drawing_has_visible_semantic_stroke(drawing: dict) -> bool:
    """判断描边是否不是白色页面上的不可见装饰线。"""
    color = drawing.get("color")
    if color is None:
        return True
    try:
        channels = [float(value) for value in color[:3]]
    except (TypeError, ValueError, IndexError):
        return True
    if len(channels) != 3:
        return True
    return not all(channel >= 0.95 for channel in channels)



def drawing_is_stroked_preclassified_horizontal_rule(
    drawing: dict,
    page_rect: fitz.Rect,
) -> bool:
    """判断drawing是否可以作为纯说明带的水平描边线。"""
    if drawing.get("fill") is not None:
        return False
    if drawing.get("type") not in {"s", "fs"}:
        return False
    if not drawing_has_visible_semantic_stroke(drawing):
        return False
    rule = drawing_axis_rule(drawing, page_rect)
    return rule is not None and rule["orientation"] == "h"



def merge_collinear_vertical_rules(
    vertical_rules: list[dict],
) -> list[dict]:
    """合并X位置一致且仅隔一个普通文本行的竖直规则线。"""
    tolerance = FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
    x_groups: list[list[dict]] = []

    for rule in sorted(vertical_rules, key=lambda item: item["position"]):
        matching_group = next(
            (
                group
                for group in x_groups
                if abs(
                    statistics.median(
                        item["position"] for item in group
                    )
                    - rule["position"]
                )
                <= tolerance
            ),
            None,
        )

        if matching_group is None:
            x_groups.append([rule])
        else:
            matching_group.append(rule)

    merged: list[dict] = []

    for group in x_groups:
        current: list[dict] = []

        for rule in sorted(group, key=lambda item: item["rect"].y0):
            if (
                current
                and rule["rect"].y0
                - max(item["rect"].y1 for item in current)
                > FALLBACK_TABLE_MAX_VERTICAL_BREAK_POINTS
            ):
                merged.append(
                    {
                        "position": statistics.median(
                            item["position"] for item in current
                        ),
                        "rect": union_rects(
                            [item["rect"] for item in current]
                        ),
                        "segment_count": len(current),
                    }
                )
                current = []

            current.append(rule)

        if current:
            merged.append(
                {
                    "position": statistics.median(
                        item["position"] for item in current
                    ),
                    "rect": union_rects(
                        [item["rect"] for item in current]
                    ),
                    "segment_count": len(current),
                }
            )

    return merged



def count_boundary_incident_vertical_rules(
    merged_vertical_rules: list[dict],
    boundary_y: float,
    band_x0: float,
    band_x1: float,
    tolerance: float,
) -> int:
    """统计在水平边界附近、具有不同X位置的竖向网格线数量。"""
    x_positions: list[float] = []
    for rule in merged_vertical_rules:
        rect = fitz.Rect(rule["rect"]).normalize()
        if rect.x1 < band_x0 - tolerance or rect.x0 > band_x1 + tolerance:
            continue
        endpoint_distance = min(
            abs(rect.y0 - boundary_y),
            abs(rect.y1 - boundary_y),
        )
        if endpoint_distance > tolerance:
            continue
        x_positions.append((rect.x0 + rect.x1) / 2.0)

    if not x_positions:
        return 0

    unique_x_positions: list[float] = []
    for x_position in sorted(x_positions):
        if not unique_x_positions:
            unique_x_positions.append(x_position)
            continue
        if (
            x_position - unique_x_positions[-1]
            > tolerance
        ):
            unique_x_positions.append(x_position)
    return len(unique_x_positions)



def horizontal_pair_has_boundary_grid_evidence(
    merged_vertical_rules: list[dict],
    top: float,
    bottom: float,
    band_x0: float,
    band_x1: float,
    tolerance: float,
) -> dict:
    """返回上下边界的局部竖线计数及是否命中网格反证。"""
    top_count = count_boundary_incident_vertical_rules(
        merged_vertical_rules=merged_vertical_rules,
        boundary_y=top,
        band_x0=band_x0,
        band_x1=band_x1,
        tolerance=tolerance,
    )
    bottom_count = count_boundary_incident_vertical_rules(
        merged_vertical_rules=merged_vertical_rules,
        boundary_y=bottom,
        band_x0=band_x0,
        band_x1=band_x1,
        tolerance=tolerance,
    )
    return {
        "top_boundary_vertical_count": top_count,
        "bottom_boundary_vertical_count": bottom_count,
        "has_boundary_grid_evidence": (
            top_count
            >= PRECLASSIFIED_RULE_BAND_MIN_BOUNDARY_VERTICAL_RULE_COUNT
            or bottom_count
            >= PRECLASSIFIED_RULE_BAND_MIN_BOUNDARY_VERTICAL_RULE_COUNT
        ),
    }



def group_horizontal_rules_by_span(
    horizontal_rules: list[dict],
) -> list[list[dict]]:
    """将左右边界近似相同的长水平线分组。"""
    tolerance = FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
    groups: list[list[dict]] = []

    for rule in sorted(
        horizontal_rules,
        key=lambda item: (
            item["rect"].x0,
            item["rect"].x1,
            item["position"],
        ),
    ):
        matching_group = next(
            (
                group
                for group in groups
                if abs(
                    statistics.median(
                        item["rect"].x0 for item in group
                    )
                    - rule["rect"].x0
                )
                <= tolerance
                and abs(
                    statistics.median(
                        item["rect"].x1 for item in group
                    )
                    - rule["rect"].x1
                )
                <= tolerance
            ),
            None,
        )

        if matching_group is None:
            groups.append([rule])
        else:
            matching_group.append(rule)

    return groups



def horizontal_pair_repeats_in_another_span(
    horizontal_rule_groups: list[list[dict]],
    current_group: list[dict],
    top: float,
    bottom: float,
    tolerance: float,
) -> bool:
    """判断相同Y边界的横线对是否在另一X跨度重复出现。"""
    current_x0 = statistics.median(
        rule["rect"].x0 for rule in current_group
    )
    current_x1 = statistics.median(
        rule["rect"].x1 for rule in current_group
    )
    current_width = max(0.0, current_x1 - current_x0)

    for other_group in horizontal_rule_groups:
        if other_group is current_group:
            continue

        other_x0 = statistics.median(
            rule["rect"].x0 for rule in other_group
        )
        other_x1 = statistics.median(
            rule["rect"].x1 for rule in other_group
        )
        other_width = max(0.0, other_x1 - other_x0)
        shorter_width = min(current_width, other_width)
        span_overlap_width = max(
            0.0,
            min(current_x1, other_x1)
            - max(current_x0, other_x0),
        )
        if (
            shorter_width > 0
            and span_overlap_width / shorter_width
            >= PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
        ):
            continue

        other_positions = unique_rule_positions(other_group)
        has_matching_top = any(
            abs(position - top) <= tolerance
            for position in other_positions
        )
        has_matching_bottom = any(
            abs(position - bottom) <= tolerance
            for position in other_positions
        )
        if has_matching_top and has_matching_bottom:
            return True

    return False



def vertical_rule_connects_horizontal_pair(
    merged_vertical_rules: list[dict],
    band_rect: fitz.Rect,
    top: float,
    bottom: float,
) -> bool:
    """判断竖线是否连接或紧贴双横线候选。"""
    band_height = bottom - top
    if band_height <= 0:
        return False

    connection_tolerance = (
        PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
    )
    structural_x0 = band_rect.x0 - connection_tolerance
    structural_x1 = band_rect.x1 + connection_tolerance

    for vertical_rule in merged_vertical_rules:
        vertical_x = vertical_rule["position"]
        if not structural_x0 <= vertical_x <= structural_x1:
            continue

        vertical_rect = vertical_rule["rect"]
        covered_height = max(
            0.0,
            min(vertical_rect.y1, bottom)
            - max(vertical_rect.y0, top),
        )
        if (
            covered_height / band_height
            >= FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO
        ):
            return True

    return False



def finalize_logical_horizontal_rule_audit(
    records: list[dict],
    page_number: int,
    preclassified_rule_drawing_ids: set[str],
    table_skeleton_drawing_ids: set[str],
    detector_claims: list[dict],
    resolution: dict,
) -> None:
    """回填逻辑横线源drawing在各检测阶段的实际去向。"""
    claims_by_id = {
        claim["claim_id"]: claim for claim in detector_claims
    }
    owner_by_element_id = resolution.get(
        "owner_by_element_id", {}
    )

    claim_element_ids_by_claim_id = {
        claim["claim_id"]: set(
            claim.get("input_element_ids", [])
        ).union(claim.get("claimed_element_ids", []))
        for claim in detector_claims
    }

    for record in records:
        if record.get("page_number") != page_number:
            continue

        source_ids = set(record.get("source_drawing_ids", []))
        preclassified_ids = sorted(
            source_ids & preclassified_rule_drawing_ids
        )
        table_skeleton_ids = sorted(
            source_ids & table_skeleton_drawing_ids
        )
        intersecting_claims = [
            claim
            for claim in detector_claims
            if not source_ids.isdisjoint(
                claim_element_ids_by_claim_id[
                    claim["claim_id"]
                ]
            )
        ]
        owner_claim_ids = sorted(
            {
                owner_by_element_id[element_id]
                for element_id in source_ids
                if element_id in owner_by_element_id
            }
        )

        record["preclassified_source_drawing_ids"] = (
            preclassified_ids
        )
        record["table_skeleton_source_drawing_ids"] = (
            table_skeleton_ids
        )
        record["intersecting_claim_ids"] = sorted(
            claim["claim_id"] for claim in intersecting_claims
        )
        record["intersecting_claim_detectors"] = sorted(
            {
                claim["detector"]
                for claim in intersecting_claims
            }
        )
        record["final_owner_claim_ids"] = owner_claim_ids
        record["final_owner_detectors"] = sorted(
            {
                claims_by_id[claim_id]["detector"]
                for claim_id in owner_claim_ids
                if claim_id in claims_by_id
            }
        )

        if preclassified_ids:
            record["table_source_status"] = (
                "excluded_from_table_detector_by_preclassified_rule_band"
            )
        else:
            record["table_source_status"] = (
                "raw_source_drawings_available_to_table_detector"
            )

        if not record.get("passed_logical_width_filter", False):
            record["visual_logical_status"] = (
                "logical_rule_not_created_below_width_threshold"
            )
        elif preclassified_ids:
            record["visual_logical_status"] = (
                "excluded_from_visual_by_preclassified_rule_band"
            )
        elif table_skeleton_ids:
            record["visual_logical_status"] = (
                "excluded_from_visual_by_table_skeleton"
            )
        else:
            record["visual_logical_status"] = (
                "available_to_visual_relation_and_framework_detection"
            )



def find_rule_attached_heading(
    position_rules: list[dict],
    available_text_lines: list[dict],
    tolerance: float,
) -> dict | None:
    """寻找与水平规则条端点相接并处于同一Y带的标题文字。"""
    if not position_rules:
        return None

    source_rule_bbox = union_rects(
        [rule["rect"] for rule in position_rules]
    ).normalize()
    source_center_y = (
        source_rule_bbox.y0 + source_rule_bbox.y1
    ) / 2
    candidates: list[dict] = []

    for line in available_text_lines:
        if (
            line.get("kind", "text") != "text"
            or line.get("angle", 0) != 0
        ):
            continue

        text_rect = fitz.Rect(line["bbox"]).normalize()
        vertical_overlap = max(
            0.0,
            min(source_rule_bbox.y1, text_rect.y1)
            - max(source_rule_bbox.y0, text_rect.y0),
        )
        vertically_aligned = (
            vertical_overlap > 0
            or text_rect.y0 - tolerance
            <= source_center_y
            <= text_rect.y1 + tolerance
        )
        if not vertically_aligned:
            continue

        left_edge_delta = (
            source_rule_bbox.x0 - text_rect.x1
        )
        right_edge_delta = (
            text_rect.x0 - source_rule_bbox.x1
        )
        side: str | None = None
        edge_delta = math.inf

        if abs(left_edge_delta) <= tolerance:
            side = "text_left_of_rule"
            edge_delta = abs(left_edge_delta)
        if (
            abs(right_edge_delta) <= tolerance
            and abs(right_edge_delta) < edge_delta
        ):
            side = "text_right_of_rule"
            edge_delta = abs(right_edge_delta)
        if side is None:
            continue

        minimum_height = min(
            source_rule_bbox.height,
            text_rect.height,
        )
        vertical_overlap_ratio = (
            vertical_overlap / minimum_height
            if minimum_height > 0
            else 1.0
        )
        candidates.append(
            {
                "attached_heading_bbox": [
                    round(value, 3) for value in text_rect
                ],
                "attached_heading_text": line["text"],
                "attachment_side": side,
                "edge_delta_points": edge_delta,
                "vertical_overlap_ratio": (
                    vertical_overlap_ratio
                ),
                "source_rule_bbox": [
                    round(value, 3)
                    for value in source_rule_bbox
                ],
            }
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: (
            item["edge_delta_points"],
            -item["vertical_overlap_ratio"],
            item["attached_heading_bbox"][0],
        ),
    )



def detect_text_separator_markdown_elements(
    page_rect: fitz.Rect,
    drawings: list[dict],
    text_lines: list[dict],
    source_images: list[dict],
    page_inventory: PageInventory,
    claimed_drawing_element_ids: set[str],
    occupied_non_text_rects: list[fitz.Rect],
    excluded_text_rects: list[fitz.Rect],
    logical_horizontal_rules: list[dict],
) -> tuple[list[dict], list[dict]]:
    """检测只分隔原生文字的重复水平规则线。"""
    horizontal_rules: list[dict] = []

    for rule in logical_horizontal_rules:
        element_ids = rule_element_ids(rule)
        if element_ids & claimed_drawing_element_ids:
            continue
        horizontal_rules.append(
            {
                **rule,
                "element_ids": sorted(element_ids),
            }
        )
    if len(horizontal_rules) < 2:
        return [], []

    vertical_rules = [
        rule
        for drawing in drawings
        if (rule := drawing_axis_rule(drawing, page_rect))
        is not None
        and rule["orientation"] == "v"
    ]
    merged_vertical_rules = merge_collinear_vertical_rules(
        vertical_rules
    )
    horizontal_rule_groups = group_horizontal_rules_by_span(
        horizontal_rules
    )

    available_text_lines = [
        line
        for line in text_lines
        if line.get("kind", "text") == "text"
        and line.get("angle", 0) == 0
        and not line_is_inside_any_table(
            line_rect=fitz.Rect(line["bbox"]),
            excluded_table_rects=excluded_text_rects,
        )
    ]
    meaningful_source_image_rects = [
        fitz.Rect(image["bbox"])
        for image in source_images
        if image_is_large_enough(
            image_rect=fitz.Rect(image["bbox"]),
            page_rect=page_rect,
        )
    ]

    separator_by_key: dict[tuple[float, float, float], dict] = {}
    records_by_key: dict[tuple[float, float, float], dict] = {}
    tolerance = FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS

    for group in horizontal_rule_groups:
        positions = unique_rule_positions(group)
        if len(positions) < 2:
            continue

        frame_x0 = min(rule["rect"].x0 for rule in group)
        frame_x1 = max(rule["rect"].x1 for rule in group)
        group_element_ids = set().union(
            *(rule_element_ids(rule) for rule in group)
        )

        for top, bottom in zip(positions, positions[1:]):
            if horizontal_pair_repeats_in_another_span(
                horizontal_rule_groups=horizontal_rule_groups,
                current_group=group,
                top=top,
                bottom=bottom,
                tolerance=tolerance,
            ):
                continue

            inner_top = top + tolerance
            inner_bottom = bottom - tolerance
            if inner_bottom <= inner_top:
                continue

            inner_band = fitz.Rect(
                frame_x0,
                inner_top,
                frame_x1,
                inner_bottom,
            )
            if vertical_rule_connects_horizontal_pair(
                merged_vertical_rules=merged_vertical_rules,
                band_rect=fitz.Rect(
                    frame_x0,
                    top,
                    frame_x1,
                    bottom,
                ),
                top=top,
                bottom=bottom,
            ):
                continue
            band_text_lines = [
                line
                for line in available_text_lines
                if inner_band.contains(
                    rect_center(fitz.Rect(line["bbox"]))
                )
            ]
            if not band_text_lines:
                continue

            if any(
                object_area_overlap(
                    inner_band,
                    occupied_rect,
                    threshold=OBJECT_PEER_OVERLAP_RATIO,
                )
                for occupied_rect in occupied_non_text_rects
            ):
                continue

            if any(
                object_area_overlap(
                    inner_band,
                    image_rect,
                    threshold=OBJECT_PEER_OVERLAP_RATIO,
                )
                for image_rect in meaningful_source_image_rects
            ):
                continue

            blocking_drawing_ids: list[str] = []
            for drawing in drawings:
                drawing_element_id = page_inventory.id_for_payload(
                    drawing,
                    "drawing",
                )
                if (
                    drawing_element_id in group_element_ids
                    or drawing_element_id
                    in claimed_drawing_element_ids
                ):
                    continue

                drawing_rect = fitz.Rect(drawing["rect"])
                drawing_center = rect_center(drawing_rect)
                near_boundary = (
                    min(
                        abs(drawing_center.y - top),
                        abs(drawing_center.y - bottom),
                    )
                    <= tolerance
                    and drawing_rect.height <= tolerance * 2
                )
                if near_boundary:
                    continue

                drawing_has_inner_presence = (
                    inner_band.contains(drawing_center)
                    or object_area_overlap(
                        inner_band,
                        drawing_rect,
                        threshold=OBJECT_PEER_OVERLAP_RATIO,
                    )
                )
                if drawing_has_inner_presence:
                    blocking_drawing_ids.append(
                        drawing_element_id
                    )

            if blocking_drawing_ids:
                continue

            for position in (top, bottom):
                position_rules = [
                    rule
                    for rule in group
                    if abs(rule["position"] - position)
                    <= tolerance
                ]
                attachment_position_rules: list[dict] = []
                for rule in position_rules:
                    source_rects = [
                        page_inventory.bbox_of(element_id)
                        for element_id in rule_element_ids(rule)
                        if element_id in page_inventory.entries
                        and page_inventory.entry(element_id)[
                            "kind"
                        ]
                        == "drawing"
                    ]
                    attachment_position_rules.append(
                        {
                            **rule,
                            "rect": (
                                union_rects(source_rects)
                                if source_rects
                                else rule["rect"]
                            ),
                        }
                    )
                attached_heading = find_rule_attached_heading(
                    position_rules=attachment_position_rules,
                    available_text_lines=available_text_lines,
                    tolerance=tolerance,
                )
                separator_rect = fitz.Rect(
                    min(rule["rect"].x0 for rule in position_rules),
                    position - 0.5,
                    max(rule["rect"].x1 for rule in position_rules),
                    position + 0.5,
                )
                key = (
                    round(separator_rect.x0, 3),
                    round(separator_rect.x1, 3),
                    round(position, 3),
                )
                separator_by_key[key] = {
                    "bbox": separator_rect,
                    "render_bbox": separator_rect,
                    "sort_bbox": separator_rect,
                    "text": "---",
                    "direction": (1.0, 0.0),
                    "angle": 0,
                    "kind": "text_separator",
                    "markdown": "---",
                    **(
                        {
                            **attached_heading,
                            "sort_before_heading": True,
                        }
                        if attached_heading is not None
                        else {}
                    ),
                }
                records_by_key[key] = {
                    "bbox": [
                        round(value, 3)
                        for value in separator_rect
                    ],
                    "source_drawing_element_ids": sorted(
                        set().union(
                            *(
                                rule_element_ids(rule)
                                for rule in position_rules
                            )
                        )
                    ),
                    "reason": "pure_text_between_repeated_horizontal_rules",
                    **(
                        {
                            **attached_heading,
                            "sort_before_heading": True,
                        }
                        if attached_heading is not None
                        else {}
                    ),
                }

    ordered_keys = sorted(
        separator_by_key,
        key=lambda key: (key[2], key[0], key[1]),
    )
    return (
        [separator_by_key[key] for key in ordered_keys],
        [records_by_key[key] for key in ordered_keys],
    )



def drawing_is_plain_rule_band_background(
    drawing: dict,
) -> bool:
    """判断drawing是否只是由一个或多个矩形组成的纯色背景。"""
    items = drawing.get("items", [])
    drawing_rect = fitz.Rect(drawing.get("rect", fitz.Rect()))
    return bool(
        drawing.get("fill") is not None
        and items
        and drawing_rect.width > 0
        and drawing_rect.height > 0
        and all(item and item[0] == "re" for item in items)
    )



def horizontal_projection_overlap_ratio(
    first_rect: fitz.Rect,
    second_rect: fitz.Rect,
) -> float:
    """返回X投影交集相对较窄对象宽度的比例。"""
    first = fitz.Rect(first_rect).normalize()
    second = fitz.Rect(second_rect).normalize()
    shorter_width = min(first.width, second.width)
    if shorter_width <= 0:
        return 0.0
    overlap_width = max(
        0.0,
        min(first.x1, second.x1) - max(first.x0, second.x0),
    )
    return overlap_width / shorter_width



def attach_empty_adjacent_rule_bands_to_tables(
    page_rect: fitz.Rect,
    bands: list[dict],
    table_candidates: list[dict],
    table_claims_by_id: dict[str, dict],
    page_inventory: PageInventory,
) -> dict:
    """把无独立内容、紧贴表格边缘的规则带底色归入表格截图。"""
    table_rects = [
        fitz.Rect(candidate["bbox"]).normalize()
        for candidate in table_candidates
    ]
    attachments: list[dict] = []
    inspected_band_count = 0

    for band_index, band in enumerate(bands):
        band["embedded_in_table"] = False
        band["adjacent_table_attachment_audit"] = {
            "eligible": False,
            "reason": "not_a_visual_rule_band",
        }
        if band.get("mode") != "visual_rule_band":
            continue
        inspected_band_count += 1

        if band.get("image_element_ids"):
            band["adjacent_table_attachment_audit"] = {
                "eligible": False,
                "reason": "contains_source_image",
            }
            continue

        content_drawing_ids = list(
            band.get("content_drawing_ids", [])
        )
        background_drawing_ids: list[str] = []
        complex_drawing_ids: list[str] = []
        for element_id in content_drawing_ids:
            if element_id not in page_inventory.entries:
                complex_drawing_ids.append(element_id)
                continue
            drawing = page_inventory.payload_of(element_id)
            if drawing_is_plain_rule_band_background(drawing):
                background_drawing_ids.append(element_id)
            else:
                complex_drawing_ids.append(element_id)

        if complex_drawing_ids:
            band["adjacent_table_attachment_audit"] = {
                "eligible": False,
                "reason": "contains_complex_vector_content",
                "complex_drawing_ids": sorted(complex_drawing_ids),
            }
            continue
        if not background_drawing_ids:
            band["adjacent_table_attachment_audit"] = {
                "eligible": False,
                "reason": "no_plain_background_to_attach",
            }
            continue

        residual_text_lines = [
            line
            for line in band.get("text_lines", [])
            if not any(
                table_rect.contains(
                    rect_center(fitz.Rect(line["bbox"]))
                )
                for table_rect in table_rects
            )
        ]
        if residual_text_lines:
            band["adjacent_table_attachment_audit"] = {
                "eligible": False,
                "reason": "contains_text_outside_confirmed_tables",
                "residual_text_count": len(residual_text_lines),
            }
            continue

        best_match: dict | None = None
        for candidate in table_candidates:
            if not candidate.get("should_render_table_image", False):
                continue
            table_rect = fitz.Rect(candidate["bbox"]).normalize()
            for element_id in background_drawing_ids:
                drawing_rect = fitz.Rect(
                    page_inventory.entry(element_id)["bbox"]
                ).normalize()
                x_overlap_ratio = horizontal_projection_overlap_ratio(
                    drawing_rect,
                    table_rect,
                )
                if (
                    x_overlap_ratio
                    < PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
                ):
                    continue

                edge_name: str | None = None
                edge_gap = math.inf
                if (
                    drawing_rect.y0 < table_rect.y0
                    and drawing_rect.y1
                    >= table_rect.y0
                    - OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
                    and drawing_rect.y1
                    <= table_rect.y0
                    + OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
                ):
                    edge_name = "top"
                    edge_gap = abs(table_rect.y0 - drawing_rect.y1)
                elif (
                    drawing_rect.y1 > table_rect.y1
                    and drawing_rect.y0
                    >= table_rect.y1
                    - OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
                    and drawing_rect.y0
                    <= table_rect.y1
                    + OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
                ):
                    edge_name = "bottom"
                    edge_gap = abs(drawing_rect.y0 - table_rect.y1)
                if edge_name is None:
                    continue

                match = {
                    "candidate": candidate,
                    "element_id": element_id,
                    "drawing_rect": drawing_rect,
                    "edge": edge_name,
                    "edge_gap": edge_gap,
                    "x_overlap_ratio": x_overlap_ratio,
                }
                if best_match is None or (
                    match["edge_gap"],
                    -match["x_overlap_ratio"],
                ) < (
                    best_match["edge_gap"],
                    -best_match["x_overlap_ratio"],
                ):
                    best_match = match

        if best_match is None:
            band["adjacent_table_attachment_audit"] = {
                "eligible": False,
                "reason": "no_table_edge_with_sufficient_x_overlap",
            }
            continue

        candidate = best_match["candidate"]
        table_number = candidate["table_number"]
        table_claim_id = (
            f"page#{page_inventory.page_number}:table#{table_number}"
        )
        table_claim = table_claims_by_id.get(table_claim_id)
        if table_claim is None:
            band["adjacent_table_attachment_audit"] = {
                "eligible": False,
                "reason": "matching_table_claim_missing",
            }
            continue

        original_render_bbox = fitz.Rect(
            candidate.get("render_bbox", candidate["bbox"])
        ).normalize()
        expanded_render_bbox = union_rects(
            [original_render_bbox, best_match["drawing_rect"]]
        ) & fitz.Rect(page_rect)
        attached_element_id = best_match["element_id"]
        candidate["render_bbox"] = expanded_render_bbox
        candidate.setdefault(
            "adjacent_rule_band_attachments", []
        ).append(
            {
                "band_index": band_index,
                "band_bbox": [
                    round(value, 3)
                    for value in fitz.Rect(band["bbox"])
                ],
                "attached_drawing_element_ids": [
                    attached_element_id
                ],
                "edge": best_match["edge"],
                "edge_gap_points": round(
                    best_match["edge_gap"], 6
                ),
                "x_overlap_ratio": round(
                    best_match["x_overlap_ratio"], 6
                ),
            }
        )

        for key in (
            "input_element_ids",
            "claimed_element_ids",
            "feature_element_ids",
        ):
            table_claim[key] = sorted(
                set(table_claim.get(key, [])).union(
                    {attached_element_id}
                )
            )
        table_claim["input_count"] = len(
            table_claim["input_element_ids"]
        )
        table_claim["claimed_count"] = len(
            table_claim["claimed_element_ids"]
        )
        table_claim["feature_count"] = len(
            table_claim["feature_element_ids"]
        )
        table_claim["bbox"] = [
            round(value, 3) for value in expanded_render_bbox
        ]
        table_claim["metadata"].setdefault(
            "adjacent_rule_band_attachments", []
        ).append(candidate["adjacent_rule_band_attachments"][-1])
        table_claim["reason_codes"] = sorted(
            set(table_claim.get("reason_codes", [])).union(
                {"adjacent_empty_rule_band_attached_to_table"}
            )
        )

        band["embedded_in_table"] = True
        band["parent_table_claim_id"] = table_claim_id
        band["adjacent_table_attachment_audit"] = {
            "eligible": True,
            "reason": "empty_background_touches_table_edge",
            "parent_table_claim_id": table_claim_id,
            "attached_drawing_element_ids": [attached_element_id],
            "edge": best_match["edge"],
            "edge_gap_points": round(best_match["edge_gap"], 6),
            "x_overlap_ratio": round(
                best_match["x_overlap_ratio"], 6
            ),
            "expanded_render_bbox": [
                round(value, 3)
                for value in expanded_render_bbox
            ],
        }
        attachments.append(
            {
                "band_index": band_index,
                **band["adjacent_table_attachment_audit"],
            }
        )

    return {
        "inspected_visual_band_count": inspected_band_count,
        "attachment_count": len(attachments),
        "attachments": attachments,
    }



def meaningful_overlap(
    target_rect: fitz.Rect,
    overlay_rect: fitz.Rect,
    threshold: float = OVERLAY_INTERSECTION_RATIO,
) -> bool:
    """按叠加元素自身面积判断它是否实质落入目标区域。"""
    overlay_area = rect_area(overlay_rect)

    if overlay_area <= 0:
        return False

    intersection = target_rect & overlay_rect

    if intersection.is_empty:
        return False

    return rect_area(intersection) / overlay_area >= threshold



def object_area_overlap(
    first_rect: fitz.Rect,
    second_rect: fitz.Rect,
    threshold: float = OBJECT_PEER_OVERLAP_RATIO,
) -> bool:
    """判断两个对象是否有达到门限的实质面积重叠。"""
    first_rect = fitz.Rect(first_rect).normalize()
    second_rect = fitz.Rect(second_rect).normalize()
    first_area = rect_area(first_rect)
    second_area = rect_area(second_rect)
    if first_area <= 0 or second_area <= 0:
        return False
    intersection_area = rect_area(first_rect & second_rect)
    return intersection_area / min(first_area, second_area) >= threshold



def rect_is_contained_with_tolerance(
    inner_rect: fitz.Rect,
    outer_rect: fitz.Rect,
    tolerance: float = OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS,
) -> bool:
    """判断inner_rect是否完全位于outer_rect的容差扩张范围内。"""
    inner_rect = fitz.Rect(inner_rect).normalize()
    outer_rect = fitz.Rect(outer_rect).normalize()
    return (
        inner_rect.x0 >= outer_rect.x0 - tolerance
        and inner_rect.y0 >= outer_rect.y0 - tolerance
        and inner_rect.x1 <= outer_rect.x1 + tolerance
        and inner_rect.y1 <= outer_rect.y1 + tolerance
    )



def drawing_bbox_overlap_ratio(
    object_rect: fitz.Rect,
    drawing_rect: fitz.Rect,
    tolerance: float = OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS,
) -> float:
    """返回drawing bbox落入对象bbox的比例，零面积线段改用轴向投影。"""
    object_rect = fitz.Rect(object_rect).normalize()
    drawing_rect = fitz.Rect(drawing_rect).normalize()
    drawing_area = rect_area(drawing_rect)

    if drawing_area > 0:
        return rect_area(object_rect & drawing_rect) / drawing_area

    if drawing_rect.width > 0:
        if not (
            object_rect.y0 - tolerance
            <= drawing_rect.y0
            <= object_rect.y1 + tolerance
        ):
            return 0.0
        overlap_width = max(
            0.0,
            min(object_rect.x1, drawing_rect.x1)
            - max(object_rect.x0, drawing_rect.x0),
        )
        return overlap_width / drawing_rect.width

    if drawing_rect.height > 0:
        if not (
            object_rect.x0 - tolerance
            <= drawing_rect.x0
            <= object_rect.x1 + tolerance
        ):
            return 0.0
        overlap_height = max(
            0.0,
            min(object_rect.y1, drawing_rect.y1)
            - max(object_rect.y0, drawing_rect.y0),
        )
        return overlap_height / drawing_rect.height

    return float(
        object_rect.x0 - tolerance
        <= drawing_rect.x0
        <= object_rect.x1 + tolerance
        and object_rect.y0 - tolerance
        <= drawing_rect.y0
        <= object_rect.y1 + tolerance
    )



def build_drawing_bbox_spatial_index(
    drawings: list[dict],
) -> dict:
    """建立只用于候选缩减的drawing bbox均匀网格。"""
    cell_size = OBJECT_VECTOR_SPATIAL_GRID_CELL_POINTS
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    drawing_rects: list[fitz.Rect] = []

    for drawing_index, drawing in enumerate(drawings):
        drawing_rect = fitz.Rect(
            drawing.get("rect", ())
        ).normalize()
        drawing_rects.append(drawing_rect)
        x_start = math.floor(drawing_rect.x0 / cell_size)
        x_end = math.floor(drawing_rect.x1 / cell_size)
        y_start = math.floor(drawing_rect.y0 / cell_size)
        y_end = math.floor(drawing_rect.y1 / cell_size)
        for cell_x in range(x_start, x_end + 1):
            for cell_y in range(y_start, y_end + 1):
                cells[(cell_x, cell_y)].append(drawing_index)

    return {
        "cell_size": cell_size,
        "cells": dict(cells),
        "drawing_rects": drawing_rects,
    }



def query_drawing_bbox_spatial_index(
    spatial_index: dict,
    object_rect: fitz.Rect,
    tolerance: float = OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS,
) -> list[int]:
    """返回可能与对象区域发生原判据关系的drawing索引。"""
    object_rect = fitz.Rect(object_rect).normalize()
    cell_size = spatial_index["cell_size"]
    x_start = math.floor((object_rect.x0 - tolerance) / cell_size)
    x_end = math.floor((object_rect.x1 + tolerance) / cell_size)
    y_start = math.floor((object_rect.y0 - tolerance) / cell_size)
    y_end = math.floor((object_rect.y1 + tolerance) / cell_size)
    candidate_indices: set[int] = set()
    cells = spatial_index["cells"]
    for cell_x in range(x_start, x_end + 1):
        for cell_y in range(y_start, y_end + 1):
            candidate_indices.update(cells.get((cell_x, cell_y), []))

    return sorted(candidate_indices)



def analyze_object_vector_overlap(
    object_rect: fitz.Rect,
    drawings: list[dict],
    drawing_spatial_index: dict | None = None,
) -> dict:
    """统计对象内部矢量和跨边界矢量。"""
    object_rect = fitz.Rect(object_rect).normalize()
    internal_count = 0
    external_count = 0
    external_bboxes: list[list[float]] = []

    drawing_indices = (
        range(len(drawings))
        if drawing_spatial_index is None
        else query_drawing_bbox_spatial_index(
            spatial_index=drawing_spatial_index,
            object_rect=object_rect,
        )
    )
    for drawing_index in drawing_indices:
        drawing = drawings[drawing_index]
        drawing_rect = (
            fitz.Rect(drawing.get("rect", ())).normalize()
            if drawing_spatial_index is None
            else drawing_spatial_index["drawing_rects"][drawing_index]
        )
        drawing_overlap_ratio = drawing_bbox_overlap_ratio(
            object_rect=object_rect,
            drawing_rect=drawing_rect,
        )
        if drawing_overlap_ratio < OBJECT_PEER_OVERLAP_RATIO:
            continue

        if rect_is_contained_with_tolerance(
            inner_rect=drawing_rect,
            outer_rect=object_rect,
        ):
            internal_count += 1
            continue

        external_count += 1
        if len(external_bboxes) < 5:
            external_bboxes.append(
                [round(value, 3) for value in drawing_rect]
            )

    return {
        "has_external_vector_connection": external_count > 0,
        "internal_drawing_count": internal_count,
        "external_drawing_count": external_count,
        "external_drawing_bboxes": external_bboxes,
    }



def build_source_image_overlap_groups(
    source_images: list[dict],
) -> list[dict]:
    """构建相互重叠的原始图片组。"""
    groups: list[set[int]] = []
    for image_index, image in enumerate(source_images):
        matching_groups = [
            group_index
            for group_index, group in enumerate(groups)
            if any(
                image_rects_overlap(
                    image["bbox"],
                    source_images[member_index]["bbox"],
                )
                for member_index in group
            )
        ]
        if not matching_groups:
            groups.append({image_index})
            continue

        merged_group = {image_index}
        for group_index in reversed(matching_groups):
            merged_group.update(groups.pop(group_index))
        groups.append(merged_group)

    result: list[dict] = []
    for group_index, members in enumerate(groups, start=1):
        group_images = [source_images[index] for index in sorted(members)]
        result.append(
            {
                "group_index": group_index,
                "source_image_indices": [
                    image["source_index"] for image in group_images
                ],
                "member_indices": sorted(members),
                "bbox": union_rects(
                    [fitz.Rect(image["bbox"]) for image in group_images]
                ),
            }
        )
    return result



def prepare_precluster_object_exclusions(
    table_candidates: list[dict],
    source_images: list[dict],
    drawings: list[dict],
    page_rect: fitz.Rect,
) -> dict:
    """为视觉聚类准备独立表格/图片的前置排除区域。"""
    drawing_spatial_index = build_drawing_bbox_spatial_index(
        drawings=drawings,
    )
    table_records: list[dict] = []
    for candidate in table_candidates:
        table_rect = fitz.Rect(candidate["bbox"])
        source_cluster_rect = fitz.Rect(
            candidate.get("source_cluster_bbox", page_rect)
        )
        vector_overlap = analyze_object_vector_overlap(
            object_rect=table_rect,
            drawings=drawings,
            drawing_spatial_index=drawing_spatial_index,
        )
        record = {
            "table_number": candidate["table_number"],
            "bbox": [round(value, 3) for value in table_rect],
            "source_cluster_bbox": [
                round(value, 3) for value in source_cluster_rect
            ],
            "contained_by_source_cluster": (
                rect_is_contained_with_tolerance(
                    inner_rect=table_rect,
                    outer_rect=source_cluster_rect,
                )
            ),
            "will_emit_if_unclaimed": candidate[
                "will_emit_if_unclaimed"
            ],
            "vector_overlap": vector_overlap,
            "overlaps_table_numbers": [],
            "overlaps_image_groups": [],
            "precluster_protected": False,
            "precluster_reason": None,
        }
        table_records.append(record)
        candidate["precluster_vector_overlap"] = vector_overlap

    image_groups = build_source_image_overlap_groups(source_images)
    image_records: list[dict] = []
    for image_group in image_groups:
        group_rect = fitz.Rect(image_group["bbox"])
        vector_overlap = analyze_object_vector_overlap(
            object_rect=group_rect,
            drawings=drawings,
            drawing_spatial_index=drawing_spatial_index,
        )
        image_records.append(
            {
                **image_group,
                "bbox": [round(value, 3) for value in group_rect],
                "vector_overlap": vector_overlap,
                "overlaps_table_numbers": [],
                "precluster_protected": False,
                "precluster_reason": None,
            }
        )

    for first_index, first_record in enumerate(table_records):
        first_rect = fitz.Rect(first_record["bbox"])
        for second_record in table_records[first_index + 1 :]:
            second_rect = fitz.Rect(second_record["bbox"])
            if not object_area_overlap(first_rect, second_rect):
                continue
            first_record["overlaps_table_numbers"].append(
                second_record["table_number"]
            )
            second_record["overlaps_table_numbers"].append(
                first_record["table_number"]
            )

    for table_record in table_records:
        table_rect = fitz.Rect(table_record["bbox"])
        for image_record in image_records:
            if not object_area_overlap(
                table_rect,
                fitz.Rect(image_record["bbox"]),
            ):
                continue
            table_record["overlaps_image_groups"].append(
                image_record["group_index"]
            )
            image_record["overlaps_table_numbers"].append(
                table_record["table_number"]
            )

    protected_rects: list[fitz.Rect] = []
    protected_table_numbers: set[int] = set()
    for table_record in table_records:
        vector_overlap = table_record["vector_overlap"]
        if not table_record["will_emit_if_unclaimed"]:
            table_record["precluster_reason"] = (
                "candidate_not_emitted_if_unclaimed"
            )
            continue
        if not table_record["contained_by_source_cluster"]:
            table_record["precluster_reason"] = (
                "extends_beyond_source_cluster"
            )
            continue
        if vector_overlap["has_external_vector_connection"]:
            table_record["precluster_reason"] = (
                "external_vector_connection"
            )
            continue
        if (
            table_record["overlaps_table_numbers"]
            or table_record["overlaps_image_groups"]
        ):
            table_record["precluster_reason"] = (
                "overlaps_another_detected_object"
            )
            continue
        table_record["precluster_protected"] = True
        table_record["precluster_reason"] = "independent_object"
        protected_table_numbers.add(table_record["table_number"])
        protected_rects.append(fitz.Rect(table_record["bbox"]))

    records_by_table_number = {
        record["table_number"]: record
        for record in table_records
    }
    for candidate in table_candidates:
        record = records_by_table_number[candidate["table_number"]]
        candidate["precluster_protected"] = record[
            "precluster_protected"
        ]
        candidate["precluster_reason"] = record[
            "precluster_reason"
        ]
        candidate["precluster_overlaps_table_numbers"] = record[
            "overlaps_table_numbers"
        ]
        candidate["precluster_overlaps_image_groups"] = record[
            "overlaps_image_groups"
        ]

    protected_image_group_indices: set[int] = set()
    for image_record in image_records:
        vector_overlap = image_record["vector_overlap"]
        if vector_overlap["has_external_vector_connection"]:
            image_record["precluster_reason"] = (
                "external_vector_connection"
            )
            continue
        if image_record["overlaps_table_numbers"]:
            image_record["precluster_reason"] = (
                "overlaps_detected_table"
            )
            continue
        if not image_is_large_enough(
            fitz.Rect(image_record["bbox"]),
            page_rect,
        ):
            image_record["precluster_reason"] = "small_image"
            continue
        image_record["precluster_protected"] = True
        image_record["precluster_reason"] = "independent_object"
        protected_image_group_indices.add(
            image_record["group_index"]
        )
        protected_rects.append(fitz.Rect(image_record["bbox"]))

    return {
        "protected_rects": protected_rects,
        "protected_table_numbers": sorted(protected_table_numbers),
        "protected_image_group_indices": sorted(
            protected_image_group_indices
        ),
        "tables": table_records,
        "image_groups": image_records,
    }



def normalized_drawing_value(
    value: object,
    rect: fitz.Rect,
) -> object:
    """将矢量路径坐标归一化到自身bbox，消除页面绝对位置对内容签名的影响。"""
    width = rect.width if rect.width > 0 else 1.0
    height = rect.height if rect.height > 0 else 1.0

    if isinstance(value, fitz.Point):
        return [
            "point",
            round((value.x - rect.x0) / width, 4),
            round((value.y - rect.y0) / height, 4),
        ]

    if isinstance(value, fitz.Rect):
        return [
            "rect",
            round((value.x0 - rect.x0) / width, 4),
            round((value.y0 - rect.y0) / height, 4),
            round((value.x1 - rect.x0) / width, 4),
            round((value.y1 - rect.y0) / height, 4),
        ]

    if isinstance(value, (tuple, list)):
        return [
            normalized_drawing_value(item, rect)
            for item in value
        ]

    if isinstance(value, float):
        return round(value, 4)

    if isinstance(value, (int, str, bool)) or value is None:
        return value

    return str(value)



def drawing_content_signature(drawing: dict) -> str:
    """返回单个矢量绘图对象的稳定内容摘要。"""
    rect = fitz.Rect(drawing["rect"])
    payload = {
        "type": drawing.get("type"),
        "items": normalized_drawing_value(
            drawing.get("items", []),
            rect,
        ),
        "color": normalized_drawing_value(
            drawing.get("color"),
            rect,
        ),
        "fill": normalized_drawing_value(
            drawing.get("fill"),
            rect,
        ),
        "width": round(float(drawing.get("width") or 0.0), 4),
        "lineCap": normalized_drawing_value(
            drawing.get("lineCap"),
            rect,
        ),
        "lineJoin": normalized_drawing_value(
            drawing.get("lineJoin"),
            rect,
        ),
        "dashes": normalized_drawing_value(
            drawing.get("dashes"),
            rect,
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def rect_is_mostly_inside_any(
    rect: fitz.Rect,
    container_rects: list[fitz.Rect],
    minimum_ratio: float = 0.5,
) -> bool:
    """判断候选对象是否大部分位于任一排除区域。"""
    area = rect_area(rect)

    if area <= 0:
        return False

    for container_rect in container_rects:
        intersection = rect & container_rect

        if (
            not intersection.is_empty
            and rect_area(intersection) / area >= minimum_ratio
        ):
            return True

    return False
