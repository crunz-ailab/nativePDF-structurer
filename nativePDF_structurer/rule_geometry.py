"""从PDF drawing中构建可复用的轴向规则线几何视图。

模块提取横向和竖向片段，
合并严格共线碎片，并按稳定drawing ID建立规范化逻辑线索引，供表格、规则
带、视觉容器和审计阶段共同使用。

Build reusable axis-rule geometry views from PDF drawings.

Extracts horizontal and vertical segments, merges exactly collinear fragments,
and creates normalized logical-rule indexes keyed by stable drawing IDs. The
result is reused by table, rule-band, visual-container, and audit stages.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import fitz

from .geometry import AxisRect
from .inventory import PageInventory


def make_normalized_axis_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> AxisRect:
    """构造规范化的轻量轴向矩形。"""
    return AxisRect(
        min(x0, x1),
        min(y0, y1),
        max(x0, x1),
        max(y0, y1),
    )


def drawing_item_axis_segments(
    item: tuple,
    element_id: str,
    *,
    maximum_rule_thickness_points: float,
) -> list[dict]:
    """把一个PDF路径item拆成不带页面长度门槛的轴向线段。"""
    item_type = item[0]
    segments: list[dict] = []

    if item_type == "l":
        first_point, second_point = item[1], item[2]
        rect = make_normalized_axis_rect(
            first_point.x,
            first_point.y,
            second_point.x,
            second_point.y,
        )
        if rect.width > 0 and rect.height <= maximum_rule_thickness_points:
            segments.append(
                {
                    "orientation": "h",
                    "rect": rect,
                    "position": (rect.y0 + rect.y1) / 2,
                    "element_ids": [element_id],
                    "item": item,
                }
            )
        elif rect.height > 0 and rect.width <= maximum_rule_thickness_points:
            segments.append(
                {
                    "orientation": "v",
                    "rect": rect,
                    "position": (rect.x0 + rect.x1) / 2,
                    "element_ids": [element_id],
                    "item": item,
                }
            )
        return segments

    if item_type != "re":
        return segments

    item_rect = item[1]
    rect = make_normalized_axis_rect(
        item_rect.x0,
        item_rect.y0,
        item_rect.x1,
        item_rect.y1,
    )
    if rect.width <= 0 or rect.height <= 0:
        return segments

    if rect.height <= maximum_rule_thickness_points:
        return [
            {
                "orientation": "h",
                "rect": rect,
                "position": (rect.y0 + rect.y1) / 2,
                "element_ids": [element_id],
                "item": item,
            }
        ]
    if rect.width <= maximum_rule_thickness_points:
        return [
            {
                "orientation": "v",
                "rect": rect,
                "position": (rect.x0 + rect.x1) / 2,
                "element_ids": [element_id],
                "item": item,
            }
        ]

    return [
        {
            "orientation": "h",
            "rect": AxisRect(rect.x0, rect.y0, rect.x1, rect.y0),
            "position": rect.y0,
            "element_ids": [element_id],
            "item": (
                "l",
                fitz.Point(rect.x0, rect.y0),
                fitz.Point(rect.x1, rect.y0),
            ),
        },
        {
            "orientation": "h",
            "rect": AxisRect(rect.x0, rect.y1, rect.x1, rect.y1),
            "position": rect.y1,
            "element_ids": [element_id],
            "item": (
                "l",
                fitz.Point(rect.x0, rect.y1),
                fitz.Point(rect.x1, rect.y1),
            ),
        },
        {
            "orientation": "v",
            "rect": AxisRect(rect.x0, rect.y0, rect.x0, rect.y1),
            "position": rect.x0,
            "element_ids": [element_id],
            "item": (
                "l",
                fitz.Point(rect.x0, rect.y0),
                fitz.Point(rect.x0, rect.y1),
            ),
        },
        {
            "orientation": "v",
            "rect": AxisRect(rect.x1, rect.y0, rect.x1, rect.y1),
            "position": rect.x1,
            "element_ids": [element_id],
            "item": (
                "l",
                fitz.Point(rect.x1, rect.y0),
                fitz.Point(rect.x1, rect.y1),
            ),
        },
    ]


def build_logical_axis_rule(
    segments: list[dict],
    orientation: str,
    bucket_position: float,
) -> dict:
    """构造一条带原始drawing来源的逻辑轴向线。"""
    if len(segments) == 1:
        segment = segments[0]
        source_rect = segment["rect"]
        return {
            "orientation": orientation,
            "rect": source_rect,
            "position": bucket_position,
            "element_ids": list(segment.get("element_ids", [])),
            "drawing_count": len(segment.get("element_ids", [])),
            "segment_count": 1,
            "source_rects": [source_rect],
            "logical": True,
        }

    source_rects = [segment["rect"] for segment in segments]
    rect = make_normalized_axis_rect(
        min(item.x0 for item in source_rects),
        min(item.y0 for item in source_rects),
        max(item.x1 for item in source_rects),
        max(item.y1 for item in source_rects),
    )
    element_ids = sorted(
        {
            element_id
            for segment in segments
            for element_id in segment.get("element_ids", [])
        }
    )
    return {
        "orientation": orientation,
        "rect": rect,
        "position": bucket_position,
        "element_ids": element_ids,
        "drawing_count": len(element_ids),
        "segment_count": len(segments),
        "source_rects": source_rects,
        "logical": True,
    }


def merge_exact_collinear_axis_segments(
    segments: list[dict],
    orientation: str,
    *,
    coordinate_decimal_places: int,
) -> list[dict]:
    """合并同一坐标且端点相接或重叠的水平/竖直片段。"""
    buckets: dict[float, list[dict]] = defaultdict(list)
    for segment in segments:
        buckets[
            round(segment["position"], coordinate_decimal_places)
        ].append(segment)

    logical_rules: list[dict] = []
    for bucket_position, bucket_segments in buckets.items():
        if len(bucket_segments) == 1:
            logical_rules.append(
                build_logical_axis_rule(
                    segments=bucket_segments,
                    orientation=orientation,
                    bucket_position=bucket_position,
                )
            )
            continue

        start_key = (
            (lambda item: item["rect"].x0)
            if orientation == "h"
            else (lambda item: item["rect"].y0)
        )
        current: list[dict] = []
        current_end = -math.inf
        for segment in sorted(bucket_segments, key=start_key):
            segment_rect = segment["rect"]
            segment_start = (
                segment_rect.x0
                if orientation == "h"
                else segment_rect.y0
            )
            segment_end = (
                segment_rect.x1
                if orientation == "h"
                else segment_rect.y1
            )
            if (
                current
                and round(segment_start, coordinate_decimal_places)
                > round(current_end, coordinate_decimal_places)
            ):
                logical_rules.append(
                    build_logical_axis_rule(
                        segments=current,
                        orientation=orientation,
                        bucket_position=bucket_position,
                    )
                )
                current = []
                current_end = -math.inf
            current.append(segment)
            current_end = max(current_end, segment_end)

        if current:
            logical_rules.append(
                build_logical_axis_rule(
                    segments=current,
                    orientation=orientation,
                    bucket_position=bucket_position,
                )
            )

    return sorted(
        logical_rules,
        key=lambda rule: (
            rule["position"],
            rule["rect"].x0,
            rule["rect"].y0,
        ),
    )


def build_axis_geometry_index(
    drawings: list[dict],
    page_rect: fitz.Rect,
    page_inventory: PageInventory,
    *,
    drawing_axis_rule: Callable[[dict, fitz.Rect], dict | None],
    maximum_rule_thickness_points: float,
    coordinate_decimal_places: int,
    minimum_horizontal_length_page_ratio: float,
) -> dict:
    """建立页面级轴向片段、逻辑线和原始drawing反向索引。"""
    horizontal_segments: list[dict] = []
    vertical_segments: list[dict] = []
    axis_segments_by_element_id: dict[str, list[dict]] = defaultdict(list)
    legacy_axis_rule_by_element_id: dict[str, dict | None] = {}
    drawing_by_element_id: dict[str, dict] = {}

    for drawing in drawings:
        element_id = page_inventory.id_for_payload(drawing, "drawing")
        if element_id is None:
            continue
        drawing_by_element_id[element_id] = drawing
        legacy_axis_rule_by_element_id[element_id] = drawing_axis_rule(
            drawing,
            page_rect,
        )
        for item in drawing.get("items", []):
            for segment in drawing_item_axis_segments(
                item=item,
                element_id=element_id,
                maximum_rule_thickness_points=(
                    maximum_rule_thickness_points
                ),
            ):
                if segment["orientation"] == "h":
                    horizontal_segments.append(segment)
                else:
                    vertical_segments.append(segment)
                axis_segments_by_element_id[element_id].append(segment)

    logical_horizontal_rules = merge_exact_collinear_axis_segments(
        segments=horizontal_segments,
        orientation="h",
        coordinate_decimal_places=coordinate_decimal_places,
    )
    logical_vertical_rules = merge_exact_collinear_axis_segments(
        segments=vertical_segments,
        orientation="v",
        coordinate_decimal_places=coordinate_decimal_places,
    )
    minimum_long_horizontal_width = (
        page_rect.width * minimum_horizontal_length_page_ratio
    )
    long_horizontal_rules = [
        rule
        for rule in logical_horizontal_rules
        if rule["rect"].width >= minimum_long_horizontal_width
    ]

    return {
        "horizontal_segments": horizontal_segments,
        "vertical_segments": vertical_segments,
        "axis_segments_by_element_id": dict(axis_segments_by_element_id),
        "logical_horizontal_rules": logical_horizontal_rules,
        "logical_vertical_rules": logical_vertical_rules,
        "long_logical_horizontal_rules": long_horizontal_rules,
        "legacy_axis_rule_by_element_id": legacy_axis_rule_by_element_id,
        "drawing_by_element_id": drawing_by_element_id,
        "metadata": {
            "drawing_count": len(drawings),
            "horizontal_segment_count": len(horizontal_segments),
            "vertical_segment_count": len(vertical_segments),
            "logical_horizontal_rule_count": len(logical_horizontal_rules),
            "logical_vertical_rule_count": len(logical_vertical_rules),
            "long_logical_horizontal_rule_count": len(long_horizontal_rules),
        },
    }


def rule_element_ids(rule: dict) -> set[str]:
    """统一读取原始规则或逻辑规则所拥有的drawing ID。"""
    if rule.get("element_ids") is not None:
        return {
            element_id
            for element_id in rule.get("element_ids", [])
            if element_id is not None
        }
    element_id = rule.get("element_id")
    return {element_id} if element_id is not None else set()


def logical_horizontal_rules_from_axis_index(
    axis_geometry_index: dict,
    page_rect: fitz.Rect,
    page_number: int | None,
    audit_records: list[dict] | None,
    *,
    minimum_horizontal_length_page_ratio: float,
) -> list[dict]:
    """从页面轴向索引取得长横线视图并写入兼容审计记录。"""
    logical_rules = axis_geometry_index["logical_horizontal_rules"]
    minimum_width = (
        page_rect.width * minimum_horizontal_length_page_ratio
    )
    if audit_records is not None:
        for logical_rule_index, rule in enumerate(logical_rules, start=1):
            if rule.get("segment_count", 0) < 2:
                continue
            rule_rect = rule["rect"]
            passed_minimum_width = rule_rect.width >= minimum_width
            audit_records.append(
                {
                    "page_number": page_number,
                    "logical_rule_id": (
                        f"page#{page_number}:logical_horizontal_rule#"
                        f"{logical_rule_index}"
                        if page_number is not None
                        else f"logical_horizontal_rule#{logical_rule_index}"
                    ),
                    "source_drawing_ids": sorted(rule_element_ids(rule)),
                    "source_drawing_count": rule.get("drawing_count", 0),
                    "source_rects": [
                        [round(value, 3) for value in source_rect]
                        for source_rect in rule.get("source_rects", [])
                    ],
                    "merged_bbox": [
                        round(value, 3) for value in rule_rect
                    ],
                    "merged_y": round(rule["position"], 3),
                    "merged_width_points": round(rule_rect.width, 3),
                    "merged_width_page_ratio": round(
                        (
                            rule_rect.width / page_rect.width
                            if page_rect.width > 0
                            else 0.0
                        ),
                        6,
                    ),
                    "minimum_width_page_ratio": (
                        minimum_horizontal_length_page_ratio
                    ),
                    "passed_logical_width_filter": passed_minimum_width,
                    "logical_view_status": (
                        "kept_for_table_and_visual_structure"
                        if passed_minimum_width
                        else "kept_for_table_skeleton_only"
                    ),
                    "table_source_status": "pending",
                    "visual_logical_status": "pending",
                    "final_owner_claim_ids": [],
                    "final_owner_detectors": [],
                }
            )
    return list(axis_geometry_index["long_logical_horizontal_rules"])


def build_logical_horizontal_rules(
    page_rect: fitz.Rect,
    axis_geometry_index: dict,
    page_number: int | None = None,
    audit_records: list[dict] | None = None,
    *,
    minimum_horizontal_length_page_ratio: float,
) -> list[dict]:
    """从页面轴向索引取得同Y逻辑横线。"""
    return logical_horizontal_rules_from_axis_index(
        axis_geometry_index=axis_geometry_index,
        page_rect=page_rect,
        page_number=page_number,
        audit_records=audit_records,
        minimum_horizontal_length_page_ratio=(
            minimum_horizontal_length_page_ratio
        ),
    )
