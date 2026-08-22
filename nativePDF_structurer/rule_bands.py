"""检测由对齐规则线构成的视觉框架。

模块围绕已有视觉区域分析恢复后的水平线，
提议框架bbox并返回审计元数据；它不执行形态学处理、不渲染图片、不读取OCR
文字，也不进行跨检测器归属仲裁。

Detect visual frameworks formed by aligned rule lines.

Examines recovered horizontal rules around an existing visual region and
proposes a framework bbox plus audit metadata. It does not run morphology,
render images, inspect OCR text, or decide cross-detector ownership.
"""

from __future__ import annotations

from collections.abc import Callable

import fitz

from .geometry import rect_area, rect_center
from .inventory import PageInventory


def detect_rule_line_framework_bbox(
    region_bbox: fitz.Rect,
    drawings: list[dict],
    page_rect: fitz.Rect,
    logical_horizontal_rules: list[dict],
    text_lines: list[dict] | None = None,
    page_inventory: PageInventory | None = None,
    *,
    alignment_tolerance_points: float,
    group_horizontal_rules_by_span: Callable[
        [list[dict]], list[list[dict]]
    ],
    unique_rule_positions: Callable[[list[dict]], list[float]],
    rule_element_ids: Callable[[dict], set[str]],
) -> tuple[fitz.Rect | None, dict | None]:
    """检测视觉簇附近的分段或单带水平框架并返回严格边界。"""
    # ``drawings``、``page_rect`` 和 ``page_inventory`` 保留在兼容签名中；
    # 当前算法只消费上游已经过滤并恢复的逻辑横线。
    del drawings, page_rect, page_inventory

    axis_rules: list[dict] = [
        {
            **rule,
            "element_ids": sorted(rule_element_ids(rule)),
        }
        for rule in logical_horizontal_rules
    ]
    if not axis_rules:
        return None, None

    tolerance = alignment_tolerance_points
    center_x = (region_bbox.x0 + region_bbox.x1) / 2

    for group in sorted(
        group_horizontal_rules_by_span(axis_rules),
        key=lambda candidate: -len(candidate),
    ):
        positions = unique_rule_positions(group)
        if len(positions) < 2:
            continue

        frame_x0 = min(rule["rect"].x0 for rule in group)
        frame_x1 = max(rule["rect"].x1 for rule in group)
        if not (
            frame_x0 - tolerance
            <= center_x
            <= frame_x1 + tolerance
        ):
            continue

        if len(positions) == 2:
            top, bottom = positions
        else:
            upper_positions = [
                position
                for position in positions
                if position <= region_bbox.y0 + tolerance
            ]
            lower_positions = [
                position
                for position in positions
                if position >= region_bbox.y1 - tolerance
            ]
            top = (
                max(upper_positions)
                if upper_positions
                else min(positions)
            )
            bottom = (
                min(lower_positions)
                if lower_positions
                else max(positions)
            )

        if bottom <= top:
            continue
        if (
            bottom < region_bbox.y0 - tolerance
            or top > region_bbox.y1 + tolerance
        ):
            continue
        if frame_x1 <= region_bbox.x0 or frame_x0 >= region_bbox.x1:
            continue

        framework_bbox = fitz.Rect(frame_x0, top, frame_x1, bottom)
        if rect_area(framework_bbox) <= 0:
            continue

        if len(positions) == 2:
            region_center = rect_center(region_bbox)
            region_overlap = region_bbox & framework_bbox
            region_area = rect_area(region_bbox)
            region_overlap_ratio = (
                rect_area(region_overlap) / region_area
                if region_area > 0 and not region_overlap.is_empty
                else 0.0
            )
            if (
                not framework_bbox.contains(region_center)
                or region_overlap_ratio < 0.50
            ):
                continue

            outside_anchor_text_lines = []
            for line in text_lines or []:
                line_rect = fitz.Rect(line["bbox"])
                line_center = rect_center(line_rect)
                if not framework_bbox.contains(line_center):
                    continue
                if region_bbox.contains(line_center):
                    continue
                outside_anchor_text_lines.append(line)

            if not outside_anchor_text_lines:
                continue

            framework_kind = "two_rule_band_framework"
            framework_text_line_count = len(
                outside_anchor_text_lines
            )
        else:
            framework_kind = "multi_rule_line_framework"
            framework_text_line_count = 0

        framework_drawing_element_ids = sorted(
            {
                element_id
                for rule in group
                if top - tolerance
                <= rule["position"]
                <= bottom + tolerance
                for element_id in rule.get("element_ids", [])
            }
        )
        return framework_bbox, {
            "kind": framework_kind,
            "horizontal_rule_count": len(positions),
            "outside_anchor_text_line_count": (
                framework_text_line_count
            ),
            "frame_x0": round(frame_x0, 3),
            "frame_x1": round(frame_x1, 3),
            "frame_y0": round(top, 3),
            "frame_y1": round(bottom, 3),
            "drawing_element_ids": framework_drawing_element_ids,
        }

    return None, None
