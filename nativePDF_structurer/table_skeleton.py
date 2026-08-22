"""从密集矢量簇中选择表格骨架输入。

模块发现横竖规则线连通组件，验证多单元格
拓扑，并为每条合并逻辑线生成一个供PyMuPDF表格检测使用的合成path；没有
完整骨架时，执行确定性、方向均衡的规则线预算选择。

关键不变量:
    * 短规则线过滤发生在共线碎片恢复之后。
    * 至少一个方向必须描述两个以上单元格。
    * 内部几何在API边界前使用轻量轴向矩形。
    * 每条合成path必须保留全部来源drawing ID。

Select table skeleton inputs from dense vector clusters.

Finds connected horizontal and vertical rule components, verifies multi-cell
topology, and emits one synthetic path per merged logical rule for PyMuPDF
table detection. When no complete skeleton exists, it applies a deterministic,
direction-balanced rule budget.

Key invariants:
    * Collinear fragments are evaluated before short-rule filtering.
    * At least one axis must describe more than one cell.
    * Internal geometry uses lightweight axis rectangles until API boundaries.
    * Every synthetic path preserves all source drawing IDs.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right
from collections import defaultdict
import fitz

from .geometry import as_fitz_rect
from .inventory import PageInventory

def find_axis_skeleton_components(
    axis_geometry_index: dict,
    cluster_bbox: fitz.Rect,
    allowed_element_ids: set[str],
    *,
    alignment_tolerance: float,
    minimum_structural_span: float,
    coordinate_decimal_places: int,
) -> list[dict]:
    """返回当前空间簇内由规则线拓扑形成的表格候选骨架。"""
    tolerance = alignment_tolerance
    cluster_rect = fitz.Rect(cluster_bbox).normalize()

    def in_cluster(rule: dict) -> bool:
        source_ids = rule.get("element_ids", [])
        if not source_ids or not all(
            element_id in allowed_element_ids
            for element_id in source_ids
        ):
            return False
        rect = rule["rect"]
        return not (
            rect.x1 < cluster_rect.x0 - tolerance
            or rect.x0 > cluster_rect.x1 + tolerance
            or rect.y1 < cluster_rect.y0 - tolerance
            or rect.y0 > cluster_rect.y1 + tolerance
        )

    horizontal_rules = [
        rule
        for rule in axis_geometry_index[
            "logical_horizontal_rules"
        ]
        if in_cluster(rule)
        and rule["rect"].width >= minimum_structural_span
    ]
    vertical_rules = [
        rule
        for rule in axis_geometry_index["logical_vertical_rules"]
        if in_cluster(rule)
        and rule["rect"].height >= minimum_structural_span
    ]
    if not horizontal_rules or not vertical_rules:
        return []

    sorted_vertical = sorted(
        (
            index,
            rule,
            rule["rect"].y0,
            rule["rect"].y1,
        )
        for index, rule in enumerate(vertical_rules)
    )
    sorted_vertical.sort(key=lambda item: item[1]["position"])
    vertical_positions = [
        rule[1]["position"] for rule in sorted_vertical
    ]
    parent = list(range(len(horizontal_rules) + len(vertical_rules)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for horizontal_index, horizontal_rule in enumerate(
        horizontal_rules
    ):
        horizontal_rect = horizontal_rule["rect"]
        first_vertical = bisect_left(
            vertical_positions,
            horizontal_rect.x0 - tolerance,
        )
        last_vertical = bisect_right(
            vertical_positions,
            horizontal_rect.x1 + tolerance,
        )
        for sorted_index in range(first_vertical, last_vertical):
            vertical_index, vertical_rule, vertical_y0, vertical_y1 = sorted_vertical[
                sorted_index
            ]
            horizontal_y = horizontal_rule["position"]
            if (
                horizontal_rect.x0 - tolerance
                <= vertical_rule["position"]
                <= horizontal_rect.x1 + tolerance
                and vertical_y0 - tolerance
                <= horizontal_y
                <= vertical_y1 + tolerance
            ):
                union(
                    horizontal_index,
                    len(horizontal_rules) + vertical_index,
                )

    components: dict[int, dict] = defaultdict(
        lambda: {
            "horizontal_rules": [],
            "vertical_rules": [],
        }
    )
    for index, rule in enumerate(horizontal_rules):
        components[find(index)]["horizontal_rules"].append(rule)
    for index, rule in enumerate(vertical_rules):
        components[find(len(horizontal_rules) + index)][
            "vertical_rules"
        ].append(rule)

    result: list[dict] = []
    for component in components.values():
        component_horizontal = component["horizontal_rules"]
        component_vertical = component["vertical_rules"]
        if len(component_horizontal) < 2 or len(component_vertical) < 2:
            continue

        horizontal_positions = sorted(
            {
                round(
                    rule["position"],
                    coordinate_decimal_places,
                )
                for rule in component_horizontal
            }
        )
        vertical_positions_in_component = sorted(
            {
                round(
                    rule["position"],
                    coordinate_decimal_places,
                )
                for rule in component_vertical
            }
        )
        spans_multiple_cells = (
            len(horizontal_positions) >= 3
            and len(vertical_positions_in_component) >= 2
        ) or (
            len(horizontal_positions) >= 2
            and len(vertical_positions_in_component) >= 3
        )
        if not spans_multiple_cells:
            continue

        all_rules = [
            *component_horizontal,
            *component_vertical,
        ]
        component_bbox = fitz.Rect(
            min(rule["rect"].x0 for rule in all_rules),
            min(rule["rect"].y0 for rule in all_rules),
            max(rule["rect"].x1 for rule in all_rules),
            max(rule["rect"].y1 for rule in all_rules),
        ).normalize()
        source_ids = sorted(
            {
                element_id
                for rule in all_rules
                for element_id in rule.get("element_ids", [])
            }
        )
        outer_horizontal_support = (
            any(
                abs(
                    rule["rect"].y0
                    - component_bbox.y0
                )
                <= tolerance
                for rule in component_horizontal
            )
            and any(
                abs(
                    rule["rect"].y1
                    - component_bbox.y1
                )
                <= tolerance
                for rule in component_horizontal
            )
        )
        outer_vertical_support = (
            any(
                abs(
                    rule["rect"].x0
                    - component_bbox.x0
                )
                <= tolerance
                for rule in component_vertical
            )
            and any(
                abs(
                    rule["rect"].x1
                    - component_bbox.x1
                )
                <= tolerance
                for rule in component_vertical
            )
        )
        result.append(
            {
                "bbox": component_bbox,
                "source_element_ids": source_ids,
                "_logical_rules": all_rules,
                "horizontal_rule_count": len(component_horizontal),
                "vertical_rule_count": len(component_vertical),
                "horizontal_position_count": len(horizontal_positions),
                "vertical_position_count": len(
                    vertical_positions_in_component
                ),
                "outer_horizontal_support": outer_horizontal_support,
                "outer_vertical_support": outer_vertical_support,
                "outer_envelope_supported": (
                    outer_horizontal_support
                    and outer_vertical_support
                ),
            }
        )

    return sorted(
        result,
        key=lambda component: (
            not component["outer_envelope_supported"],
            component["bbox"].y0,
            component["bbox"].x0,
        ),
    )




def select_budgeted_axis_rules_without_complete_skeleton(
    rules: list[dict],
    cluster_bbox: fitz.Rect,
    budget: int,
) -> list[dict]:
    """Select a bounded, direction-balanced logical-rule subset."""

    if len(rules) <= budget:
        return list(rules)

    cluster_rect = fitz.Rect(cluster_bbox).normalize()

    def structural_score(rule: dict) -> tuple[float, float, float]:
        rect = rule["rect"]
        if rule.get("orientation") == "h":
            page_span = max(cluster_rect.width, 1e-9)
            span = rect.width
            edge_distance = min(
                abs(rule["position"] - cluster_rect.y0),
                abs(rule["position"] - cluster_rect.y1),
            ) / max(cluster_rect.height, 1e-9)
        else:
            page_span = max(cluster_rect.height, 1e-9)
            span = rect.height
            edge_distance = min(
                abs(rule["position"] - cluster_rect.x0),
                abs(rule["position"] - cluster_rect.x1),
            ) / max(cluster_rect.width, 1e-9)
        return (span / page_span, -edge_distance, span)

    horizontal_rules = [
        rule for rule in rules if rule.get("orientation") == "h"
    ]
    vertical_rules = [
        rule for rule in rules if rule.get("orientation") == "v"
    ]
    orientation_budget = max(1, budget // 2)
    selected = [
        *heapq.nlargest(
            min(len(horizontal_rules), orientation_budget),
            horizontal_rules,
            key=structural_score,
        ),
        *heapq.nlargest(
            min(len(vertical_rules), orientation_budget),
            vertical_rules,
            key=structural_score,
        ),
    ]
    selected_rule_ids = {id(rule) for rule in selected}
    remaining_budget = budget - len(selected)
    if remaining_budget > 0:
        selected.extend(
            heapq.nlargest(
                remaining_budget,
                (
                    rule
                    for rule in rules
                    if id(rule) not in selected_rule_ids
                ),
                key=structural_score,
            )
        )
        selected_rule_ids = {id(rule) for rule in selected}
    return [
        rule for rule in rules if id(rule) in selected_rule_ids
    ]


def select_cluster_table_budget(
    cluster_drawings: list[dict],
    *,
    drawing_budget: int,
) -> list[dict]:
    """Select a deterministic spatial prefix when no axis skeleton exists.

    当轴向骨架不可用时，按页面坐标稳定选择有限数量的drawing。
    """

    return sorted(
        cluster_drawings,
        key=lambda drawing: (
            fitz.Rect(drawing["rect"]).y0,
            fitz.Rect(drawing["rect"]).x0,
            drawing.get("seqno", 0),
        ),
    )[:drawing_budget]


def select_cluster_table_skeleton_drawings(
    cluster_drawings: list[dict],
    cluster_bbox: fitz.Rect,
    axis_geometry_index: dict,
    page_inventory: PageInventory,
    drawing_budget: int,
    alignment_tolerance: float,
    minimum_structural_span: float,
    coordinate_decimal_places: int,
) -> dict:
    """为大型簇选择表格骨架输入，并返回可审计选择元数据。"""
    allowed_element_ids = {
        element_id
        for drawing in cluster_drawings
        for element_id in [
            page_inventory.id_for_payload(drawing, "drawing")
        ]
        if element_id is not None
    }
    components = find_axis_skeleton_components(
        axis_geometry_index=axis_geometry_index,
        cluster_bbox=cluster_bbox,
        allowed_element_ids=allowed_element_ids,
        alignment_tolerance=alignment_tolerance,
        minimum_structural_span=minimum_structural_span,
        coordinate_decimal_places=coordinate_decimal_places,
    )
    selected_logical_rules: list[dict] = []
    selected_rule_object_ids: set[int] = set()
    for component in components:
        for rule in component.get("_logical_rules", []):
            if id(rule) in selected_rule_object_ids:
                continue
            selected_rule_object_ids.add(id(rule))
            selected_logical_rules.append(rule)

    if not selected_logical_rules:
        cluster_tolerance = (
            alignment_tolerance
        )
        cluster_x0 = cluster_bbox.x0 - cluster_tolerance
        cluster_x1 = cluster_bbox.x1 + cluster_tolerance
        cluster_y0 = cluster_bbox.y0 - cluster_tolerance
        cluster_y1 = cluster_bbox.y1 + cluster_tolerance
        eligible_rules = [
            rule
            for rule in [
                *axis_geometry_index["logical_horizontal_rules"],
                *axis_geometry_index["logical_vertical_rules"],
            ]
            if all(
                element_id in allowed_element_ids
                for element_id in rule.get("element_ids", [])
            )
            and rule.get("element_ids")
            and not (
                rule["rect"].x1 < cluster_x0
                or rule["rect"].x0 > cluster_x1
                or rule["rect"].y1 < cluster_y0
                or rule["rect"].y0 > cluster_y1
            )
        ]
        selected_logical_rules = (
            select_budgeted_axis_rules_without_complete_skeleton(
                rules=eligible_rules,
                cluster_bbox=cluster_bbox,
                budget=drawing_budget,
            )
        )

    skeleton_ids = sorted(
        {
            element_id
            for rule in selected_logical_rules
            for element_id in rule.get("element_ids", [])
        }
    )

    drawing_by_id = {
        page_inventory.id_for_payload(drawing, "drawing"): drawing
        for drawing in cluster_drawings
        if page_inventory.id_for_payload(drawing, "drawing")
        is not None
    }
    selected_drawings: list[dict] = []
    for rule in selected_logical_rules:
        source_element_ids = sorted(
            element_id
            for element_id in rule.get("element_ids", [])
            if element_id in drawing_by_id
        )
        if not source_element_ids:
            continue
        source_drawing = drawing_by_id.get(source_element_ids[0])
        if source_drawing is None:
            continue
        rule_rect = as_fitz_rect(rule["rect"])
        if rule.get("orientation") == "h":
            rule_item = (
                "l",
                fitz.Point(rule_rect.x0, rule["position"]),
                fitz.Point(rule_rect.x1, rule["position"]),
            )
        else:
            rule_item = (
                "l",
                fitz.Point(rule["position"], rule_rect.y0),
                fitz.Point(rule["position"], rule_rect.y1),
            )
        skeleton_drawing = dict(source_drawing)
        skeleton_drawing["items"] = [rule_item]
        skeleton_drawing["rect"] = rule_rect
        skeleton_drawing["_source_element_ids"] = source_element_ids
        skeleton_drawing["_axis_skeleton_item"] = True
        selected_drawings.append(skeleton_drawing)
    if not selected_drawings:
        return {
            "drawings": [],
            "mode": "axis_skeleton_empty",
            "source_element_ids": [],
            "skeleton_components": components,
        }
    return {
        "drawings": selected_drawings,
        "mode": (
            "axis_skeleton_components"
            if components
            else "axis_rules_without_complete_skeleton"
        ),
        "source_element_ids": skeleton_ids,
        "skeleton_components": components,
    }
