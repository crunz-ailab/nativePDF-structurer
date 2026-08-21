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

#@GPT 轴向规则线组件检测使用显式的连接容差、最小跨度和坐标精度，不读取全局策略。
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
        #@GPT 直接在已有set中逐ID查询，避免为密集页面的每条逻辑线创建临时集合。
        source_ids = rule.get("element_ids", [])
        if not source_ids or not all(
            element_id in allowed_element_ids
            for element_id in source_ids
        ):
            return False
        #@GPT 逻辑线矩形在索引构建阶段已经规范化，此处只读复用。
        rect = rule["rect"]
        return not (
            rect.x1 < cluster_rect.x0 - tolerance
            or rect.x0 > cluster_rect.x1 + tolerance
            or rect.y1 < cluster_rect.y0 - tolerance
            or rect.y0 > cluster_rect.y1 + tolerance
        )

    #@GPT 极短轴向碎片先在“共线恢复之后”排除：不影响由多个短片段拼成的真实表格边，但避免器件图中微小矩形进入交点索引。
    horizontal_rules = [
        rule
        for rule in axis_geometry_index[
            "logical_horizontal_rules"
        ]
        if in_cluster(rule)
        #@GPT 直接读取规范化矩形的跨度，避免重复物化fitz.Rect。
        and rule["rect"].width >= minimum_structural_span
    ]
    vertical_rules = [
        rule
        for rule in axis_geometry_index["logical_vertical_rules"]
        if in_cluster(rule)
        #@GPT 竖向结构跨度同样直接读取缓存矩形。
        and rule["rect"].height >= minimum_structural_span
    ]
    if not horizontal_rules or not vertical_rules:
        return []

    #@GPT 竖线按X位置排序后用二分查找候选区间；只比较横线跨度内可能相交的竖线，避免密集页的全量两两运算。
    sorted_vertical = sorted(
        (
            index,
            rule,
            #@GPT 直接读取轻量轴向矩形的纵向边界，减少拓扑扫描中的对象构造。
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
        #@GPT 横线拓扑扫描只读取已缓存跨度，不在循环中重复构造矩形。
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
            #@GPT 规则线通常是零厚度矩形，不能用面积交集；用“竖线X落在横线跨度、横线Y落在竖线跨度”判断拓扑交点，并允许既有表格连接容差。
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
        #@GPT 单个矩形的两条横边和两条竖边不足以证明是表格；至少一个方向必须形成两个单元格。
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
            #@GPT 组件bbox由全部轴向规则的四个极值构造，最终在PyMuPDF边界处物化为fitz.Rect。
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
                    #@GPT 外沿支撑判断直接读取规则坐标。
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
                #@GPT 组件内部保留实际逻辑线，选择阶段可直接生成“一条逻辑线一个path”；该内部字段不写入页面审计。
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




#@GPT 无完整骨架时对合并轴向规则执行方向均衡的固定预算选择，预算由调用方显式传入。
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


#@GPT 轴向骨架为空时按稳定空间顺序选取有限drawing，保证结果可重复且受调用方预算约束。
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


#@GPT 大簇骨架选择只依赖显式的页面索引、台账、预算和几何容差。
def select_cluster_table_skeleton_drawings(
    cluster_drawings: list[dict],
    cluster_bbox: fitz.Rect,
    #@GPT 页面轴向索引和台账是必需输入，确保骨架来源可追溯且无需重复扫描PDF路径。
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
    #@GPT 完整骨架直接消费组件中的合并逻辑线，避免按父drawing重新展开无关路径。
    selected_logical_rules: list[dict] = []
    selected_rule_object_ids: set[int] = set()
    for component in components:
        for rule in component.get("_logical_rules", []):
            if id(rule) in selected_rule_object_ids:
                continue
            selected_rule_object_ids.add(id(rule))
            selected_logical_rules.append(rule)

    #@GPT 即使暂时没有形成完整网格，也优先把当前簇内的轴向规则线交给find_tables；比按空间位置截断更可能保留真实外框。
    if not selected_logical_rules:
        #@GPT 簇边界容差预先展开为四个标量，避免每条逻辑线重复构造矩形和执行相同加减法。
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
            #@GPT 逐项查询已有set并拒绝空来源列表，避免在密集簇中创建大量临时集合。
            if all(
                element_id in allowed_element_ids
                for element_id in rule.get("element_ids", [])
            )
            and rule.get("element_ids")
            and not (
                #@GPT 逻辑轴线矩形已规范化，可直接与预计算簇边界比较。
                rule["rect"].x1 < cluster_x0
                or rule["rect"].x0 > cluster_x1
                or rule["rect"].y1 < cluster_y0
                or rule["rect"].y0 > cluster_y1
            )
        ]
        #@GPT 未形成完整网格时仍对合并后的逻辑线执行预算选择，避免将海量原始路径直接交给find_tables。
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
    #@GPT 每条合并逻辑规则只生成一个合成path；几何来自原片段并集，Claim通过_source_element_ids追溯全部来源drawing。
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
        #@GPT 内部使用轻量AxisRect，仅在调用PyMuPDF find_tables()前物化fitz.Rect。
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
