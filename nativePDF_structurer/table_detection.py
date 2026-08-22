"""根据PDF原生几何检测并准备表格候选。

模块协调空间预聚簇、PyMuPDF原生表格
检测、几何补检、局部框补检、候选合并、页面级表格策略和结构Claim准备；
密集矢量簇优先选择轴向骨架，而不是将全部内部路径送入 ``find_tables``。

处理流程:
    1. 页面或局部簇超过预算时执行drawing预聚簇。
    2. 对密集簇优先选择完整轴向骨架。
    3. 运行原生检测及几何、局部框补检。
    4. 合并、去重、验证并排序表格候选。
    5. 构建表格Claim并完成页面级表格归属数据。

关键不变量:
    * 候选Claim只能包含检测器真实输入ID。
    * 单元格内部复杂矢量不会降低外层表格有效性。
    * detection、render和source-cluster bbox职责分离。
    * 表格输出延迟到跨检测器归属完成之后。

Detect and prepare table candidates from native PDF geometry.

Coordinates spatial preclustering, PyMuPDF native table detection, geometry
fallbacks, local-frame detection, candidate merging, page-level table policy,
and structural Claim preparation. Dense vector clusters use an axis-skeleton
budget instead of submitting all internal paths to ``find_tables``.

Processing flow:
    1. Precluster drawings when a page exceeds the per-cluster budget.
    2. Select complete axis skeletons for dense clusters when available.
    3. Run native detection and geometry or local-frame fallbacks.
    4. Merge, deduplicate, validate, and order table candidates.
    5. Build table Claims and finalize page-local table ownership data.

Key invariants:
    * Candidate Claims contain only actual detector input IDs.
    * Complex cell drawings do not reduce outer-table validity.
    * Detection, render, and source-cluster bboxes retain distinct roles.
    * Table output is delayed until cross-detector ownership is resolved.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import fitz
import numpy as np

from .geometry import rect_area, rect_center
from .inventory import PageInventory
from .page_context import PageRunContext
from .run_state import (
    DocumentRunMetrics,
    PageTableCandidateBuildResult,
    PageTableCandidateFinalization,
    PageTableDetectionResult,
)
from .table_skeleton import (
    select_cluster_table_budget,
    select_cluster_table_skeleton_drawings,
)
from .table_structure import (
    TableStructureDependencies,
    GeometryFallbackTable,
    build_table_claims,
    classify_table_candidate_structure,
    combine_native_and_fallback_tables,
    configure_table_container_claim,
    count_nonempty_table_cells,
    deduplicate_overlapping_table_candidates,
    extract_geometry_table_data,
    merge_horizontal_rules_for_table_skeleton,
    refine_table_claim_confidence,
    table_claim_confidence,
    table_contains_complex_visual_content,
    unique_rule_positions,
)
from .table_output import infer_table_rotation, rotate_table_data
from .visual_assets import draw_rect_on_visual_mask, image_is_large_enough


@dataclass(frozen=True)
class TablePreclusterDependencies:
    mask_pixels_per_point: float
    close_gap_points: float
    attach_margin_points: float
    drawing_overlap_ratio: float
    max_background_area_page_ratio: float
    drawing_is_filled_horizontal_bar: Callable[[dict], bool]
    drawing_is_long_horizontal_single_line: Callable[[dict], bool]


def build_pre_table_drawing_clusters(
    page: fitz.Page,
    excluded_header_footer_rects: list[fitz.Rect],
    drawings: list[dict],
    page_inventory: PageInventory,
    *,
    dependencies: TablePreclusterDependencies,
) -> list[dict]:
    """Build initial vector clusters used only by table detection."""

    candidates: list[dict[str, fitz.Rect]] = []
    eligible_drawings: list[
        tuple[dict, fitz.Rect, float, float, float]
    ] = []
    page_rect = fitz.Rect(page.rect)
    page_area = rect_area(page_rect)

    for drawing in drawings:
        drawing_element_id = page_inventory.id_for_payload(
            drawing,
            "drawing",
        )
        drawing_entry = (
            page_inventory.entry(drawing_element_id)
            if drawing_element_id is not None
            else None
        )
        drawing_labels = (
            drawing_entry.get("labels", {})
            if drawing_entry is not None
            else {}
        )
        if drawing_entry is not None and drawing_labels:
            drawing_rect = drawing_entry["bbox"]
            drawing_is_degenerate = bool(
                drawing_labels.get("degenerate", False)
            )
            drawing_is_background = bool(
                drawing_labels.get("background", False)
            )
            drawing_is_filled_bar = bool(
                drawing_labels.get("filled_horizontal_bar", False)
            )
            drawing_is_long_stroke = bool(
                drawing_labels.get("long_horizontal_stroke", False)
            )
        else:
            drawing_rect = fitz.Rect(drawing["rect"])
            drawing_is_degenerate = (
                drawing_rect.width <= 0
                and drawing_rect.height <= 0
            )
            fallback_drawing_area = rect_area(drawing_rect)
            drawing_is_background = (
                page_area > 0
                and fallback_drawing_area / page_area
                >= dependencies.max_background_area_page_ratio
            )
            drawing_is_filled_bar = (
                dependencies.drawing_is_filled_horizontal_bar(drawing)
            )
            drawing_is_long_stroke = (
                dependencies.drawing_is_long_horizontal_single_line(drawing)
            )

        drawing_area = rect_area(drawing_rect)
        if drawing_is_degenerate:
            continue
        center = rect_center(drawing_rect)
        if any(
            band.contains(center)
            for band in excluded_header_footer_rects
        ):
            continue
        if drawing_is_background:
            continue

        eligible_drawings.append(
            (
                drawing,
                drawing_rect,
                center.x,
                center.y,
                drawing_area,
            )
        )
        if drawing_is_filled_bar or drawing_is_long_stroke:
            continue
        candidates.append({"bbox": drawing_rect})

    if not candidates:
        return [
            {
                "bbox": fitz.Rect(page_rect),
                "drawings": list(drawings),
                "drawing_count": len(drawings),
            }
        ]

    scale = dependencies.mask_pixels_per_point
    mask_width = max(1, int(math.ceil(page_rect.width * scale)))
    mask_height = max(1, int(math.ceil(page_rect.height * scale)))
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    for candidate in candidates:
        draw_rect_on_visual_mask(
            mask=mask,
            rect=candidate["bbox"],
            page_rect=page_rect,
            scale=scale,
        )

    kernel_radius = max(
        1,
        int(round(dependencies.close_gap_points * scale)),
    )
    kernel_size = kernel_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_mask,
        connectivity=8,
    )

    candidate_regions: dict[int, fitz.Rect] = {}
    for component_index in range(1, component_count):
        x, y, width, height, _ = stats[component_index]
        component_rect = fitz.Rect(
            page_rect.x0 + x / scale,
            page_rect.y0 + y / scale,
            page_rect.x0 + (x + width) / scale,
            page_rect.y0 + (y + height) / scale,
        )
        if image_is_large_enough(
            image_rect=component_rect,
            page_rect=page_rect,
        ):
            candidate_regions[component_index] = component_rect

    clusters_by_label: dict[int, dict[str, Any]] = {
        label: {"bbox": region, "drawings": []}
        for label, region in candidate_regions.items()
    }
    unassigned_records: list[
        tuple[dict, fitz.Rect, float, float, float]
    ] = []
    for record in eligible_drawings:
        drawing, _, center_x, center_y, _ = record
        pixel_x = max(
            0,
            min(
                mask_width - 1,
                int((center_x - page_rect.x0) * scale),
            ),
        )
        pixel_y = max(
            0,
            min(
                mask_height - 1,
                int((center_y - page_rect.y0) * scale),
            ),
        )
        label = int(labels[pixel_y, pixel_x])
        if label in clusters_by_label:
            clusters_by_label[label]["drawings"].append(drawing)
        else:
            unassigned_records.append(record)

    expanded_regions = {
        label: (
            max(
                page_rect.x0,
                region.x0 - dependencies.attach_margin_points,
            ),
            max(
                page_rect.y0,
                region.y0 - dependencies.attach_margin_points,
            ),
            min(
                page_rect.x1,
                region.x1 + dependencies.attach_margin_points,
            ),
            min(
                page_rect.y1,
                region.y1 + dependencies.attach_margin_points,
            ),
        )
        for label, region in candidate_regions.items()
    }
    for (
        drawing,
        drawing_rect,
        center_x,
        center_y,
        drawing_area,
    ) in unassigned_records:
        for label, (x0, y0, x1, y1) in expanded_regions.items():
            intersection_width = max(
                0.0,
                min(drawing_rect.x1, x1) - max(drawing_rect.x0, x0),
            )
            intersection_height = max(
                0.0,
                min(drawing_rect.y1, y1) - max(drawing_rect.y0, y0),
            )
            overlap_ratio = (
                intersection_width * intersection_height / drawing_area
                if drawing_area > 0
                else 0.0
            )
            if (
                x0 <= center_x <= x1
                and y0 <= center_y <= y1
            ) or overlap_ratio >= dependencies.drawing_overlap_ratio:
                clusters_by_label[label]["drawings"].append(drawing)

    clusters = [
        {
            **cluster,
            "drawing_count": len(cluster["drawings"]),
        }
        for cluster in clusters_by_label.values()
        if cluster["drawings"]
    ]
    if not clusters:
        return [
            {
                "bbox": fitz.Rect(page_rect),
                "drawings": [record[0] for record in eligible_drawings],
                "drawing_count": len(eligible_drawings),
            }
        ]

    return sorted(
        clusters,
        key=lambda cluster: (
            fitz.Rect(cluster["bbox"]).y0,
            fitz.Rect(cluster["bbox"]).x0,
        ),
    )


def detect_geometry_fallback_tables(
    page: fitz.Page,
    text_lines: list[dict],
    excluded_header_footer_rects: list[fitz.Rect],
    drawings: list[dict],
    *,
    axis_geometry_index: dict | None = None,
    rule_alignment_tolerance_points: float,
    min_horizontal_rules: int,
    min_populated_rows: int,
    min_populated_columns: int,
    min_vertical_coverage_ratio: float,
    native_replacement_overlap_ratio: float,
    drawing_axis_rule: Callable[[dict, fitz.Rect], dict | None],
    merge_collinear_vertical_rules: Callable[[list[dict]], list[dict]],
    group_horizontal_rules_by_span: Callable[[list[dict]], list[list[dict]]],
    table_structure_dependencies: TableStructureDependencies,
) -> list[GeometryFallbackTable]:
    """Detect open-frame and cross-column tables from axis-rule geometry."""

    page_drawings = drawings
    has_synthetic_skeleton_inputs = any(
        drawing.get("_axis_skeleton_item")
        for drawing in page_drawings
    )

    if axis_geometry_index is not None and not has_synthetic_skeleton_inputs:
        index_by_payload_identity = {
            id(drawing): element_id
            for element_id, drawing in axis_geometry_index[
                "drawing_by_element_id"
            ].items()
        }
        allowed_ids = {
            index_by_payload_identity[id(drawing)]
            for drawing in page_drawings
            if id(drawing) in index_by_payload_identity
        }
        horizontal_rules = [
            {
                **rule,
                "element_ids": sorted(rule.get("element_ids", [])),
            }
            for rule in axis_geometry_index["logical_horizontal_rules"]
            if set(rule.get("element_ids", [])).issubset(allowed_ids)
        ]
        vertical_rules = [
            {
                **rule,
                "element_ids": [element_id],
            }
            for element_id, rule in axis_geometry_index[
                "legacy_axis_rule_by_element_id"
            ].items()
            if (
                element_id in allowed_ids
                and rule is not None
                and rule["orientation"] == "v"
            )
        ]
    else:
        axis_rules = [
            rule
            for drawing in page_drawings
            if (rule := drawing_axis_rule(drawing, page.rect)) is not None
        ]
        horizontal_rules = [
            rule for rule in axis_rules if rule["orientation"] == "h"
        ]
        vertical_rules = [
            rule for rule in axis_rules if rule["orientation"] == "v"
        ]

    horizontal_rules = merge_horizontal_rules_for_table_skeleton(
        horizontal_rules,
        dependencies=table_structure_dependencies,
    )
    merged_vertical_rules = merge_collinear_vertical_rules(vertical_rules)
    candidates: list[GeometryFallbackTable] = []
    tolerance = rule_alignment_tolerance_points

    for anchor_vertical in merged_vertical_rules:
        anchor_x = anchor_vertical["position"]
        anchor_rect = anchor_vertical["rect"]
        crossing_horizontal_rules = [
            rule
            for rule in horizontal_rules
            if (
                rule["rect"].x0 - tolerance
                <= anchor_x
                <= rule["rect"].x1 + tolerance
                and anchor_rect.y0 - tolerance
                <= rule["position"]
                <= anchor_rect.y1 + tolerance
            )
        ]
        span_groups = group_horizontal_rules_by_span(
            crossing_horizontal_rules
        )
        if not span_groups:
            continue

        dominant_span_group = max(span_groups, key=len)
        horizontal_positions = unique_rule_positions(
            dominant_span_group,
            tolerance=table_structure_dependencies.alignment_tolerance_points,
        )
        if len(horizontal_positions) < 3:
            continue

        x0 = statistics.median(
            rule["rect"].x0 for rule in dominant_span_group
        )
        x1 = statistics.median(
            rule["rect"].x1 for rule in dominant_span_group
        )
        candidate_bbox = fitz.Rect(
            x0,
            min(rule["rect"].y0 for rule in dominant_span_group),
            x1,
            max(rule["rect"].y1 for rule in dominant_span_group),
        )
        if rect_area(candidate_bbox) <= 0:
            continue

        candidate_area = rect_area(candidate_bbox)
        if any(
            band.contains(rect_center(candidate_bbox))
            or rect_area(candidate_bbox & band) / candidate_area >= 0.50
            for band in excluded_header_footer_rects
        ):
            continue

        separator_positions = sorted(
            rule["position"]
            for rule in merged_vertical_rules
            if (
                candidate_bbox.x0 + tolerance
                < rule["position"]
                < candidate_bbox.x1 - tolerance
                and (
                    max(
                        0.0,
                        min(candidate_bbox.y1, rule["rect"].y1)
                        - max(candidate_bbox.y0, rule["rect"].y0),
                    )
                    / candidate_bbox.height
                    >= min_vertical_coverage_ratio
                )
            )
        )
        if not separator_positions:
            continue

        data, populated_rows, populated_columns = extract_geometry_table_data(
            bbox=candidate_bbox,
            horizontal_positions=horizontal_positions,
            separator_positions=separator_positions,
            text_lines=text_lines,
        )

        internal_complex_drawing_count = 0
        for drawing in page_drawings:
            if not candidate_bbox.contains(
                rect_center(fitz.Rect(drawing["rect"]))
            ):
                continue
            if axis_geometry_index is None or has_synthetic_skeleton_inputs:
                drawing_rule = drawing_axis_rule(drawing, page.rect)
            else:
                element_id = index_by_payload_identity.get(id(drawing))
                drawing_rule = (
                    axis_geometry_index[
                        "legacy_axis_rule_by_element_id"
                    ].get(element_id)
                    if element_id is not None
                    else None
                )
            if drawing_rule is None:
                internal_complex_drawing_count += 1

        standard_text_grid = (
            len(horizontal_positions) >= min_horizontal_rules
            and populated_rows >= min_populated_rows
            and populated_columns >= min_populated_columns
        )
        two_row_visual_grid = (
            len(horizontal_positions) >= 3
            and len(separator_positions) >= 2
            and populated_rows >= 2
            and populated_columns >= 2
            and internal_complex_drawing_count > 0
        )
        if not standard_text_grid and not two_row_visual_grid:
            continue

        table = GeometryFallbackTable(
            bbox=candidate_bbox,
            data=data,
            metadata={
                "horizontal_rule_count": len(horizontal_positions),
                "separator_count": len(separator_positions),
                "merged_vertical_segment_count": sum(
                    rule["segment_count"]
                    for rule in merged_vertical_rules
                    if rule["position"] in separator_positions
                ),
                "populated_row_count": populated_rows,
                "populated_column_count": populated_columns,
                "standard_text_grid": standard_text_grid,
                "two_row_visual_grid": two_row_visual_grid,
                "internal_complex_drawing_count": (
                    internal_complex_drawing_count
                ),
            },
        )
        duplicate = next(
            (
                existing
                for existing in candidates
                if rect_area(
                    fitz.Rect(existing.bbox) & fitz.Rect(table.bbox)
                )
                / min(
                    rect_area(fitz.Rect(existing.bbox)),
                    rect_area(fitz.Rect(table.bbox)),
                )
                >= native_replacement_overlap_ratio
            ),
            None,
        )
        if duplicate is None:
            candidates.append(table)

    return candidates


def detect_simple_cluster_frame_tables(
    page: fitz.Page,
    text_lines: list[dict],
    excluded_header_footer_rects: list[fitz.Rect],
    cluster_bbox: fitz.Rect,
    drawings: list[dict],
    *,
    axis_geometry_index: dict | None,
    page_inventory: PageInventory | None,
    drawing_axis_rule: Callable[[dict, fitz.Rect], dict | None],
    group_horizontal_rules_by_span: Callable[[list[dict]], list[list[dict]]],
    rect_is_mostly_inside_any: Callable[[fitz.Rect, list[fitz.Rect]], bool],
    alignment_tolerance: float,
    table_structure_dependencies: TableStructureDependencies,
) -> list[GeometryFallbackTable]:
    """Detect small local table frames without scanning the full device drawing."""

    has_synthetic_skeleton_inputs = any(
        drawing.get("_axis_skeleton_item")
        for drawing in drawings
    )
    if (
        axis_geometry_index is not None
        and page_inventory is not None
        and not has_synthetic_skeleton_inputs
    ):
        allowed_ids = {
            element_id
            for drawing in drawings
            for element_id in [
                page_inventory.id_for_payload(drawing, "drawing")
            ]
            if element_id is not None
        }
        horizontal_rules = [
            {
                **rule,
                "element_ids": sorted(rule.get("element_ids", [])),
            }
            for rule in axis_geometry_index["logical_horizontal_rules"]
            if set(rule.get("element_ids", [])).issubset(allowed_ids)
        ]
        vertical_rules = [
            {
                **rule,
                "element_ids": [element_id],
            }
            for element_id, rule in axis_geometry_index[
                "legacy_axis_rule_by_element_id"
            ].items()
            if element_id in allowed_ids
            and rule is not None
            and rule["orientation"] == "v"
        ]
    else:
        axis_rules = [
            rule
            for drawing in drawings
            if (rule := drawing_axis_rule(drawing, page.rect)) is not None
        ]
        horizontal_rules = [
            rule for rule in axis_rules if rule["orientation"] == "h"
        ]
        vertical_rules = [
            rule for rule in axis_rules if rule["orientation"] == "v"
        ]

    candidates: list[GeometryFallbackTable] = []
    tolerance = alignment_tolerance
    for group in group_horizontal_rules_by_span(horizontal_rules):
        positions = unique_rule_positions(
            group,
            tolerance=table_structure_dependencies.alignment_tolerance_points,
        )
        if len(positions) < 2:
            continue
        x0 = max(
            cluster_bbox.x0 - tolerance,
            min(rule["rect"].x0 for rule in group),
        )
        x1 = min(
            cluster_bbox.x1 + tolerance,
            max(rule["rect"].x1 for rule in group),
        )
        y0, y1 = min(positions), max(positions)
        if x1 <= x0 or y1 <= y0:
            continue

        crossing_verticals = [
            rule
            for rule in vertical_rules
            if rule["rect"].y0 <= y0 + tolerance
            and rule["rect"].y1 >= y1 - tolerance
            and x0 - tolerance <= rule["position"] <= x1 + tolerance
        ]
        separator_positions = unique_rule_positions(
            crossing_verticals,
            tolerance=table_structure_dependencies.alignment_tolerance_points,
        )
        if len(separator_positions) < 2:
            continue

        candidate_bbox = fitz.Rect(x0, y0, x1, y1)
        if any(
            band.contains(rect_center(candidate_bbox))
            or rect_is_mostly_inside_any(candidate_bbox, [band])
            for band in excluded_header_footer_rects
        ):
            continue

        data, populated_rows, populated_columns = extract_geometry_table_data(
            bbox=candidate_bbox,
            horizontal_positions=positions,
            separator_positions=separator_positions,
            text_lines=text_lines,
        )
        if populated_rows < 1 or populated_columns < 2:
            continue

        candidates.append(
            GeometryFallbackTable(
                bbox=candidate_bbox,
                data=data,
                metadata={
                    "horizontal_rule_count": len(positions),
                    "separator_count": len(separator_positions),
                    "populated_row_count": populated_rows,
                    "populated_column_count": populated_columns,
                },
                detection_source="cluster_frame_fallback",
            )
        )
    return candidates
def detect_tables_by_drawing_clusters(
    page: fitz.Page,
    text_lines: list[dict],
    excluded_header_footer_rects: list[fitz.Rect],
    drawings: list[dict],
    page_inventory: PageInventory,
    axis_geometry_index: dict,
    *,
    cluster_table_drawing_budget: int,
    fallback_table_max_vertical_break_points: float,
    fallback_table_min_horizontal_rules: int,
    fallback_table_min_populated_columns: int,
    fallback_table_min_populated_rows: int,
    fallback_table_min_vertical_coverage_ratio: float,
    fallback_table_min_vertical_segment_points: float,
    fallback_table_native_expansion_area_ratio: float,
    fallback_table_native_replacement_overlap_ratio: float,
    fallback_table_rule_alignment_tolerance_points: float,
    logical_horizontal_rule_coordinate_decimal_places: int,
    table_precluster_dependencies: TablePreclusterDependencies,
    table_structure_dependencies: TableStructureDependencies,
    drawing_axis_rule: Callable[..., dict | None],
    group_horizontal_rules_by_span: Callable[..., list[list[dict]]],
    merge_collinear_vertical_rules: Callable[..., list[dict]],
    meaningful_overlap: Callable[..., bool],
    rect_is_mostly_inside_any: Callable[..., bool],
) -> tuple[list, dict, list[dict]]:
    """按初始空间簇检测原生表格和局部规则线表格。"""
    if len(drawings) <= cluster_table_drawing_budget:
        native_tables = page.find_tables(paths=drawings).tables
        for table in native_tables:
            table.source_cluster_bbox = fitz.Rect(page.rect)
        native_tables = [
            table
            for table in native_tables
            if not rect_is_mostly_inside_any(
                fitz.Rect(table.bbox),
                excluded_header_footer_rects,
            )
            and not any(
                band.contains(rect_center(fitz.Rect(table.bbox)))
                for band in excluded_header_footer_rects
            )
        ]
        fallback_tables = detect_geometry_fallback_tables(
            page=page,
            text_lines=text_lines,
            excluded_header_footer_rects=excluded_header_footer_rects,
            drawings=drawings,
            axis_geometry_index=axis_geometry_index,
            rule_alignment_tolerance_points=(
                fallback_table_rule_alignment_tolerance_points
            ),
            min_horizontal_rules=fallback_table_min_horizontal_rules,
            min_populated_rows=fallback_table_min_populated_rows,
            min_populated_columns=fallback_table_min_populated_columns,
            min_vertical_coverage_ratio=(
                fallback_table_min_vertical_coverage_ratio
            ),
            native_replacement_overlap_ratio=(
                fallback_table_native_replacement_overlap_ratio
            ),
            drawing_axis_rule=drawing_axis_rule,
            merge_collinear_vertical_rules=merge_collinear_vertical_rules,
            group_horizontal_rules_by_span=group_horizontal_rules_by_span,
            table_structure_dependencies=table_structure_dependencies,
        )
        for table in fallback_tables:
            table.source_cluster_bbox = fitz.Rect(page.rect)

        combined_tables = combine_native_and_fallback_tables(
            native_tables=native_tables,
            fallback_tables=fallback_tables,
            max_vertical_break_points=(
                fallback_table_max_vertical_break_points
            ),
            native_replacement_overlap_ratio=(
                fallback_table_native_replacement_overlap_ratio
            ),
            native_expansion_area_ratio=(
                fallback_table_native_expansion_area_ratio
            ),
        )
        combined_tables = deduplicate_overlapping_table_candidates(
            combined_tables
        )

        combined_tables = sorted(
            combined_tables,
            key=lambda table: (
                fitz.Rect(table.bbox).y0,
                fitz.Rect(table.bbox).x0,
            ),
        )

        whole_page_input_ids = page_inventory.ids_for_payloads(
            drawings,
            "drawing",
        )
        for table in combined_tables:
            table.claim_input_element_ids = list(
                whole_page_input_ids
            )

        claims = build_table_claims(
            page_number=page.number + 1,
            tables=combined_tables,
            page_inventory=page_inventory,
            detection_mode="whole_page_under_budget",
            confidence_fn=table_claim_confidence,
        )

        return combined_tables, {
            "mode": "whole_page_under_budget",
            "cluster_count": 1,
            "tested_drawing_count": len(drawings),
            "budget_limited_cluster_count": 0,
            "non_table_early_stop_count": 0,
            "early_stopped_cluster_count": 0,
        }, claims

    clusters = build_pre_table_drawing_clusters(
        page=page,
        excluded_header_footer_rects=excluded_header_footer_rects,
        drawings=drawings,
        page_inventory=page_inventory,
        dependencies=table_precluster_dependencies,
    )
    detected_tables: list = []
    budget_limited_cluster_count = 0
    non_table_early_stop_count = 0
    tested_drawing_count = 0
    skeleton_selected_cluster_count = 0
    skeleton_selected_drawing_count = 0
    skeleton_component_count = 0
    skeleton_selection_modes: Counter = Counter()

    for cluster in clusters:
        cluster_bbox = fitz.Rect(cluster["bbox"])
        cluster_drawings = cluster["drawings"]
        if len(cluster_drawings) > cluster_table_drawing_budget:
            skeleton_selection = (
                select_cluster_table_skeleton_drawings(
                    cluster_drawings=cluster_drawings,
                    cluster_bbox=cluster_bbox,
                    axis_geometry_index=axis_geometry_index,
                    page_inventory=page_inventory,
                    drawing_budget=cluster_table_drawing_budget,
                    alignment_tolerance=(
                        fallback_table_rule_alignment_tolerance_points
                    ),
                    minimum_structural_span=(
                        fallback_table_min_vertical_segment_points
                    ),
                    coordinate_decimal_places=(
                        logical_horizontal_rule_coordinate_decimal_places
                    ),
                )
            )
            tested_drawings = skeleton_selection["drawings"]
            if tested_drawings:
                skeleton_selected_cluster_count += 1
                skeleton_selected_drawing_count += len(
                    tested_drawings
                )
                skeleton_component_count += len(
                    skeleton_selection["skeleton_components"]
                )
                skeleton_selection_modes[
                    skeleton_selection["mode"]
                ] += 1
            else:
                tested_drawings = select_cluster_table_budget(
                    cluster_drawings,
                    drawing_budget=cluster_table_drawing_budget,
                )
                skeleton_selection_modes[
                    "legacy_spatial_budget_fallback"
                ] += 1
            budget_limited_cluster_count += 1
        else:
            tested_drawings = cluster_drawings
            skeleton_selection_modes["whole_cluster_under_budget"] += 1
        tested_drawing_count += len(tested_drawings)

        native_tables = page.find_tables(paths=tested_drawings).tables
        native_tables = [
            table
            for table in native_tables
            if meaningful_overlap(
                cluster_bbox,
                fitz.Rect(table.bbox),
            )
            and not rect_is_mostly_inside_any(
                fitz.Rect(table.bbox),
                excluded_header_footer_rects,
            )
        ]
        fallback_tables = detect_geometry_fallback_tables(
            page=page,
            text_lines=text_lines,
            excluded_header_footer_rects=excluded_header_footer_rects,
            drawings=tested_drawings,
            axis_geometry_index=None,
            rule_alignment_tolerance_points=(
                fallback_table_rule_alignment_tolerance_points
            ),
            min_horizontal_rules=fallback_table_min_horizontal_rules,
            min_populated_rows=fallback_table_min_populated_rows,
            min_populated_columns=fallback_table_min_populated_columns,
            min_vertical_coverage_ratio=(
                fallback_table_min_vertical_coverage_ratio
            ),
            native_replacement_overlap_ratio=(
                fallback_table_native_replacement_overlap_ratio
            ),
            drawing_axis_rule=drawing_axis_rule,
            merge_collinear_vertical_rules=merge_collinear_vertical_rules,
            group_horizontal_rules_by_span=group_horizontal_rules_by_span,
            table_structure_dependencies=table_structure_dependencies,
        )
        fallback_tables.extend(
            detect_simple_cluster_frame_tables(
                page=page,
                text_lines=text_lines,
                excluded_header_footer_rects=excluded_header_footer_rects,
                cluster_bbox=cluster_bbox,
                drawings=tested_drawings,
                axis_geometry_index=None,
                page_inventory=None,
                drawing_axis_rule=drawing_axis_rule,
                group_horizontal_rules_by_span=(
                    group_horizontal_rules_by_span
                ),
                rect_is_mostly_inside_any=rect_is_mostly_inside_any,
                alignment_tolerance=(
                    fallback_table_rule_alignment_tolerance_points
                ),
                table_structure_dependencies=table_structure_dependencies,
            )
        )
        for table in [*native_tables, *fallback_tables]:
            table.source_cluster_bbox = fitz.Rect(cluster_bbox)
        cluster_tables = combine_native_and_fallback_tables(
            native_tables=native_tables,
            fallback_tables=fallback_tables,
            max_vertical_break_points=(
                fallback_table_max_vertical_break_points
            ),
            native_replacement_overlap_ratio=(
                fallback_table_native_replacement_overlap_ratio
            ),
            native_expansion_area_ratio=(
                fallback_table_native_expansion_area_ratio
            ),
        )
        cluster_input_ids = sorted(
            {
                element_id
                for drawing in tested_drawings
                for element_id in (
                    drawing.get("_source_element_ids", [])
                    or [
                        page_inventory.id_for_payload(
                            drawing,
                            "drawing",
                        )
                    ]
                )
                if element_id is not None
            }
        )
        for table in cluster_tables:
            table.claim_input_element_ids = list(
                cluster_input_ids
            )
        if not cluster_tables and len(cluster_drawings) > cluster_table_drawing_budget:
            non_table_early_stop_count += 1
            continue
        detected_tables.extend(cluster_tables)

    deduplicated = deduplicate_overlapping_table_candidates(
        detected_tables
    )

    final_tables = sorted(
        deduplicated,
        key=lambda item: (
            fitz.Rect(item.bbox).y0,
            fitz.Rect(item.bbox).x0,
        ),
    )

    claims = build_table_claims(
        page_number=page.number + 1,
        tables=final_tables,
        page_inventory=page_inventory,
        detection_mode="cluster_budgeted",
        confidence_fn=table_claim_confidence,
    )

    return final_tables, {
        "mode": "cluster_budgeted",
        "cluster_count": len(clusters),
        "tested_drawing_count": tested_drawing_count,
        "skeleton_selected_cluster_count": (
            skeleton_selected_cluster_count
        ),
        "skeleton_selected_drawing_count": (
            skeleton_selected_drawing_count
        ),
        "skeleton_component_count": skeleton_component_count,
        "skeleton_selection_modes": dict(skeleton_selection_modes),
        "budget_limited_cluster_count": budget_limited_cluster_count,
        "non_table_early_stop_count": non_table_early_stop_count,
        # "early_stopped_cluster_count": early_stopped_cluster_count,
        "early_stopped_cluster_count": non_table_early_stop_count,
    }, claims


@dataclass(frozen=True)
class TablePipelineDependencies:
    metrics: DocumentRunMetrics
    tables_dir: Path
    cluster_table_drawing_budget: int
    fallback_table_max_vertical_break_points: float
    fallback_table_min_horizontal_rules: int
    fallback_table_min_populated_columns: int
    fallback_table_min_populated_rows: int
    fallback_table_min_vertical_coverage_ratio: float
    fallback_table_min_vertical_segment_points: float
    fallback_table_native_expansion_area_ratio: float
    fallback_table_native_replacement_overlap_ratio: float
    fallback_table_rule_alignment_tolerance_points: float
    logical_horizontal_rule_coordinate_decimal_places: int
    max_small_table_cells: int
    min_direction_text_chars: int
    rotation_dominance_threshold: float
    math_font_hints: Sequence[str]
    drawing_axis_rule: Callable[..., Any]
    table_precluster_dependencies: TablePreclusterDependencies
    table_structure_dependencies: TableStructureDependencies
    group_horizontal_rules_by_span: Callable[..., list[list[dict]]]
    merge_collinear_vertical_rules: Callable[..., list[dict]]
    meaningful_overlap: Callable[..., bool]
    rect_is_mostly_inside_any: Callable[..., bool]
    attach_empty_adjacent_rule_bands_to_tables: Callable[..., dict]


def detect_page_tables(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    dependencies: TablePipelineDependencies,
) -> PageTableDetectionResult:
    """Run page-level native and geometry-fallback table detection."""

    detection_started = time.perf_counter()
    gate_started = time.perf_counter()
    rule_band_result = page_context.preclassified_rule_band_result
    (
        page_context.detected_tables,
        page_context.table_detection_metadata,
        page_context.table_claims,
    ) = detect_tables_by_drawing_clusters(
        page=page,
        text_lines=page_context.text_lines,
        excluded_header_footer_rects=page_context.vector_exclusion_rects,
        drawings=rule_band_result["table_detection_drawings"],
        page_inventory=page_context.page_inventory,
        axis_geometry_index=page_context.axis_geometry_index,
        cluster_table_drawing_budget=(
            dependencies.cluster_table_drawing_budget
        ),
        fallback_table_max_vertical_break_points=(
            dependencies.fallback_table_max_vertical_break_points
        ),
        fallback_table_min_horizontal_rules=(
            dependencies.fallback_table_min_horizontal_rules
        ),
        fallback_table_min_populated_columns=(
            dependencies.fallback_table_min_populated_columns
        ),
        fallback_table_min_populated_rows=(
            dependencies.fallback_table_min_populated_rows
        ),
        fallback_table_min_vertical_coverage_ratio=(
            dependencies.fallback_table_min_vertical_coverage_ratio
        ),
        fallback_table_min_vertical_segment_points=(
            dependencies.fallback_table_min_vertical_segment_points
        ),
        fallback_table_native_expansion_area_ratio=(
            dependencies.fallback_table_native_expansion_area_ratio
        ),
        fallback_table_native_replacement_overlap_ratio=(
            dependencies.fallback_table_native_replacement_overlap_ratio
        ),
        fallback_table_rule_alignment_tolerance_points=(
            dependencies.fallback_table_rule_alignment_tolerance_points
        ),
        logical_horizontal_rule_coordinate_decimal_places=(
            dependencies.logical_horizontal_rule_coordinate_decimal_places
        ),
        table_precluster_dependencies=(
            dependencies.table_precluster_dependencies
        ),
        table_structure_dependencies=(
            dependencies.table_structure_dependencies
        ),
        drawing_axis_rule=dependencies.drawing_axis_rule,
        group_horizontal_rules_by_span=(
            dependencies.group_horizontal_rules_by_span
        ),
        merge_collinear_vertical_rules=(
            dependencies.merge_collinear_vertical_rules
        ),
        meaningful_overlap=dependencies.meaningful_overlap,
        rect_is_mostly_inside_any=dependencies.rect_is_mostly_inside_any,
    )

    gate_seconds = time.perf_counter() - gate_started
    dependencies.metrics.table_detection_gate_seconds += gate_seconds
    dense_vector_page = (
        len(page_context.drawings)
        > dependencies.cluster_table_drawing_budget
    )
    detection_skipped = (
        page_context.table_detection_metadata[
            "budget_limited_cluster_count"
        ]
        > 0
    )
    skip_reason = (
        "cluster_drawing_budget_applied" if detection_skipped else None
    )
    page_context.dense_vector_page = dense_vector_page
    page_context.table_detection_skipped = detection_skipped
    page_context.table_detection_skip_reason = skip_reason
    detection_seconds = time.perf_counter() - detection_started
    dependencies.metrics.table_detection_seconds += detection_seconds
    page_context.record_timing("table_detection_gate", gate_seconds)
    page_context.record_timing("table_detection", detection_seconds)
    return PageTableDetectionResult(
        detected_tables=page_context.detected_tables,
        metadata=page_context.table_detection_metadata,
        table_claims=page_context.table_claims,
        gate_seconds=gate_seconds,
        detection_seconds=detection_seconds,
        dense_vector_page=dense_vector_page,
        detection_skipped=detection_skipped,
        skip_reason=skip_reason,
    )


def build_page_table_candidates(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    force_pure_text_table_to_markdown: bool,
    dependencies: TablePipelineDependencies,
) -> PageTableCandidateBuildResult:
    """Build stable page-local table candidates from detected tables."""

    page_context.table_markdown_elements = []
    page_context.table_records = []
    page_context.embedded_table_count = 0
    page_context.auto_rotated_table_count = 0
    page_context.anomalous_table_claimed_count = 0
    page_context.anomalous_table_unclaimed_count = 0
    page_context.excluded_table_rects = []
    page_context.table_candidates = []

    for table_number, table in enumerate(
        page_context.detected_tables,
        start=1,
    ):
        extracted_raw = table.extract()
        raw_row_count = len(extracted_raw)
        raw_column_count = max(
            (len(row) for row in extracted_raw),
            default=0,
        )
        nonempty_cell_count = count_nonempty_table_cells(extracted_raw)
        is_anomalous_single_value_table = nonempty_cell_count == 1
        table_cell_count = raw_row_count * raw_column_count
        exceeds_threshold = (
            table_cell_count > dependencies.max_small_table_cells
        )
        (
            table_has_complex_visual_content,
            table_visual_metadata,
        ) = table_contains_complex_visual_content(
            table_bbox=fitz.Rect(table.bbox),
            page_dict=page_context.page_dict,
            drawings=page_context.drawings,
            page_rect=page.rect,
            drawing_axis_rule=dependencies.drawing_axis_rule,
            math_font_hints=dependencies.math_font_hints,
        )
        force_markdown_table = (
            force_pure_text_table_to_markdown
            and not table_has_complex_visual_content
        )
        should_render_table_image = (
            not force_markdown_table
            and (exceeds_threshold or table_has_complex_visual_content)
        )
        rotation_info = infer_table_rotation(
            text_lines=page_context.text_lines,
            table_bbox=table.bbox,
            minimum_direction_text_chars=(
                dependencies.min_direction_text_chars
            ),
            rotation_dominance_threshold=(
                dependencies.rotation_dominance_threshold
            ),
        )
        rotation_degrees = rotation_info["suggested_rotation_degrees"]
        extracted_reflowed = rotate_table_data(
            table=extracted_raw,
            rotation_degrees=rotation_degrees,
        )
        reflowed_row_count = len(extracted_reflowed)
        reflowed_column_count = max(
            (len(row) for row in extracted_reflowed),
            default=0,
        )
        csv_name = (
            f"page_{page_number:03d}_table_{table_number:02d}"
            f"_rotated_{rotation_degrees:+d}.csv"
        )
        structurally_valid_table, table_structure_audit = (
            classify_table_candidate_structure(
                extracted_raw=extracted_raw,
                table_visual_metadata=table_visual_metadata,
                raw_row_count=raw_row_count,
                raw_column_count=raw_column_count,
                nonempty_cell_count=nonempty_cell_count,
            )
        )
        page_context.table_candidates.append(
            {
                "table_number": table_number,
                "table": table,
                "bbox": fitz.Rect(table.bbox),
                "extracted_raw": extracted_raw,
                "raw_row_count": raw_row_count,
                "raw_column_count": raw_column_count,
                "nonempty_cell_count": nonempty_cell_count,
                "is_anomalous_single_value_table": (
                    is_anomalous_single_value_table
                ),
                "table_cell_count": table_cell_count,
                "exceeds_threshold": exceeds_threshold,
                "table_has_complex_visual_content": (
                    table_has_complex_visual_content
                ),
                "source_cluster_bbox": fitz.Rect(
                    getattr(table, "source_cluster_bbox", page.rect)
                ),
                "table_visual_metadata": table_visual_metadata,
                "force_markdown_table": force_markdown_table,
                "should_render_table_image": should_render_table_image,
                "will_emit_if_unclaimed": (
                    should_render_table_image
                    or force_markdown_table
                    or is_anomalous_single_value_table
                ),
                "rotation_info": rotation_info,
                "rotation_degrees": rotation_degrees,
                "extracted_reflowed": extracted_reflowed,
                "reflowed_row_count": reflowed_row_count,
                "reflowed_column_count": reflowed_column_count,
                "csv_name": csv_name,
                "csv_path": dependencies.tables_dir / csv_name,
                "structurally_valid_table": structurally_valid_table,
                "table_structure_audit": table_structure_audit,
                "table_owns_full_visual_bbox": structurally_valid_table,
            }
        )

    return PageTableCandidateBuildResult(
        candidates=page_context.table_candidates,
        detected_table_count=len(page_context.detected_tables),
    )


def finalize_page_table_candidates(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    rule_bands: list[dict],
    page_inventory_seconds: float,
    dependencies: TablePipelineDependencies,
) -> PageTableCandidateFinalization:
    """Finalize table candidates and configure their structural claims."""

    claims_by_id = {
        claim["claim_id"]: claim
        for claim in page_context.table_claims
    }
    page_context.table_skeleton_drawing_ids = set()
    page_context.table_full_drawing_ids = set()
    active_candidates: list[dict] = []

    for candidate in page_context.table_candidates:
        claim_id = f"page#{page_number}:table#{candidate['table_number']}"
        claim = claims_by_id.get(claim_id)
        if claim is None:
            continue
        if not candidate.get("structurally_valid_table", False):
            claim["claimed_element_ids"] = []
            claim["feature_element_ids"] = []
            claim["claimed_count"] = 0
            claim["feature_count"] = 0
            claim["metadata"]["suppressed_before_ownership"] = True
            claim["metadata"]["table_structure_audit"] = candidate.get(
                "table_structure_audit", {}
            )
            claim["reason_codes"] = sorted(
                set(claim.get("reason_codes", [])).union(
                    {"rejected_sparse_non_table_candidate"}
                )
            )
            claim["output_artifacts"] = []
            candidate["suppressed_before_ownership"] = True
            continue

        refine_table_claim_confidence(
            claim=claim,
            table_candidate=candidate,
        )
        container_result = configure_table_container_claim(
            claim=claim,
            candidate=candidate,
            page_inventory=page_context.page_inventory,
            page_rect=page.rect,
            dependencies=dependencies.table_structure_dependencies,
        )
        page_context.table_skeleton_drawing_ids.update(
            container_result["skeleton_drawing_ids"]
        )
        page_context.table_full_drawing_ids.update(
            element_id
            for element_id in claim.get("claimed_element_ids", [])
            if element_id.startswith("d:")
        )
        active_candidates.append(candidate)

    page_context.table_candidates = active_candidates
    adjacent_attachment = (
        dependencies.attach_empty_adjacent_rule_bands_to_tables(
            page_rect=page.rect,
            bands=rule_bands,
            table_candidates=page_context.table_candidates,
            table_claims_by_id=claims_by_id,
            page_inventory=page_context.page_inventory,
        )
    )
    page_context.preclassified_rule_band_result["metadata"][
        "adjacent_table_attachment"
    ] = adjacent_attachment

    for display_number, candidate in enumerate(
        page_context.table_candidates,
        start=1,
    ):
        candidate["display_table_number"] = display_number
        candidate["csv_name"] = (
            f"page_{page_number:03d}_table_{display_number:02d}"
            f"_rotated_{candidate['rotation_degrees']:+d}.csv"
        )
        candidate["csv_path"] = dependencies.tables_dir / candidate["csv_name"]

    page_context.inventory_audit_record = {
        **page_context.page_inventory.to_audit_record(),
        "build_seconds": round(page_inventory_seconds, 6),
    }
    return PageTableCandidateFinalization(
        candidates=page_context.table_candidates,
        claims_by_id=claims_by_id,
        skeleton_drawing_ids=page_context.table_skeleton_drawing_ids,
        full_drawing_ids=page_context.table_full_drawing_ids,
        adjacent_rule_band_attachment=adjacent_attachment,
    )
