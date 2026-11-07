"""定义表格结构、候选和Claim策略。

模块提供规则位置聚合、骨架碎片合并、纯
几何单元格提取、原生/补检兼容对象、候选验证、候选合并以及逻辑表格容器
Claim配置；它不执行跨检测器仲裁，也不渲染表格制品。

关键不变量:
    * 外层表格骨架与单元格内部drawing分离。
    * 全截图表格可以拥有bbox内部视觉内容。
    * 父容器表格为下游保留子视觉内容。
    * 候选置信度用于记录证据，不能替代拓扑结构判断。

Define table structure, candidate, and Claim policies.

Provides rule-position grouping, skeleton-fragment merging, geometry-only
cell extraction, native/fallback compatibility objects, candidate validation,
candidate merging, and logical table-container Claim configuration. It does
not perform cross-detector arbitration or render table artifacts.

Key invariants:
    * Outer table skeletons are separated from cell-internal drawings.
    * Full-screenshot tables may own visual content inside their bbox.
    * Parent-container tables leave child visual content available downstream.
    * Candidate confidence records evidence but does not replace topology.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import fitz

from .claims import create_claim_manifest
from .geometry import as_fitz_rect, rect_area, rect_center, union_rects
from .inventory import PageInventory


@dataclass(frozen=True)
class TableStructureDependencies:
    alignment_tolerance_points: float
    skeleton_horizontal_gap_points: float
    drawing_axis_rule: Callable[[dict, fitz.Rect], dict | None]


def unique_rule_positions(
    rules: list[dict],
    *,
    tolerance: float = 3.0,
) -> list[float]:
    """Return sorted horizontal/vertical rule positions after tolerance grouping."""

    groups: list[list[float]] = []
    for position in sorted(rule["position"] for rule in rules):
        if groups and position - statistics.median(groups[-1]) <= tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [statistics.median(group) for group in groups]


def merge_horizontal_rules_for_table_skeleton(
    horizontal_rules: list[dict],
    *,
    dependencies: TableStructureDependencies,
) -> list[dict]:
    """Merge aligned horizontal fragments into logical table skeleton rules."""

    tolerance = dependencies.alignment_tolerance_points
    maximum_gap = dependencies.skeleton_horizontal_gap_points
    y_groups: list[list[dict]] = []
    for rule in sorted(
        horizontal_rules,
        key=lambda item: (item["position"], item["rect"].x0),
    ):
        matching_group = next(
            (
                group
                for group in y_groups
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
            y_groups.append([rule])
        else:
            matching_group.append(rule)

    merged: list[dict] = []

    def rule_element_ids(rule: dict) -> set[str]:
        if rule.get("element_ids") is not None:
            return {
                element_id
                for element_id in rule.get("element_ids", [])
                if element_id is not None
            }
        element_id = rule.get("element_id")
        return {element_id} if element_id is not None else set()

    for group in y_groups:
        current: list[dict] = []

        def flush() -> None:
            if not current:
                return
            source_rects = [
                fitz.Rect(item["rect"]).normalize()
                for item in current
            ]
            merged.append(
                {
                    "orientation": "h",
                    "rect": union_rects(source_rects),
                    "position": statistics.median(
                        item["position"] for item in current
                    ),
                    "element_ids": sorted(
                        set().union(
                            *(rule_element_ids(item) for item in current)
                        )
                    ),
                    "source_rects": source_rects,
                    "segment_count": sum(
                        item.get("segment_count", 1)
                        for item in current
                    ),
                    "logical": True,
                }
            )

        for rule in sorted(group, key=lambda item: item["rect"].x0):
            if current:
                current_x1 = max(item["rect"].x1 for item in current)
                if rule["rect"].x0 > current_x1 + maximum_gap:
                    flush()
                    current = []
            current.append(rule)
        flush()
    return merged


def table_cell_boundary_positions(
    table: object,
) -> tuple[list[float], list[float]]:
    """Return unique X/Y positions from a native table's cell rectangles."""

    cells = getattr(table, "cells", None) or []
    x_positions: list[float] = []
    y_positions: list[float] = []
    for cell in cells:
        if not cell:
            continue
        rect = fitz.Rect(cell).normalize()
        x_positions.extend([rect.x0, rect.x1])
        y_positions.extend([rect.y0, rect.y1])
    return sorted(set(x_positions)), sorted(set(y_positions))


def drawing_is_table_container_structure(
    drawing: dict,
    table: object,
    table_bbox: fitz.Rect,
    page_rect: fitz.Rect,
    *,
    dependencies: TableStructureDependencies,
) -> bool:
    """Return whether a drawing is part of the table's outer/container structure."""

    tolerance = dependencies.alignment_tolerance_points
    drawing_rect = fitz.Rect(drawing["rect"]).normalize()
    if (drawing_rect & table_bbox).is_empty:
        return False

    x_boundaries, y_boundaries = table_cell_boundary_positions(table)

    def near(value: float, positions: list[float]) -> bool:
        return any(abs(value - position) <= tolerance for position in positions)

    axis_rule = dependencies.drawing_axis_rule(drawing, page_rect)
    if axis_rule is not None:
        rule_rect = as_fitz_rect(axis_rule["rect"]).normalize()
        if axis_rule["orientation"] == "h":
            aligned = not y_boundaries or near(
                axis_rule["position"], y_boundaries
            )
            projected = max(
                0.0,
                min(rule_rect.x1, table_bbox.x1)
                - max(rule_rect.x0, table_bbox.x0),
            )
            return aligned and projected > 0.0

        aligned = not x_boundaries or near(
            axis_rule["position"], x_boundaries
        )
        projected = max(
            0.0,
            min(rule_rect.y1, table_bbox.y1)
            - max(rule_rect.y0, table_bbox.y0),
        )
        return aligned and projected > 0.0

    items = drawing.get("items", [])
    if (
        drawing.get("fill") is None
        or not items
        or not x_boundaries
        or not y_boundaries
        or any(item[0] != "re" for item in items)
    ):
        return False

    for item in items:
        item_rect = fitz.Rect(item[1]).normalize()
        if not (
            near(item_rect.x0, x_boundaries)
            and near(item_rect.x1, x_boundaries)
            and near(item_rect.y0, y_boundaries)
            and near(item_rect.y1, y_boundaries)
        ):
            return False
    return True


def configure_table_container_claim(
    claim: dict,
    candidate: dict,
    page_inventory: PageInventory,
    page_rect: fitz.Rect,
    *,
    dependencies: TableStructureDependencies,
) -> dict:
    """Configure table container ownership and return its structure audit."""

    table = candidate["table"]
    table_bbox = fitz.Rect(candidate["bbox"]).normalize()
    original_claimed_ids = [
        element_id
        for element_id in claim.get("claimed_element_ids", [])
        if element_id in page_inventory.entries
        and page_inventory.entry(element_id)["kind"] == "drawing"
    ]
    skeleton_ids: list[str] = []
    internal_visual_ids: list[str] = []
    for element_id in original_claimed_ids:
        drawing = page_inventory.payload_of(element_id)
        if drawing_is_table_container_structure(
            drawing=drawing,
            table=table,
            table_bbox=table_bbox,
            page_rect=page_rect,
            dependencies=dependencies,
        ):
            skeleton_ids.append(element_id)
        else:
            internal_visual_ids.append(element_id)

    logically_tabular = bool(skeleton_ids) or (
        candidate.get("raw_row_count", 0) >= 2
        and candidate.get("raw_column_count", 0) >= 2
        and candidate.get("nonempty_cell_count", 0) >= 2
    )
    container_id: str | None = None
    if logically_tabular:
        container_id = page_inventory.register_table_container(
            table_number=candidate["table_number"],
            bbox=table_bbox,
            payload={
                "table_number": candidate["table_number"],
                "row_count": candidate.get("raw_row_count", 0),
                "column_count": candidate.get("raw_column_count", 0),
            },
        )
        table_owns_full_bbox = bool(
            candidate.get("table_owns_full_visual_bbox", False)
        )
        if table_owns_full_bbox:
            full_content_ids = page_inventory.ids_touching_rect(
                rect=table_bbox,
                kinds=("drawing", "image"),
            )
            structural_ids = sorted({container_id, *full_content_ids})
            internal_visual_ids = []
        else:
            structural_ids = sorted({container_id, *skeleton_ids})
        claim["claimed_element_ids"] = structural_ids
        claim["feature_element_ids"] = structural_ids
        claim["claimed_count"] = len(structural_ids)
        claim["feature_count"] = len(structural_ids)
        claim["input_element_ids"] = sorted(
            set(claim.get("input_element_ids", [])).union({container_id})
        )
        claim["input_count"] = len(claim["input_element_ids"])

    claim["metadata"]["logical_table_container"] = logically_tabular
    claim["metadata"]["table_container_element_id"] = container_id
    claim["metadata"]["table_skeleton_drawing_ids"] = sorted(skeleton_ids)
    claim["metadata"]["table_owns_full_visual_bbox"] = bool(
        candidate.get("table_owns_full_visual_bbox", False)
    )
    claim["metadata"]["table_render_mode"] = (
        "full_table_screenshot"
        if candidate.get("table_owns_full_visual_bbox", False)
        else "parent_container_with_child_visuals"
    )
    claim["metadata"]["cell_internal_visual_drawing_ids"] = sorted(
        internal_visual_ids
    )
    claim["metadata"]["original_bbox_drawing_ids"] = sorted(
        original_claimed_ids
    )
    claim["reason_codes"] = sorted(
        set(claim.get("reason_codes", [])).union(
            (
                {
                    "logical_table_container",
                    "table_skeleton_separated_from_cell_visuals",
                }
                if logically_tabular
                else {"no_logical_table_container"}
            )
        )
    )

    candidate["logical_table_container"] = logically_tabular
    candidate["table_container_element_id"] = container_id
    candidate["table_skeleton_drawing_ids"] = sorted(skeleton_ids)
    candidate["cell_internal_visual_drawing_ids"] = sorted(internal_visual_ids)
    candidate["child_visual_claim_ids"] = []
    return {
        "logical_table_container": logically_tabular,
        "container_element_id": container_id,
        "skeleton_drawing_ids": sorted(skeleton_ids),
        "internal_visual_drawing_ids": sorted(internal_visual_ids),
    }


def refine_table_claim_confidence(
    claim: dict,
    table_candidate: dict,
) -> None:
    """Record table-local evidence while preserving detector confidence."""

    grid_cell_count = max(1, int(table_candidate.get("table_cell_count", 0)))
    nonempty_cell_count = int(table_candidate.get("nonempty_cell_count", 0))
    visual_metadata = table_candidate.get("table_visual_metadata", {})
    image_block_count = int(visual_metadata.get("image_block_count", 0))
    non_rule_drawing_count = int(
        visual_metadata.get("non_rule_drawing_count", 0)
    )
    semantic_cell_count = min(
        grid_cell_count,
        nonempty_cell_count + image_block_count,
    )
    semantic_fill_ratio = semantic_cell_count / grid_cell_count
    non_rule_pressure = min(1.0, non_rule_drawing_count / grid_cell_count)
    source_confidence = float(claim.get("confidence", 0.0))
    claim["metadata"]["source_confidence"] = round(source_confidence, 6)
    claim["metadata"]["semantic_fill_ratio"] = round(
        semantic_fill_ratio, 6
    )
    claim["metadata"]["non_rule_drawing_pressure"] = round(
        non_rule_pressure, 6
    )
    claim["metadata"]["local_evidence_confidence"] = round(
        source_confidence, 6
    )
    claim["confidence"] = round(source_confidence, 6)
    claim["metadata"]["confidence_used_for_ownership"] = False


def deduplicate_overlapping_table_candidates(
    tables: list[Any],
    overlap_ratio: float = 0.80,
) -> list[Any]:
    """Remove table candidates substantially covered by larger candidates."""

    deduplicated: list[Any] = []
    for table in sorted(
        tables,
        key=lambda item: rect_area(fitz.Rect(item.bbox)),
        reverse=True,
    ):
        table_rect = fitz.Rect(table.bbox)
        table_area = rect_area(table_rect)
        if table_area <= 0:
            deduplicated.append(table)
            continue

        duplicate = False
        for existing in deduplicated:
            existing_rect = fitz.Rect(existing.bbox)
            denominator = min(
                table_area,
                rect_area(existing_rect),
            )
            if denominator <= 0:
                continue
            if (
                rect_area(table_rect & existing_rect) / denominator
                >= overlap_ratio
            ):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(table)
    return deduplicated


def table_claim_confidence(table: object) -> float:
    """Return the initial table Claim confidence for a detector source."""

    detection_source = getattr(
        table,
        "detection_source",
        "pymupdf_native",
    )
    return {
        "pymupdf_native": 0.95,
        "geometry_fallback": 0.85,
        "cluster_frame_fallback": 0.75,
    }.get(detection_source, 0.70)


def table_contains_complex_visual_content(
    table_bbox: fitz.Rect,
    page_dict: dict,
    drawings: Sequence[dict],
    page_rect: fitz.Rect,
    *,
    drawing_axis_rule: Callable[[dict, fitz.Rect], object | None],
    math_font_hints: Iterable[str],
    image_overlap_threshold: float = 0.10,
    math_text_overlap_threshold: float = 0.50,
) -> tuple[bool, dict]:
    """Return whether a table contains visual content requiring a rendered image."""

    image_block_count = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1 or "bbox" not in block:
            continue
        block_rect = fitz.Rect(block["bbox"])
        intersection = block_rect & table_bbox
        if rect_area(intersection) <= 0:
            continue
        if (
            rect_area(intersection) / max(rect_area(block_rect), 1e-9)
            >= image_overlap_threshold
        ):
            image_block_count += 1

    non_rule_drawing_count = 0
    for drawing in drawings:
        drawing_rect = fitz.Rect(drawing.get("rect", ())).normalize()
        intersection = drawing_rect & table_bbox
        if rect_area(intersection) <= 0:
            continue
        if (
            rect_area(intersection) / max(rect_area(drawing_rect), 1e-9)
            < image_overlap_threshold
        ):
            continue
        if drawing_axis_rule(drawing, page_rect) is not None:
            continue
        non_rule_drawing_count += 1

    math_font_line_count = 0
    normalized_hints = tuple(str(hint).lower() for hint in math_font_hints)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_rect = fitz.Rect(line.get("bbox", ()))
            intersection = line_rect & table_bbox
            if rect_area(intersection) <= 0:
                continue
            if (
                rect_area(intersection) / max(rect_area(line_rect), 1e-9)
                < math_text_overlap_threshold
            ):
                continue
            if any(
                any(
                    hint in span.get("font", "").lower()
                    for hint in normalized_hints
                )
                for span in line.get("spans", [])
            ):
                math_font_line_count += 1

    metadata = {
        "image_block_count": image_block_count,
        "non_rule_drawing_count": non_rule_drawing_count,
        "math_font_line_count": math_font_line_count,
    }
    return (
        image_block_count > 0
        or non_rule_drawing_count > 0
        or math_font_line_count > 0,
        metadata,
    )


def classify_table_candidate_structure(
    extracted_raw: list[list[str | None]],
    table_visual_metadata: dict,
    raw_row_count: int,
    raw_column_count: int,
    nonempty_cell_count: int,
) -> tuple[bool, dict]:
    """Classify whether extracted cells form a minimally useful table structure."""

    row_occupancies = [
        sum(bool(str(cell).strip()) for cell in row if cell is not None)
        for row in extracted_raw
    ]
    column_occupancies = [
        sum(
            bool(str(row[column_index]).strip())
            if column_index < len(row) and row[column_index] is not None
            else False
            for row in extracted_raw
        )
        for column_index in range(raw_column_count)
    ]
    occupied_row_count = sum(value > 0 for value in row_occupancies)
    occupied_column_count = sum(value > 0 for value in column_occupancies)
    multi_value_row_count = sum(value >= 2 for value in row_occupancies)
    image_block_count = int(table_visual_metadata.get("image_block_count", 0))

    if raw_column_count <= 1:
        structurally_valid = occupied_row_count >= 2 or image_block_count > 0
        rejection_reason = (
            None
            if structurally_valid
            else "single_column_without_repeated_content"
        )
    else:
        structurally_valid = (
            occupied_column_count >= 2
            and (multi_value_row_count >= 1 or image_block_count > 0)
            and nonempty_cell_count >= 2
        )
        rejection_reason = (
            None
            if structurally_valid
            else "sparse_cells_without_multi_column_row"
        )

    return structurally_valid, {
        "raw_row_count": raw_row_count,
        "raw_column_count": raw_column_count,
        "nonempty_cell_count": nonempty_cell_count,
        "occupied_row_count": occupied_row_count,
        "occupied_column_count": occupied_column_count,
        "multi_value_row_count": multi_value_row_count,
        "image_block_count": image_block_count,
        "structurally_valid": structurally_valid,
        "rejection_reason": rejection_reason,
    }


def count_nonempty_table_cells(table_data: list[list[str | None]]) -> int:
    """Count cells containing non-whitespace text."""

    return sum(
        bool(cell and str(cell).strip())
        for row in table_data
        for cell in row
    )


def build_table_claims(
    *,
    page_number: int,
    tables: list[Any],
    page_inventory: PageInventory,
    detection_mode: str,
    confidence_fn: Callable[[Any], float],
) -> list[dict[str, Any]]:
    """Build normalized Claims from already-detected table objects."""

    claims: list[dict[str, Any]] = []
    for table_index, table in enumerate(tables):
        table_rect = fitz.Rect(table.bbox)
        input_ids = list(getattr(table, "claim_input_element_ids", []))
        claimed_ids = page_inventory.ids_touching_rect(
            rect=table_rect,
            kinds=("drawing",),
            allowed_element_ids=input_ids,
        )
        detection_source = getattr(
            table,
            "detection_source",
            "pymupdf_native",
        )
        metadata = {
            "detection_mode": detection_mode,
            "table_index": table_index,
            "detection_source": detection_source,
        }
        source_cluster_bbox = getattr(table, "source_cluster_bbox", None)
        if source_cluster_bbox is not None:
            metadata["source_cluster_bbox"] = [
                round(value, 3)
                for value in fitz.Rect(source_cluster_bbox)
            ]
        claims.append(
            create_claim_manifest(
                detector_name="table_detector",
                output_element_id=(
                    f"page#{page_number}:table#{table_index + 1}"
                ),
                input_element_ids=input_ids,
                claimed_element_ids=claimed_ids,
                feature_element_ids=claimed_ids,
                bbox=table_rect,
                confidence=confidence_fn(table),
                reason_codes=[
                    f"detected_by:{detection_source}",
                    "drawing_touches_table_bbox_within_detector_input",
                ],
                output_artifacts=[
                    {
                        "type": "table_candidate",
                        "table_index": table_index,
                        "bbox": [round(value, 3) for value in table_rect],
                    }
                ],
                metadata=metadata,
            )
        )
    return claims


def extract_geometry_table_data(
    bbox: fitz.Rect,
    horizontal_positions: list[float],
    separator_positions: list[float],
    text_lines: list[dict],
) -> tuple[list[list[str | None]], int, int]:
    """按几何行列边界构造二维文本，并返回有效行列数量。"""

    column_boundaries = [bbox.x0, *separator_positions, bbox.x1]
    row_cells: list[list[list[dict]]] = [
        [[] for _ in range(len(column_boundaries) - 1)]
        for _ in range(len(horizontal_positions) - 1)
    ]

    for line in text_lines:
        if line.get("angle") != 0:
            continue
        center = rect_center(line["bbox"])
        if not bbox.contains(center):
            continue
        row_index = next(
            (
                index
                for index in range(len(horizontal_positions) - 1)
                if horizontal_positions[index]
                <= center.y
                <= horizontal_positions[index + 1]
            ),
            None,
        )
        if row_index is None:
            continue
        column_index = min(
            len(column_boundaries) - 2,
            next(
                (
                    index
                    for index in range(len(column_boundaries) - 1)
                    if column_boundaries[index]
                    <= center.x
                    <= column_boundaries[index + 1]
                ),
                len(column_boundaries) - 2,
            ),
        )
        row_cells[row_index][column_index].append(line)

    data: list[list[str | None]] = []
    for row in row_cells:
        output_row: list[str | None] = []
        for cell_lines in row:
            ordered = sorted(
                cell_lines,
                key=lambda item: (item["bbox"].y0, item["bbox"].x0),
            )
            cell_text = "\n".join(
                item["text"].strip()
                for item in ordered
                if item["text"].strip()
            )
            output_row.append(cell_text or None)
        data.append(output_row)

    populated_rows = sum(any(cell for cell in row) for row in data)
    populated_columns = sum(
        any(row[column_index] for row in data)
        for column_index in range(len(column_boundaries) - 1)
    )
    return data, populated_rows, populated_columns


class GeometryFallbackTable:
    """Minimal table object compatible with the parser's table consumers."""

    def __init__(
        self,
        bbox: fitz.Rect,
        data: list[list[str | None]],
        metadata: dict[str, Any] | None = None,
        detection_source: str = "geometry_fallback",
    ) -> None:
        self.bbox = tuple(bbox)
        self._data = data
        self.metadata = metadata or {}
        self.detection_source = detection_source

    def extract(self) -> list[list[str | None]]:
        return [list(row) for row in self._data]


def combine_native_and_fallback_tables(
    native_tables: list[Any],
    fallback_tables: list[GeometryFallbackTable],
    *,
    max_vertical_break_points: float,
    native_replacement_overlap_ratio: float,
    native_expansion_area_ratio: float,
    alignment_tolerance_points: float,
) -> list[Any]:
    """先消解同域候选，再拼接有几何证据的上下互补片段。

    同域结果优先保留原生单元格结构；完整补检候选可同时替换多个被覆盖的
    原生片段。整表数据不因行数少或内容相同而相加、删行。部分重叠但不能
    证明包含关系的候选不在这里拼接，交给既有候选去重阶段处理。

    Resolve alternative detections before joining complementary grid fragments.
    Reuse the existing coverage, expansion, and alignment policy parameters;
    row counts and text equality do not establish a spatial relationship.
    """

    combined = list(native_tables)

    def contains_with_alignment(outer: fitz.Rect, inner: fitz.Rect) -> bool:
        """允许已有规则线对齐误差，但不把一般相交当作包含。"""
        tolerance = alignment_tolerance_points
        return (
            outer.x0 - tolerance <= inner.x0
            and outer.y0 - tolerance <= inner.y0
            and outer.x1 + tolerance >= inner.x1
            and outer.y1 + tolerance >= inner.y1
        )

    def record_decision(table: Any, decision: str, alternatives: list[Any]) -> None:
        """保留选择依据，不把审计元数据当作检测分数或新阈值。"""
        records = list(getattr(table, "candidate_merge_audit", []))
        records.append({
            "decision": decision,
            "selected_bbox": list(table.bbox),
            "selected_row_count": len(table.extract()),
            "alternative_bboxes": [list(other.bbox) for other in alternatives],
            "alternative_row_counts": [len(other.extract()) for other in alternatives],
        })
        table.candidate_merge_audit = records

    def grid_positions(table: Any) -> tuple[list[float], list[float]]:
        """使用已解析的单元格或补检规则坐标，不重新扫描页面图元。"""
        metadata = getattr(table, "metadata", {}) or {}
        if "column_positions" in metadata and "row_positions" in metadata:
            return list(metadata["column_positions"]), list(metadata["row_positions"])
        x_positions, y_positions = table_cell_boundary_positions(table)
        return (
            unique_rule_positions(
                [{"position": value} for value in x_positions],
                tolerance=alignment_tolerance_points,
            ),
            unique_rule_positions(
                [{"position": value} for value in y_positions],
                tolerance=alignment_tolerance_points,
            ),
        )

    def merge_complementary_fragments(
        first: Any, second: GeometryFallbackTable
    ) -> GeometryFallbackTable | None:
        """只拼接不重叠且在相同列网格上共享行边界的片段。"""
        upper, lower = sorted(
            (first, second),
            key=lambda table: (fitz.Rect(table.bbox).y0, fitz.Rect(table.bbox).x0),
        )
        upper_rect, lower_rect = fitz.Rect(upper.bbox), fitz.Rect(lower.bbox)
        if min(rect_area(upper_rect), rect_area(lower_rect)) <= 0:
            return None
        gap = lower_rect.y0 - upper_rect.y1
        # 行区域有重叠时不能直接相加；间距上限仅作粗筛，共享行边界还需满足对齐容差。
        if gap < 0 or gap > max_vertical_break_points:
            return None
        if gap > alignment_tolerance_points:
            return None

        upper_columns, upper_rows = grid_positions(upper)
        lower_columns, lower_rows = grid_positions(lower)
        upper_data, lower_data = upper.extract(), lower.extract()
        for columns, rows, data, rect in (
            (upper_columns, upper_rows, upper_data, upper_rect),
            (lower_columns, lower_rows, lower_data, lower_rect),
        ):
            if not data or len(columns) < 2 or len(rows) != len(data) + 1:
                return None
            if any(b <= a for a, b in zip(columns, columns[1:])):
                return None
            if any(b <= a for a, b in zip(rows, rows[1:])):
                return None
            if any(len(row) != len(columns) - 1 for row in data):
                return None
            if (
                abs(rows[0] - rect.y0) > alignment_tolerance_points
                or abs(rows[-1] - rect.y1) > alignment_tolerance_points
            ):
                return None
        if len(upper_columns) != len(lower_columns) or any(
            abs(a - b) > alignment_tolerance_points
            for a, b in zip(upper_columns, lower_columns)
        ):
            return None
        if not 0 <= lower_rows[0] - upper_rows[-1] <= alignment_tolerance_points:
            return None

        # 不同空间行可以具有相同文字；按文字相等去重会误删原表中的合法重复值。
        merged = GeometryFallbackTable(
            bbox=union_rects([upper_rect, lower_rect]),
            data=[*upper_data, *lower_data],
            metadata={
                "merged_native_fragment": True,
                "merge_relation": "complementary_nonoverlapping_grid",
                "column_positions": upper_columns,
                "row_positions": [*upper_rows, *lower_rows[1:]],
                "fragment_bboxes": [list(upper.bbox), list(lower.bbox)],
                "fragment_row_counts": [len(upper_data), len(lower_data)],
            },
            detection_source="merged_native_fragment",
        )
        # 仅保留两片共同且一致的来源簇，不凭整个页面 bbox 推断它们相连。
        first_cluster = getattr(first, "source_cluster_bbox", None)
        second_cluster = getattr(second, "source_cluster_bbox", None)
        if first_cluster is not None and second_cluster is not None:
            if fitz.Rect(first_cluster) == fitz.Rect(second_cluster):
                merged.source_cluster_bbox = fitz.Rect(first_cluster)
        record_decision(merged, "join_complementary_grid_fragments", [upper, lower])
        return merged

    # 较大的补检候选先参与覆盖判断，避免较小候选的输入顺序影响选择。
    for fallback_table in sorted(
        fallback_tables,
        key=lambda table: (
            -rect_area(fitz.Rect(table.bbox)),
            fitz.Rect(table.bbox).y0,
            fitz.Rect(table.bbox).x0,
        ),
    ):
        fallback_rect = fitz.Rect(fallback_table.bbox)
        fallback_area = rect_area(fallback_rect)
        if fallback_area <= 0:
            combined.append(fallback_table)
            continue

        replacements: list[int] = []
        keeper: Any | None = None
        overlapping: list[Any] = []
        for index, existing in enumerate(combined):
            existing_rect = fitz.Rect(existing.bbox)
            existing_area = rect_area(existing_rect)
            if existing_area <= 0:
                continue
            overlap = rect_area(fallback_rect & existing_rect)
            if overlap <= 0:
                continue
            overlapping.append(existing)
            fallback_covers_existing = (
                contains_with_alignment(fallback_rect, existing_rect)
                and overlap / existing_area >= native_replacement_overlap_ratio
            )
            existing_covers_fallback = (
                contains_with_alignment(existing_rect, fallback_rect)
                and overlap / fallback_area >= native_replacement_overlap_ratio
            )
            if (
                fallback_covers_existing
                and fallback_area >= existing_area * native_expansion_area_ratio
            ):
                replacements.append(index)
            elif existing_covers_fallback:
                keeper = existing
                break

        # 先确认整个选择结果，再移除被覆盖的候选，避免抑制补检时误删其他片段。
        if keeper is not None:
            record_decision(keeper, "keep_existing_same_or_covering_region", [fallback_table])
            continue
        if replacements:
            replaced = [combined[index] for index in replacements]
            indices = set(replacements)
            combined = [table for index, table in enumerate(combined) if index not in indices]
            record_decision(fallback_table, "replace_all_covered_fragments", replaced)
            combined.append(fallback_table)
            continue

        if overlapping:
            record_decision(fallback_table, "partial_overlap_not_concatenated", overlapping)
            combined.append(fallback_table)
            continue

        for index, existing in enumerate(combined):
            merged = merge_complementary_fragments(existing, fallback_table)
            if merged is not None:
                combined[index] = merged
                break
        else:
            combined.append(fallback_table)

    return sorted(
        combined,
        key=lambda table: (fitz.Rect(table.bbox).y0, fitz.Rect(table.bbox).x0),
    )
