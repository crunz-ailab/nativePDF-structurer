"""根据PDF原生图元构建并合并视觉区域。

模块提供原始图片提取、visual drawing
过滤、OpenCV掩码构建、闭合容器检测、连通区域生成、空间合并规则、共享
图片关系、连接线分析、文字附着和视觉输出记录构建。

处理流程:
    1. 过滤非内容drawing和已确认的保护结构。
    2. 构建闭合容器及矢量连通区域。
    3. 通过明确几何或共享对象关系合并区域。
    4. 单次附着原始图片、连接drawing和已归属原生文字。
    5. 为编排层生成渲染、排序、Claim和provenance数据。

关键不变量:
    * ``render_bbox`` 与 ``sort_bbox`` 职责不同。
    * 闭合视觉容器阻止跨容器边界合并。
    * 文字和连接线只执行一跳附着，不递归传播。
    * 每次区域合并都必须保留稳定图元ID。

Build and merge visual regions from native PDF elements.

Provides source-image extraction, visual drawing filters, OpenCV mask
construction, closed-container detection, connected-region creation, spatial
merge rules, shared-image relationships, connector analysis, text attachment,
and visual output record construction.

Processing flow:
    1. Filter non-content drawings and confirmed protected structures.
    2. Build closed containers and connected vector regions.
    3. Merge regions through explicit geometry or shared-object relationships.
    4. Attach source images, connector drawings, and owned native text once.
    5. Produce render, sort, Claim, and provenance data for orchestration.

Key invariants:
    * ``render_bbox`` and ``sort_bbox`` serve different purposes.
    * Closed visual containers block merges across container boundaries.
    * Text and connector attachment is one-hop and non-recursive.
    * Stable element IDs are preserved through every region merge.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections import defaultdict
from collections.abc import Callable
import statistics
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from .geometry import as_fitz_rect, rect_area, rect_center, union_rects
from .inventory import PageInventory

VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO = 0.04
VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO = 0.02
VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO = 0.80
VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO = 0.04
VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO = 0.10
VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO = 0.50
MAX_HORIZONTAL_LINE_SLOPE_RATIO = 0.05
VISUAL_ENDPOINT_BOUNDARY_TOLERANCE_POINTS = 4.0
SORT_BBOX_TOP_HEIGHT_POINTS = 2.0
CLOSED_VISUAL_CONTAINER_RULE_TOLERANCE_POINTS = 1.0
VISUAL_MASK_PIXELS_PER_POINT = 0.5
VISUAL_MASK_CLOSE_GAP_PAGE_RATIO = 0.015
VISUAL_REGION_TEXT_MARGIN_PAGE_RATIO = 0.015
VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER = 2.0

IMAGE_PADDING_POINTS = 2.0
MIN_IMAGE_THIN_SIDE_POINTS = 12.0
MIN_IMAGE_LINE_ASPECT_RATIO = 5.0
MIN_IMAGE_AREA_POINTS_SQUARED = 160.0
COMPLEX_TRANSFORM_EPSILON = 0.01
HEADER_FOOTER_IMAGE_HASH_SIZE = (32, 16)


def image_rects_overlap(
    first_rect: fitz.Rect,
    second_rect: fitz.Rect,
) -> bool:
    """判断两张原生图片是否有足够的面积重叠。"""

    intersection = first_rect & second_rect
    if intersection.is_empty:
        return False

    minimum_area = min(rect_area(first_rect), rect_area(second_rect))
    return (
        minimum_area > 0
        and rect_area(intersection) / minimum_area >= 0.01
    )


def image_is_large_enough(
    image_rect: fitz.Rect,
    page_rect: fitz.Rect,
) -> bool:
    """判断图片是否不是细长装饰条或极小面积块。"""

    minor_side = min(image_rect.width, image_rect.height)
    major_side = max(image_rect.width, image_rect.height)
    aspect_ratio = (
        major_side / minor_side
        if minor_side > 0
        else math.inf
    )
    line_like = (
        minor_side < MIN_IMAGE_THIN_SIDE_POINTS
        and aspect_ratio > MIN_IMAGE_LINE_ASPECT_RATIO
    )
    tiny_area = rect_area(image_rect) < MIN_IMAGE_AREA_POINTS_SQUARED
    _ = page_rect
    return not (line_like or tiny_area)


def image_has_complex_transform(transform: tuple[float, ...]) -> bool:
    """判断原生图片变换是否包含旋转或倾斜。"""

    if len(transform) < 4:
        return True

    _, b, c, _, *_ = transform
    return (
        abs(b) > COMPLEX_TRANSFORM_EPSILON
        or abs(c) > COMPLEX_TRANSFORM_EPSILON
    )


def padded_image_clip(
    bbox: fitz.Rect,
    page_rect: fitz.Rect,
) -> fitz.Rect:
    """为视觉资产截图增加少量留白并限制在页面边界内。"""

    return fitz.Rect(
        max(page_rect.x0, bbox.x0 - IMAGE_PADDING_POINTS),
        max(page_rect.y0, bbox.y0 - IMAGE_PADDING_POINTS),
        min(page_rect.x1, bbox.x1 + IMAGE_PADDING_POINTS),
        min(page_rect.y1, bbox.y1 + IMAGE_PADDING_POINTS),
    )


def extract_source_image_blocks(page_dict: dict) -> list[dict]:
    """提取原PDF图片块并保留直接复制所需的编码和变换信息。"""

    image_blocks: list[dict] = []
    for source_index, block in enumerate(page_dict["blocks"], start=1):
        if block["type"] != 1:
            continue
        image_blocks.append(
            {
                "source_index": source_index,
                "bbox": fitz.Rect(block["bbox"]),
                "extension": block.get("ext", "png"),
                "image_bytes": block.get("image"),
                "has_mask": bool(block.get("mask")),
                "source_width": block.get("width"),
                "source_height": block.get("height"),
                "transform": tuple(block.get("transform", ())),
            }
        )
    return image_blocks


def image_content_signature(
    image_bytes: bytes | None,
) -> str | None:
    """返回图片固定尺寸RGB缩略图的稳定摘要。"""

    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            normalized = image.convert("RGB").resize(
                HEADER_FOOTER_IMAGE_HASH_SIZE,
                Image.Resampling.BILINEAR,
            )
            return hashlib.sha256(normalized.tobytes()).hexdigest()
    except Exception:
        return None


def collect_visual_drawing_candidates(
    page: fitz.Page,
    all_table_rects: list[fitz.Rect],
    excluded_header_footer_rects: list[fitz.Rect],
    drawings: list[dict],
    page_inventory: PageInventory,
    excluded_drawing_element_ids: set[str] | None = None,
    *,
    max_background_area_page_ratio: float,
    build_endpoint_marker_grid: Callable[[list[dict]], dict],
    drawing_has_endpoint_marker: Callable[..., bool],
    drawing_is_filled_horizontal_bar: Callable[[dict], bool],
    drawing_is_long_horizontal_single_line: Callable[[dict], bool],
    drawing_is_page_enclosing_stroke_frame: Callable[..., bool],
    rect_is_mostly_inside_any: Callable[..., bool],
) -> list[dict]:
    """筛选进入视觉占用掩码的页面矢量对象。"""

    page_rect = page_inventory.page_rect
    page_area = rect_area(page_rect)
    excluded_ids = set(excluded_drawing_element_ids or set())
    eligible_drawings: list[dict] = []

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
        drawing_rect = (
            drawing_entry["bbox"]
            if drawing_entry is not None
            else fitz.Rect(drawing["rect"])
        )
        drawing_area = rect_area(drawing_rect)

        if drawing_element_id is not None and drawing_element_id in excluded_ids:
            continue

        drawing_is_degenerate = (
            bool(drawing_labels.get("degenerate", False))
            if drawing_labels
            else drawing_rect.width <= 0 and drawing_rect.height <= 0
        )
        if drawing_is_degenerate:
            continue

        drawing_is_page_frame = (
            bool(
                drawing_labels.get(
                    "page_enclosing_stroke_frame",
                    False,
                )
            )
            if drawing_labels
            else drawing_is_page_enclosing_stroke_frame(
                drawing=drawing,
                page_rect=page_rect,
            )
        )
        if drawing_is_page_frame:
            continue

        if rect_is_mostly_inside_any(
            rect=drawing_rect,
            container_rects=excluded_header_footer_rects,
        ) or any(
            band.contains(rect_center(drawing_rect))
            for band in excluded_header_footer_rects
        ):
            continue

        if rect_is_mostly_inside_any(
            rect=drawing_rect,
            container_rects=all_table_rects,
        ) or any(
            table_rect.contains(rect_center(drawing_rect))
            for table_rect in all_table_rects
        ):
            continue

        drawing_is_background = (
            bool(drawing_labels.get("background", False))
            if drawing_labels
            else (
                page_area > 0
                and drawing_area / page_area
                >= max_background_area_page_ratio
            )
        )
        if drawing_is_background:
            continue

        eligible_drawings.append(drawing)

    marker_grid = build_endpoint_marker_grid(eligible_drawings)
    candidates: list[dict] = []
    for drawing_index, drawing in enumerate(eligible_drawings):
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
        drawing_rect = (
            drawing_entry["bbox"]
            if drawing_entry is not None
            else fitz.Rect(drawing["rect"])
        )

        if drawing_labels:
            is_filled_horizontal_bar = bool(
                drawing_labels.get("filled_horizontal_bar", False)
            )
            is_long_horizontal_line = bool(
                drawing_labels.get("long_horizontal_stroke", False)
            )
        else:
            is_filled_horizontal_bar = drawing_is_filled_horizontal_bar(
                drawing
            )
            is_long_horizontal_line = (
                drawing_is_long_horizontal_single_line(drawing)
            )

        if (
            (is_filled_horizontal_bar or is_long_horizontal_line)
            and not drawing_has_endpoint_marker(
                drawing_index=drawing_index,
                connector_drawing=drawing,
                drawings=eligible_drawings,
                marker_grid=marker_grid,
            )
        ):
            continue

        candidates.append(
            {
                "bbox": drawing_rect,
                "type": drawing.get("type"),
                "seqno": drawing.get("seqno"),
                "element_id": drawing_element_id,
                "drawing_payload": drawing,
            }
        )

    _ = page
    return candidates


def draw_rect_on_visual_mask(
    mask: np.ndarray,
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    scale: float,
) -> None:
    """在占用掩码中填充一个矢量对象bbox。"""

    x0 = max(
        0,
        min(
            mask.shape[1] - 1,
            int(math.floor((rect.x0 - page_rect.x0) * scale)),
        ),
    )
    y0 = max(
        0,
        min(
            mask.shape[0] - 1,
            int(math.floor((rect.y0 - page_rect.y0) * scale)),
        ),
    )
    x1 = max(
        x0,
        min(
            mask.shape[1] - 1,
            int(math.ceil((rect.x1 - page_rect.x0) * scale)),
        ),
    )
    y1 = max(
        y0,
        min(
            mask.shape[0] - 1,
            int(math.ceil((rect.y1 - page_rect.y0) * scale)),
        ),
    )
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)


def clear_rect_on_visual_mask(
    mask: np.ndarray,
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    scale: float,
) -> None:
    """在占用掩码中清除一个保护区域bbox。"""

    rect = fitz.Rect(rect).normalize()
    x0 = max(
        0,
        min(
            mask.shape[1] - 1,
            int(math.floor((rect.x0 - page_rect.x0) * scale)),
        ),
    )
    y0 = max(
        0,
        min(
            mask.shape[0] - 1,
            int(math.floor((rect.y0 - page_rect.y0) * scale)),
        ),
    )
    x1 = max(
        x0,
        min(
            mask.shape[1] - 1,
            int(math.ceil((rect.x1 - page_rect.x0) * scale)),
        ),
    )
    y1 = max(
        y0,
        min(
            mask.shape[0] - 1,
            int(math.ceil((rect.y1 - page_rect.y0) * scale)),
        ),
    )
    cv2.rectangle(mask, (x0, y0), (x1, y1), 0, thickness=-1)


def build_visual_regions_from_candidates(
    page_rect: fitz.Rect,
    candidates: list[dict],
    closed_visual_containers: list[dict],
    all_table_rects: list[fitz.Rect],
    *,
    mask_pixels_per_point: float,
    close_gap_page_ratio: float,
) -> list[dict]:
    """从矢量候选和闭合容器生成初始视觉区域记录。"""

    closed_container_by_id = {
        container["container_element_id"]: container
        for container in closed_visual_containers
    }

    for candidate in candidates:
        candidate_rect = fitz.Rect(candidate["bbox"])
        matching_containers = [
            container
            for container in closed_visual_containers
            if container["bbox"].contains(rect_center(candidate_rect))
        ]
        if not matching_containers:
            candidate["closed_visual_container_element_id"] = None
            continue
        selected_container = min(
            matching_containers,
            key=lambda container: rect_area(container["bbox"]),
        )
        candidate["closed_visual_container_element_id"] = (
            selected_container["container_element_id"]
        )

    scale = mask_pixels_per_point
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

    close_gap_points = (
        min(page_rect.width, page_rect.height)
        * close_gap_page_ratio
    )
    kernel_radius = max(1, int(round(close_gap_points * scale)))
    kernel_size = kernel_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    for protected_rect in all_table_rects:
        clear_rect_on_visual_mask(
            mask=closed_mask,
            rect=protected_rect,
            page_rect=page_rect,
            scale=scale,
        )

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        closed_mask,
        connectivity=8,
    )
    regions: list[dict] = []
    for component_index in range(1, component_count):
        x, y, width, height, _ = stats[component_index]
        component_rect = fitz.Rect(
            page_rect.x0 + x / scale,
            page_rect.y0 + y / scale,
            page_rect.x0 + (x + width) / scale,
            page_rect.y0 + (y + height) / scale,
        )
        component_candidates = [
            candidate
            for candidate in candidates
            if component_rect.contains(rect_center(candidate["bbox"]))
        ]
        candidates_by_container: dict[str | None, list[dict]] = defaultdict(list)
        for candidate in component_candidates:
            candidates_by_container[
                candidate.get("closed_visual_container_element_id")
            ].append(candidate)

        for container_id, container_candidates in candidates_by_container.items():
            container = (
                closed_container_by_id.get(container_id)
                if container_id is not None
                else None
            )
            region_bbox = component_rect
            boundary_ids: list[str] = []
            closed_container_ids: list[str] = []
            if container is not None:
                region_bbox = union_rects(
                    [component_rect, container["bbox"]]
                )
                boundary_ids = list(
                    container["boundary_drawing_element_ids"]
                )
                closed_container_ids = [container_id]

            component_drawing_ids = sorted(
                {
                    candidate["element_id"]
                    for candidate in container_candidates
                    if candidate.get("element_id") is not None
                }.union(boundary_ids)
            )
            regions.append(
                {
                    "bbox": region_bbox,
                    "drawing_count": len(container_candidates),
                    "drawing_element_ids": component_drawing_ids,
                    "closed_visual_container_element_ids": (
                        closed_container_ids
                    ),
                    "component_count": 1,
                    "included_text_line_count": 0,
                    "included_image_count": 0,
                }
            )

    return [
        region
        for region in regions
        if image_is_large_enough(
            image_rect=region["bbox"],
            page_rect=page_rect,
        )
    ]

def merged_closed_visual_container_ids(
    regions: list[dict],
) -> list[str]:
    """返回一组视觉区域涉及的闭合视觉容器ID并集。"""
    return sorted(
        {
            container_id
            for region in regions
            for container_id in region.get(
                "closed_visual_container_element_ids",
                [],
            )
            if container_id is not None
        }
    )

def visual_regions_cross_distinct_closed_containers(
    first: dict,
    second: dict,
) -> bool:
    """判断两个视觉区域是否分别属于不同闭合视觉容器。"""
    first_ids = set(
        first.get("closed_visual_container_element_ids", [])
    )
    second_ids = set(
        second.get("closed_visual_container_element_ids", [])
    )
    return bool(first_ids and second_ids and first_ids.isdisjoint(second_ids))

def visual_regions_should_merge(
    first_rect: fitz.Rect,
    second_rect: fitz.Rect,
    page_rect: fitz.Rect,
) -> bool:
    """判断两个视觉连通域是否应合并为更粗粒度区域。"""
    horizontal_gap = max(
        0.0,
        max(first_rect.x0, second_rect.x0)
        - min(first_rect.x1, second_rect.x1),
    )
    vertical_gap = max(
        0.0,
        max(first_rect.y0, second_rect.y0)
        - min(first_rect.y1, second_rect.y1),
    )
    horizontal_overlap = (
        min(first_rect.x1, second_rect.x1)
        - max(first_rect.x0, second_rect.x0)
    )
    vertical_overlap = (
        min(first_rect.y1, second_rect.y1)
        - max(first_rect.y0, second_rect.y0)
    )
    maximum_horizontal_gap = (
        page_rect.width
        * VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO
    )
    maximum_vertical_gap = (
        page_rect.height
        * VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO
    )

    return (
        (
            horizontal_overlap > 0
            and vertical_gap <= maximum_vertical_gap
        )
        or (
            vertical_overlap > 0
            and horizontal_gap <= maximum_horizontal_gap
        )
        or (
            horizontal_gap <= maximum_horizontal_gap
            and vertical_gap <= maximum_vertical_gap
        )
    )

def merged_region_element_ids(
    regions: list[dict],
    key: str,
) -> list[str]:
    return sorted(
        {
            element_id
            for region in regions
            for element_id in region.get(key, [])
        }
    )

def merge_visual_region_records(
    regions: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """迭代合并留白距离较小的视觉区域。"""
    merged_regions = [dict(region) for region in regions]
    changed = True

    while changed:
        changed = False

        for first_index in range(len(merged_regions)):
            if changed:
                break

            for second_index in range(
                first_index + 1,
                len(merged_regions),
            ):
                first = merged_regions[first_index]
                second = merged_regions[second_index]

                if visual_regions_cross_distinct_closed_containers(
                    first=first,
                    second=second,
                ):
                    continue

                if not visual_regions_should_merge(
                    first_rect=first["bbox"],
                    second_rect=second["bbox"],
                    page_rect=page_rect,
                ):
                    continue

                combined = {
                    "bbox": union_rects(
                        [first["bbox"], second["bbox"]]
                    ),
                    "sort_source_bbox": union_rects(
                        [
                            fitz.Rect(
                                first.get(
                                    "sort_source_bbox",
                                    first["bbox"],
                                )
                            ),
                            fitz.Rect(
                                second.get(
                                    "sort_source_bbox",
                                    second["bbox"],
                                )
                            ),
                        ]
                    ),
                    "drawing_count": (
                        first.get("drawing_count", 0)
                        + second.get("drawing_count", 0)
                    ),
                    "drawing_element_ids": merged_region_element_ids(
                        [first, second],
                        "drawing_element_ids",
                    ),
                    "closed_visual_container_element_ids": merged_closed_visual_container_ids(
                        [first, second]
                    ),
                    "component_count": (
                        first.get("component_count", 1)
                        + second.get("component_count", 1)
                    ),
                    "included_text_line_count": (
                        first.get("included_text_line_count", 0)
                        + second.get("included_text_line_count", 0)
                    ),
                    "included_image_count": (
                        first.get("included_image_count", 0)
                        + second.get("included_image_count", 0)
                    ),
                }
                merged_regions[first_index] = combined
                del merged_regions[second_index]
                changed = True
                break

    return merged_regions

def pure_vector_same_layer_should_merge(
    first_rect: fitz.Rect,
    second_rect: fitz.Rect,
    page_rect: fitz.Rect,
) -> bool:
    """判断两个已聚合矢量区域是否处于同一视觉层。"""
    vertical_overlap = max(
        0.0,
        min(first_rect.y1, second_rect.y1)
        - max(first_rect.y0, second_rect.y0),
    )
    minimum_height = min(first_rect.height, second_rect.height)
    if minimum_height <= 0:
        return False

    vertical_overlap_ratio = vertical_overlap / minimum_height
    horizontal_gap = max(
        0.0,
        max(first_rect.x0, second_rect.x0)
        - min(first_rect.x1, second_rect.x1),
    )
    maximum_horizontal_gap = (
        page_rect.width
        * VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO
    )

    normal_same_layer = (
        vertical_overlap_ratio
        >= VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO
        and horizontal_gap <= maximum_horizontal_gap
    )


    return normal_same_layer

def merge_pure_vector_same_layer_once(
    regions: list[dict],
    page_rect: fitz.Rect,
) -> list[dict]:
    """按已聚合区域极限坐标执行一轮保守的纯矢量同层合并。"""
    merged_regions: list[dict] = []
    consumed_indices: set[int] = set()

    for first_index, first in enumerate(regions):
        if first_index in consumed_indices:
            continue

        partner_index: int | None = None
        for second_index in range(
            first_index + 1,
            len(regions),
        ):
            if second_index in consumed_indices:
                continue

            second = regions[second_index]
            if visual_regions_cross_distinct_closed_containers(
                first=first,
                second=second,
            ):
                continue
            if pure_vector_same_layer_should_merge(
                first_rect=fitz.Rect(first["bbox"]),
                second_rect=fitz.Rect(second["bbox"]),
                page_rect=page_rect,
            ):
                partner_index = second_index
                break

        if partner_index is None:
            merged_regions.append(dict(first))
            continue

        second = regions[partner_index]
        consumed_indices.add(partner_index)
        merged_regions.append(
            {
                "bbox": union_rects(
                    [first["bbox"], second["bbox"]]
                ),
                "drawing_count": (
                    first.get("drawing_count", 0)
                    + second.get("drawing_count", 0)
                ),
                "drawing_element_ids": merged_region_element_ids(
                    [first, second],
                    "drawing_element_ids",
                ),
                "closed_visual_container_element_ids": merged_closed_visual_container_ids(
                    [first, second]
                ),
                "component_count": (
                    first.get("component_count", 1)
                    + second.get("component_count", 1)
                ),
                "included_text_line_count": (
                    first.get("included_text_line_count", 0)
                    + second.get("included_text_line_count", 0)
                ),
                "included_image_count": (
                    first.get("included_image_count", 0)
                    + second.get("included_image_count", 0)
                ),
            }
        )

    return merged_regions

def merge_visual_regions_by_horizontal_relations_once(
    regions: list[dict],
    drawings: list[dict],
    page_rect: fitz.Rect,
    logical_horizontal_rules: list[dict],
    drawing_is_filled_horizontal_bar: Callable[[dict], bool],
    drawing_is_long_horizontal_single_line: Callable[[dict], bool],
    group_horizontal_rules_by_span: Callable[[list[dict]], list[list[dict]]],
    unique_rule_positions: Callable[[list[dict]], list[float]],
) -> list[dict]:
    """按水平连接线和共享水平框架做一次视觉区域合并。"""
    if len(regions) < 2:
        return regions

    parent = list(range(len(regions)))
    container_ids_by_root = {
        index: set(
            region.get(
                "closed_visual_container_element_ids",
                [],
            )
        )
        for index, region in enumerate(regions)
    }

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> bool:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root == second_root:
            return True
        combined_container_ids = (
            container_ids_by_root[first_root]
            | container_ids_by_root[second_root]
        )
        if len(combined_container_ids) > 1:
            return False
        parent[second_root] = first_root
        container_ids_by_root[first_root] = combined_container_ids
        return True

    horizontal_relation_lines: list[fitz.Rect] = []
    horizontal_frame_rules: list[dict] = []

    for drawing in drawings:
        drawing_rect = fitz.Rect(drawing["rect"]).normalize()
        if (
            drawing_is_filled_horizontal_bar(drawing)
            or drawing_is_long_horizontal_single_line(drawing)
        ):
            horizontal_relation_lines.append(drawing_rect)

    horizontal_frame_rules = [
        dict(rule) for rule in logical_horizontal_rules
    ]
    horizontal_relation_lines.extend(
        as_fitz_rect(rule["rect"])
        for rule in logical_horizontal_rules
    )

    for first_index, first in enumerate(regions):
        first_rect = fitz.Rect(first["bbox"])
        for second_index in range(first_index + 1, len(regions)):
            second_rect = fitz.Rect(regions[second_index]["bbox"])
            left_rect, right_rect = sorted(
                [first_rect, second_rect],
                key=lambda rect: (rect.x0, rect.x1),
            )
            overlap_y0 = max(left_rect.y0, right_rect.y0)
            overlap_y1 = min(left_rect.y1, right_rect.y1)
            if overlap_y1 <= overlap_y0:
                continue

            if any(
                line.x0 <= left_rect.x1
                and line.x1 >= right_rect.x0
                and overlap_y0
                <= (line.y0 + line.y1) / 2
                <= overlap_y1
                for line in horizontal_relation_lines
            ):
                union(first_index, second_index)

    for frame_group in group_horizontal_rules_by_span(
        horizontal_frame_rules
    ):
        positions = unique_rule_positions(frame_group)
        if len(positions) < 2:
            continue

        frame_x0 = min(rule["rect"].x0 for rule in frame_group)
        frame_x1 = max(rule["rect"].x1 for rule in frame_group)
        for band_y0, band_y1 in zip(positions, positions[1:]):
            members = [
                region_index
                for region_index, region in enumerate(regions)
                if frame_x0
                <= rect_center(fitz.Rect(region["bbox"])).x
                <= frame_x1
                and band_y0
                <= rect_center(fitz.Rect(region["bbox"])).y
                <= band_y1
            ]
            for member_index in members[1:]:
                union(members[0], member_index)

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(regions)):
        grouped_indices.setdefault(find(index), []).append(index)

    if all(len(indices) == 1 for indices in grouped_indices.values()):
        return regions

    merged_regions: list[dict] = []
    for indices in grouped_indices.values():
        if len(indices) == 1:
            merged_regions.append(dict(regions[indices[0]]))
            continue

        members = [regions[index] for index in indices]
        merged_regions.append(
            {
                "bbox": union_rects(
                    [fitz.Rect(member["bbox"]) for member in members]
                ),
                "drawing_count": sum(
                    member.get("drawing_count", 0)
                    for member in members
                ),
                "drawing_element_ids": merged_region_element_ids(
                    members,
                    "drawing_element_ids",
                ),
                "closed_visual_container_element_ids": merged_closed_visual_container_ids(
                    members
                ),
                "component_count": sum(
                    member.get("component_count", 1)
                    for member in members
                ),
                "included_text_line_count": sum(
                    member.get("included_text_line_count", 0)
                    for member in members
                ),
                "included_image_count": sum(
                    member.get("included_image_count", 0)
                    for member in members
                ),
            }
        )

    return sorted(
        merged_regions,
        key=lambda region: (
            fitz.Rect(region["bbox"]).y0,
            fitz.Rect(region["bbox"]).x0,
        ),
    )

def visual_regions_should_merge_shared_image(
    first: dict,
    second: dict,
) -> bool:
    """判断两个区域是否关联了同一个原始图片对象。"""
    first_image_ids = set(
        first.get("attached_source_image_indices", [])
    )
    second_image_ids = set(
        second.get("attached_source_image_indices", [])
    )
    if not first_image_ids.intersection(second_image_ids):
        return False

    #     minimum_height > 0
    #     and vertical_overlap / minimum_height >= 0.80
    #     and horizontal_gap
    #     <= page_rect.width * VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO
    # )
    return True

def merge_visual_regions_sharing_images(
    regions: list[dict],
) -> list[dict]:
    """合并引用同一原始图片的视觉内容候选。"""
    merged_regions = [dict(region) for region in regions]
    changed = True

    while changed:
        changed = False

        for first_index in range(len(merged_regions)):
            if changed:
                break

            for second_index in range(
                first_index + 1,
                len(merged_regions),
            ):
                first = merged_regions[first_index]
                second = merged_regions[second_index]

                if visual_regions_cross_distinct_closed_containers(
                    first=first,
                    second=second,
                ):
                    continue

                if not visual_regions_should_merge_shared_image(
                    first=first,
                    second=second,
                ):
                    continue

                structural_rects_by_bbox: dict[
                    tuple[float, ...], fitz.Rect
                ] = {}
                for structural_rect in (
                    first.get("structural_rects", [])
                    + second.get("structural_rects", [])
                ):
                    structural_rect = fitz.Rect(structural_rect)
                    structural_rects_by_bbox[
                        tuple(
                            round(value, 4)
                            for value in structural_rect
                        )
                    ] = structural_rect
                attached_image_rects_by_bbox: dict[
                    tuple[float, ...], fitz.Rect
                ] = {}
                for image_rect in (
                    first.get("attached_image_rects", [])
                    + second.get("attached_image_rects", [])
                ):
                    image_rect = fitz.Rect(image_rect)
                    attached_image_rects_by_bbox[
                        tuple(
                            round(value, 4)
                            for value in image_rect
                        )
                    ] = image_rect
                combined_content_bbox = union_rects(
                    list(structural_rects_by_bbox.values())
                )
                combined = {
                    "bbox": combined_content_bbox,
                    "content_bbox": combined_content_bbox,
                    "sort_source_bbox": combined_content_bbox,
                    "structural_rects": list(
                        structural_rects_by_bbox.values()
                    ),
                    "attached_image_rects": list(
                        attached_image_rects_by_bbox.values()
                    ),
                    "drawing_count": (
                        first.get("drawing_count", 0)
                        + second.get("drawing_count", 0)
                    ),
                    "drawing_element_ids": merged_region_element_ids(
                        [first, second],
                        "drawing_element_ids",
                    ),
                    "closed_visual_container_element_ids": merged_closed_visual_container_ids(
                        [first, second]
                    ),
                    "component_count": (
                        first.get("component_count", 1)
                        + second.get("component_count", 1)
                    ),
                    "included_image_count": len(
                        set(
                            first.get(
                                "attached_source_image_indices", []
                            )
                            + second.get(
                                "attached_source_image_indices", []
                            )
                        )
                    ),
                    "attached_source_image_indices": sorted(
                        set(
                            first.get(
                                "attached_source_image_indices", []
                            )
                            + second.get(
                                "attached_source_image_indices", []
                            )
                        )
                    ),
                    "attached_image_element_ids": (
                        merged_region_element_ids(
                            [first, second],
                            "attached_image_element_ids",
                        )
                    ),
                    "rule_line_frameworks": [
                        *first.get("rule_line_frameworks", []),
                        *second.get("rule_line_frameworks", []),
                    ],
                }
                merged_regions[first_index] = combined
                del merged_regions[second_index]
                changed = True
                break

    return merged_regions
def detect_closed_visual_rule_containers(
    page_rect: fitz.Rect,
    axis_geometry_index: dict,
    page_inventory: PageInventory,
    candidate_element_ids: set[str],
    excluded_drawing_element_ids: set[str],
    all_table_rects: list[fitz.Rect],
    *,
    container_rule_tolerance_points: float,
    max_background_area_page_ratio: float,
    drawing_has_visible_semantic_stroke: Callable[[dict], bool],
    rule_element_ids: Callable[[dict], set[str]],
    rect_is_mostly_inside_any: Callable[..., bool],
) -> list[dict]:
    """检测由上下横线和左右竖线构成的独立视觉容器。"""
    tolerance = container_rule_tolerance_points
    available_ids = set(candidate_element_ids)
    excluded_ids = set(excluded_drawing_element_ids)
    page_area = rect_area(page_rect)

    visible_semantic_stroke_by_element_id: dict[str, bool] = {}
    for element_id in page_inventory.ids_of_kind("drawing"):
        entry = page_inventory.entry(element_id)
        labels = entry.get("labels", {})
        if "visible_semantic_stroke" in labels:
            visible_semantic_stroke_by_element_id[element_id] = bool(
                labels["visible_semantic_stroke"]
            )
        else:
            visible_semantic_stroke_by_element_id[element_id] = (
                drawing_has_visible_semantic_stroke(entry["payload"])
            )

    available_candidate_centers = {
        element_id: rect_center(page_inventory.bbox_of(element_id))
        for element_id in available_ids
        if element_id in page_inventory.entries
    }

    horizontal_rules: list[dict] = []
    for rule in axis_geometry_index.get(
        "logical_horizontal_rules", []
    ):
        source_ids = rule_element_ids(rule)
        if not source_ids or source_ids & excluded_ids:
            continue
        rule_rect = rule["rect"]
        if rule_rect.width <= 0:
            continue
        if not all(
            visible_semantic_stroke_by_element_id.get(
                element_id,
                False,
            )
            for element_id in source_ids
        ):
            continue
        horizontal_rules.append(
            {
                **rule,
                "rect": rule_rect,
                "source_ids": source_ids,
            }
        )

    vertical_rules: list[dict] = []
    for rule in axis_geometry_index.get(
        "logical_vertical_rules", []
    ):
        source_ids = rule_element_ids(rule)
        if not source_ids or source_ids & excluded_ids:
            continue
        rule_rect = rule["rect"]
        if rule_rect.height <= 0:
            continue
        if not all(
            visible_semantic_stroke_by_element_id.get(
                element_id,
                False,
            )
            for element_id in source_ids
        ):
            continue
        vertical_rules.append(
            {
                **rule,
                "rect": rule_rect,
                "source_ids": source_ids,
            }
        )

    horizontal_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rule in horizontal_rules:
        rect = rule["rect"]
        horizontal_groups[
            (
                round(rect.x0 / tolerance),
                round(rect.x1 / tolerance),
            )
        ].append(rule)

    vertical_by_x: dict[int, list[dict]] = defaultdict(list)
    for rule in vertical_rules:
        vertical_by_x[
            round(rule["position"] / tolerance)
        ].append(rule)

    raw_containers: list[dict] = []
    for group in horizontal_groups.values():
        ordered = sorted(group, key=lambda item: item["position"])
        unique_positions: list[float] = []
        rules_by_position: dict[float, list[dict]] = defaultdict(list)
        for rule in ordered:
            position = float(rule["position"])
            matched_position = next(
                (
                    existing
                    for existing in unique_positions
                    if abs(existing - position) <= tolerance
                ),
                None,
            )
            if matched_position is None:
                unique_positions.append(position)
                matched_position = position
            rules_by_position[matched_position].append(rule)

        for top, bottom in zip(unique_positions, unique_positions[1:]):
            top_rules = rules_by_position[top]
            bottom_rules = rules_by_position[bottom]
            x0 = statistics.median(
                rule["rect"].x0 for rule in [*top_rules, *bottom_rules]
            )
            x1 = statistics.median(
                rule["rect"].x1 for rule in [*top_rules, *bottom_rules]
            )
            if x1 <= x0:
                continue
            container_rect = fitz.Rect(
                x0,
                top,
                x1,
                bottom,
            ).normalize()
            if not image_is_large_enough(
                image_rect=container_rect,
                page_rect=page_rect,
            ):
                continue
            if (
                page_area > 0
                and rect_area(container_rect) / page_area
                >= max_background_area_page_ratio
            ):
                continue
            if any(
                rect_is_mostly_inside_any(
                    rect=container_rect,
                    container_rects=[table_rect],
                )
                for table_rect in all_table_rects
            ):
                continue

            left_rules: list[dict] = []
            right_rules: list[dict] = []
            for x_bucket in (
                round(x0 / tolerance) - 1,
                round(x0 / tolerance),
                round(x0 / tolerance) + 1,
            ):
                left_rules.extend(vertical_by_x.get(x_bucket, []))
            for x_bucket in (
                round(x1 / tolerance) - 1,
                round(x1 / tolerance),
                round(x1 / tolerance) + 1,
            ):
                right_rules.extend(vertical_by_x.get(x_bucket, []))

            left_rules = [
                rule
                for rule in left_rules
                if abs(rule["position"] - x0) <= tolerance
                and rule["rect"].y0 <= top + tolerance
                and rule["rect"].y1 >= bottom - tolerance
            ]
            right_rules = [
                rule
                for rule in right_rules
                if abs(rule["position"] - x1) <= tolerance
                and rule["rect"].y0 <= top + tolerance
                and rule["rect"].y1 >= bottom - tolerance
            ]
            if not left_rules or not right_rules:
                continue

            boundary_ids = sorted(
                set().union(
                    *(rule["source_ids"] for rule in [
                        *top_rules,
                        *bottom_rules,
                        *left_rules,
                        *right_rules,
                    ])
                )
            )
            interior_candidate_ids = [
                element_id
                for element_id, candidate_center
                in available_candidate_centers.items()
                if element_id not in boundary_ids
                and container_rect.contains(
                    candidate_center
                )
            ]
            if not interior_candidate_ids:
                continue

            raw_containers.append(
                {
                    "bbox": container_rect,
                    "boundary_drawing_element_ids": boundary_ids,
                    "interior_candidate_element_ids": sorted(
                        interior_candidate_ids
                    ),
                }
            )

    unique_by_bbox: dict[tuple[float, ...], dict] = {}
    for container in raw_containers:
        rect = container["bbox"]
        key = tuple(round(value, 3) for value in rect)
        if key not in unique_by_bbox:
            unique_by_bbox[key] = container
        else:
            unique_by_bbox[key]["boundary_drawing_element_ids"] = sorted(
                set(
                    unique_by_bbox[key][
                        "boundary_drawing_element_ids"
                    ]
                ).union(container["boundary_drawing_element_ids"])
            )

    outermost: list[dict] = []
    for container in sorted(
        unique_by_bbox.values(),
        key=lambda item: -rect_area(item["bbox"]),
    ):
        rect = container["bbox"]
        if any(
            parent["bbox"].contains(rect.tl)
            and parent["bbox"].contains(rect.br)
            and rect_area(parent["bbox"]) > rect_area(rect)
            for parent in outermost
        ):
            continue
        outermost.append(container)

    outermost.sort(
        key=lambda item: (
            item["bbox"].y0,
            item["bbox"].x0,
            item["bbox"].y1,
            item["bbox"].x1,
        )
    )
    for container_number, container in enumerate(outermost, start=1):
        container["container_element_id"] = (
            page_inventory.register_visual_container(
                container_number=container_number,
                bbox=container["bbox"],
                payload={
                    "boundary_drawing_element_ids": container[
                        "boundary_drawing_element_ids"
                    ],
                    "interior_candidate_element_ids": container[
                        "interior_candidate_element_ids"
                    ],
                },
            )
        )
    return outermost

def text_bbox_overlaps_visual_crop(
    text_rect: fitz.Rect,
    visual_rect: fitz.Rect,
    expanded_rect: fitz.Rect,
    maximum_gap: float,
    overlap_threshold: float = VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO,
) -> bool:
    """判断文字框是否足够贴近视觉区域，可作为截图边界信息来源。"""
    text_area = rect_area(text_rect)
    if text_area <= 0:
        return False

    if text_rect.y1 <= visual_rect.y0:
        return False

    intersection = text_rect & expanded_rect
    if intersection.is_empty:
        return False

    if rect_area(intersection) / text_area >= overlap_threshold:
        return True

    horizontal_gap = max(
        visual_rect.x0 - text_rect.x1,
        text_rect.x0 - visual_rect.x1,
        0.0,
    )
    vertical_gap = max(
        visual_rect.y0 - text_rect.y1,
        text_rect.y0 - visual_rect.y1,
        0.0,
    )
    horizontal_projection = max(
        0.0,
        min(text_rect.x1, visual_rect.x1)
        - max(text_rect.x0, visual_rect.x0),
    )
    vertical_projection = max(
        0.0,
        min(text_rect.y1, visual_rect.y1)
        - max(text_rect.y0, visual_rect.y0),
    )

    return (
        (
            horizontal_gap <= maximum_gap
            and vertical_projection >= min(text_rect.height, visual_rect.height) * 0.5
        )
        or (
            vertical_gap <= maximum_gap
            and horizontal_projection >= min(text_rect.width, visual_rect.width) * 0.5
        )
    )

def text_is_owned_by_visual_structure(
    text_rect: fitz.Rect,
    structural_rects: list[fitz.Rect],
    overlap_threshold: float = (
        VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO
    ),
) -> bool:
    """判断文字是否实际落入最终矢量、规则线框架或原始图片结构。"""
    text_area = rect_area(text_rect)
    if text_area <= 0:
        return False

    center = rect_center(text_rect)
    for structural_rect in structural_rects:
        structural_rect = fitz.Rect(structural_rect)
        if structural_rect.contains(center):
            return True

        intersection = text_rect & structural_rect
        if (
            not intersection.is_empty
            and rect_area(intersection) / text_area
            >= overlap_threshold
        ):
            return True

    return False

def promote_right_attached_crop_text_lines(
    crop_text_lines: list[dict],
    attached_text_lines: list[dict],
    visual_rect: fitz.Rect,
    attached_image_rects: list[fitz.Rect],
) -> list[dict]:
    """返回可与视觉截图组成同一Markdown原子的右侧附着文字。"""
    if not crop_text_lines or not attached_image_rects:
        return []

    if any(
        image_rect.height > visual_rect.width
        for image_rect in attached_image_rects
    ):
        return []

    def is_right_attached(line: dict) -> bool:
        line_rect = fitz.Rect(line["bbox"])
        vertical_projection = max(
            0.0,
            min(line_rect.y1, visual_rect.y1)
            - max(line_rect.y0, visual_rect.y0),
        )
        return (
            line_rect.x0 >= visual_rect.x1
            and vertical_projection > 0.0
        )

    attached_ids = {id(line) for line in attached_text_lines}
    return [
        line
        for line in crop_text_lines
        if id(line) not in attached_ids and is_right_attached(line)
    ]

def drawing_extension_segments(
    drawing: dict,
) -> list[tuple[fitz.Point, fitz.Point]]:
    """提取可用于视觉区域端点附着的直线/曲线端点段。"""
    segments: list[tuple[fitz.Point, fitz.Point]] = []

    for item in drawing.get("items", []):
        item_type = item[0]
        if item_type == "l":
            first, second = fitz.Point(item[1]), fitz.Point(item[2])
            if math.dist((first.x, first.y), (second.x, second.y)) >= 4.0:
                segments.append((first, second))
        elif item_type == "c":
            first, second = fitz.Point(item[1]), fitz.Point(item[4])
            if math.dist((first.x, first.y), (second.x, second.y)) >= 4.0:
                segments.append((first, second))
        elif item_type == "re":
            rect = fitz.Rect(item[1]).normalize()
            major = max(rect.width, rect.height)
            minor = min(rect.width, rect.height)
            if major < 4.0 or minor > 3.0:
                continue
            if rect.width >= rect.height:
                center_y = (rect.y0 + rect.y1) / 2
                segments.append(
                    (
                        fitz.Point(rect.x0, center_y),
                        fitz.Point(rect.x1, center_y),
                    )
                )
            else:
                center_x = (rect.x0 + rect.x1) / 2
                segments.append(
                    (
                        fitz.Point(center_x, rect.y0),
                        fitz.Point(center_x, rect.y1),
                    )
                )

    return segments

def segment_rect_intersection_interval(
    first: fitz.Point,
    second: fitz.Point,
    rect: fitz.Rect,
) -> tuple[float, float] | None:
    """返回线段落入矩形的参数区间；无相交时返回None。"""
    rect = fitz.Rect(rect).normalize()
    dx = second.x - first.x
    dy = second.y - first.y
    lower, upper = 0.0, 1.0

    for coefficient, constant in (
        (-dx, first.x - rect.x0),
        (dx, rect.x1 - first.x),
        (-dy, first.y - rect.y0),
        (dy, rect.y1 - first.y),
    ):
        if abs(coefficient) <= 1e-9:
            if constant < 0:
                return None
            continue

        parameter = constant / coefficient
        if coefficient < 0:
            lower = max(lower, parameter)
        else:
            upper = min(upper, parameter)
        if lower > upper:
            return None

    return lower, upper

def segment_bridges_opposite_visual_sides(
    first: fitz.Point,
    second: fitz.Point,
    visual_rect: fitz.Rect,
    boundary_tolerance: float,
) -> bool:
    """判断轴对齐线段是否连接视觉区域的两条相对边。"""
    visual_rect = fitz.Rect(visual_rect).normalize()
    delta_x = abs(second.x - first.x)
    delta_y = abs(second.y - first.y)

    if (
        delta_x > 0
        and delta_y / delta_x
        <= MAX_HORIZONTAL_LINE_SLOPE_RATIO
    ):
        left_x = min(first.x, second.x)
        right_x = max(first.x, second.x)
        return (
            abs(left_x - visual_rect.x0) <= boundary_tolerance
            and abs(right_x - visual_rect.x1)
            <= boundary_tolerance
        )

    if (
        delta_y > 0
        and delta_x / delta_y
        <= MAX_HORIZONTAL_LINE_SLOPE_RATIO
    ):
        top_y = min(first.y, second.y)
        bottom_y = max(first.y, second.y)
        return (
            abs(top_y - visual_rect.y0) <= boundary_tolerance
            and abs(bottom_y - visual_rect.y1)
            <= boundary_tolerance
        )

    return False

def collect_visual_region_terminal_endpoints(
    visual_rect: fitz.Rect,
    drawings: list[dict],
    boundary_tolerance: float = (
        VISUAL_ENDPOINT_BOUNDARY_TOLERANCE_POINTS
    ),
) -> list[tuple[fitz.Point, fitz.Rect]]:
    """收集说明引线的边界/外部端点，并排除跨满区域的结构线。"""
    visual_rect = fitz.Rect(visual_rect).normalize()
    terminal_endpoints: list[tuple[fitz.Point, fitz.Rect]] = []

    for drawing in drawings:
        drawing_rect = fitz.Rect(drawing["rect"]).normalize()
        if not drawing_rect.intersects(visual_rect):
            continue

        for first, second in drawing_extension_segments(drawing):
            interval = segment_rect_intersection_interval(
                first,
                second,
                visual_rect,
            )
            if interval is None:
                continue

            if segment_bridges_opposite_visual_sides(
                first=first,
                second=second,
                visual_rect=visual_rect,
                boundary_tolerance=boundary_tolerance,
            ):
                continue

            for endpoint in (first, second):
                if visual_rect.contains(endpoint):
                    boundary_distance = min(
                        endpoint.x - visual_rect.x0,
                        visual_rect.x1 - endpoint.x,
                        endpoint.y - visual_rect.y0,
                        visual_rect.y1 - endpoint.y,
                    )
                    if boundary_distance > boundary_tolerance:
                        continue

                terminal_endpoints.append((endpoint, drawing_rect))

    return terminal_endpoints

def promote_text_lines_at_visual_endpoints(
    candidate_text_lines: list[dict],
    attached_text_lines: list[dict],
    visual_rect: fitz.Rect,
    drawings: list[dict],
    maximum_gap: float,
) -> tuple[list[dict], list[fitz.Rect]]:
    """提升靠近视觉区域外部/边界端点的文字，并返回用于截图的矢量bbox。"""
    if not candidate_text_lines:
        return [], []

    terminal_endpoints = collect_visual_region_terminal_endpoints(
        visual_rect=visual_rect,
        drawings=drawings,
    )
    if not terminal_endpoints:
        return [], []

    attached_ids = {id(line) for line in attached_text_lines}
    promoted: list[dict] = []
    promoted_ids: set[int] = set()
    extension_rects: list[fitz.Rect] = []

    for line in candidate_text_lines:
        line_id = id(line)
        if line_id in attached_ids or line_id in promoted_ids:
            continue

        text_rect = fitz.Rect(line["bbox"])
        nearest_endpoint_distance = math.inf
        nearest_drawing_rect: fitz.Rect | None = None

        for endpoint, drawing_rect in terminal_endpoints:
            distance = point_to_rect_distance(endpoint, text_rect)
            endpoint_gap = min(
                maximum_gap,
                max(4.0, text_rect.height * 0.85),
            )
            if distance > endpoint_gap or distance >= nearest_endpoint_distance:
                continue

            nearest_endpoint_distance = distance
            nearest_drawing_rect = drawing_rect

        if nearest_drawing_rect is None:
            continue

        promoted.append(line)
        promoted_ids.add(line_id)
        extension_rects.append(nearest_drawing_rect)

    return promoted, extension_rects

def collect_region_external_connector_endpoints(
    region_rect: fitz.Rect,
    drawings: list[dict],
    boundary_tolerance: float = (
        VISUAL_ENDPOINT_BOUNDARY_TOLERANCE_POINTS
    ),
) -> list[tuple[fitz.Point, dict]]:
    """返回从区域内部穿到外部的连接矢量外侧端点。"""
    region_rect = fitz.Rect(region_rect).normalize()
    external_endpoints: list[tuple[fitz.Point, dict]] = []

    for drawing in drawings:
        for first, second in drawing_extension_segments(drawing):
            interval = segment_rect_intersection_interval(
                first,
                second,
                region_rect,
            )
            if interval is None:
                continue

            lower, upper = interval
            first_is_outside = lower > 1e-9
            second_is_outside = upper < 1.0 - 1e-9

            if first_is_outside == second_is_outside:
                continue
            if segment_bridges_opposite_visual_sides(
                first=first,
                second=second,
                visual_rect=region_rect,
                boundary_tolerance=boundary_tolerance,
            ):
                continue

            external_endpoints.append(
                (
                    first if first_is_outside else second,
                    drawing,
                )
            )

    return external_endpoints

def promote_text_lines_at_region_external_connectors(
    candidate_text_lines: list[dict],
    attached_text_lines: list[dict],
    region_rect: fitz.Rect,
    drawings: list[dict],
    maximum_gap: float,
) -> tuple[list[dict], list[dict]]:
    """一跳提升外连矢量端点附近文字，并返回对应原始drawing。"""
    if not candidate_text_lines:
        return [], []

    terminal_endpoints = collect_region_external_connector_endpoints(
        region_rect=region_rect,
        drawings=drawings,
    )
    if not terminal_endpoints:
        return [], []

    attached_ids = {id(line) for line in attached_text_lines}
    promoted: list[dict] = []
    promoted_ids: set[int] = set()
    extension_drawings: list[dict] = []
    extension_drawing_ids: set[int] = set()

    for line in candidate_text_lines:
        line_id = id(line)
        if line_id in attached_ids or line_id in promoted_ids:
            continue

        text_rect = fitz.Rect(line["bbox"]).normalize()
        nearest_endpoint_distance = math.inf
        nearest_drawing: dict | None = None

        for endpoint, drawing in terminal_endpoints:
            distance = point_to_rect_distance(endpoint, text_rect)
            endpoint_gap = min(
                maximum_gap,
                max(4.0, text_rect.height * 0.85),
            )
            if distance > endpoint_gap or distance >= nearest_endpoint_distance:
                continue
            nearest_endpoint_distance = distance
            nearest_drawing = drawing

        if nearest_drawing is None:
            continue

        promoted.append(line)
        promoted_ids.add(line_id)
        if id(nearest_drawing) not in extension_drawing_ids:
            extension_drawing_ids.add(id(nearest_drawing))
            extension_drawings.append(nearest_drawing)

    return promoted, extension_drawings


def format_visual_region_native_text(
    text_lines: list[dict],
    page_rect: fitz.Rect,
    *,
    sort_lines_in_reading_order: Callable[..., list[dict]],
) -> tuple[str, list[dict]]:
    """将已归属视觉区域的原生文字格式化为Markdown引用块。"""

    ordered_lines: list[dict] = []
    for angle in [0, -90, 90, 180]:
        direction_lines = [
            line for line in text_lines if line["angle"] == angle
        ]
        if not direction_lines:
            continue
        ordered_lines.extend(
            sort_lines_in_reading_order(
                lines=direction_lines,
                source_angle=angle,
                page_rect=page_rect,
            )
        )

    if not ordered_lines:
        return "", []

    quote_rows = ["> **图内文字（PDF原生文本）**"]
    for line in ordered_lines:
        text = " ".join(line["text"].splitlines()).strip()
        text = text.replace("\\", "\\\\").replace("|", "\\|")
        if text:
            quote_rows.append(f"> {text}  ")
    return "\n".join(quote_rows) + "\n", ordered_lines


def build_top_sort_bbox(render_bbox: fitz.Rect) -> fitz.Rect:
    """为视觉资产生成仅用于阅读顺序的顶部锚点框。"""

    rect = fitz.Rect(render_bbox).normalize()
    if rect.width <= 0 or rect.height <= 0:
        return rect
    return fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x1,
        min(rect.y1, rect.y0 + SORT_BBOX_TOP_HEIGHT_POINTS),
    )


def build_visual_region_output_record(
    *,
    page_number: int,
    region_number: int,
    region: dict,
    output_bbox: fitz.Rect,
    output_path: object,
    output_width: int,
    output_height: int,
    render_seconds: float,
    page_rect: fitz.Rect,
    sort_lines_in_reading_order: Callable[..., list[dict]],
) -> tuple[dict, dict, fitz.Rect, list[fitz.Rect]]:
    """构造一个视觉区域的Markdown元素、记录和文字去重框。"""

    render_bbox = fitz.Rect(output_bbox)
    output_name = getattr(output_path, "name", str(output_path).rsplit("/", 1)[-1])
    relative_path = f"visual_regions/{output_name}"
    image_markdown = (
        f"![Page {page_number} Visual Region {region_number}]"
        f"({relative_path})"
    )
    native_text_markdown, ordered_native_text_lines = (
        format_visual_region_native_text(
            text_lines=region.get("included_text_lines", []),
            page_rect=page_rect,
            sort_lines_in_reading_order=sort_lines_in_reading_order,
        )
    )
    markdown_text = (
        f"{image_markdown}\n\n{native_text_markdown}"
        if native_text_markdown
        else image_markdown
    )
    sort_source_bbox = fitz.Rect(
        region.get("sort_source_bbox", render_bbox)
    )
    sort_bbox = (
        build_top_sort_bbox(sort_source_bbox)
        if ordered_native_text_lines
        else fitz.Rect(render_bbox)
    )
    claim_id = f"page#{page_number}:visual_region#{region_number}"
    markdown_element = {
        "bbox": render_bbox,
        "render_bbox": render_bbox,
        "sort_bbox": sort_bbox,
        "text": markdown_text,
        "direction": (1.0, 0.0),
        "angle": 0,
        "kind": "image",
        "claim_id": claim_id,
        "markdown": markdown_text,
        "mode": "rendered_visual_region",
        "output_path": output_path,
    }
    record = {
        "page_number": page_number,
        "region_number": region_number,
        "claim_id": claim_id,
        "bbox": [round(value, 3) for value in render_bbox],
        "render_bbox": [round(value, 3) for value in render_bbox],
        "sort_bbox": [round(value, 3) for value in sort_bbox],
        "sort_source_bbox": [
            round(value, 3) for value in sort_source_bbox
        ],
        "content_bbox": [
            round(value, 3)
            for value in fitz.Rect(
                region.get("content_bbox", sort_source_bbox)
            )
        ],
        "drawing_count": region["drawing_count"],
        "component_count": region["component_count"],
        "closed_visual_container_element_ids": region.get(
            "closed_visual_container_element_ids",
            [],
        ),
        "included_text_line_count": region["included_text_line_count"],
        "crop_text_line_count": region.get("crop_text_line_count", 0),
        "promoted_crop_text_line_count": region.get(
            "promoted_crop_text_line_count",
            0,
        ),
        "leader_promoted_text_line_count": region.get(
            "leader_promoted_text_line_count",
            0,
        ),
        "endpoint_promoted_text_line_count": region.get(
            "endpoint_promoted_text_line_count",
            0,
        ),
        "endpoint_extension_drawing_count": region.get(
            "endpoint_extension_drawing_count",
            0,
        ),
        "rule_line_frameworks": region.get("rule_line_frameworks", []),
        "native_text_lines": [
            {
                "text": line["text"],
                "bbox": [round(value, 3) for value in line["bbox"]],
                "angle": line["angle"],
            }
            for line in ordered_native_text_lines
        ],
        "included_image_count": region["included_image_count"],
        "output_path": str(output_path),
        "output_width": output_width,
        "output_height": output_height,
        "render_seconds": round(render_seconds, 6),
    }
    owned_text_rects = [
        fitz.Rect(line["bbox"])
        for line in ordered_native_text_lines
    ]
    return markdown_element, record, render_bbox, owned_text_rects


def build_visual_region_claims(
    *,
    page_number: int,
    regions: list[dict],
    records: list[dict],
    page_inventory: PageInventory,
    create_claim_manifest: Callable[..., dict],
) -> list[dict]:
    """为已渲染视觉区域构造稳定ID Claim记录。"""

    claims: list[dict] = []
    for region_number, region in enumerate(regions, start=1):
        consumed_drawing_ids = list(
            region.get("drawing_element_ids", [])
        )
        consumed_image_ids = list(
            region.get("attached_image_element_ids", [])
        )
        consumed_text_ids = page_inventory.ids_for_payloads(
            region.get("included_text_lines", []),
            "text",
        )
        consumed_container_ids = list(
            region.get("closed_visual_container_element_ids", [])
        )
        feature_ids = sorted(
            set(consumed_drawing_ids).union(
                consumed_image_ids,
                consumed_container_ids,
            )
        )
        claimed_ids = sorted(set(feature_ids).union(consumed_text_ids))
        reason_codes = ["connected_vector_region"]
        if region.get("rule_line_frameworks"):
            reason_codes.append("rule_line_framework")
        if consumed_image_ids:
            reason_codes.append("attached_source_image")
        if consumed_text_ids:
            reason_codes.append("owned_native_text")
        confidence = (
            0.85
            if region.get("rule_line_frameworks") or consumed_image_ids
            else 0.75
        )

        output_path = records[region_number - 1]["output_path"]
        claims.append(
            create_claim_manifest(
                detector_name="visual_region",
                output_element_id=(
                    f"page#{page_number}:visual_region#{region_number}"
                ),
                input_element_ids=claimed_ids,
                claimed_element_ids=claimed_ids,
                feature_element_ids=feature_ids,
                bbox=fitz.Rect(region["bbox"]),
                confidence=confidence,
                reason_codes=reason_codes,
                output_artifacts=[
                    {
                        "type": "visual_region_png",
                        "region_number": region_number,
                        "path": str(output_path),
                        "bbox": [
                            round(value, 3)
                            for value in region["bbox"]
                        ],
                    }
                ],
                metadata={
                    "region_number": region_number,
                    "drawing_count": region["drawing_count"],
                    "included_image_count": region[
                        "included_image_count"
                    ],
                    "included_text_line_count": region[
                        "included_text_line_count"
                    ],
                    "drawing_element_ids": consumed_drawing_ids,
                    "image_element_ids": consumed_image_ids,
                    "visual_container_element_ids": consumed_container_ids,
                    "text_element_ids": consumed_text_ids,
                },
            )
        )
    return claims


def render_visual_regions(
    *,
    page: fitz.Page,
    page_number: int,
    regions: list[dict],
    regions_dir: Path,
    page_inventory: PageInventory,
    render_dpi: int,
    sort_lines_in_reading_order: Callable[..., list[dict]],
    create_claim_manifest: Callable[..., dict],
) -> tuple[
    list[dict],
    list[dict],
    list[fitz.Rect],
    list[fitz.Rect],
    list[dict],
]:
    """渲染视觉区域并返回记录、Markdown、bbox和Claim。"""

    records: list[dict] = []
    markdown_elements: list[dict] = []
    output_rects: list[fitz.Rect] = []
    owned_text_rects: list[fitz.Rect] = []

    for region_number, region in enumerate(regions, start=1):
        output_bbox = padded_image_clip(
            bbox=region["bbox"],
            page_rect=page.rect,
        )
        output_name = (
            f"page_{page_number:03d}_visual_region_"
            f"{region_number:02d}.png"
        )
        output_path = regions_dir / output_name
        render_started = time.perf_counter()
        pixmap = page.get_pixmap(
            clip=output_bbox,
            dpi=render_dpi,
            alpha=False,
            annots=True,
        )
        pixmap.save(output_path)
        render_seconds = time.perf_counter() - render_started

        markdown_element, record, render_bbox, region_owned_text_rects = (
            build_visual_region_output_record(
                page_number=page_number,
                region_number=region_number,
                region=region,
                output_bbox=output_bbox,
                output_path=output_path,
                output_width=pixmap.width,
                output_height=pixmap.height,
                render_seconds=render_seconds,
                page_rect=page.rect,
                sort_lines_in_reading_order=sort_lines_in_reading_order,
            )
        )
        output_rects.append(render_bbox)
        owned_text_rects.extend(region_owned_text_rects)
        markdown_elements.append(markdown_element)
        records.append(record)

    markdown_elements.sort(
        key=lambda item: (item["bbox"].y0, item["bbox"].x0)
    )
    claims = build_visual_region_claims(
        page_number=page_number,
        regions=regions,
        records=records,
        page_inventory=page_inventory,
        create_claim_manifest=create_claim_manifest,
    )
    return (
        records,
        markdown_elements,
        output_rects,
        owned_text_rects,
        claims,
    )

def point_to_rect_distance(
    point: fitz.Point,
    rect: fitz.Rect,
) -> float:
    """返回点到矩形的最短欧氏距离；点在矩形内时为 0。"""
    delta_x = max(rect.x0 - point.x, 0.0, point.x - rect.x1)
    delta_y = max(rect.y0 - point.y, 0.0, point.y - rect.y1)
    return math.hypot(delta_x, delta_y)

def detect_coarse_visual_regions(
    page: fitz.Page,
    page_dict: dict,
    text_lines: list[dict],
    all_table_rects: list[fitz.Rect],
    excluded_header_footer_rects: list[fitz.Rect],
    logical_horizontal_rules: list[dict],
    drawings: list[dict],
    source_images: list[dict],
    page_inventory: PageInventory,
    axis_geometry_index: dict,
    excluded_drawing_element_ids: set[str] | None = None,
    *,
    max_background_area_page_ratio: float,
    build_endpoint_marker_grid: Callable[[list[dict]], dict],
    drawing_has_endpoint_marker: Callable[..., bool],
    drawing_has_visible_semantic_stroke: Callable[[dict], bool],
    drawing_is_filled_horizontal_bar: Callable[[dict], bool],
    drawing_is_long_horizontal_single_line: Callable[[dict], bool],
    drawing_is_page_enclosing_stroke_frame: Callable[..., bool],
    detect_rule_line_framework_bbox: Callable[..., tuple],
    group_horizontal_rules_by_span: Callable[[list[dict]], list[list[dict]]],
    unique_rule_positions: Callable[[list[dict]], list[float]],
    rule_element_ids: Callable[[dict], set[str]],
    rect_is_contained_with_tolerance: Callable[..., bool],
    rect_is_mostly_inside_any: Callable[..., bool],
) -> list[dict]:
    """检测表格外的粗粒度矢量视觉区域。"""
    page_drawings = drawings
    page_frame_drawing_ids = {
        element_id
        for element_id in page_inventory.ids_of_kind("drawing")
        if page_inventory.entry(element_id)
        .get("labels", {})
        .get("page_enclosing_stroke_frame", False)
    }
    page_frame_drawing_ids.discard(None)
    visual_drawings = [
        drawing
        for drawing in page_drawings
        if page_inventory.id_for_payload(
            drawing,
            "drawing",
        ) not in (
            set(excluded_drawing_element_ids or set()).union(
                page_frame_drawing_ids
            )
        )
    ]
    excluded_visual_drawing_ids = set(
        excluded_drawing_element_ids or set()
    ).union(page_frame_drawing_ids)
    visual_logical_horizontal_rules = [
        rule
        for rule in logical_horizontal_rules
        if not (
            rule_element_ids(rule)
            & excluded_visual_drawing_ids
        )
    ]
    candidates = collect_visual_drawing_candidates(
        page=page,
        all_table_rects=all_table_rects,
        excluded_header_footer_rects=excluded_header_footer_rects,
        drawings=visual_drawings,
        page_inventory=page_inventory,
        excluded_drawing_element_ids=(
            excluded_drawing_element_ids
        ),
        max_background_area_page_ratio=(
            max_background_area_page_ratio
        ),
        build_endpoint_marker_grid=build_endpoint_marker_grid,
        drawing_has_endpoint_marker=drawing_has_endpoint_marker,
        drawing_is_filled_horizontal_bar=(
            drawing_is_filled_horizontal_bar
        ),
        drawing_is_long_horizontal_single_line=(
            drawing_is_long_horizontal_single_line
        ),
        drawing_is_page_enclosing_stroke_frame=(
            drawing_is_page_enclosing_stroke_frame
        ),
        rect_is_mostly_inside_any=rect_is_mostly_inside_any,
    )

    if not candidates:
        return []

    closed_visual_containers = detect_closed_visual_rule_containers(
        page_rect=page.rect,
        axis_geometry_index=axis_geometry_index,
        page_inventory=page_inventory,
        candidate_element_ids={
            candidate["element_id"]
            for candidate in candidates
            if candidate.get("element_id") is not None
        },
        excluded_drawing_element_ids=(
            set(excluded_drawing_element_ids or set()).union(
                page_frame_drawing_ids
            )
        ),
        all_table_rects=all_table_rects,
        container_rule_tolerance_points=(
            CLOSED_VISUAL_CONTAINER_RULE_TOLERANCE_POINTS
        ),
        max_background_area_page_ratio=(
            max_background_area_page_ratio
        ),
        drawing_has_visible_semantic_stroke=(
            drawing_has_visible_semantic_stroke
        ),
        rule_element_ids=rule_element_ids,
        rect_is_mostly_inside_any=rect_is_mostly_inside_any,
    )
    regions = build_visual_regions_from_candidates(
        page_rect=page.rect,
        candidates=candidates,
        closed_visual_containers=closed_visual_containers,
        all_table_rects=all_table_rects,
        mask_pixels_per_point=VISUAL_MASK_PIXELS_PER_POINT,
        close_gap_page_ratio=VISUAL_MASK_CLOSE_GAP_PAGE_RATIO,
    )

    regions = [
        region
        for region in regions
        if image_is_large_enough(
            image_rect=region["bbox"],
            page_rect=page.rect,
        )
    ]
    regions = merge_visual_region_records(
        regions=regions,
        page_rect=page.rect,
    )
    regions = merge_pure_vector_same_layer_once(
        regions=regions,
        page_rect=page.rect,
    )
    regions = merge_visual_regions_by_horizontal_relations_once(
        regions=regions,
        drawings=visual_drawings,
        page_rect=page.rect,
        logical_horizontal_rules=(
            visual_logical_horizontal_rules
        ),
        drawing_is_filled_horizontal_bar=(
            drawing_is_filled_horizontal_bar
        ),
        drawing_is_long_horizontal_single_line=(
            drawing_is_long_horizontal_single_line
        ),
        group_horizontal_rules_by_span=group_horizontal_rules_by_span,
        unique_rule_positions=unique_rule_positions,
    )

    for region in regions:
        framework_bbox, framework_metadata = (
            detect_rule_line_framework_bbox(
                region_bbox=fitz.Rect(region["bbox"]),
                drawings=visual_drawings,
                page_rect=page.rect,
                text_lines=text_lines,
                page_inventory=page_inventory,
                logical_horizontal_rules=(
                    visual_logical_horizontal_rules
                ),
            )
        )
        if framework_bbox is not None:
            closed_container_ids = region.get(
                "closed_visual_container_element_ids",
                [],
            )
            if closed_container_ids:
                closed_container_rect = page_inventory.bbox_of(
                    closed_container_ids[0]
                )
                if not rect_is_contained_with_tolerance(
                    inner_rect=fitz.Rect(framework_bbox),
                    outer_rect=closed_container_rect,
                    tolerance=CLOSED_VISUAL_CONTAINER_RULE_TOLERANCE_POINTS,
                ):
                    framework_bbox = None

        if framework_bbox is not None:
            original_vector_cluster_bbox = fitz.Rect(
                region["bbox"]
            )
            region["vector_cluster_bbox"] = (
                original_vector_cluster_bbox
            )
            region["bbox"] = union_rects(
                [
                    original_vector_cluster_bbox,
                    framework_bbox,
                ]
            )
            region["rule_line_frameworks"] = [framework_metadata]
            region["drawing_element_ids"] = sorted(
                set(region.get("drawing_element_ids", [])).union(
                    framework_metadata.get(
                        "drawing_element_ids", []
                    )
                )
            )
        else:
            region["rule_line_frameworks"] = []

    source_images = [
        image
        for image in source_images
        if not rect_is_mostly_inside_any(
            rect=image["bbox"],
            container_rects=excluded_header_footer_rects,
        )
        and not any(
            band.contains(rect_center(image["bbox"]))
            for band in excluded_header_footer_rects
        )
    ]
    attachment_margin = (
        min(page.rect.width, page.rect.height)
        * VISUAL_REGION_TEXT_MARGIN_PAGE_RATIO
    )

    for region in regions:
        structural_bbox = fitz.Rect(region["bbox"])
        expanded = fitz.Rect(
            max(page.rect.x0, structural_bbox.x0 - attachment_margin),
            max(page.rect.y0, structural_bbox.y0 - attachment_margin),
            min(page.rect.x1, structural_bbox.x1 + attachment_margin),
            min(page.rect.y1, structural_bbox.y1 + attachment_margin),
        )
        attached_images = [
            image
            for image in source_images
            if expanded.intersects(image["bbox"])
            and not rect_is_mostly_inside_any(
                image["bbox"],
                all_table_rects,
            )
        ]
        attached_image_rects = [
            fitz.Rect(image["bbox"])
            for image in attached_images
        ]
        structural_rects = [
            structural_bbox,
            *attached_image_rects,
        ]
        content_bbox = union_rects(structural_rects)

        attached_source_image_indices = [
            image["source_index"] for image in attached_images
        ]
        attached_image_element_ids = page_inventory.ids_for_payloads(
            attached_images,
            "image",
        )
        region["bbox"] = content_bbox
        region["content_bbox"] = content_bbox
        region["sort_source_bbox"] = content_bbox
        region["structural_rects"] = structural_rects
        region["attached_image_rects"] = attached_image_rects
        region[
            "attached_source_image_indices"
        ] = attached_source_image_indices
        region[
            "attached_image_element_ids"
        ] = attached_image_element_ids
        region["included_image_count"] = len(
            set(attached_source_image_indices)
        )

    regions = merge_visual_regions_sharing_images(
        regions=regions,
    )

    assigned_text_lines: list[list[dict]] = [
        [] for _ in regions
    ]
    assigned_owner_by_line_id: dict[int, int] = {}
    closed_container_entries = [
        page_inventory.entry(element_id)
        for element_id in page_inventory.ids_of_kind(
            "visual_container"
        )
    ]

    for line in text_lines:
        line_rect = fitz.Rect(line["bbox"])
        line_center = rect_center(line_rect)
        if any(
            table_rect.contains(line_center)
            for table_rect in all_table_rects
        ) or any(
            band.contains(line_center)
            for band in excluded_header_footer_rects
        ):
            continue

        line_container_ids = {
            entry["id"]
            for entry in closed_container_entries
            if entry["bbox"].contains(line_center)
        }

        eligible_region_indices = [
            region_index
            for region_index, region in enumerate(regions)
            if (
                not line_container_ids
                or line_container_ids.intersection(
                    region.get(
                        "closed_visual_container_element_ids",
                        [],
                    )
                )
            )
            if text_is_owned_by_visual_structure(
                text_rect=line_rect,
                structural_rects=[
                    fitz.Rect(region["content_bbox"]),
                    *(
                        fitz.Rect(rect)
                        for rect in region.get(
                            "structural_rects", []
                        )
                    ),
                ],
            )
        ]
        if not eligible_region_indices:
            continue

        owner_index = min(
            eligible_region_indices,
            key=lambda region_index: (
                point_to_rect_distance(
                    line_center,
                    fitz.Rect(
                        regions[region_index]["content_bbox"]
                    ),
                ),
                region_index,
            ),
        )
        assigned_text_lines[owner_index].append(line)
        assigned_owner_by_line_id[id(line)] = owner_index

    promoted_line_ids: set[int] = set()

    for region_index, region in enumerate(regions):
        content_bbox = fitz.Rect(region["content_bbox"])
        region["sort_source_bbox"] = fitz.Rect(content_bbox)
        visual_expanded = fitz.Rect(
            max(page.rect.x0, content_bbox.x0 - attachment_margin),
            max(page.rect.y0, content_bbox.y0 - attachment_margin),
            min(page.rect.x1, content_bbox.x1 + attachment_margin),
            min(page.rect.y1, content_bbox.y1 + attachment_margin),
        )
        crop_text_lines = [
            line
            for line in text_lines
            if not any(
                table_rect.contains(rect_center(line["bbox"]))
                for table_rect in all_table_rects
            )
            and not any(
                band.contains(rect_center(line["bbox"]))
                for band in excluded_header_footer_rects
            )
            and assigned_owner_by_line_id.get(
                id(line), region_index
            ) == region_index
            and id(line) not in promoted_line_ids
            and text_bbox_overlaps_visual_crop(
                text_rect=fitz.Rect(line["bbox"]),
                visual_rect=content_bbox,
                expanded_rect=visual_expanded,
                maximum_gap=(
                    attachment_margin
                    * VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER
                ),
            )
        ]
        attached_text_lines = assigned_text_lines[region_index]
        promoted_crop_text_lines = promote_right_attached_crop_text_lines(
            crop_text_lines=crop_text_lines,
            attached_text_lines=attached_text_lines,
            visual_rect=content_bbox,
            attached_image_rects=region.get(
                "attached_image_rects", []
            ),
        )
        leader_promoted_text_lines: list[dict] = []

        endpoint_candidate_text_lines = [
            line
            for line in text_lines
            if not any(
                table_rect.contains(rect_center(line["bbox"]))
                for table_rect in all_table_rects
            )
            and not any(
                band.contains(rect_center(line["bbox"]))
                for band in excluded_header_footer_rects
            )
            and assigned_owner_by_line_id.get(
                id(line), region_index
            ) == region_index
            and id(line) not in promoted_line_ids
        ]
        endpoint_promoted_text_lines, endpoint_extension_rects = (
            promote_text_lines_at_visual_endpoints(
                candidate_text_lines=endpoint_candidate_text_lines,
                attached_text_lines=[
                    *attached_text_lines,
                    *promoted_crop_text_lines,
                ],
                visual_rect=content_bbox,
                drawings=visual_drawings,
                maximum_gap=(
                    attachment_margin
                    * VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER
                ),
            )
        )
        promoted_line_ids.update(
            id(line)
            for line in [
                *promoted_crop_text_lines,
                *leader_promoted_text_lines,
                *endpoint_promoted_text_lines,
            ]
        )
        effective_by_id = {
            id(line): line
            for line in [
                *attached_text_lines,
                *promoted_crop_text_lines,
                *leader_promoted_text_lines,
                *endpoint_promoted_text_lines,
            ]
        }
        effective_text_lines = list(effective_by_id.values())

        render_rects = [content_bbox]
        render_rects.extend(endpoint_extension_rects)
        render_rects.extend(
            fitz.Rect(line["bbox"])
            for line in effective_text_lines
        )
        region["bbox"] = union_rects(render_rects)
        region["included_text_line_count"] = len(effective_text_lines)
        region["crop_text_line_count"] = len(crop_text_lines)
        region[
            "promoted_crop_text_line_count"
        ] = len(
            {
                id(line)
                for line in [
                    *promoted_crop_text_lines,
                    *leader_promoted_text_lines,
                    *endpoint_promoted_text_lines,
                ]
            }
        )
        region[
            "leader_promoted_text_line_count"
        ] = len(leader_promoted_text_lines)
        region[
            "endpoint_promoted_text_line_count"
        ] = len(endpoint_promoted_text_lines)
        region[
            "endpoint_extension_drawing_count"
        ] = len(endpoint_extension_rects)
        region["included_text_lines"] = effective_text_lines
    return regions
