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

#@GPT 视觉区域合并的四个页面比例策略集中到同一主题模块；核心输出摘要从这里读取同一组值。
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

#@GPT 图片视觉资产的几何策略也集中在本主题模块中；图片输入和矢量合并共享同一文件，但仍由独立函数维护。
IMAGE_PADDING_POINTS = 2.0
MIN_IMAGE_THIN_SIDE_POINTS = 12.0
MIN_IMAGE_LINE_ASPECT_RATIO = 5.0
MIN_IMAGE_AREA_POINTS_SQUARED = 160.0
COMPLEX_TRANSFORM_EPSILON = 0.01
HEADER_FOOTER_IMAGE_HASH_SIZE = (32, 16)


#@GPT 只有达到最小面积比例的图片才视为同一复合图片；零面积或仅边界接触不应触发合并。
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


#@GPT 图片尺寸过滤只删除细长装饰条和极小块；page_rect保留在公共签名中供统一调用。
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


#@GPT 图片有旋转/倾斜变换时必须渲染页面；纯水平或垂直缩放仍可直接复制原始编码。
def image_has_complex_transform(transform: tuple[float, ...]) -> bool:
    """判断原生图片变换是否包含旋转或倾斜。"""

    if len(transform) < 4:
        return True

    _, b, c, _, *_ = transform
    return (
        abs(b) > COMPLEX_TRANSFORM_EPSILON
        or abs(c) > COMPLEX_TRANSFORM_EPSILON
    )


#@GPT 复合图片渲染只增加固定留白，并裁剪到页面范围；图片和矢量截图共用此几何语义。
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


#@GPT 图片块清点只读取PDF已有dict，不在此阶段判断是否截图或是否被表格/视觉区域接管。
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


#@GPT 页眉/页尾图片模板和视觉资产去重共用解码后的缩略图摘要；压缩格式差异不应破坏同图匹配。
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


#@GPT 矢量候选生成只消费阶段A/B已有台账和局部标签；端点邻域判断仍由调用方提供，避免本模块重复构造另一套页面状态。
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


#@GPT 矢量聚类使用低分辨率占用掩码；坐标映射只决定分区，不改变最终页面渲染精度。
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


#@GPT 前置保护对象必须在闭运算结果中再次清零，避免其两侧矢量被形态学操作重新桥接。
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


#@GPT 将候选矢量转换为粗粒度视觉区域：候选容器归属、闭运算、表格保护和连通域拆分全部在此完成；文字归属与渲染留给页面编排层。
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

    #@GPT 一个候选只绑定最小的闭合容器；容器边界不会改变候选的原始身份，只用于阻止跨框合并和补全渲染范围。
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

    #@GPT 表格保护区域在闭运算后再次清零，防止保护区两侧的矢量通过形态学闭运算重新桥接。
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

                #@GPT 两个独立闭合图片框即使间距很小，也不能被普通空间合并重新桥接。
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
                    #@GPT 共享原始图片导致区域合并时，排序来源框也只合并文字扩张前的视觉内容，
                    #@GPT 避免后续重新退化为包含crop文字的完整截图bbox。
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
                    #@GPT 区域合并只做成员ID并集，确保最终视觉声明可追溯到初始连通域中的原drawing。
                    "drawing_element_ids": merged_region_element_ids(
                        [first, second],
                        "drawing_element_ids",
                    ),
                    #@GPT 普通视觉区域合并只保留闭合容器ID并集；不同容器已在上方阻止，避免后续阶段丢失隔离证据。
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
    #@GPT 同层合并只依赖区域几何，不使用drawing数量作为隐藏条件。
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

    #@GPT 采用两个已聚合bbox的上下左右极限坐标；不使用中心点，也不使用合并后新bbox递归扩张。
    #@GPT “高度投影+页面比例间隔”是窄间隔区域的主合并路径。
    normal_same_layer = (
        vertical_overlap_ratio
        >= VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO
        and horizontal_gap <= maximum_horizontal_gap
    )

    #@GPT 更远区域必须由水平连接线或共享水平框架提供明确结构关系。

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
            #@GPT 同一页的上下闭合图片框不能因纯矢量同层投影重叠而再次合并；该规则只允许同框组件合并。
            if visual_regions_cross_distinct_closed_containers(
                first=first,
                second=second,
            ):
                continue
            if pure_vector_same_layer_should_merge(
                first_rect=fitz.Rect(first["bbox"]),
                second_rect=fitz.Rect(second["bbox"]),
                page_rect=page_rect,
                #@GPT 同层关系只消费双方已聚合bbox和页面尺寸。
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
                #@GPT 一次性同层合并保留双方drawing身份，Claim阶段无需再按bbox反查来源。
                "drawing_element_ids": merged_region_element_ids(
                    [first, second],
                    "drawing_element_ids",
                ),
                #@GPT 一次性同层合并沿用闭合容器身份；不同容器已被前置判定阻止。
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
    #@GPT 水平关系合并使用阶段B逻辑横线，保证框架几何与来源ID一致。
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
    #@GPT 并查集同时维护闭合视觉容器ID；通过普通水平关系合并时，联合集合若包含两个不同容器则拒绝union，防止间接桥接。
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
        #@GPT 复用已有分隔线分类；这些线仍不进入闭运算，只在这里作为横向关系证据。
        if (
            drawing_is_filled_horizontal_bar(drawing)
            or drawing_is_long_horizontal_single_line(drawing)
        ):
            horizontal_relation_lines.append(drawing_rect)

    #@GPT 逻辑横线既提供共享框架，也可作为跨越左右区域的关系证据；这里只读取几何rect，不向视觉掩码新增drawing。
    horizontal_frame_rules = [
        dict(rule) for rule in logical_horizontal_rules
    ]
    horizontal_relation_lines.extend(
        #@GPT 风险1：逻辑横线内部可能是AxisRect，渲染/关系列表是PyMuPDF几何API边界，集中转换并保留轻量索引内部不物化的收益。
        as_fitz_rect(rule["rect"])
        for rule in logical_horizontal_rules
    )

    #@GPT 水平线直接跨过左右两个区域的投影重叠带时，认为它们属于同一横向视觉域；不检查垂直连接线。
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

    #@GPT 共享表格框架按相邻水平边界形成行带；同一行带内的多个簇合并，跨行的簇不会因上下接近而合并。
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
                #@GPT 水平关系合并后的候选继承全部成员drawing ID；关系线本身若已进入任一成员也随之保留。
                "drawing_element_ids": merged_region_element_ids(
                    members,
                    "drawing_element_ids",
                ),
                #@GPT 水平关系合并保留闭合容器ID并集；并查集已确保不会出现两个不同容器的非法并集。
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

    #@GPT 共享同一个PDF原始图片块是直接对象身份关系，不再附加Y投影或距离门槛。
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

                #@GPT 共享原图关系不能跨越两个明确闭合图片框；原图身份只合并同一容器内被拆开的矢量子簇。
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
                    #@GPT 共享原图合并发生在声明输出前，必须同时合并矢量与图片稳定ID，避免最终区域只保留最后一个子簇身份。
                    "drawing_element_ids": merged_region_element_ids(
                        [first, second],
                        "drawing_element_ids",
                    ),
                    #@GPT 共享图片合并保留闭合容器ID，供最终声明和后续审计核对。
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
                    #@GPT 框架元数据只用于审计；文字归属读取structural_rects，合并后仍保留框内关系。
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
#@GPT 闭合视觉容器检测独立于OpenCV连通域：只依据轴向规则、可见描边和候选台账登记容器，不负责文字归属或渲染。
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

    #@GPT 描边可见性只依赖单个drawing；按稳定ID缓存结果，避免规则线多次引用同一对象时重复解析。
    visible_semantic_stroke_by_element_id: dict[str, bool] = {}
    for element_id in page_inventory.ids_of_kind("drawing"):
        entry = page_inventory.entry(element_id)
        labels = entry.get("labels", {})
        if "visible_semantic_stroke" in labels:
            visible_semantic_stroke_by_element_id[element_id] = bool(
                labels["visible_semantic_stroke"]
            )
        else:
            #@GPT 独立调试调用若未提供阶段B标签，则现场执行一次等价判据。
            visible_semantic_stroke_by_element_id[element_id] = (
                drawing_has_visible_semantic_stroke(entry["payload"])
            )

    #@GPT 候选中心在全部闭合框判断中保持不变，因此按稳定ID预计算一次。
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
        #@GPT 轴向索引中的AxisRect已规范化，容器筛选直接读取坐标属性。
        rule_rect = rule["rect"]
        if rule_rect.width <= 0:
            continue
        #@GPT 规则线的全部来源drawing都必须具有可见语义描边，判断复用阶段B布尔缓存。
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
        #@GPT 竖线与横线共用轻量AxisRect，只在最终形成真实container_rect时进入PyMuPDF几何对象。
        rule_rect = rule["rect"]
        if rule_rect.height <= 0:
            continue
        #@GPT 竖线与横线复用可见描边缓存，白色或不可见线条继续被排除。
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

    #@GPT 使用1pt量化桶查找相同左右端点，避免在密集页对所有横线执行两两跨度比较。
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

    #@GPT 同一矩形可能由重复描边或分段横线重复提出；按bbox去重后再移除被更大闭框完全包住的内部框，保留最外层语义容器。
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

#@GPT 视觉文字附着基础层：先证明文字与视觉结构的几何关系，再由页面编排层决定是否写入Markdown。
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

    #@GPT 仅允许与原始视觉簇存在纵向投影的上方文字参与截图；
    #@GPT 文字即使落入固定扩张边距，只要与区域Y投影为0，就不应被视觉区域接管。
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

    #@GPT 文字在左右两侧时要求Y投影重叠，在上下两侧时要求X投影重叠；仅凭距离不能把邻栏正文并入截图。
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

    #@GPT 大图四周标注不走本规则；只有小图标重复排列时，
    #@GPT 右侧文字才可以安全视为截图中的原生文字，而不会接管普通图外说明。
    #@GPT 比较图标高度与视觉簇宽度：小而近似方形的图标可附着右侧文字，纵向大图不会触发该规则。
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
            #@GPT 曲线路径使用首尾控制点作为端点，保留其整体bbox用于截图；不把控制点当作额外文字连接点。
            first, second = fitz.Point(item[1]), fitz.Point(item[4])
            if math.dist((first.x, first.y), (second.x, second.y)) >= 4.0:
                segments.append((first, second))
        elif item_type == "re":
            #@GPT PDF常把细竖线或横线编码成填充矩形；只将细长矩形的长轴中心线作为连接段。
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

    #@GPT 近水平线必须左右两端分别触及区域左右边；它可以是外框线，也可以是框内分段线。
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

    #@GPT 垂直情形与水平情形对称；两端跨满上下边的线归入容器/分隔结构，而不是文字说明引线。
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

            #@GPT 结构线的两端可能也位于边界附近，必须在逐端点判断之前整段排除，否则仍会用其中一端吸收框外文字。
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
    #@GPT 端点附着比普通crop候选更严格：按文字行高度限制端点间距，避免图片上方正文仅因边界线而被吸收。
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

            #@GPT 要求线段恰好一个端点位于区域外，保证扩张是一跳的“图内→图外”关系；穿图线不参与。
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


#@GPT 视觉区域的原生文字只做Markdown引用格式化；阅读顺序算法由调用方注入，避免视觉模块反向依赖核心排序实现。
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


#@GPT 视觉资产保留完整渲染框，阅读顺序只在有图内文字时使用顶部窄锚点；该函数不修改原始bbox。
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


#@GPT 最终视觉输出契约集中构造Markdown原子和调试记录；截图执行与Claim仲裁仍由页面编排层负责。
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


#@GPT 视觉区域Claim仅依据候选已保存的稳定ID和输出记录构造；不再从最终扩张bbox反查页面图元。
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


#@GPT 页面渲染循环与视觉输出/Claim构造放在同一主题模块；调用方只提供已完成检测的区域和现有策略函数。
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

#@GPT 粗视觉区域总编排在同一主题模块中串联候选、容器、连通域、结构合并和文字/图片附着；外部算法依赖通过显式回调传入。
def detect_coarse_visual_regions(
    page: fitz.Page,
    page_dict: dict,
    text_lines: list[dict],
    all_table_rects: list[fitz.Rect],
    #@GPT 传入已确认页眉/页尾带，使视觉区域检测在候选提取阶段就排除重复装饰。
    excluded_header_footer_rects: list[fitz.Rect],
    #@GPT 2026-08-15：阶段B逻辑横线是视觉关系/框架检测的必需输入；删除向下传播None的假兼容接口。
    logical_horizontal_rules: list[dict],
    #@GPT 视觉粗区域接收页面drawing、原图和台账缓存，避免重复提取和对象身份漂移。
    drawings: list[dict],
    source_images: list[dict],
    page_inventory: PageInventory,
    #@GPT 封闭视觉容器复用阶段B轴向索引；不再为每个页面重新扫描drawing寻找上下左右边界。
    axis_geometry_index: dict,
    #@GPT 表格父容器只移交骨架drawing ID；表内复杂矢量仍保留在视觉检测器输入中，稍后建立child_of关系。
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
    #@GPT 水平关系阶段与候选阶段共用main()传入的页面drawing缓存。
    page_drawings = drawings
    #@GPT 表格骨架ID不仅不能进入初始视觉掩码，也不能在后续水平连接、规则框架或端点扩张阶段重新进入；建立一份视觉阶段专用drawing视图，原始页面缓存保持不变。
    #@GPT 整页描边框同样属于“页面结构而非内容”：它即使没有进入初始候选，也可能通过逻辑横线索引重新成为框架边界，因此先转换为稳定ID统一排除。
    #@GPT 阶段B缓存整页框标签，视觉候选直接读取，避免密集页重复扫描drawing路径。
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
    #@GPT visual_drawings和logical_horizontal_rules使用同一图元排除视图，防止已归表格或说明带的横线重新参与视觉合并。
    #@GPT 逻辑横线携带全部源drawing ID；只要其中任一片段已被上游表格/说明带认领，整条合成线就不再向视觉阶段提供跨区连接或框架证据，避免部分复用造成边界泄漏。
    #@GPT 逻辑横线过滤与初始drawing过滤保持同一ID集合，防止页面结构线绕过候选过滤扩展区域。
    excluded_visual_drawing_ids = set(
        excluded_drawing_element_ids or set()
    ).union(page_frame_drawing_ids)
    #@GPT 逻辑横线始终使用列表契约；无规则时传空列表。
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
        #@GPT 初始候选与后续结构阶段统一使用已移除父表格骨架的drawing视图，防止同一网格线在框架阶段被重新认领。
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

    #@GPT 先登记闭合图片框；容器检测只消费当前视觉候选的drawing ID，不把表格或页眉规则线引回视觉通道。
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
    #@GPT 候选到OpenCV初始区域的空间阶段由build_visual_regions_from_candidates统一处理。
    regions = build_visual_regions_from_candidates(
        page_rect=page.rect,
        candidates=candidates,
        closed_visual_containers=closed_visual_containers,
        all_table_rects=all_table_rects,
        mask_pixels_per_point=VISUAL_MASK_PIXELS_PER_POINT,
        close_gap_page_ratio=VISUAL_MASK_CLOSE_GAP_PAGE_RATIO,
    )

    #@GPT build_visual_regions_from_candidates统一完成容器归属、闭运算、表格保护和连通域拆分。
    #@GPT OpenCV初级连通域先经过图片尺寸门限，阻止细线噪声在粗粒度空间聚合中相互放大。
    regions = [
        region
        for region in regions
        if image_is_large_enough(
            image_rect=region["bbox"],
            page_rect=page.rect,
        )
    ]
    #@GPT 初级连通域过滤后才执行通用空间聚合，保留不规则矢量簇所需的横向、纵向及双间距合并条件。
    regions = merge_visual_region_records(
        regions=regions,
        page_rect=page.rect,
    )
    #@GPT 普通小间距聚合完成后，只做一轮纯矢量同层合并；不把文字、图片附着后的扩张bbox再次用于递归合并。
    regions = merge_pure_vector_same_layer_once(
        regions=regions,
        page_rect=page.rect,
    )
    #@GPT 普通同层合并后再依据水平连接线/共享水平框架合并同一行簇；垂直线不作为区域扩张依据。
    regions = merge_visual_regions_by_horizontal_relations_once(
        regions=regions,
        #@GPT 表格骨架不能充当两个表内/表外视觉簇之间的水平连接件。
        drawings=visual_drawings,
        page_rect=page.rect,
        #@GPT 与visual_drawings保持同一图元视图；已被上游认领的逻辑横线不能重新连接视觉簇。
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

    #@GPT 【规则线框架阶段】位于闭运算和视觉簇合并之后、文字/原图附着之前；被分隔线过滤掉的横线不重新进入闭运算，只作为最终结构边界。
    #@GPT 三条以上横线可恢复完整分段框；两条等跨度横线在框内存在内容时也可统一渲染区域。
    for region in regions:
        framework_bbox, framework_metadata = (
            detect_rule_line_framework_bbox(
                region_bbox=fitz.Rect(region["bbox"]),
                #@GPT 规则线框架只能读取视觉阶段剩余drawing；已归父表格的横竖线不能再次把整个表格包装成视觉框架。
                drawings=visual_drawings,
                page_rect=page.rect,
                #@GPT 两线模式使用get_text('dict')提取的原生文字确认框内存在说明内容，不新增OCR或第三方依赖。
                text_lines=text_lines,
                page_inventory=page_inventory,
                #@GPT 框架检测同样只读取视觉阶段尚未被认领的逻辑横线，防止表格边框再次扩张视觉截图。
                logical_horizontal_rules=(
                    visual_logical_horizontal_rules
                ),
            )
        )
        if framework_bbox is not None:
            #@GPT 多个相邻图片框可能共享上下边界；闭合容器边界优先，跨出当前容器的框架不参与扩张。
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
                    #@GPT 跨出当前闭合容器的框架不参与扩张，避免重复渲染相邻图片。
                    framework_bbox = None

        if framework_bbox is not None:
            #@GPT 规则线框架只能补全视觉内容，不能反向缩小已经检测到的矢量簇。
            original_vector_cluster_bbox = fitz.Rect(
                region["bbox"]
            )
            region["vector_cluster_bbox"] = (
                original_vector_cluster_bbox
            )
            #@GPT 取原簇与框架候选的并集：误分组不会裁掉原内容，真实外围框架仍可向外扩张。
            region["bbox"] = union_rects(
                [
                    original_vector_cluster_bbox,
                    framework_bbox,
                ]
            )
            #@GPT rule_line_frameworks使用列表保存，使共享图片合并后仍能保留每个框架的审计信息。
            region["rule_line_frameworks"] = [framework_metadata]
            #@GPT 框架横线是该视觉候选成立和扩边的直接证据，必须进入同一drawing声明集合；只合并ID，不改变bbox。
            region["drawing_element_ids"] = sorted(
                set(region.get("drawing_element_ids", [])).union(
                    framework_metadata.get(
                        "drawing_element_ids", []
                    )
                )
            )
        else:
            region["rule_line_frameworks"] = []

    #@GPT 2026-08-16：视觉阶段直接消费阶段A提取的原图缓存，不再从page_dict重复构造图片列表。
    #@GPT 不允许已识别页眉/页尾图片作为视觉区域附件，否则矢量区域仍会把被排除的图像重新带回。
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

    #@GPT 先完成“矢量/框架+原始图片”的结构合并，再在最终结构上统一判定文字归属。
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

        #@GPT 保存页面图片块的稳定source_index，供“同一原始图片多次被区域引用”的结构合并使用。
        attached_source_image_indices = [
            image["source_index"] for image in attached_images
        ]
        #@GPT 原图的source_index属于page_dict block编号，不能直接拼接成i:ID；始终通过台账对象反查得到同一编号域。
        #@GPT 原始图片通过页面台账反查稳定ID，保证声明与页面inventory使用同一编号域。
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

    #@GPT 共享图片身份是结构关系，必须在文字归属和截图扩张之前合并；合并后的sort_source_bbox明确等于结构content_bbox。
    regions = merge_visual_regions_sharing_images(
        regions=regions,
    )

    assigned_text_lines: list[list[dict]] = [
        [] for _ in regions
    ]
    assigned_owner_by_line_id: dict[int, int] = {}
    #@GPT 文本归属也必须尊重闭合视觉容器；否则第二个图片框内的编号可能因“未分配文字默认当前区域”被第一个框抢走。
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
                #@GPT content_bbox是共享图片合并后的最终结构外框；独立结构框同时保留，便于后续审计与扩展。
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
        #@GPT 附近文字只作为“小图标+右侧说明”的提升候选；普通crop候选不无条件扩大截图。
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
        #@GPT 水平、垂直和倾斜引线统一由端点附着规则处理，避免维护多套方向特例。
        leader_promoted_text_lines: list[dict] = []

        #@GPT 端点规则需要审查固定crop框之外的文字；归属仍受表格、页眉和已有区域owner约束，不会把整页文字作为截图候选。
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
                #@GPT 端点文字扩张同样不能借用父表格骨架，否则表格边线会把框外标题或正文吸入子视觉截图。
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

        #@GPT render_bbox只允许最终结构和已确认归属/提升的文字扩张；sort_source_bbox始终停留在文字扩张前的content_bbox。
        render_rects = [content_bbox]
        #@GPT 端点附着命中的矢量本身也属于截图内容；只加入本次命中文字对应的drawing，不递归扩大视觉区域。
        render_rects.extend(endpoint_extension_rects)
        render_rects.extend(
            fitz.Rect(line["bbox"])
            for line in effective_text_lines
        )
        region["bbox"] = union_rects(render_rects)
        region["included_text_line_count"] = len(effective_text_lines)
        #@GPT crop_text_line_count现在表示审查过的邻近候选数；它不再等价于截图扩张数量。
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
        #@GPT leader字段继续作为稳定报告字段；所有方向的端点标签统一记录到endpoint字段。
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
