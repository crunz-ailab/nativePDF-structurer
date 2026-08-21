"""编排原始图片和矢量视觉区域的输出处理。

模块协调台账局部标签、图片叠加
分析、原图直拷或复合渲染、视觉区域检测、端点证据以及阶段E/F边界重建；
底层区域几何和渲染记录构建委托给 ``visual_assets``。

处理流程:
    1. 为页面台账图元标记可复用的局部证据。
    2. 检测图片叠加并选择直拷或渲染模式。
    3. 通过共享视觉模块检测和渲染矢量区域。
    4. 应用Claim存活结果，重建结构边界并附着阶段F文字。
    5. 使用最终渲染框和排序框更新Markdown原子及记录。

关键不变量:
    * 原始图片身份是source-image Claim的生存feature。
    * overlay drawing可以扩张边界，但不能单独维持原图候选存活。
    * 最终重渲染使用获胜结构图元和唯一归属文字。
    * 页面drawing、图片和台账缓存必须跨阶段复用。

Orchestrate source-image and vector-region output processing.

Coordinates inventory labeling, image-overlay analysis, source-image copying
or composite rendering, vector-region detection, endpoint evidence, and
Stage-E/F boundary rebuilding. Low-level region geometry and rendering-record
construction are delegated to :mod:`visual_assets`.

Processing flow:
    1. Label page inventory entries with reusable local evidence.
    2. Detect source-image overlays and choose direct-copy or rendered modes.
    3. Detect and render vector visual regions through shared visual assets.
    4. Apply Claim survival, rebuild structure bounds, and attach Stage-F text.
    5. Update Markdown atoms and records with final render and sort bounds.

Key invariants:
    * Native image identity is the survival feature for source-image Claims.
    * Overlay drawings may expand bounds but cannot keep a source image alive.
    * Final rerendering uses won structure plus uniquely owned text.
    * Page drawings, images, and inventory caches are reused across stages.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
import math
import time

import fitz

from .claims import create_claim_manifest
from .formula_regions import EQUATION_NUMBER_PATTERN
from .geometry import rect_area, rect_center, union_rects
from .inventory import PageInventory
from .layout_geometry import (
    drawing_axis_rule,
    drawing_has_visible_semantic_stroke,
    group_horizontal_rules_by_span,
    meaningful_overlap,
    rect_is_contained_with_tolerance,
    rect_is_mostly_inside_any,
    rule_element_ids,
    unique_rule_positions,
)
from .reading_order_backend import sort_lines_in_reading_order
from .rule_bands import detect_rule_line_framework_bbox
from .visual_assets import (
    MAX_HORIZONTAL_LINE_SLOPE_RATIO,
    MIN_IMAGE_LINE_ASPECT_RATIO,
    MIN_IMAGE_THIN_SIDE_POINTS,
    VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER,
    VISUAL_REGION_TEXT_MARGIN_PAGE_RATIO,
    build_top_sort_bbox,
    detect_coarse_visual_regions,
    format_visual_region_native_text,
    image_has_complex_transform,
    image_is_large_enough,
    image_rects_overlap,
    padded_image_clip,
    point_to_rect_distance,
    promote_text_lines_at_region_external_connectors,
    render_visual_regions,
)


#@GPT 视觉处理模块集中维护渲染和端点策略；参数不包含页码或固定坐标特例。
TEXT_ROW_TOLERANCE_POINTS = 3.0
VISUAL_REGION_DPI = 200
COMPOSITE_IMAGE_DPI = 200
MAX_BACKGROUND_AREA_PAGE_RATIO = 0.80
FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS = 3.0
VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS = 8.0
VECTOR_ENDPOINT_GRID_CELL_POINTS = 12.0
MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS = 24.0




def apply_stage_e_f_to_visual_outputs(
    page: fitz.Page,
    page_inventory: PageInventory,
    resolution: dict,
    stage_f_preview: dict,
    visual_records: list[dict],
    visual_markdown_elements: list[dict],
) -> dict:
    """按阶段D/F结果重建视觉区域边界并重新渲染。"""
    candidate_by_id = {
        candidate["claim_id"]: candidate
        for candidate in resolution.get("candidates", [])
    }
    text_owner_by_id = stage_f_preview.get(
        "owner_by_text_element_id", {}
    )
    assigned_text_ids_by_claim: dict[str, list[str]] = defaultdict(list)
    for text_id, claim_id in text_owner_by_id.items():
        assigned_text_ids_by_claim[claim_id].append(text_id)

    markdown_by_claim = {
        element.get("claim_id"): element
        for element in visual_markdown_elements
        if element.get("claim_id") is not None
    }
    rerendered_count = 0
    rerender_seconds = 0.0
    final_text_rects: list[fitz.Rect] = []

    for record in visual_records:
        claim_id = record.get("claim_id")
        candidate = candidate_by_id.get(claim_id)
        if candidate is None:
            continue
        owned_feature_bbox_values = candidate.get(
            "owned_feature_bbox"
        )
        if owned_feature_bbox_values is None:
            continue

        owned_feature_bbox = fitz.Rect(
            owned_feature_bbox_values
        ).normalize()
        proposed_bbox = fitz.Rect(
            candidate.get(
                "proposed_bbox",
                owned_feature_bbox,
            )
        ).normalize()
        assigned_text_ids = sorted(
            assigned_text_ids_by_claim.get(claim_id, [])
        )
        assigned_text_lines = [
            page_inventory.entry(text_id)["payload"]
            for text_id in assigned_text_ids
            if text_id in page_inventory.entries
            and page_inventory.entry(text_id)["kind"] == "text"
        ]
        #@GPT 视觉自身内容受proposed_bbox约束；已确认的复合成员在其后独立并入截图边界。
        #@GPT 视觉自身的feature与阶段F文字仍受原提议边界约束；成员表格bbox在此后单独并入，因此不会借正文扩张制造组合，也不会使用其他Claim的drawing重算边界。
        visual_owned_content_rects = [owned_feature_bbox]
        visual_owned_content_rects.extend(
            fitz.Rect(line["bbox"])
            for line in assigned_text_lines
        )
        visual_owned_content_bbox = union_rects(
            visual_owned_content_rects
        ) & proposed_bbox
        if visual_owned_content_bbox.is_empty:
            visual_owned_content_bbox = owned_feature_bbox

        #@GPT 复合表格仍由自己的table Claim独占图元；这里只在输出层把已确认成员的bbox并入同一张视觉PNG，不改变owner_by_element_id。
        composite_member_table_rects = [
            fitz.Rect(bbox).normalize()
            for bbox in record.get(
                "composite_member_table_bboxes", []
            )
            if not fitz.Rect(bbox).normalize().is_empty
        ]
        #@GPT 并列visual成员没有表格覆盖关系时，renderer截图需要显式并入成员content bbox。
        composite_member_visual_rects = [
            fitz.Rect(bbox).normalize()
            for bbox in record.get(
                "composite_member_visual_bboxes", []
            )
            if not fitz.Rect(bbox).normalize().is_empty
        ]
        render_content_bbox = union_rects(
            [
                visual_owned_content_bbox,
                *composite_member_table_rects,
                *composite_member_visual_rects,
            ]
        ) & page.rect
        if render_content_bbox.is_empty:
            render_content_bbox = owned_feature_bbox

        render_bbox = padded_image_clip(
            render_content_bbox,
            page.rect,
        )
        output_path = Path(record["output_path"])
        render_started = time.perf_counter()
        pixmap = page.get_pixmap(
            clip=render_bbox,
            dpi=VISUAL_REGION_DPI,
            alpha=False,
            annots=True,
        )
        pixmap.save(output_path)
        current_render_seconds = time.perf_counter() - render_started
        rerender_seconds += current_render_seconds
        rerendered_count += 1

        native_text_markdown, ordered_native_text_lines = (
            format_visual_region_native_text(
                text_lines=assigned_text_lines,
                page_rect=page.rect,
                sort_lines_in_reading_order=partial(
                    sort_lines_in_reading_order,
                    row_tolerance_points=TEXT_ROW_TOLERANCE_POINTS,
                ),
            )
        )
        image_name = output_path.name
        relative_path = f"visual_regions/{image_name}"
        image_markdown = (
            f"![Page {record['page_number']} Visual Region "
            f"{record['region_number']}]({relative_path})"
        )
        markdown_text = (
            f"{image_markdown}\n\n{native_text_markdown}"
            if native_text_markdown
            else image_markdown
        )
        sort_source_bbox = fitz.Rect(owned_feature_bbox)
        #@GPT 组合输出只用renderer自身owned bbox排序，复合成员并集仅用于截图边界。
        sort_bbox = (
            build_top_sort_bbox(sort_source_bbox)
            if ordered_native_text_lines
            else (
                fitz.Rect(sort_source_bbox)
                if (
                    composite_member_table_rects
                    or composite_member_visual_rects
                )
                else fitz.Rect(render_bbox)
            )
        )

        record["bbox"] = [round(value, 3) for value in render_bbox]
        record["render_bbox"] = [
            round(value, 3) for value in render_bbox
        ]
        record["sort_bbox"] = [
            round(value, 3) for value in sort_bbox
        ]
        record["sort_source_bbox"] = [
            round(value, 3) for value in sort_source_bbox
        ]
        record["content_bbox"] = [
            round(value, 3)
            for value in render_content_bbox
        ]
        record["native_text_lines"] = [
            {
                "text": line["text"],
                "bbox": [round(value, 3) for value in line["bbox"]],
                "angle": line["angle"],
            }
            for line in ordered_native_text_lines
        ]
        record["included_text_line_count"] = len(
            ordered_native_text_lines
        )
        record["stage_e_owned_feature_bbox"] = [
            round(value, 3)
            for value in owned_feature_bbox
        ]
        record["stage_f_text_element_ids"] = assigned_text_ids
        #@GPT 审计同时保留视觉自身边界和复合成员并集；可直接确认PNG扩大来自成员表格，而不是正文文字或他人drawing。
        record["stage_e_visual_owned_content_bbox"] = [
            round(value, 3)
            for value in visual_owned_content_bbox
        ]
        record["stage_e_composite_member_table_bboxes"] = [
            [round(value, 3) for value in rect]
            for rect in composite_member_table_rects
        ]
        record["stage_e_composite_member_visual_bboxes"] = [
            [round(value, 3) for value in rect]
            for rect in composite_member_visual_rects
        ]
        record["stage_e_f_applied"] = True
        record["render_seconds"] = round(
            current_render_seconds,
            6,
        )
        record["output_width"] = pixmap.width
        record["output_height"] = pixmap.height

        markdown_element = markdown_by_claim.get(claim_id)
        if markdown_element is not None:
            markdown_element.update(
                {
                    "bbox": fitz.Rect(render_bbox),
                    "render_bbox": fitz.Rect(render_bbox),
                    "sort_bbox": fitz.Rect(sort_bbox),
                    "text": markdown_text,
                    "markdown": markdown_text,
                    "output_path": output_path,
                    "stage_e_owned_feature_bbox": fitz.Rect(
                        owned_feature_bbox
                    ),
                    "stage_f_text_element_ids": assigned_text_ids,
                }
            )

        final_text_rects.extend(
            fitz.Rect(line["bbox"])
            for line in ordered_native_text_lines
        )

    return {
        "applied": True,
        "rerendered_visual_region_count": rerendered_count,
        "rerender_seconds": round(rerender_seconds, 6),
        #@GPT 审计结果只写数量，fitz.Rect对象不进入JSON；真实去重范围由调用点从已更新record重建。
        "assigned_text_rect_count": len(final_text_rects),
    }



def apply_stage_e_to_source_outputs(
    page: fitz.Page,
    #@GPT 台账用于从阶段F文字ID精确反查bbox和payload，不通过最终截图bbox猜测文字归属。
    page_inventory: PageInventory,
    resolution: dict,
    #@GPT 阶段F结果明确区分source Claim真正拥有的端点文字。
    stage_f_preview: dict,
    source_records: list[dict],
    source_markdown_elements: list[dict],
) -> dict:
    """按D获胜结构与F唯一文字归属重建原图/复合图输出。"""
    candidate_by_id = {
        candidate["claim_id"]: candidate
        for candidate in resolution.get("candidates", [])
    }
    markdown_by_claim = {
        element.get("claim_id"): element
        for element in source_markdown_elements
        if element.get("claim_id") is not None
    }
    text_owner_by_id = stage_f_preview.get(
        "owner_by_text_element_id", {}
    )
    assigned_text_ids_by_claim: dict[str, list[str]] = defaultdict(list)
    for text_id, owner_claim_id in text_owner_by_id.items():
        assigned_text_ids_by_claim[owner_claim_id].append(text_id)
    rerendered_count = 0
    rerender_seconds = 0.0
    final_text_rects: list[fitz.Rect] = []

    for record in source_records:
        claim_id = record.get("claim_id")
        candidate = candidate_by_id.get(claim_id)
        if candidate is None or not record.get("output_path"):
            continue
        owned_values = candidate.get("owned_feature_bbox")
        if owned_values is None:
            continue
        owned_survival_feature_bbox = fitz.Rect(
            owned_values
        ).normalize()
        if owned_survival_feature_bbox.is_empty:
            continue

        #@GPT 存活由原始image feature确认；渲染边界使用本Claim赢得的全部非文字ID，使overlay影响边界但不能单独维持候选。
        won_structural_element_ids = [
            element_id
            for element_id in candidate.get("won_element_ids", [])
            if element_id in page_inventory.entries
            and page_inventory.entry(element_id)["kind"] != "text"
        ]
        owned_structural_bbox = (
            union_rects(
                [
                    page_inventory.bbox_of(element_id)
                    for element_id in won_structural_element_ids
                ]
            )
            if won_structural_element_ids
            else owned_survival_feature_bbox
        )

        assigned_text_ids = sorted(
            assigned_text_ids_by_claim.get(claim_id, [])
        )
        assigned_text_lines = [
            page_inventory.entry(text_id)["payload"]
            for text_id in assigned_text_ids
            if text_id in page_inventory.entries
            and page_inventory.entry(text_id)["kind"] == "text"
        ]
        proposed_bbox = fitz.Rect(
            candidate.get(
                "proposed_bbox",
                owned_structural_bbox,
            )
        ).normalize()
        #@GPT 最终内容边界合并阶段D结构图元和阶段F文字，并裁剪到检测器提议范围。
        render_content_bbox = union_rects(
            [
                owned_structural_bbox,
                *(
                    fitz.Rect(line["bbox"])
                    for line in assigned_text_lines
                ),
            ]
        ) & proposed_bbox
        if render_content_bbox.is_empty:
            render_content_bbox = owned_structural_bbox

        old_bbox = fitz.Rect(
            record.get("render_bbox")
            or record.get("output_bbox")
            or render_content_bbox
        ).normalize()
        mode = record.get("mode", "")
        #@GPT 边界变化比较使用结构与阶段F文字的最终内容范围。
        bbox_changed = any(
            abs(float(left) - float(right)) > 0.01
            for left, right in zip(old_bbox, render_content_bbox)
        )
        should_rerender = (
            mode == "rendered_composite"
            or bbox_changed
        )
        output_path = Path(record["output_path"])
        if should_rerender:
            #@GPT 重渲染使用包含阶段F文字的一次性最终范围。
            render_bbox = padded_image_clip(
                render_content_bbox,
                page.rect,
            )
            render_started = time.perf_counter()
            pixmap = page.get_pixmap(
                clip=render_bbox,
                dpi=COMPOSITE_IMAGE_DPI,
                alpha=False,
                annots=True,
            )
            pixmap.save(output_path)
            current_render_seconds = (
                time.perf_counter() - render_started
            )
            rerender_seconds += current_render_seconds
            rerendered_count += 1
            record["output_bbox"] = [
                round(value, 3)
                for value in render_bbox
            ]
            record["render_bbox"] = [
                round(value, 3)
                for value in render_bbox
            ]
            record["output_width"] = pixmap.width
            record["output_height"] = pixmap.height
            record["processing_seconds"] = round(
                current_render_seconds,
                6,
            )
        else:
            render_bbox = old_bbox

        #@GPT render_bbox可因端点文字向外扩张，但排序仍锚定原始图片本体顶部，避免外侧编号把图片提前到上一段正文之前。
        source_sort_bbox = union_rects(
            [
                fitz.Rect(bbox)
                for bbox in record.get("source_bboxes", [])
            ]
        )
        sort_bbox = build_top_sort_bbox(source_sort_bbox)
        native_text_markdown, ordered_native_text_lines = (
            format_visual_region_native_text(
                text_lines=assigned_text_lines,
                page_rect=page.rect,
                sort_lines_in_reading_order=partial(
                    sort_lines_in_reading_order,
                    row_tolerance_points=TEXT_ROW_TOLERANCE_POINTS,
                ),
            )
        )
        relative_path = f"source_images/{output_path.name}"
        image_markdown = (
            f"![Page {record['page_number']} Source Image "
            f"{record['claim_id'].rsplit('#', 1)[-1]}]"
            f"({relative_path})"
        )
        markdown_text = (
            f"{image_markdown}\n\n{native_text_markdown}"
            if native_text_markdown
            else image_markdown
        )

        record["output_bbox"] = [
            round(value, 3) for value in render_bbox
        ]
        record["render_bbox"] = [
            round(value, 3) for value in render_bbox
        ]
        record["sort_bbox"] = [
            round(value, 3) for value in sort_bbox
        ]
        record["native_text_lines"] = [
            {
                "text": line["text"],
                "bbox": [round(value, 3) for value in line["bbox"]],
                "angle": line["angle"],
            }
            for line in ordered_native_text_lines
        ]
        record["stage_f_text_element_ids"] = assigned_text_ids

        markdown_element = markdown_by_claim.get(claim_id)
        if markdown_element is not None:
            markdown_element.update(
                {
                    "bbox": fitz.Rect(render_bbox),
                    "render_bbox": fitz.Rect(render_bbox),
                    "sort_bbox": fitz.Rect(sort_bbox),
                    "text": markdown_text,
                    "markdown": markdown_text,
                    "stage_f_text_element_ids": assigned_text_ids,
                }
            )

        record["stage_e_owned_feature_bbox"] = [
            round(value, 3)
            for value in owned_survival_feature_bbox
        ]
        record["stage_e_owned_structural_bbox"] = [
            round(value, 3)
            for value in owned_structural_bbox
        ]
        record["stage_e_applied"] = True
        final_text_rects.extend(
            fitz.Rect(line["bbox"])
            for line in ordered_native_text_lines
        )

    return {
        "applied": True,
        "rerendered_source_image_count": rerendered_count,
        "rerender_seconds": round(rerender_seconds, 6),
        "assigned_text_rect_count": len(final_text_rects),
    }



def annotate_inventory_labels(
    inventory: PageInventory,
) -> None:
    """为台账中的每个图元写入局部标签。"""
    page_rect = inventory.page_rect
    page_area = rect_area(page_rect)

    drawing_ids = inventory.ids_of_kind("drawing")
    drawing_payloads = [
        inventory.payload_of(element_id) for element_id in drawing_ids
    ]
    #@GPT 端点标记索引一次建立、全页复用，避免逐对象重复扫描。
    marker_grid = build_endpoint_marker_grid(drawing_payloads)

    for local_index, element_id in enumerate(drawing_ids):
        entry = inventory.entry(element_id)
        drawing = entry["payload"]
        drawing_bbox = entry["bbox"]
        drawing_area = rect_area(drawing_bbox)

        axis_rule = drawing_axis_rule(drawing, page_rect)
        is_filled_bar = drawing_is_filled_horizontal_bar(drawing)
        is_long_stroke = drawing_is_long_horizontal_single_line(drawing)

        #@GPT 端点复核只对“可能是连接件”的对象执行，避免对全部矢量做邻域搜索。
        has_marker = False
        if is_filled_bar or is_long_stroke:
            has_marker = drawing_has_endpoint_marker(
                drawing_index=local_index,
                connector_drawing=drawing,
                drawings=drawing_payloads,
                marker_grid=marker_grid,
            )

        entry["labels"] = {
            #@GPT 退化对象：宽高同时为0，现有流程在候选提取阶段直接跳过。
            "degenerate": (
                drawing_bbox.width <= 0 and drawing_bbox.height <= 0
            ),
            #@GPT 整页背景：面积占比达到MAX_BACKGROUND_AREA_PAGE_RATIO。
            "background": (
                page_area > 0
                and drawing_area / page_area
                >= MAX_BACKGROUND_AREA_PAGE_RATIO
            ),
            #@GPT P5性能优化：闭合视觉容器检测会重复判断同一drawing是否为白色不可见描边；该判断只依赖drawing自身，阶段B计算一次后供后续规则复用。
            "visible_semantic_stroke": drawing_has_visible_semantic_stroke(
                drawing
            ),
            #@GPT P5性能优化：视觉候选阶段和闭合容器阶段都需要排除整页描边框；缓存同一局部判据，避免密集页重复扫描drawing items。
            "page_enclosing_stroke_frame": drawing_is_page_enclosing_stroke_frame(
                drawing=drawing,
                page_rect=page_rect,
            ),
            #@GPT 规则线方向：h/v，非轴对齐细线为None。用于表格结构和框架识别。
            "axis_rule_orientation": (
                axis_rule["orientation"] if axis_rule else None
            ),
            "filled_horizontal_bar": is_filled_bar,
            "long_horizontal_stroke": is_long_stroke,
            #@GPT 端点带紧凑矢头/端帽：说明它更可能是指引线而非分隔条。
            "endpoint_marker": has_marker,
            #@GPT 单路径类型，供阶段C区分“单线段/单矩形”与复杂路径。
            "single_path_item": (
                drawing.get("items", [{}])[0][0]
                if len(drawing.get("items", [])) == 1
                else None
            ),
            "item_count": len(drawing.get("items", [])),
            "filled": drawing.get("fill") is not None,
            "stroke_type": drawing.get("type"),
            #@GPT 复用图片尺寸门限，用于判断该矢量单体是否足以构成可见内容。
            "passes_size_filter": image_is_large_enough(
                drawing_bbox,
                page_rect,
            ),
        }

    for element_id in inventory.ids_of_kind("image"):
        entry = inventory.entry(element_id)
        image = entry["payload"]
        entry["labels"] = {
            "passes_size_filter": image_is_large_enough(
                entry["bbox"],
                page_rect,
            ),
            "has_mask": bool(image.get("has_mask")),
            "complex_transform": image_has_complex_transform(
                image.get("transform", ())
            ),
            "missing_bytes": not image.get("image_bytes"),
        }

    for element_id in inventory.ids_of_kind("text"):
        entry = inventory.entry(element_id)
        line = entry["payload"]
        normalized_text = "".join(line.get("text", "").split())
        entry["labels"] = {
            #@GPT 文字方向沿用现有量化结果，供阶段D按方向分组归属。
            "angle": line.get("angle", 0),
            "character_count": len(normalized_text),
            #@GPT 独立公式编号是高置信锚点，此处只做纯局部正则识别，不做聚类。
            "equation_number": bool(
                EQUATION_NUMBER_PATTERN.fullmatch(
                    line.get("text", "").strip()
                )
            ),
        }



def collect_page_overlay_rects(
    page: fitz.Page,
    text_lines: list[dict],
    #@GPT 图片叠加判定消费页面级drawing缓存和台账，保持图元身份可追踪。
    drawings: list[dict],
    page_inventory: PageInventory,
) -> dict[str, list[dict]]:
    """收集可能叠加在原始图片上的文字、矢量对象、批注和表单。"""
    #@GPT 原图叠加分析复用main()提供的页面drawing缓存。
    page_drawings = drawings
    #@GPT overlay记录携带父drawing稳定ID；只有候选冲突时才惰性展开原子路径。
    drawing_overlays = [
        {
            "bbox": fitz.Rect(drawing["rect"]),
            "element_id": page_inventory.id_for_payload(
                drawing,
                "drawing",
            ),
        }
        for drawing in page_drawings
        if rect_area(fitz.Rect(drawing["rect"])) > 0
        #@GPT 与视觉候选使用同一整页描边框过滤；否则原图复合截图仍会把出血框作为overlay扩张到整页。
        and not drawing_is_page_enclosing_stroke_frame(
            drawing=drawing,
            page_rect=page.rect,
        )
    ]
    annotations = [
        {
            "bbox": fitz.Rect(annotation.rect),
            "element_id": None,
        }
        for annotation in (page.annots() or [])
    ]
    widgets = [
        {
            "bbox": fitz.Rect(widget.rect),
            "element_id": None,
        }
        for widget in (page.widgets() or [])
    ]

    return {
        "text_overlay": [
            {
                "bbox": fitz.Rect(line["bbox"]),
                "element_id": page_inventory.id_for_payload(
                    line,
                    "text",
                ),
            }
            for line in text_lines
        ],
        "vector_drawing_overlap": drawing_overlays,
        "annotation_overlap": annotations,
        "form_widget_overlap": widgets,
    }



def process_page_source_images(
    page: fitz.Page,
    page_dict: dict,
    text_lines: list[dict],
    excluded_table_rects: list[fitz.Rect],
    #@GPT 矢量视觉区域会接管其中的小图片和混合图片，新增排除区域避免重复输出。
    excluded_visual_region_rects: list[fitz.Rect],
    #@GPT 页眉/页尾带在图片判定前应用，已确认重复的装饰图不再进入原图复制或复合截图流程。
    excluded_header_footer_rects: list[fitz.Rect],
    page_number: int,
    images_dir: Path,
    #@GPT 原图阶段复用阶段A的drawing、图片和台账缓存，避免重复提取和身份漂移。
    drawings: list[dict],
    source_images: list[dict],
    page_inventory: PageInventory,
) -> tuple[list[dict], list[dict], float, list[dict]]:
    """
    处理原 PDF 图片：

    - 独立且无复杂变换的图片直接复制原始编码；
    - 有文字/矢量/批注/表单/蒙版/旋转/多图重叠时截图；
    - 已被大表格截图覆盖的图片只记录，不重复嵌入 Markdown。
    """
    started = time.perf_counter()
    #@GPT 原图列表直接使用阶段A缓存，不再遍历page_dict重复提取。
    overlay_rects = collect_page_overlay_rects(
        page,
        text_lines,
        drawings=drawings,
        page_inventory=page_inventory,
    )
    records: list[dict] = []
    markdown_elements: list[dict] = []

    eligible_indices: list[int] = []

    for image_index, image in enumerate(source_images):
        image_rect = image["bbox"]
        #@GPT 已确认的页眉/页尾区域在图片判定前排除，避免重复装饰进入正文输出。
        covered_by_header_footer = rect_is_mostly_inside_any(
            rect=image_rect,
            container_rects=excluded_header_footer_rects,
        ) or any(
            band.contains(rect_center(image_rect))
            for band in excluded_header_footer_rects
        )
        covered_by_table = any(
            table_rect.contains(rect_center(image_rect))
            for table_rect in excluded_table_rects
        )
        #@GPT 视觉区域的优先级高于独立图片；图片中心落入区域时只保留归属记录。
        covered_by_visual_region = any(
            region_rect.contains(rect_center(image_rect))
            for region_rect in excluded_visual_region_rects
        )
        large_enough = image_is_large_enough(image_rect, page.rect)

        #@GPT 覆盖模式在尺寸过滤前判定，使被表格或视觉区域接管的小图片仍保留可审计原因。
        if (
            covered_by_header_footer
            or covered_by_table
            or covered_by_visual_region
            or not large_enough
        ):
            mode = (
                "covered_by_header_footer"
                if covered_by_header_footer
                else (
                    "covered_by_table"
                    if covered_by_table
                else (
                    "covered_by_visual_region"
                    if covered_by_visual_region
                    else "filtered_small_image"
                    )
                )
            )
            records.append(
                {
                    "page_number": page_number,
                    "source_image_indices": [image["source_index"]],
                    "mode": mode,
                    "reason": [mode],
                    "source_bboxes": [
                        [round(value, 3) for value in image_rect]
                    ],
                    "output_bbox": None,
                    "output_path": None,
                    "source_width": image["source_width"],
                    "source_height": image["source_height"],
                    "output_width": None,
                    "output_height": None,
                    #@GPT source_index属于page_dict block编号；声明始终保存由原对象反查得到的台账图片ID。
                    "source_element_ids": page_inventory.ids_for_payloads(
                        [image],
                        "image",
                    ),
                    "overlay_element_ids": [],
                    #@GPT 过滤图片没有输出原子，但仍保持外连矢量/端点文字字段结构一致，便于报告按统一schema读取。
                    "external_connector_drawing_element_ids": [],
                    "endpoint_text_element_ids": [],
                }
            )
            continue

        eligible_indices.append(image_index)

    handled: set[int] = set()
    output_number = 0

    for initial_index in eligible_indices:
        if initial_index in handled:
            continue

        group = {initial_index}
        changed = True

        while changed:
            changed = False
            group_rect = union_rects(
                [source_images[index]["bbox"] for index in group]
            )

            for candidate_index in eligible_indices:
                if candidate_index in group or candidate_index in handled:
                    continue

                if image_rects_overlap(
                    group_rect,
                    source_images[candidate_index]["bbox"],
                ):
                    group.add(candidate_index)
                    changed = True

        handled.update(group)
        group_images = [source_images[index] for index in sorted(group)]
        source_rect = union_rects([image["bbox"] for image in group_images])
        reasons: list[str] = []
        composite_rects = [image["bbox"] for image in group_images]
        overlay_element_ids: list[str] = []
        #@GPT 外连矢量和端点文字单独记录；它们既用于截图边界，也作为阶段C/F可审计的稳定ID关系。
        external_connector_drawing_element_ids: list[str] = []
        endpoint_text_element_ids: list[str] = []
        endpoint_promoted_text_lines: list[dict] = []
        external_connector_drawings: list[dict] = []

        if len(group_images) > 1:
            reasons.append("multiple_images_overlap")

        for reason, candidate_items in overlay_rects.items():
            matched_items = [
                candidate_item
                for candidate_item in candidate_items
                if meaningful_overlap(
                    source_rect,
                    candidate_item["bbox"],
                )
            ]

            if matched_items:
                reasons.append(reason)
                composite_rects.extend(
                    item["bbox"] for item in matched_items
                )
                overlay_element_ids.extend(
                    item["element_id"]
                    for item in matched_items
                    if item.get("element_id") is not None
                )

        #@GPT 除面积重叠外，恰好一端伸出图片的零面积引线也可触发复合渲染。
        #@GPT 新路径从原始source_rect出发检查真实路径穿越，只做一次端点文字附着；后续不使用扩张后的bbox再次搜索，避免递归吞入附近正文。
        attached_overlay_text_lines = [
            page_inventory.payload_of(element_id)
            for element_id in set(overlay_element_ids)
            if element_id in page_inventory.entries
            and page_inventory.entry(element_id)["kind"] == "text"
        ]
        endpoint_candidate_text_lines = [
            line
            for line in text_lines
            if not any(
                table_rect.contains(rect_center(line["bbox"]))
                for table_rect in excluded_table_rects
            )
            and not any(
                region_rect.contains(rect_center(line["bbox"]))
                for region_rect in excluded_visual_region_rects
            )
            and not any(
                band.contains(rect_center(line["bbox"]))
                for band in excluded_header_footer_rects
            )
        ]
        (
            endpoint_promoted_text_lines,
            external_connector_drawings,
        ) = promote_text_lines_at_region_external_connectors(
            candidate_text_lines=endpoint_candidate_text_lines,
            attached_text_lines=attached_overlay_text_lines,
            region_rect=source_rect,
            drawings=drawings,
            maximum_gap=(
                min(page.rect.width, page.rect.height)
                * VISUAL_REGION_TEXT_MARGIN_PAGE_RATIO
                * VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER
            ),
        )
        if endpoint_promoted_text_lines:
            reasons.append("external_connector_endpoint_text")
            composite_rects.extend(
                fitz.Rect(drawing["rect"])
                for drawing in external_connector_drawings
            )
            composite_rects.extend(
                fitz.Rect(line["bbox"])
                for line in endpoint_promoted_text_lines
            )
            external_connector_drawing_element_ids = (
                page_inventory.ids_for_payloads(
                    external_connector_drawings,
                    "drawing",
                )
            )
            endpoint_text_element_ids = (
                page_inventory.ids_for_payloads(
                    endpoint_promoted_text_lines,
                    "text",
                )
            )
            overlay_element_ids.extend(
                external_connector_drawing_element_ids
            )
            overlay_element_ids.extend(endpoint_text_element_ids)

        if any(image["has_mask"] for image in group_images):
            reasons.append("soft_mask")

        if any(
            image_has_complex_transform(image["transform"])
            for image in group_images
        ):
            reasons.append("complex_transform")

        if any(not image["image_bytes"] for image in group_images):
            reasons.append("missing_original_bytes")

        output_number += 1
        decision_started = time.perf_counter()

        if reasons:
            output_bbox = padded_image_clip(
                union_rects(composite_rects),
                page.rect,
            )
            output_name = (
                f"page_{page_number:03d}_composite_{output_number:02d}.png"
            )
            output_path = images_dir / output_name
            pixmap = page.get_pixmap(
                clip=output_bbox,
                dpi=COMPOSITE_IMAGE_DPI,
                alpha=False,
                annots=True,
            )
            pixmap.save(output_path)
            output_width = pixmap.width
            output_height = pixmap.height
            mode = "rendered_composite"
        else:
            image = group_images[0]
            extension = image["extension"] or "png"
            output_name = (
                f"page_{page_number:03d}_image_{output_number:02d}."
                f"{extension}"
            )
            output_path = images_dir / output_name
            output_path.write_bytes(image["image_bytes"])
            output_bbox = source_rect
            output_width = image["source_width"]
            output_height = image["source_height"]
            mode = "direct_copy"

        processing_seconds = time.perf_counter() - decision_started
        relative_path = f"source_images/{output_name}"
        #@GPT 图片元素同时保存完整渲染框和顶部排序锚点，使截图扩张不改变阅读顺序锚点。
        render_bbox = fitz.Rect(output_bbox)
        sort_bbox = build_top_sort_bbox(render_bbox)
        markdown_elements.append(
            {
                # "bbox": source_rect,
                "bbox": render_bbox,
                "render_bbox": render_bbox,
                "sort_bbox": sort_bbox,
                #@GPT 原 PDF 图片作为虚拟0°页面元素进入现有坐标重排，而不是集中堆在页首；排序实际使用sort_bbox。
                "text": (
                    f"![Page {page_number} Source Image {output_number}]"
                    f"({relative_path})"
                ),
                "direction": (1.0, 0.0),
                "angle": 0,
                "kind": "image",
                "markdown": (
                    f"![Page {page_number} Source Image {output_number}]"
                    f"({relative_path})"
                ),
                "mode": mode,
                "output_path": output_path,
            }
        )
        records.append(
            {
                "page_number": page_number,
                "source_image_indices": [
                    image["source_index"] for image in group_images
                ],
                "mode": mode,
                "reason": sorted(set(reasons)),
                "source_bboxes": [
                    [round(value, 3) for value in image["bbox"]]
                    for image in group_images
                ],
                "output_bbox": [
                    round(value, 3) for value in output_bbox
                ],
                #@GPT 报告显式保存render_bbox和sort_bbox；output_bbox作为公开清单字段继续提供。
                "render_bbox": [
                    round(value, 3) for value in render_bbox
                ],
                "sort_bbox": [
                    round(value, 3) for value in sort_bbox
                ],
                "output_path": str(output_path),
                "source_width": group_images[0]["source_width"],
                "source_height": group_images[0]["source_height"],
                "output_width": output_width,
                "output_height": output_height,
                "processing_seconds": round(processing_seconds, 6),
                #@GPT 每个复合/直拷候选保留原图ID和实际命中的叠加图元ID，后续清单不再从output_bbox反查。
                "source_element_ids": page_inventory.ids_for_payloads(
                    group_images,
                    "image",
                ),
                "overlay_element_ids": sorted(
                    set(overlay_element_ids)
                ),
                "external_connector_drawing_element_ids": sorted(
                    set(external_connector_drawing_element_ids)
                ),
                "endpoint_text_element_ids": sorted(
                    set(endpoint_text_element_ids)
                ),
            }
        )

    markdown_elements.sort(
        key=lambda item: (item["bbox"].y0, item["bbox"].x0)
    )

    #@GPT 阶段C：收集原图处理消费的图元ID
    claims: list[dict] = []
    #@GPT page_inventory已成为必需参数；阶段C原图Claim无条件建立，不再保留空台账分支。
    #@GPT 阶段D通过实际输出路径把原图记录和Markdown原子绑定到同一claim_id；过滤图片记录没有输出路径，但仍保留声明用于审计。
    markdown_element_by_output_path = {
        str(element["output_path"]): element
        for element in markdown_elements
        if element.get("output_path") is not None
    }
    for claim_number, record in enumerate(records, start=1):
        mode = record["mode"]
        claim_id = (
            f"page#{page_number}:source_image#{claim_number}"
        )
        record["claim_id"] = claim_id
        if record.get("output_path"):
            markdown_element = markdown_element_by_output_path.get(
                str(record["output_path"])
            )
            if markdown_element is not None:
                markdown_element["claim_id"] = claim_id
        #@GPT source元素ID来自页面台账，过滤记录也保留声明以审计表格、视觉区域和独立图片冲突。
        source_element_ids = list(
            record.get("source_element_ids", [])
        )
        overlay_element_ids = list(
            record.get("overlay_element_ids", [])
        )
        claimed_ids = sorted(
            set(source_element_ids).union(
                overlay_element_ids
            )
        )
        #@GPT 【生存/边界契约分离】原始image ID是source Claim唯一生存feature；获胜overlay只在阶段E扩张render_bbox，不能单独维持候选存活。
        source_survival_feature_ids = source_element_ids
        record_bbox_values = (
            record.get("output_bbox")
            or union_rects(
                [
                    fitz.Rect(bbox)
                    for bbox in record.get(
                        "source_bboxes", []
                    )
                ]
            )
        )
        confidence = {
            "direct_copy": 0.95,
            "rendered_composite": 0.90,
            "covered_by_table": 0.90,
            "covered_by_visual_region": 0.90,
            "covered_by_header_footer": 0.20,
            "filtered_small_image": 0.10,
        }.get(mode, 0.70)

        claims.append(create_claim_manifest(
            detector_name="source_image_processor",
            output_element_id=claim_id,
            input_element_ids=claimed_ids,
            claimed_element_ids=claimed_ids,
            #@GPT 原始image必须由本Claim赢得；连接矢量属于claimed结构但不是生存feature，避免被视觉区域接管的图片靠残余线条继续输出。
            feature_element_ids=source_survival_feature_ids,
            bbox=fitz.Rect(record_bbox_values),
            confidence=confidence,
            reason_codes=[
                f"legacy_mode:{mode}",
                *record.get("reason", []),
            ],
            output_artifacts=(
                [
                    {
                        "type": mode,
                        "path": str(record["output_path"]),
                        "bbox": record["output_bbox"],
                    }
                ]
                if record.get("output_path")
                else []
            ),
            metadata={
                "page_number": record["page_number"],
                "mode": mode,
                "reason": record.get("reason", []),
                "legacy_output_emitted": bool(
                    record.get("output_path")
                ),
                "source_image_indices": record.get(
                    "source_image_indices", []
                ),
                "source_element_ids": source_element_ids,
                "overlay_element_ids": overlay_element_ids,
                "external_connector_drawing_element_ids": record.get(
                    "external_connector_drawing_element_ids", []
                ),
                "endpoint_text_element_ids": record.get(
                    "endpoint_text_element_ids", []
                ),
            },
        ))

    return records, markdown_elements, time.perf_counter() - started, claims



def drawing_single_line_endpoints(
    drawing: dict,
) -> tuple[fitz.Point, fitz.Point] | None:
    """返回单线段路径的两个端点，其他路径返回 None。"""
    items = drawing.get("items", [])

    if len(items) != 1 or items[0][0] != "l":
        return None

    return fitz.Point(items[0][1]), fitz.Point(items[0][2])



def drawing_is_long_horizontal_single_line(drawing: dict) -> bool:
    """判断对象是否为需要端点复核的单条水平描边线。"""
    endpoints = drawing_single_line_endpoints(drawing)

    if endpoints is None or drawing.get("type") not in {"s", "fs"}:
        return False

    first, second = endpoints
    delta_x = abs(second.x - first.x)
    delta_y = abs(second.y - first.y)
    stroke_width = float(drawing.get("width") or 1.0)
    effective_minor = max(delta_y, stroke_width)

    return (
        delta_x > 0
        and delta_y / delta_x <= MAX_HORIZONTAL_LINE_SLOPE_RATIO
        and effective_minor < MIN_IMAGE_THIN_SIDE_POINTS
        and delta_x / effective_minor > MIN_IMAGE_LINE_ASPECT_RATIO
    )



def drawing_is_filled_horizontal_bar(drawing: dict) -> bool:
    """判断对象是否为单个填充矩形构成的水平细条。"""
    items = drawing.get("items", [])

    if (
        len(items) != 1
        or items[0][0] != "re"
        or drawing.get("fill") is None
    ):
        return False

    drawing_rect = fitz.Rect(drawing["rect"])

    if drawing_rect.width <= 0 or drawing_rect.height <= 0:
        return False

    return (
        drawing_rect.width > drawing_rect.height
        and drawing_rect.height < MIN_IMAGE_THIN_SIDE_POINTS
        and (
            drawing_rect.width / drawing_rect.height
            > MIN_IMAGE_LINE_ASPECT_RATIO
        )
    )



def filled_bars_are_collinear_neighbors(
    first_drawing: dict,
    second_drawing: dict,
    tolerance: float = FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS,
) -> bool:
    """判断两个填充水平细矩形是否为同Y首尾相接的相邻切片。"""
    if not (
        drawing_is_filled_horizontal_bar(first_drawing)
        and drawing_is_filled_horizontal_bar(second_drawing)
    ):
        return False

    first_rect = fitz.Rect(first_drawing["rect"]).normalize()
    second_rect = fitz.Rect(second_drawing["rect"]).normalize()
    minimum_height = min(first_rect.height, second_rect.height)
    if minimum_height <= 0:
        return False

    center_y_delta = abs(
        (first_rect.y0 + first_rect.y1) / 2
        - (second_rect.y0 + second_rect.y1) / 2
    )
    vertical_overlap = max(
        0.0,
        min(first_rect.y1, second_rect.y1)
        - max(first_rect.y0, second_rect.y0),
    )
    if (
        center_y_delta > tolerance
        or vertical_overlap / minimum_height < 0.80
    ):
        return False

    if first_rect.x1 <= second_rect.x0:
        horizontal_gap = second_rect.x0 - first_rect.x1
    elif second_rect.x1 <= first_rect.x0:
        horizontal_gap = first_rect.x0 - second_rect.x1
    else:
        #@GPT 两个矩形发生水平覆盖时不是“首尾相接切片”，不能用该规则替代端点证据判断。
        return False

    return horizontal_gap <= tolerance



def drawing_connector_endpoints(
    drawing: dict,
) -> tuple[fitz.Point, fitz.Point] | None:
    """返回细长连接对象的左右端点，支持描边线和填充细矩形。"""
    line_endpoints = drawing_single_line_endpoints(drawing)

    if line_endpoints is not None:
        return line_endpoints

    if not drawing_is_filled_horizontal_bar(drawing):
        return None

    drawing_rect = fitz.Rect(drawing["rect"])
    center_y = (drawing_rect.y0 + drawing_rect.y1) / 2
    return (
        fitz.Point(drawing_rect.x0, center_y),
        fitz.Point(drawing_rect.x1, center_y),
    )



def drawing_can_mark_line_endpoint(drawing: dict) -> bool:
    """判断矢量对象是否可作为紧凑的矢头、端帽或连接标记。"""
    drawing_rect = fitz.Rect(drawing["rect"])
    major_dimension = max(drawing_rect.width, drawing_rect.height)

    if major_dimension > MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS:
        return False

    endpoints = drawing_single_line_endpoints(drawing)

    #@GPT 一条较长的水平线不能为另一条水平线作“矢头”，防止多条目录色条互相保护。
    if endpoints is not None:
        first, second = endpoints
        delta_x = abs(second.x - first.x)
        delta_y = abs(second.y - first.y)

        if (
            delta_x > 0
            and delta_y / delta_x
            <= MAX_HORIZONTAL_LINE_SLOPE_RATIO
        ):
            return False

    return bool(drawing.get("items"))



def build_endpoint_marker_grid(
    drawings: list[dict],
) -> dict[tuple[int, int], list[int]]:
    """将紧凑端点标记按 bbox 覆盖的空间桶建立索引。"""
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cell_size = VECTOR_ENDPOINT_GRID_CELL_POINTS

    for drawing_index, drawing in enumerate(drawings):
        if not drawing_can_mark_line_endpoint(drawing):
            continue

        drawing_rect = fitz.Rect(drawing["rect"])
        first_column = math.floor(drawing_rect.x0 / cell_size)
        last_column = math.floor(drawing_rect.x1 / cell_size)
        first_row = math.floor(drawing_rect.y0 / cell_size)
        last_row = math.floor(drawing_rect.y1 / cell_size)

        for column in range(first_column, last_column + 1):
            for row in range(first_row, last_row + 1):
                grid[(column, row)].append(drawing_index)

    return grid



def drawing_has_endpoint_marker(
    drawing_index: int,
    connector_drawing: dict,
    drawings: list[dict],
    marker_grid: dict[tuple[int, int], list[int]],
) -> bool:
    """判断连接对象任一端点附近是否存在紧凑矢头、端帽或连接标记。"""
    endpoints = drawing_connector_endpoints(connector_drawing)

    if endpoints is None:
        return False

    cell_size = VECTOR_ENDPOINT_GRID_CELL_POINTS
    bucket_radius = max(
        1,
        math.ceil(
            VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS / cell_size
        ),
    )

    for endpoint in endpoints:
        endpoint_column = math.floor(endpoint.x / cell_size)
        endpoint_row = math.floor(endpoint.y / cell_size)
        neighbor_indices: set[int] = set()

        for column in range(
            endpoint_column - bucket_radius,
            endpoint_column + bucket_radius + 1,
        ):
            for row in range(
                endpoint_row - bucket_radius,
                endpoint_row + bucket_radius + 1,
            ):
                neighbor_indices.update(
                    marker_grid.get((column, row), [])
                )

        for neighbor_index in neighbor_indices:
            if neighbor_index == drawing_index:
                continue

            neighbor_drawing = drawings[neighbor_index]
            #@GPT 同Y首尾相接的细矩形不能互相保护；这正是PDF把“图26”一条下划线拆成两个蓝色填充矩形的情况。
            if filled_bars_are_collinear_neighbors(
                first_drawing=connector_drawing,
                second_drawing=neighbor_drawing,
            ):
                continue

            neighbor_rect = fitz.Rect(
                neighbor_drawing["rect"]
            )

            if (
                point_to_rect_distance(endpoint, neighbor_rect)
                <= VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS
            ):
                return True

    return False



def drawing_is_page_enclosing_stroke_frame(
    drawing: dict,
    page_rect: fitz.Rect,
) -> bool:
    """判断drawing是否为不应参与内容聚类的整页描边框。"""
    drawing_rect = fitz.Rect(drawing["rect"]).normalize()
    if drawing.get("type") != "s" or drawing.get("fill") is not None:
        return False

    items = drawing.get("items", [])
    if len(items) != 1 or items[0][0] != "re":
        return False

    stroke_width = float(drawing.get("width") or 0.0)
    if stroke_width > 1.0:
        return False

    #@GPT 允许少量浮点误差和出血延伸；两条边都必须覆盖页面主体，避免误伤普通横线或局部框。
    width_coverage = (
        max(
            0.0,
            min(drawing_rect.x1, page_rect.x1)
            - max(drawing_rect.x0, page_rect.x0),
        )
        / page_rect.width
        if page_rect.width > 0
        else 0.0
    )
    height_coverage = (
        max(
            0.0,
            min(drawing_rect.y1, page_rect.y1)
            - max(drawing_rect.y0, page_rect.y0),
        )
        / page_rect.height
        if page_rect.height > 0
        else 0.0
    )
    return width_coverage >= 0.98 and height_coverage >= 0.98



def process_page_visual_regions(
    page: fitz.Page,
    page_dict: dict,
    text_lines: list[dict],
    all_table_rects: list[fitz.Rect],
    #@GPT 页眉/页尾排除带由文档级模板先验生成，并在矢量候选提取前生效。
    excluded_header_footer_rects: list[fitz.Rect],
    page_number: int,
    regions_dir: Path,
    #@GPT 2026-08-15：包装层同步要求阶段B逻辑横线，防止签名允许None而下游只接受列表。
    logical_horizontal_rules: list[dict],
    #@GPT 视觉输出包装层沿用页面级drawing、原图和台账缓存。
    drawings: list[dict],
    source_images: list[dict],
    page_inventory: PageInventory,
    #@GPT 视觉输出与粗区域检测共用页面轴向索引，封闭框边界由阶段B一次性分类结果提供。
    axis_geometry_index: dict,
    #@GPT 父表格骨架在视觉掩码前按ID剔除；与all_table_rects整块挖空不同，单元格内部子视觉仍会形成候选。
    excluded_drawing_element_ids: set[str] | None = None,
) -> tuple[
    list[dict],
    list[dict],
    list[fitz.Rect],
    list[fitz.Rect],
    float,
    list[dict],
]:
    """检测、渲染并记录一页中的表格外粗粒度视觉区域。"""
    started = time.perf_counter()
    regions = detect_coarse_visual_regions(
        page=page,
        page_dict=page_dict,
        text_lines=text_lines,
        all_table_rects=all_table_rects,
        excluded_header_footer_rects=excluded_header_footer_rects,
        drawings=drawings,
        source_images=source_images,
        page_inventory=page_inventory,
        axis_geometry_index=axis_geometry_index,
        excluded_drawing_element_ids=(
            excluded_drawing_element_ids
        ),
        logical_horizontal_rules=logical_horizontal_rules,
        max_background_area_page_ratio=(
            MAX_BACKGROUND_AREA_PAGE_RATIO
        ),
        build_endpoint_marker_grid=build_endpoint_marker_grid,
        drawing_has_endpoint_marker=drawing_has_endpoint_marker,
        drawing_has_visible_semantic_stroke=(
            drawing_has_visible_semantic_stroke
        ),
        drawing_is_filled_horizontal_bar=(
            drawing_is_filled_horizontal_bar
        ),
        drawing_is_long_horizontal_single_line=(
            drawing_is_long_horizontal_single_line
        ),
        drawing_is_page_enclosing_stroke_frame=(
            drawing_is_page_enclosing_stroke_frame
        ),
        detect_rule_line_framework_bbox=partial(
            detect_rule_line_framework_bbox,
            alignment_tolerance_points=(
                FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
            ),
            group_horizontal_rules_by_span=group_horizontal_rules_by_span,
            unique_rule_positions=unique_rule_positions,
            rule_element_ids=rule_element_ids,
        ),
        group_horizontal_rules_by_span=group_horizontal_rules_by_span,
        unique_rule_positions=unique_rule_positions,
        rule_element_ids=rule_element_ids,
        rect_is_contained_with_tolerance=(
            rect_is_contained_with_tolerance
        ),
        rect_is_mostly_inside_any=rect_is_mostly_inside_any,
    )
    #@GPT visual_assets.render_visual_regions统一生成截图、输出记录、Markdown原子和Claim；本函数负责编排和总耗时。
    (
        records,
        markdown_elements,
        output_rects,
        owned_text_rects,
        claims,
    ) = render_visual_regions(
        page=page,
        page_number=page_number,
        regions=regions,
        regions_dir=regions_dir,
        page_inventory=page_inventory,
        render_dpi=VISUAL_REGION_DPI,
        sort_lines_in_reading_order=partial(
            sort_lines_in_reading_order,
            row_tolerance_points=TEXT_ROW_TOLERANCE_POINTS,
        ),
        create_claim_manifest=create_claim_manifest,
    )

    #@GPT 视觉区域的渲染、Markdown原子、记录字段和Claim由render_visual_regions统一生成。
    return (
        records,
        markdown_elements,
        output_rects,
        owned_text_rects,
        time.perf_counter() - started,
        claims,
    )



def render_preclassified_rule_band_visuals(
    page: fitz.Page,
    page_number: int,
    bands: list[dict],
    regions_dir: Path,
    page_inventory: PageInventory,
    start_region_number: int,
) -> tuple[list[dict], list[dict], list[fitz.Rect], list[fitz.Rect], float, list[dict]]:
    """渲染表格检测前已分流的双水平规则复合带。"""
    started = time.perf_counter()
    records: list[dict] = []
    markdown_elements: list[dict] = []
    output_rects: list[fitz.Rect] = []
    owned_text_rects: list[fitz.Rect] = []
    claims: list[dict] = []

    for offset, band in enumerate(bands):
        region_number = start_region_number + offset
        render_bbox = padded_image_clip(
            bbox=fitz.Rect(band["bbox"]),
            page_rect=page.rect,
        )
        output_name = (
            f"page_{page_number:03d}_rule_band_"
            f"{offset + 1:02d}.png"
        )
        output_path = regions_dir / output_name
        render_started = time.perf_counter()
        pixmap = page.get_pixmap(
            clip=render_bbox,
            dpi=VISUAL_REGION_DPI,
            alpha=False,
            annots=True,
        )
        pixmap.save(output_path)
        render_seconds = time.perf_counter() - render_started

        native_text_markdown, ordered_text_lines = (
            format_visual_region_native_text(
                text_lines=band["text_lines"],
                page_rect=page.rect,
                sort_lines_in_reading_order=partial(
                    sort_lines_in_reading_order,
                    row_tolerance_points=TEXT_ROW_TOLERANCE_POINTS,
                ),
            )
        )
        relative_path = f"visual_regions/{output_name}"
        image_markdown = (
            f"![Page {page_number} Visual Rule Band {offset + 1}]"
            f"({relative_path})"
        )
        markdown_text = (
            f"{image_markdown}\n\n{native_text_markdown}"
            if native_text_markdown
            else image_markdown
        )
        sort_source_bbox = fitz.Rect(band["bbox"])
        sort_bbox = (
            build_top_sort_bbox(sort_source_bbox)
            if ordered_text_lines
            else fitz.Rect(render_bbox)
        )
        claim_id = (
            f"page#{page_number}:visual_region#{region_number}"
        )
        text_ids = list(band["text_element_ids"])
        feature_ids = sorted(
            set(band["drawing_element_ids"]).union(
                band["image_element_ids"]
            )
        )
        claimed_ids = sorted(set(feature_ids).union(text_ids))

        markdown_elements.append(
            {
                "bbox": fitz.Rect(render_bbox),
                "render_bbox": fitz.Rect(render_bbox),
                "sort_bbox": fitz.Rect(sort_bbox),
                "text": markdown_text,
                "direction": (1.0, 0.0),
                "angle": 0,
                "kind": "image",
                "claim_id": claim_id,
                "markdown": markdown_text,
                "mode": "preclassified_visual_rule_band",
                "output_path": output_path,
            }
        )
        records.append(
            {
                "page_number": page_number,
                "region_number": region_number,
                "claim_id": claim_id,
                "bbox": [round(value, 3) for value in render_bbox],
                "render_bbox": [
                    round(value, 3) for value in render_bbox
                ],
                "sort_bbox": [round(value, 3) for value in sort_bbox],
                "sort_source_bbox": [
                    round(value, 3) for value in sort_source_bbox
                ],
                "content_bbox": [
                    round(value, 3) for value in band["bbox"]
                ],
                "drawing_count": len(
                    band["drawing_element_ids"]
                ),
                "component_count": 1,
                "included_text_line_count": len(ordered_text_lines),
                "crop_text_line_count": 0,
                "promoted_crop_text_line_count": 0,
                "leader_promoted_text_line_count": 0,
                "endpoint_promoted_text_line_count": 0,
                "endpoint_extension_drawing_count": 0,
                "rule_line_frameworks": [
                    {
                        "kind": "preclassified_two_rule_band",
                        "horizontal_rule_count": 2,
                        "drawing_element_ids": band[
                            "rule_drawing_ids"
                        ],
                        "reason": band["reason"],
                    }
                ],
                "native_text_lines": [
                    {
                        "text": line["text"],
                        "bbox": [
                            round(value, 3)
                            for value in line["bbox"]
                        ],
                        "angle": line["angle"],
                    }
                    for line in ordered_text_lines
                ],
                "included_image_count": len(
                    band["image_element_ids"]
                ),
                "output_path": str(output_path),
                "output_width": pixmap.width,
                "output_height": pixmap.height,
                "render_seconds": round(render_seconds, 6),
                "mode": "preclassified_visual_rule_band",
                "preclassified_rule_band": True,
                "source_drawing_element_ids": band[
                    "drawing_element_ids"
                ],
                "source_image_element_ids": band[
                    "image_element_ids"
                ],
                "source_text_element_ids": text_ids,
            }
        )
        output_rects.append(render_bbox)
        owned_text_rects.extend(
            fitz.Rect(line["bbox"])
            for line in ordered_text_lines
        )
        claims.append(create_claim_manifest(
            detector_name="visual_region",
            output_element_id=claim_id,
            input_element_ids=claimed_ids,
            claimed_element_ids=claimed_ids,
            feature_element_ids=feature_ids,
            bbox=fitz.Rect(band["bbox"]),
            confidence=0.90,
            reason_codes=[
                "preclassified_two_rule_band",
                "visual_rule_band",
                "native_text_inside_band",
                *(
                    ["source_image_inside_band"]
                    if band["image_element_ids"]
                    else []
                ),
                *(
                    ["non_rule_vector_inside_band"]
                    if band["content_drawing_ids"]
                    else []
                ),
            ],
            output_artifacts=[
                {
                    "type": "visual_rule_band_png",
                    "region_number": region_number,
                    "path": str(output_path),
                    "bbox": [
                        round(value, 3)
                        for value in band["bbox"]
                    ],
                }
            ],
            metadata={
                "preclassified": True,
                "rule_drawing_ids": band["rule_drawing_ids"],
                "drawing_element_ids": band[
                    "drawing_element_ids"
                ],
                "image_element_ids": band[
                    "image_element_ids"
                ],
                "text_element_ids": text_ids,
                "mode": band["mode"],
            },
        ))

    return (
        records,
        markdown_elements,
        output_rects,
        owned_text_rects,
        time.perf_counter() - started,
        claims,
    )
