"""编排文档级和页面级处理流水线。

模块定义页面阶段适配器、执行顺序、文档
迭代、共享运行依赖、结果收尾和制品写出；具体检测算法保留在各主题模块，
本模块只决定调用时机和完成态如何传递。

处理流程:
    1. 校验配置并初始化文档级缓存。
    2. 按声明的 ``PageStage`` 顺序处理每个选定页面。
    3. 仲裁结构归属并重建最终输出边界。
    4. 提交页面结果、关闭文档并写出摘要制品。

关键不变量:
    * 页面原生文字、图片和drawing按所需视图只提取一次。
    * 阶段顺序显式定义，并可在调试模式校验。
    * 检测器接收冻结的依赖对象，不读取隐藏全局状态。
    * 最简模式和调试模式返回同一个 ``ParseResult`` 实例。

Orchestrate the document and page processing pipeline.

Defines page-stage adapters, stage order, document iteration, shared runtime
dependencies, result finalization, and artifact emission. Detection algorithms
remain in focused topic modules; this module controls when they run and how
their completed states are transferred.

Processing flow:
    1. Validate configuration and initialize document-wide caches.
    2. Process each selected page through the declared ``PageStage`` order.
    3. Resolve structural ownership and rebuild final output boundaries.
    4. Commit page results, close the document, and emit summary artifacts.

Key invariants:
    * A page extracts native text, images, and drawings once per required view.
    * Stage order is explicit and optionally validated in debug mode.
    * Detectors receive frozen dependency objects rather than hidden globals.
    * Minimal and debug modes return the same ``ParseResult`` instance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable
import json
import time

import fitz

from .page_context import PageRunContext
from .page_contract import PageStage
from .run_state import (
    PageBoundaryOwnershipResult,
    PageClaimResolutionResult,
    PageGeometryPreparation,
    PageHeaderFooterResult,
    PageInitialization,
    PageOutputCandidateProposal,
    PageReadingOrderProcessingResult,
    PageRuleBandPreparation,
    PageSemanticProcessingResult,
    PageSourceImageDetectionResult,
    PageVisualDetectionResult,
    PageVisualProtectionPreparation,
)


@dataclass(frozen=True)
class PageStageAdapterDependencies:
    ENABLE_FONT_HEADING_INFERENCE: Any
    ENABLE_HEADING_DETECTION: Any
    ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT: Any
    FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS: Any
    FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO: Any
    FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO: Any
    FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS: Any
    FIGURE_CAPTION_LABEL_PATTERN: Any
    FIGURE_CAPTION_MAX_TEXT_CHARACTERS: Any
    FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER: Any
    FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS: Any
    FORMULA_IMAGE_DPI: Any
    FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO: Any
    FORMULA_PADDING_POINTS: Any
    FORMULA_OUTPUT_MODE: Any
    GAPTREE_DEBUG_DPI: Any
    HEADER_FOOTER_BAND_PADDING_POINTS: Any
    HEADER_FOOTER_CANDIDATE_COUNT: Any
    HEADER_FOOTER_DETECT_FOOTER: Any
    HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE: Any
    HEADER_FOOTER_EDGE_TOLERANCE_POINTS: Any
    HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS: Any
    HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE: Any
    HEADER_FOOTER_X_TOLERANCE_POINTS: Any
    HEADER_TEXT_OVERLAP_THRESHOLD: Any
    HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT: Any
    HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS: Any
    HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO: Any
    HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS: Any
    HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS: Any
    HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS: Any
    HEADING_MAX_MARKDOWN_LEVEL: Any
    HEADING_MAX_TEXT_CHARACTERS: Any
    HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS: Any
    HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS: Any
    HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO: Any
    HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO: Any
    HEADING_SAME_BODY_REQUIRES_BOLD: Any
    LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES: Any
    MAX_BACKGROUND_AREA_PAGE_RATIO: Any
    MIN_IMAGE_AREA_POINTS_SQUARED: Any
    PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO: Any
    PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS: Any
    TABLE_IMAGE_DPI: Any
    VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO: Any
    annotate_heading_levels: Any
    annotate_inventory_labels: Any
    append_reflowed_text_to_markdown: Any
    apply_claim_resolution_to_outputs: Any
    apply_stage_e_f_to_visual_outputs: Any
    apply_stage_e_to_source_outputs: Any
    build_axis_geometry_index: Any
    build_logical_horizontal_rules: Any
    build_page_inventory: Any
    build_preclassified_text_separator_outputs: Any
    build_reflowed_page_text: Any
    build_stage_f_text_ownership_preview: Any
    build_table_visual_composite_output_groups: Any
    classify_figure_caption_roles: Any
    create_claim_manifest: Any
    cross_detector_ownership_dependencies: Any
    detect_page_header_footer_bands: Any
    detect_preclassified_rule_bands: Any
    detect_text_separator_markdown_elements: Any
    document_run_state: Any
    drawing_axis_rule: Any
    drawing_content_signature: Any
    drawing_is_stroked_preclassified_horizontal_rule: Any
    emit_table_candidate: Any
    extract_header_text_for_markdown: Any
    extract_source_image_blocks: Any
    extract_text_lines: Any
    filter_confirmed_header_footer_drawings: Any
    finalize_logical_horizontal_rule_audit: Any
    finalize_table_visual_composite_output_groups: Any
    group_horizontal_rules_by_span: Any
    horizontal_pair_has_boundary_grid_evidence: Any
    horizontal_pair_repeats_in_another_span: Any
    image_content_signature: Any
    image_is_large_enough: Any
    line_is_inside_any_table: Any
    line_matches_running_header_layout: Any
    meaningful_overlap: Any
    merge_collinear_vertical_rules: Any
    padded_table_candidate_clip: Any
    prepare_precluster_object_exclusions: Any
    process_page_formula_regions: Any
    process_page_source_images: Any
    process_page_visual_regions: Any
    rect_is_mostly_inside_any: Any
    render_gaptree_debug_image: Any
    render_preclassified_rule_band_visuals: Any
    resolve_table_visual_candidate_ownership: Any
    restore_inactive_composite_table_outputs: Any
    rule_element_ids: Any
    run_metrics: Any
    save_rotated_table_image: Any
    sort_elements_with_gap_tree: Any
    unique_rule_positions: Any
    vertical_rule_connects_horizontal_pair: Any

def initialize_page_context(
    page_index: int,
    *,
    dependencies: PageStageAdapterDependencies,
) -> PageInitialization:
    """Load one page's raw objects and build its initial inventory."""
    annotate_inventory_labels = dependencies.annotate_inventory_labels
    build_page_inventory = dependencies.build_page_inventory
    document_run_state = dependencies.document_run_state
    extract_source_image_blocks = dependencies.extract_source_image_blocks
    extract_text_lines = dependencies.extract_text_lines
    run_metrics = dependencies.run_metrics

    page_number = page_index + 1
    page = document_run_state.document.load_page(page_index)
    page_started = time.perf_counter()

    drawing_started = time.perf_counter()
    if page_index in document_run_state.drawings_cache:
        page_drawings = document_run_state.drawings_cache[page_index]
    else:
        page_drawings = page.get_drawings()
        document_run_state.drawings_cache[page_index] = page_drawings
    page_drawing_seconds = time.perf_counter() - drawing_started
    run_metrics.drawing_extraction_seconds += page_drawing_seconds

    page_dict = page.get_text("dict", sort=False)
    text_lines = extract_text_lines(page_dict)
    source_images_for_precluster = extract_source_image_blocks(
        page_dict
    )
    page_context = PageRunContext(
        page_index=page_index,
        page_number=page_number,
        page=page,
        page_rect=page.rect,
        page_dict=page_dict,
        text_lines=text_lines,
        source_images=source_images_for_precluster,
        raw_drawings=page_drawings,
    )
    page_context.record_timing(
        "drawing_extraction",
        page_drawing_seconds,
    )

    inventory_started = time.perf_counter()
    page_inventory = build_page_inventory(
        page_number=page_number,
        page_rect=page.rect,
        drawings=page_context.drawings,
        image_blocks=page_context.source_images,
        text_lines=page_context.text_lines,
    )
    annotate_inventory_labels(page_inventory)
    page_context.page_inventory = page_inventory
    page_inventory_seconds = (
        time.perf_counter() - inventory_started
    )
    run_metrics.inventory_build_seconds += page_inventory_seconds
    page_context.inventory_audit_record = {
        **page_inventory.to_audit_record(),
        "build_seconds": round(page_inventory_seconds, 6),
    }
    page_context.record_timing(
        "inventory_build",
        page_inventory_seconds,
    )

    return PageInitialization(
        page_number=page_number,
        page=page,
        page_started=page_started,
        page_drawings=page_context.drawings,
        page_dict=page_context.page_dict,
        text_lines=page_context.text_lines,
        source_images=page_context.source_images,
        page_context=page_context,
        page_inventory=page_inventory,
        page_drawing_seconds=page_drawing_seconds,
        page_inventory_seconds=page_inventory_seconds,
    )

def process_page_header_footer(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    keep_header_text: bool,
    dependencies: PageStageAdapterDependencies,
) -> PageHeaderFooterResult:
    """Detect, filter, and annotate page-header/footer content."""
    HEADER_FOOTER_BAND_PADDING_POINTS = dependencies.HEADER_FOOTER_BAND_PADDING_POINTS
    HEADER_FOOTER_CANDIDATE_COUNT = dependencies.HEADER_FOOTER_CANDIDATE_COUNT
    HEADER_FOOTER_DETECT_FOOTER = dependencies.HEADER_FOOTER_DETECT_FOOTER
    HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE = dependencies.HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE
    HEADER_FOOTER_EDGE_TOLERANCE_POINTS = dependencies.HEADER_FOOTER_EDGE_TOLERANCE_POINTS
    HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS = dependencies.HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS
    HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE = dependencies.HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE
    HEADER_FOOTER_X_TOLERANCE_POINTS = dependencies.HEADER_FOOTER_X_TOLERANCE_POINTS
    HEADER_TEXT_OVERLAP_THRESHOLD = dependencies.HEADER_TEXT_OVERLAP_THRESHOLD
    MAX_BACKGROUND_AREA_PAGE_RATIO = dependencies.MAX_BACKGROUND_AREA_PAGE_RATIO
    detect_page_header_footer_bands = dependencies.detect_page_header_footer_bands
    document_run_state = dependencies.document_run_state
    drawing_content_signature = dependencies.drawing_content_signature
    extract_header_text_for_markdown = dependencies.extract_header_text_for_markdown
    filter_confirmed_header_footer_drawings = dependencies.filter_confirmed_header_footer_drawings
    image_content_signature = dependencies.image_content_signature

    started = time.perf_counter()
    (
        page_context.header_footer_rects,
        page_context.header_footer_record,
    ) = detect_page_header_footer_bands(
        page=page,
        page_dict=page_context.page_dict,
        templates=document_run_state.header_footer_templates,
        drawings=page_context.drawings,
        detect_footer=HEADER_FOOTER_DETECT_FOOTER,
        minimum_matched_objects_by_edge=(
            HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE
        ),
        band_padding_points=HEADER_FOOTER_BAND_PADDING_POINTS,
        edge_page_ratio_by_edge=(
            HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE
        ),
        edge_tolerance_points=(
            HEADER_FOOTER_EDGE_TOLERANCE_POINTS
        ),
        candidate_count=HEADER_FOOTER_CANDIDATE_COUNT,
        max_background_area_page_ratio=(
            MAX_BACKGROUND_AREA_PAGE_RATIO
        ),
        image_content_signature=image_content_signature,
        drawing_content_signature=drawing_content_signature,
        height_tolerance_points=(
            HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS
        ),
        x_tolerance_points=HEADER_FOOTER_X_TOLERANCE_POINTS,
    )
    (
        page_context.drawings,
        page_context.header_footer_record,
    ) = filter_confirmed_header_footer_drawings(
        drawings=page_context.drawings,
        header_footer_record=page_context.header_footer_record,
    )
    page_context.vector_exclusion_rects = []
    header_band = next(
        (
            rect
            for rect in page_context.header_footer_rects
            if rect.y0 <= page.rect.y0
            and rect.y1 < page.rect.y1
        ),
        None,
    )
    header_text = ""
    header_text_lines: list[dict] = []
    if keep_header_text:
        (
            header_text,
            header_text_lines,
        ) = extract_header_text_for_markdown(
            text_lines=page_context.text_lines,
            header_band=header_band,
            overlap_threshold=HEADER_TEXT_OVERLAP_THRESHOLD,
        )
    page_context.header_footer_record["header"][
        "text_retention_enabled"
    ] = keep_header_text
    page_context.header_footer_record["header"]["text"] = (
        header_text if keep_header_text else None
    )
    page_context.header_footer_record["header"][
        "text_line_count"
    ] = len(header_text_lines)
    detection_seconds = time.perf_counter() - started
    document_run_state.metrics.header_footer_page_detection_seconds += (
        detection_seconds
    )
    page_context.record_timing(
        "header_footer_detection",
        detection_seconds,
    )
    return PageHeaderFooterResult(
        page_drawings=page_context.drawings,
        header_footer_rects=page_context.header_footer_rects,
        header_footer_record=page_context.header_footer_record,
        vector_exclusion_rects=page_context.vector_exclusion_rects,
        header_text=header_text,
        header_text_lines=header_text_lines,
        detection_seconds=detection_seconds,
    )

def prepare_page_geometry(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    dependencies: PageStageAdapterDependencies,
) -> PageGeometryPreparation:
    """Build the shared axis index and logical horizontal rules once."""
    FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS = dependencies.FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS
    FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO = dependencies.FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO
    LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES = dependencies.LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES
    build_axis_geometry_index = dependencies.build_axis_geometry_index
    build_logical_horizontal_rules = dependencies.build_logical_horizontal_rules
    document_run_state = dependencies.document_run_state
    drawing_axis_rule = dependencies.drawing_axis_rule

    axis_started = time.perf_counter()
    page_context.axis_geometry_index = build_axis_geometry_index(
        drawings=page_context.drawings,
        page_rect=page_context.page_rect,
        page_inventory=page_context.page_inventory,
        drawing_axis_rule=drawing_axis_rule,
        maximum_rule_thickness_points=(
            FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS
        ),
        coordinate_decimal_places=(
            LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES
        ),
        minimum_horizontal_length_page_ratio=(
            FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO
        ),
    )
    axis_seconds = time.perf_counter() - axis_started
    document_run_state.metrics.axis_geometry_index_seconds_total += (
        axis_seconds
    )
    page_context.record_timing("axis_geometry_index", axis_seconds)

    logical_started = time.perf_counter()
    page_context.logical_horizontal_rule_audit = []
    page_context.logical_horizontal_rules = (
        build_logical_horizontal_rules(
            page_rect=page_context.page_rect,
            axis_geometry_index=page_context.axis_geometry_index,
            page_number=page_number,
            audit_records=(
                page_context.logical_horizontal_rule_audit
            ),
            minimum_horizontal_length_page_ratio=(
                FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO
            ),
        )
    )
    logical_seconds = time.perf_counter() - logical_started
    document_run_state.metrics.logical_horizontal_rule_seconds_total += (
        logical_seconds
    )
    page_context.record_timing(
        "logical_horizontal_rules",
        logical_seconds,
    )
    return PageGeometryPreparation(
        axis_geometry_index=page_context.axis_geometry_index,
        logical_horizontal_rules=(
            page_context.logical_horizontal_rules
        ),
        logical_horizontal_rule_audit=(
            page_context.logical_horizontal_rule_audit
        ),
        axis_seconds=axis_seconds,
        logical_rule_seconds=logical_seconds,
    )

def prepare_page_rule_bands(
    page_context: PageRunContext,
    *,
    page_number: int,
    dependencies: PageStageAdapterDependencies,
) -> PageRuleBandPreparation:
    """Route paired horizontal-rule bands before table detection."""
    FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO = dependencies.FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO
    FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS = dependencies.FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
    MIN_IMAGE_AREA_POINTS_SQUARED = dependencies.MIN_IMAGE_AREA_POINTS_SQUARED
    PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO = dependencies.PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
    PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS = dependencies.PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
    build_preclassified_text_separator_outputs = dependencies.build_preclassified_text_separator_outputs
    create_claim_manifest = dependencies.create_claim_manifest
    detect_preclassified_rule_bands = dependencies.detect_preclassified_rule_bands
    document_run_state = dependencies.document_run_state
    drawing_is_stroked_preclassified_horizontal_rule = dependencies.drawing_is_stroked_preclassified_horizontal_rule
    group_horizontal_rules_by_span = dependencies.group_horizontal_rules_by_span
    horizontal_pair_has_boundary_grid_evidence = dependencies.horizontal_pair_has_boundary_grid_evidence
    horizontal_pair_repeats_in_another_span = dependencies.horizontal_pair_repeats_in_another_span
    image_is_large_enough = dependencies.image_is_large_enough
    meaningful_overlap = dependencies.meaningful_overlap
    merge_collinear_vertical_rules = dependencies.merge_collinear_vertical_rules
    rule_element_ids = dependencies.rule_element_ids
    unique_rule_positions = dependencies.unique_rule_positions
    vertical_rule_connects_horizontal_pair = dependencies.vertical_rule_connects_horizontal_pair

    started = time.perf_counter()
    page_context.preclassified_rule_band_result = (
        detect_preclassified_rule_bands(
            page_rect=page_context.page_rect,
            drawings=page_context.drawings,
            text_lines=page_context.text_lines,
            source_images=page_context.source_images,
            page_inventory=page_context.page_inventory,
            logical_horizontal_rules=(
                page_context.logical_horizontal_rules
            ),
            axis_geometry_index=page_context.axis_geometry_index,
            drawing_is_stroked_preclassified_horizontal_rule=(
                drawing_is_stroked_preclassified_horizontal_rule
            ),
            rule_element_ids=rule_element_ids,
            merge_collinear_vertical_rules=merge_collinear_vertical_rules,
            group_horizontal_rules_by_span=group_horizontal_rules_by_span,
            unique_rule_positions=unique_rule_positions,
            horizontal_pair_repeats_in_another_span=(
                horizontal_pair_repeats_in_another_span
            ),
            image_is_large_enough=image_is_large_enough,
            meaningful_overlap=meaningful_overlap,
            vertical_rule_connects_horizontal_pair=(
                vertical_rule_connects_horizontal_pair
            ),
            horizontal_pair_has_boundary_grid_evidence=(
                horizontal_pair_has_boundary_grid_evidence
            ),
            minimum_x_overlap_ratio=(
                PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
            ),
            alignment_tolerance_points=(
                FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
            ),
            minimum_vertical_coverage_ratio=(
                FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO
            ),
            minimum_image_area_points_squared=(
                MIN_IMAGE_AREA_POINTS_SQUARED
            ),
            vector_connection_tolerance_points=(
                PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
            ),
        )
    )
    result = page_context.preclassified_rule_band_result
    bands = result["bands"]
    drawing_ids = set(result["preclassified_rule_drawing_ids"])
    (
        page_context.preclassified_text_separator_markdown_elements,
        page_context.preclassified_text_separator_records,
        page_context.preclassified_text_separator_claims,
    ) = build_preclassified_text_separator_outputs(
        page_number=page_number,
        bands=bands,
        create_claim_manifest=create_claim_manifest,
    )
    detection_seconds = time.perf_counter() - started
    document_run_state.metrics.preclassified_rule_band_detection_seconds += (
        detection_seconds
    )
    document_run_state.metrics.preclassified_rule_band_output_count += len(
        bands
    )
    page_context.record_timing(
        "preclassified_rule_bands",
        detection_seconds,
    )
    return PageRuleBandPreparation(
        rule_band_result=result,
        bands=bands,
        drawing_ids=drawing_ids,
        detection_seconds=detection_seconds,
        output_count=len(bands),
    )

def prepare_page_visual_protection(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    dependencies: PageStageAdapterDependencies,
) -> PageVisualProtectionPreparation:
    """Build pre-morphology object and table protection rectangles."""
    PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS = dependencies.PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
    prepare_precluster_object_exclusions = dependencies.prepare_precluster_object_exclusions

    page_context.precluster_object_exclusions = (
        prepare_precluster_object_exclusions(
            table_candidates=page_context.table_candidates,
            source_images=page_context.source_images,
            drawings=page_context.drawings,
            page_rect=page.rect,
        )
    )
    page_context.precluster_protected_rects = [
        fitz.Rect(rect)
        for rect in page_context.precluster_object_exclusions[
            "protected_rects"
        ]
    ]
    page_context.table_visual_protection_rects = [
        (
            fitz.Rect(
                candidate.get("render_bbox", candidate["bbox"])
            ).normalize()
            + (
                -PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS,
                -PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS,
                PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS,
                PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS,
            )
        )
        & page.rect
        for candidate in page_context.table_candidates
        if candidate.get("table_owns_full_visual_bbox", False)
    ]
    return PageVisualProtectionPreparation(
        object_exclusions=page_context.precluster_object_exclusions,
        protected_rects=page_context.precluster_protected_rects,
        table_visual_protection_rects=(
            page_context.table_visual_protection_rects
        ),
    )

def detect_page_visual_regions(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    rule_bands: list[dict],
    dependencies: PageStageAdapterDependencies,
) -> PageVisualDetectionResult:
    """Detect ordinary and preclassified visual regions for one page."""
    document_run_state = dependencies.document_run_state
    process_page_visual_regions = dependencies.process_page_visual_regions
    rect_is_mostly_inside_any = dependencies.rect_is_mostly_inside_any
    render_preclassified_rule_band_visuals = dependencies.render_preclassified_rule_band_visuals

    (
        page_context.visual_region_records,
        page_context.visual_region_markdown_elements,
        page_context.visual_region_rects,
        page_context.visual_region_text_rects,
        page_context.visual_region_seconds,
        page_context.visual_region_claims,
    ) = process_page_visual_regions(
        page=page,
        page_dict=page_context.page_dict,
        text_lines=page_context.text_lines,
        all_table_rects=page_context.table_visual_protection_rects,
        excluded_header_footer_rects=(
            page_context.vector_exclusion_rects
        ),
        page_number=page_number,
        regions_dir=document_run_state.output_layout.visual_regions,
        drawings=page_context.drawings,
        source_images=page_context.source_images,
        axis_geometry_index=page_context.axis_geometry_index,
        page_inventory=page_context.page_inventory,
        excluded_drawing_element_ids=(
            set(page_context.table_full_drawing_ids).union(
                page_context.preclassified_rule_band_result[
                    "preclassified_rule_drawing_ids"
                ]
            )
        ),
        logical_horizontal_rules=(
            page_context.logical_horizontal_rules
        ),
    )
    document_run_state.metrics.visual_region_processing_seconds += (
        page_context.visual_region_seconds
    )
    (
        preclassified_records,
        preclassified_markdown,
        preclassified_rects,
        preclassified_text_rects,
        preclassified_seconds,
        preclassified_claims,
    ) = render_preclassified_rule_band_visuals(
        page=page,
        page_number=page_number,
        bands=[
            band
            for band in rule_bands
            if band["mode"] == "visual_rule_band"
            and not band.get("embedded_in_table", False)
            and not rect_is_mostly_inside_any(
                rect=fitz.Rect(band["bbox"]).normalize(),
                container_rects=(
                    page_context.table_visual_protection_rects
                ),
            )
        ],
        regions_dir=document_run_state.output_layout.visual_regions,
        page_inventory=page_context.page_inventory,
        start_region_number=(
            len(page_context.visual_region_records) + 1
        ),
    )
    document_run_state.metrics.visual_region_processing_seconds += (
        preclassified_seconds
    )
    page_context.visual_region_records.extend(preclassified_records)
    page_context.visual_region_markdown_elements.extend(
        preclassified_markdown
    )
    page_context.visual_region_rects.extend(preclassified_rects)
    page_context.visual_region_text_rects.extend(
        preclassified_text_rects
    )
    page_context.visual_region_claims.extend(preclassified_claims)
    page_context.embedded_child_visual_count = 0
    page_context.record_timing(
        "visual_regions",
        page_context.visual_region_seconds,
    )
    page_context.record_timing(
        "preclassified_visual_regions",
        preclassified_seconds,
    )
    return PageVisualDetectionResult(
        records=page_context.visual_region_records,
        markdown_elements=(
            page_context.visual_region_markdown_elements
        ),
        rects=page_context.visual_region_rects,
        text_rects=page_context.visual_region_text_rects,
        claims=page_context.visual_region_claims,
        base_seconds=page_context.visual_region_seconds,
        preclassified_seconds=preclassified_seconds,
        embedded_child_visual_count=(
            page_context.embedded_child_visual_count
        ),
    )

def propose_page_output_candidates(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    dependencies: PageStageAdapterDependencies,
) -> PageOutputCandidateProposal:
    """Create table atoms and table/visual composite proposals."""
    TABLE_IMAGE_DPI = dependencies.TABLE_IMAGE_DPI
    build_table_visual_composite_output_groups = dependencies.build_table_visual_composite_output_groups
    cross_detector_ownership_dependencies = dependencies.cross_detector_ownership_dependencies
    document_run_state = dependencies.document_run_state
    emit_table_candidate = dependencies.emit_table_candidate
    padded_table_candidate_clip = dependencies.padded_table_candidate_clip
    resolve_table_visual_candidate_ownership = dependencies.resolve_table_visual_candidate_ownership
    save_rotated_table_image = dependencies.save_rotated_table_image

    legacy_decisions, _legacy_suppressed_indices = (
        resolve_table_visual_candidate_ownership(
            table_candidates=page_context.table_candidates,
            visual_region_rects=page_context.visual_region_rects,
            dependencies=cross_detector_ownership_dependencies,
        )
    )
    table_decisions = [
        {
            **decision,
            "legacy_claimed_by_visual_region": decision[
                "claimed_by_visual_region"
            ],
            "legacy_visual_region_index": decision[
                "visual_region_index"
            ],
            "claimed_by_visual_region": False,
            "visual_region_index": None,
        }
        for decision in legacy_decisions
    ]
    suppressed_visual_indices: set[int] = set()
    page_context.table_ownership_decisions = table_decisions
    page_context.suppressed_visual_indices = suppressed_visual_indices
    decisions_by_index = {
        decision["table_index"]: decision
        for decision in table_decisions
    }
    for table_index, candidate in enumerate(
        page_context.table_candidates
    ):
        candidate.update(decisions_by_index[table_index])

    page_context.composite_output_groups = (
        build_table_visual_composite_output_groups(
            page_number=page_number,
            page_rect=page.rect,
            page_inventory=page_context.page_inventory,
            text_lines=page_context.text_lines,
            excluded_header_footer_rects=(
                page_context.vector_exclusion_rects
            ),
            table_candidates=page_context.table_candidates,
            visual_records=page_context.visual_region_records,
            visual_markdown_elements=(
                page_context.visual_region_markdown_elements
            ),
            table_claims=page_context.table_claims,
            visual_claims=page_context.visual_region_claims,
            dependencies=cross_detector_ownership_dependencies,
        )
    )
    for table_index, candidate in enumerate(
        page_context.table_candidates
    ):
        decision = decisions_by_index[table_index]
        emitted = emit_table_candidate(
            candidate=candidate,
            page=page,
            page_number=page_number,
            tables_dir=document_run_state.output_layout.tables,
            table_images_dir=(
                document_run_state.output_layout.table_images
            ),
            table_image_dpi=TABLE_IMAGE_DPI,
            padded_table_candidate_clip=padded_table_candidate_clip,
            save_rotated_table_image=save_rotated_table_image,
        )
        page_context.table_records.append(emitted["record"])
        if emitted["markdown_element"] is not None:
            emitted["markdown_element"]["claim_id"] = (
                emitted["record"]["claim_id"]
            )
            page_context.table_markdown_elements.append(
                emitted["markdown_element"]
            )
        if emitted["excluded_rect"] is not None:
            page_context.excluded_table_rects.append(
                emitted["excluded_rect"]
            )
        page_context.embedded_table_count += int(emitted["embedded"])
        page_context.auto_rotated_table_count += int(
            emitted["auto_rotated"]
        )
        document_run_state.metrics.table_screenshot_seconds += (
            emitted["screenshot_seconds"]
        )
        if candidate["is_anomalous_single_value_table"]:
            if decision["claimed_by_visual_region"]:
                page_context.anomalous_table_claimed_count += 1
            else:
                page_context.anomalous_table_unclaimed_count += 1

    page_context.visual_region_records = [
        record
        for index, record in enumerate(
            page_context.visual_region_records
        )
        if index not in suppressed_visual_indices
    ]
    page_context.visual_region_markdown_elements = [
        element
        for index, element in enumerate(
            page_context.visual_region_markdown_elements
        )
        if index not in suppressed_visual_indices
    ]
    page_context.visual_region_rects = [
        rect
        for index, rect in enumerate(page_context.visual_region_rects)
        if index not in suppressed_visual_indices
    ]
    page_context.visual_region_text_rects = [
        fitz.Rect(line["bbox"])
        for record in page_context.visual_region_records
        for line in record.get("native_text_lines", [])
    ]
    return PageOutputCandidateProposal(
        table_ownership_decisions=table_decisions,
        suppressed_visual_indices=suppressed_visual_indices,
        decisions_by_table_index=decisions_by_index,
        composite_output_groups=(
            page_context.composite_output_groups
        ),
    )

def detect_page_source_images(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    dependencies: PageStageAdapterDependencies,
) -> PageSourceImageDetectionResult:
    """Detect copied or rendered source-image candidates for one page."""
    document_run_state = dependencies.document_run_state
    process_page_source_images = dependencies.process_page_source_images

    (
        page_context.source_image_records,
        page_context.source_image_markdown_elements,
        page_context.source_image_seconds,
        page_context.source_image_claims,
    ) = process_page_source_images(
        page=page,
        page_dict=page_context.page_dict,
        text_lines=page_context.text_lines,
        excluded_table_rects=[],
        excluded_visual_region_rects=[],
        excluded_header_footer_rects=(
            page_context.vector_exclusion_rects
        ),
        page_number=page_number,
        images_dir=document_run_state.output_layout.source_images,
        drawings=page_context.drawings,
        source_images=page_context.source_images,
        page_inventory=page_context.page_inventory,
    )
    document_run_state.metrics.source_image_processing_seconds += (
        page_context.source_image_seconds
    )
    page_context.record_timing(
        "source_images",
        page_context.source_image_seconds,
    )
    return PageSourceImageDetectionResult(
        records=page_context.source_image_records,
        markdown_elements=(
            page_context.source_image_markdown_elements
        ),
        claims=page_context.source_image_claims,
        detection_seconds=page_context.source_image_seconds,
    )

def resolve_page_claims(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    dependencies: PageStageAdapterDependencies,
) -> PageClaimResolutionResult:
    """Resolve structural claims and keep only surviving output atoms."""
    apply_claim_resolution_to_outputs = dependencies.apply_claim_resolution_to_outputs
    cross_detector_ownership_dependencies = dependencies.cross_detector_ownership_dependencies
    document_run_state = dependencies.document_run_state
    finalize_table_visual_composite_output_groups = dependencies.finalize_table_visual_composite_output_groups
    restore_inactive_composite_table_outputs = dependencies.restore_inactive_composite_table_outputs

    page_context.detector_claims = [
        *page_context.table_claims,
        *page_context.visual_region_claims,
        *page_context.source_image_claims,
        *page_context.preclassified_text_separator_claims,
    ]
    page_context.claim_resolution = (
        page_context.page_inventory.resolve_conflicts(
            page_context.detector_claims
        )
    )
    resolution = page_context.claim_resolution
    resolution["policy"]["applied_to_outputs"] = True
    resolution["policy"]["application_scope"] = (
        "candidate_survival_only"
    )
    resolution["policy"]["render_bbox_rebuilt"] = False

    group_finalization = finalize_table_visual_composite_output_groups(
        output_groups=page_context.composite_output_groups,
        resolution=resolution,
        table_candidates=page_context.table_candidates,
        visual_records=page_context.visual_region_records,
        visual_markdown_elements=(
            page_context.visual_region_markdown_elements
        ),
        table_claims=page_context.table_claims,
        visual_claims=page_context.visual_region_claims,
        dependencies=cross_detector_ownership_dependencies,
    )
    (
        page_context.table_records,
        page_context.table_markdown_elements,
        restored_seconds,
        restore_actions,
    ) = restore_inactive_composite_table_outputs(
        restored_table_claim_ids=(
            group_finalization["restored_table_claim_ids"]
        ),
        table_candidates=page_context.table_candidates,
        page_table_records=page_context.table_records,
        page_table_markdown_elements=(
            page_context.table_markdown_elements
        ),
        page=page,
        page_number=page_number,
        tables_dir=document_run_state.output_layout.tables,
        table_images_dir=(
            document_run_state.output_layout.table_images
        ),
        dependencies=cross_detector_ownership_dependencies,
    )
    document_run_state.metrics.table_screenshot_seconds += (
        restored_seconds
    )
    resolution["output_groups"] = page_context.composite_output_groups
    resolution["output_group_finalization"] = group_finalization
    resolution["output_group_restore_actions"] = restore_actions

    (
        page_context.table_records,
        page_context.table_markdown_elements,
        suppressed_table_actions,
    ) = apply_claim_resolution_to_outputs(
        records=page_context.table_records,
        markdown_elements=page_context.table_markdown_elements,
        resolution=resolution,
        output_kind="table",
    )
    (
        page_context.visual_region_records,
        page_context.visual_region_markdown_elements,
        suppressed_visual_actions,
    ) = apply_claim_resolution_to_outputs(
        records=page_context.visual_region_records,
        markdown_elements=(
            page_context.visual_region_markdown_elements
        ),
        resolution=resolution,
        output_kind="visual_region",
    )
    (
        page_context.source_image_records,
        page_context.source_image_markdown_elements,
        suppressed_source_actions,
    ) = apply_claim_resolution_to_outputs(
        records=page_context.source_image_records,
        markdown_elements=(
            page_context.source_image_markdown_elements
        ),
        resolution=resolution,
        output_kind="source_image",
    )
    page_context.suppressed_claim_actions = [
        *suppressed_table_actions,
        *suppressed_visual_actions,
        *suppressed_source_actions,
    ]
    return PageClaimResolutionResult(
        resolution=resolution,
        suppressed_actions=page_context.suppressed_claim_actions,
        output_group_finalization=group_finalization,
        restore_actions=restore_actions,
        restored_table_screenshot_seconds=restored_seconds,
    )

def apply_page_boundary_ownership(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    dependencies: PageStageAdapterDependencies,
) -> PageBoundaryOwnershipResult:
    """Apply final structural boundaries and native-text ownership."""
    VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO = dependencies.VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO
    apply_stage_e_f_to_visual_outputs = dependencies.apply_stage_e_f_to_visual_outputs
    apply_stage_e_to_source_outputs = dependencies.apply_stage_e_to_source_outputs
    build_stage_f_text_ownership_preview = dependencies.build_stage_f_text_ownership_preview
    document_run_state = dependencies.document_run_state
    finalize_logical_horizontal_rule_audit = dependencies.finalize_logical_horizontal_rule_audit

    resolution = page_context.claim_resolution
    active_table_claim_ids = {
        element.get("claim_id")
        for element in page_context.table_markdown_elements
        if element.get("claim_id") is not None
    }
    page_context.excluded_table_rects = [
        fitz.Rect(record.get("render_bbox", record["bbox"]))
        for record in page_context.table_records
        if record.get("claim_id") in active_table_claim_ids
    ]
    page_context.excluded_table_rects.extend(
        fitz.Rect(bbox)
        for group in page_context.composite_output_groups
        if group.get("active", False)
        for bbox in group.get("member_table_bboxes", [])
    )
    page_context.visual_region_rects = [
        fitz.Rect(record["render_bbox"])
        for record in page_context.visual_region_records
    ]
    page_context.visual_region_text_rects = [
        fitz.Rect(line["bbox"])
        for record in page_context.visual_region_records
        for line in record.get("native_text_lines", [])
    ]
    emitted_claim_ids = {
        element["claim_id"]
        for element in [
            *page_context.table_markdown_elements,
            *page_context.visual_region_markdown_elements,
            *page_context.source_image_markdown_elements,
            *page_context.preclassified_text_separator_markdown_elements,
        ]
        if element.get("claim_id") is not None
    }
    active_composite_table_claim_ids = {
        claim_id
        for group in page_context.composite_output_groups
        if group.get("active", False)
        for claim_id in group.get("member_table_claim_ids", [])
    }
    resolution["stage_f_text_preview"] = (
        build_stage_f_text_ownership_preview(
            inventory=page_context.page_inventory,
            claims=page_context.detector_claims,
            resolution=resolution,
            emitted_claim_ids=emitted_claim_ids,
            overlap_threshold=(
                VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO
            ),
        )
    )
    visual_application = apply_stage_e_f_to_visual_outputs(
        page=page,
        page_inventory=page_context.page_inventory,
        resolution=resolution,
        stage_f_preview=resolution["stage_f_text_preview"],
        visual_records=page_context.visual_region_records,
        visual_markdown_elements=(
            page_context.visual_region_markdown_elements
        ),
    )
    source_application = apply_stage_e_to_source_outputs(
        page=page,
        page_inventory=page_context.page_inventory,
        resolution=resolution,
        stage_f_preview=resolution["stage_f_text_preview"],
        source_records=page_context.source_image_records,
        source_markdown_elements=(
            page_context.source_image_markdown_elements
        ),
    )
    resolution["stage_f_text_preview"]["applied_to_markdown"] = True
    resolution["stage_f_text_preview"]["policy"][
        "applied_to_markdown"
    ] = True
    resolution["policy"][
        "render_bbox_rebuilt_from_owned_elements"
    ] = True
    resolution["policy"]["stage_e_f_application"] = (
        visual_application
    )
    resolution["policy"]["stage_e_source_application"] = (
        source_application
    )

    audit_started = time.perf_counter()
    finalize_logical_horizontal_rule_audit(
        records=page_context.logical_horizontal_rule_audit,
        page_number=page_number,
        preclassified_rule_drawing_ids=set(
            page_context.preclassified_rule_band_result[
                "preclassified_rule_drawing_ids"
            ]
        ),
        table_skeleton_drawing_ids=(
            page_context.table_skeleton_drawing_ids
        ),
        detector_claims=page_context.detector_claims,
        resolution=resolution,
    )
    document_run_state.metrics.logical_horizontal_rule_audit_finalize_seconds_total += (
        time.perf_counter() - audit_started
    )

    page_context.visual_region_rects = [
        fitz.Rect(record["render_bbox"])
        for record in page_context.visual_region_records
    ]
    page_context.visual_region_text_rects = [
        fitz.Rect(line["bbox"])
        for record in page_context.visual_region_records
        for line in record.get("native_text_lines", [])
    ]
    page_context.source_image_text_rects = [
        fitz.Rect(line["bbox"])
        for record in page_context.source_image_records
        for line in record.get("native_text_lines", [])
    ]
    return PageBoundaryOwnershipResult(
        excluded_table_rects=page_context.excluded_table_rects,
        visual_region_rects=page_context.visual_region_rects,
        visual_region_text_rects=(
            page_context.visual_region_text_rects
        ),
        source_image_text_rects=page_context.source_image_text_rects,
        emitted_claim_ids=emitted_claim_ids,
        active_composite_table_claim_ids=(
            active_composite_table_claim_ids
        ),
        visual_application=visual_application,
        source_application=source_application,
    )

def process_page_semantics(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    formula_enabled: bool,
    emitted_claim_ids: set[str],
    active_composite_table_claim_ids: set[str],
    dependencies: PageStageAdapterDependencies,
) -> PageSemanticProcessingResult:
    """Run final semantic annotations after structural ownership."""
    ENABLE_FONT_HEADING_INFERENCE = dependencies.ENABLE_FONT_HEADING_INFERENCE
    ENABLE_HEADING_DETECTION = dependencies.ENABLE_HEADING_DETECTION
    ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT = dependencies.ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT
    FIGURE_CAPTION_LABEL_PATTERN = dependencies.FIGURE_CAPTION_LABEL_PATTERN
    FIGURE_CAPTION_MAX_TEXT_CHARACTERS = dependencies.FIGURE_CAPTION_MAX_TEXT_CHARACTERS
    FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER = dependencies.FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER
    FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS = dependencies.FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS
    FORMULA_IMAGE_DPI = dependencies.FORMULA_IMAGE_DPI
    FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO = dependencies.FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO
    FORMULA_PADDING_POINTS = dependencies.FORMULA_PADDING_POINTS
    FORMULA_OUTPUT_MODE = dependencies.FORMULA_OUTPUT_MODE
    HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT = dependencies.HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT
    HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS = dependencies.HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS
    HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO = dependencies.HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO
    HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS = dependencies.HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS
    HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS = dependencies.HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS
    HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS = dependencies.HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS
    HEADING_MAX_MARKDOWN_LEVEL = dependencies.HEADING_MAX_MARKDOWN_LEVEL
    HEADING_MAX_TEXT_CHARACTERS = dependencies.HEADING_MAX_TEXT_CHARACTERS
    HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS = dependencies.HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS
    HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS = dependencies.HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS
    HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO = dependencies.HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO
    HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO = dependencies.HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO
    HEADING_SAME_BODY_REQUIRES_BOLD = dependencies.HEADING_SAME_BODY_REQUIRES_BOLD
    annotate_heading_levels = dependencies.annotate_heading_levels
    classify_figure_caption_roles = dependencies.classify_figure_caption_roles
    detect_text_separator_markdown_elements = dependencies.detect_text_separator_markdown_elements
    document_run_state = dependencies.document_run_state
    line_is_inside_any_table = dependencies.line_is_inside_any_table
    line_matches_running_header_layout = dependencies.line_matches_running_header_layout
    process_page_formula_regions = dependencies.process_page_formula_regions
    rect_is_mostly_inside_any = dependencies.rect_is_mostly_inside_any

    page_context.figure_caption_records = (
        classify_figure_caption_roles(
            text_lines=page_context.text_lines,
            visual_records=page_context.visual_region_records,
            excluded_rects=[
                *page_context.header_footer_rects,
                *page_context.excluded_table_rects,
                *page_context.visual_region_text_rects,
                *page_context.source_image_text_rects,
            ],
            page_rect=page.rect,
            label_pattern=FIGURE_CAPTION_LABEL_PATTERN,
            max_text_characters=FIGURE_CAPTION_MAX_TEXT_CHARACTERS,
            max_vertical_gap_multiplier=(
                FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER
            ),
            max_vertical_gap_points=FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS,
            rect_is_mostly_inside_any=rect_is_mostly_inside_any,
        )
    )
    page_context.embedded_table_count = sum(
        record.get("rendered_as_image", False)
        for record in page_context.table_records
    )
    page_context.auto_rotated_table_count = sum(
        record.get("rendered_as_image", False)
        and record.get("csv_rotation_degrees", 0) != 0
        for record in page_context.table_records
    )

    if formula_enabled:
        (
            page_context.formula_records,
            page_context.formula_markdown_elements,
            page_context.formula_text_rects,
            page_context.formula_seconds,
        ) = process_page_formula_regions(
            page=page,
            page_dict=page_context.page_dict,
            excluded_table_rects=page_context.excluded_table_rects,
            excluded_visual_text_rects=[
                *page_context.visual_region_text_rects,
                *page_context.source_image_text_rects,
            ],
            excluded_header_footer_rects=(
                page_context.vector_exclusion_rects
            ),
            page_number=page_number,
            formula_dir=document_run_state.output_layout.formula_images,
            formula_image_dpi=FORMULA_IMAGE_DPI,
            formula_min_unnumbered_width_page_ratio=(
                FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO
            ),
            formula_padding_points=FORMULA_PADDING_POINTS,
            line_is_inside_any_table=line_is_inside_any_table,
            formula_output_mode=FORMULA_OUTPUT_MODE,
        )
    else:
        page_context.formula_records = []
        page_context.formula_markdown_elements = []
        page_context.formula_text_rects = []
        page_context.formula_seconds = 0.0
    document_run_state.metrics.formula_processing_seconds += (
        page_context.formula_seconds
    )
    page_context.record_timing(
        "formula_regions",
        page_context.formula_seconds,
    )

    active_claimed_drawing_element_ids = {
        element_id
        for claim in page_context.detector_claims
        if claim.get("claim_id") in (
            emitted_claim_ids
            | active_composite_table_claim_ids
        )
        for element_id in claim.get("claimed_element_ids", [])
        if element_id.startswith("d:")
    }
    active_claimed_drawing_element_ids.update(
        page_context.preclassified_rule_band_result[
            "preclassified_rule_drawing_ids"
        ]
    )
    occupied_non_text_rects = [
        fitz.Rect(
            element.get("render_bbox", element["bbox"])
        )
        for element in [
            *page_context.table_markdown_elements,
            *page_context.visual_region_markdown_elements,
            *page_context.source_image_markdown_elements,
            *page_context.formula_markdown_elements,
        ]
        if element.get("kind", "text") != "text"
    ]
    separator_started = time.perf_counter()
    detected_markdown, detected_records = (
        detect_text_separator_markdown_elements(
            page_rect=page.rect,
            drawings=page_context.drawings,
            text_lines=page_context.text_lines,
            source_images=page_context.source_images,
            page_inventory=page_context.page_inventory,
            claimed_drawing_element_ids=(
                active_claimed_drawing_element_ids
            ),
            occupied_non_text_rects=occupied_non_text_rects,
            excluded_text_rects=[
                *page_context.vector_exclusion_rects,
                *page_context.excluded_table_rects,
                *page_context.visual_region_text_rects,
                *page_context.source_image_text_rects,
                *page_context.formula_text_rects,
            ],
            logical_horizontal_rules=(
                page_context.logical_horizontal_rules
            ),
        )
    )
    page_context.text_separator_markdown_elements = [
        *page_context.preclassified_text_separator_markdown_elements,
        *detected_markdown,
    ]
    page_context.text_separator_records = [
        *page_context.preclassified_text_separator_records,
        *detected_records,
    ]
    page_context.text_separator_seconds = (
        time.perf_counter() - separator_started
    )
    document_run_state.metrics.text_separator_detection_seconds += (
        page_context.text_separator_seconds
    )
    document_run_state.metrics.text_separator_output_count += len(
        page_context.text_separator_markdown_elements
    )
    page_context.record_timing(
        "text_separators",
        page_context.text_separator_seconds,
    )

    heading_started = time.perf_counter()
    page_context.heading_records = annotate_heading_levels(
        text_lines=page_context.text_lines,
        typography_profile=document_run_state.typography_profile,
        excluded_rects=[
            *page_context.header_footer_rects,
            *[
                fitz.Rect(
                    candidate.get(
                        "render_bbox", candidate["bbox"]
                    )
                )
                for candidate in page_context.table_candidates
            ],
            *page_context.visual_region_text_rects,
            *page_context.source_image_text_rects,
            *page_context.formula_text_rects,
        ],
        page_rect=page.rect,
        outline_entries=(
            document_run_state.outline_heading_index.get(
                page_number,
                [],
            )
        ),
        page_title_outline_entry_index=(
            int(page_context.outline_page_name["entry_index"])
            if (
                page_context.outline_page_name is not None
                and page_context.outline_page_name.get("source")
                == "exact_outline_target"
            )
            else None
        ),
        enable_font_heading_inference=ENABLE_FONT_HEADING_INFERENCE,
        enable_heading_detection=ENABLE_HEADING_DETECTION,
        enable_pdf_outline_heading_assignment=(
            ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT
        ),
        candidate_max_block_line_count=(
            HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT
        ),
        font_size_cluster_tolerance_points=(
            HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS
        ),
        local_body_cluster_weight_ratio=(
            HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO
        ),
        local_body_min_cluster_characters=(
            HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS
        ),
        local_body_min_size_delta_points=(
            HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS
        ),
        local_body_min_visible_characters=(
            HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS
        ),
        max_markdown_level=HEADING_MAX_MARKDOWN_LEVEL,
        max_text_characters=HEADING_MAX_TEXT_CHARACTERS,
        outline_shared_target_tolerance_points=(
            HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS
        ),
        outline_target_max_vertical_distance_points=(
            HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS
        ),
        outline_target_max_x_distance_page_ratio=(
            HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO
        ),
        same_body_min_bold_character_ratio=(
            HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO
        ),
        same_body_requires_bold=HEADING_SAME_BODY_REQUIRES_BOLD,
        line_matches_running_header_layout=(
            line_matches_running_header_layout
        ),
        rect_is_mostly_inside_any=rect_is_mostly_inside_any,
    )
    for heading_record in page_context.heading_records:
        heading_record["page_number"] = page_number
    page_context.heading_seconds = (
        time.perf_counter() - heading_started
    )
    document_run_state.metrics.heading_detection_seconds += (
        page_context.heading_seconds
    )
    page_context.record_timing(
        "heading_detection",
        page_context.heading_seconds,
    )
    return PageSemanticProcessingResult(
        figure_caption_records=page_context.figure_caption_records,
        formula_records=page_context.formula_records,
        formula_markdown_elements=(
            page_context.formula_markdown_elements
        ),
        formula_text_rects=page_context.formula_text_rects,
        formula_seconds=page_context.formula_seconds,
        text_separator_markdown_elements=(
            page_context.text_separator_markdown_elements
        ),
        text_separator_records=page_context.text_separator_records,
        text_separator_seconds=page_context.text_separator_seconds,
        heading_records=page_context.heading_records,
        heading_seconds=page_context.heading_seconds,
    )

def process_page_reading_order(
    page_context: PageRunContext,
    *,
    page: fitz.Page,
    page_number: int,
    debug_output: bool,
    dependencies: PageStageAdapterDependencies,
) -> PageReadingOrderProcessingResult:
    """Sort page atoms and append the ordered Markdown body."""
    GAPTREE_DEBUG_DPI = dependencies.GAPTREE_DEBUG_DPI
    append_reflowed_text_to_markdown = dependencies.append_reflowed_text_to_markdown
    build_reflowed_page_text = dependencies.build_reflowed_page_text
    document_run_state = dependencies.document_run_state
    line_is_inside_any_table = dependencies.line_is_inside_any_table
    render_gaptree_debug_image = dependencies.render_gaptree_debug_image
    sort_elements_with_gap_tree = dependencies.sort_elements_with_gap_tree

    reflow_elements = [
        *page_context.text_lines,
        *page_context.text_separator_markdown_elements,
        *page_context.table_markdown_elements,
        *page_context.visual_region_markdown_elements,
        *page_context.source_image_markdown_elements,
        *page_context.formula_markdown_elements,
    ]
    (
        page_context.reading_order_metadata,
        page_context.grouped_markdown,
    ) = build_reflowed_page_text(
        text_lines=reflow_elements,
        page_rect=page.rect,
        excluded_header_footer_rects=(
            page_context.vector_exclusion_rects
        ),
        excluded_table_rects=page_context.excluded_table_rects,
        excluded_visual_region_rects=[
            *page_context.visual_region_text_rects,
            *page_context.source_image_text_rects,
        ],
        excluded_formula_rects=page_context.formula_text_rects,
        line_is_inside_any_table=line_is_inside_any_table,
        sort_elements_with_gap_tree=sort_elements_with_gap_tree,
    )
    normal_direction_metadata = next(
        (
            direction["gaptree"]
            for direction in page_context.reading_order_metadata[
                "direction_groups"
            ]
            if direction["angle"] == 0
        ),
        {
            "row_count": 0,
            "cut_count": 0,
            "node_count": 0,
            "cuts": [],
            "nodes": [],
        },
    )
    page_context.reading_order_debug_path = None
    if debug_output:
        page_context.reading_order_debug_path = (
            render_gaptree_debug_image(
                page=page,
                page_number=page_number,
                gaptree_metadata=normal_direction_metadata,
                output_dir=(
                    document_run_state.output_layout.reading_order_debug
                ),
                debug_dpi=GAPTREE_DEBUG_DPI,
            )
        )
    if page_context.embedded_table_count:
        page_context.markdown_lines.extend(
            [
                "",
                "> 大表格区域的原始文本层已省略，"
                "以避免表格单元格内容与正文、脚注混排。",
                "> 表格内容请以转正后的 PNG 和 CSV 为准。",
            ]
        )
    if page_context.visual_region_rects:
        page_context.markdown_lines.extend(
            [
                "",
                "> 粗粒度视觉区域采用双通道保留："
                "PNG 保留版面，图片下方保留 PDF 原生文字。",
            ]
        )
    append_reflowed_text_to_markdown(
        page_markdown=page_context.markdown_lines,
        grouped_markdown=page_context.grouped_markdown,
    )
    return PageReadingOrderProcessingResult(
        metadata=page_context.reading_order_metadata,
        grouped_markdown=page_context.grouped_markdown,
        normal_direction_metadata=normal_direction_metadata,
        debug_path=page_context.reading_order_debug_path,
        markdown_lines=page_context.markdown_lines,
    )

@dataclass(frozen=True)
class PagePipelineDependencies:
    document_run_state: Any
    table_pipeline_dependencies: Any
    initialize_page_context: Callable[..., Any]
    process_page_header_footer: Callable[..., Any]
    prepare_page_geometry: Callable[..., Any]
    prepare_page_rule_bands: Callable[..., Any]
    detect_page_tables: Callable[..., Any]
    build_page_table_candidates: Callable[..., Any]
    finalize_page_table_candidates: Callable[..., Any]
    prepare_page_visual_protection: Callable[..., Any]
    detect_page_visual_regions: Callable[..., Any]
    propose_page_output_candidates: Callable[..., Any]
    detect_page_source_images: Callable[..., Any]
    resolve_page_claims: Callable[..., Any]
    apply_page_boundary_ownership: Callable[..., Any]
    process_page_semantics: Callable[..., Any]
    process_page_reading_order: Callable[..., Any]
    finalize_page_context: Callable[..., Any]
    format_markdown_pdf_page_title: Callable[..., str]
    keep_header_text: bool
    formula_detection_enabled: bool
    debug_output_enabled: bool
    force_pure_text_table_to_markdown: bool

def process_page(
    page_index: int,
    *,
    dependencies: PagePipelineDependencies,
) -> PageRunContext:
    """Process one page through the complete, contract-checked stage chain."""

    initialization = dependencies.initialize_page_context(page_index)
    page_number = initialization.page_number
    page = initialization.page
    page_started = initialization.page_started
    page_context = initialization.page_context
    page_context.record_stage(PageStage.INITIALIZATION)

    header_footer_result = dependencies.process_page_header_footer(
        page_context,
        page=page,
        keep_header_text=dependencies.keep_header_text,
    )
    header_text = header_footer_result.header_text
    page_context.record_stage(PageStage.HEADER_FOOTER)

    dependencies.prepare_page_geometry(
        page_context,
        page=page,
        page_number=page_number,
    )
    page_context.record_stage(PageStage.GEOMETRY)

    rule_band_preparation = dependencies.prepare_page_rule_bands(
        page_context,
        page_number=page_number,
    )
    preclassified_rule_bands = rule_band_preparation.bands
    page_context.record_stage(PageStage.RULE_BANDS)

    dependencies.detect_page_tables(
        page_context,
        page=page,
        dependencies=dependencies.table_pipeline_dependencies,
    )
    page_context.record_stage(PageStage.TABLE_DETECTION)

    page_context.outline_page_name = (
        dependencies.document_run_state.outline_page_name_index.get(
            page_number
        )
    )
    page_context.markdown_page_title = (
        dependencies.format_markdown_pdf_page_title(
            page_number=page_number,
            outline_page_name=page_context.outline_page_name,
        )
    )
    page_context.markdown_lines = [page_context.markdown_page_title]
    if header_text:
        page_context.markdown_lines.extend(["", f"页眉:{header_text}"])

    dependencies.build_page_table_candidates(
        page_context,
        page=page,
        page_number=page_number,
        force_pure_text_table_to_markdown=(
            dependencies.force_pure_text_table_to_markdown
        ),
        dependencies=dependencies.table_pipeline_dependencies,
    )
    page_context.record_stage(PageStage.TABLE_CANDIDATES)

    dependencies.finalize_page_table_candidates(
        page_context,
        page=page,
        page_number=page_number,
        rule_bands=preclassified_rule_bands,
        page_inventory_seconds=page_context.timings["inventory_build"],
        dependencies=dependencies.table_pipeline_dependencies,
    )
    page_context.record_stage(PageStage.TABLE_FINALIZATION)

    dependencies.prepare_page_visual_protection(
        page_context,
        page=page,
    )
    page_context.record_stage(PageStage.VISUAL_PROTECTION)

    dependencies.detect_page_visual_regions(
        page_context,
        page=page,
        page_number=page_number,
        rule_bands=preclassified_rule_bands,
    )
    page_context.record_stage(PageStage.VISUAL_DETECTION)

    dependencies.propose_page_output_candidates(
        page_context,
        page=page,
        page_number=page_number,
    )
    page_context.record_stage(PageStage.OUTPUT_PROPOSAL)

    dependencies.detect_page_source_images(
        page_context,
        page=page,
        page_number=page_number,
    )
    page_context.record_stage(PageStage.SOURCE_IMAGES)

    dependencies.resolve_page_claims(
        page_context,
        page=page,
        page_number=page_number,
    )
    page_context.record_stage(PageStage.CLAIM_RESOLUTION)

    boundary_ownership = dependencies.apply_page_boundary_ownership(
        page_context,
        page=page,
        page_number=page_number,
    )
    page_context.record_stage(PageStage.BOUNDARY_OWNERSHIP)

    dependencies.process_page_semantics(
        page_context,
        page=page,
        page_number=page_number,
        formula_enabled=dependencies.formula_detection_enabled,
        emitted_claim_ids=boundary_ownership.emitted_claim_ids,
        active_composite_table_claim_ids=(
            boundary_ownership.active_composite_table_claim_ids
        ),
    )
    page_context.record_stage(PageStage.SEMANTICS)

    dependencies.process_page_reading_order(
        page_context,
        page=page,
        page_number=page_number,
        debug_output=dependencies.debug_output_enabled,
    )
    page_context.record_stage(PageStage.READING_ORDER)

    dependencies.finalize_page_context(
        page_context,
        page_started=page_started,
    )
    page_context.record_stage(PageStage.FINALIZATION)
    return page_context

@dataclass(frozen=True)
class DocumentPipelineDependencies:
    document_run_state: Any
    parse_result: Any
    page_pipeline_dependencies: Any
    process_page: Callable[..., Any]
    commit_page_context: Callable[..., None]
    validate_page_contracts: bool

def process_document_pages(
    *,
    dependencies: DocumentPipelineDependencies,
) -> Any:
    """Process configured pages, commit their results, and close the document."""

    for page_index in dependencies.document_run_state.page_indices:
        page_context = dependencies.process_page(
            page_index,
            dependencies=dependencies.page_pipeline_dependencies,
        )
        dependencies.commit_page_context(
            dependencies.parse_result,
            page_context,
            validate=dependencies.validate_page_contracts,
        )

    dependencies.document_run_state.document.close()
    return dependencies.parse_result

@dataclass(frozen=True)
class DocumentFinalizeDependencies:
    CLAIM_DETECTOR_PRIORITY: Any
    CLUSTER_TABLE_CLOSE_GAP_POINTS: Any
    CLUSTER_TABLE_DRAWING_BUDGET: Any
    COMPOSITE_IMAGE_DPI: Any
    FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO: Any
    FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO: Any
    FORMULA_IMAGE_DPI: Any
    HEADER_FOOTER_CANDIDATE_COUNT: Any
    HEADER_FOOTER_DETECT_FOOTER: Any
    HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE: Any
    HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS: Any
    HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE: Any
    HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE: Any
    HEADER_FOOTER_SAMPLE_MAX_PAGE: Any
    HEADER_FOOTER_SAMPLE_START_PAGE: Any
    HEADER_FOOTER_X_TOLERANCE_POINTS: Any
    HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS: Any
    HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO: Any
    HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS: Any
    HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS: Any
    HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS: Any
    HEADING_MAX_MARKDOWN_LEVEL: Any
    LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES: Any
    MAX_HORIZONTAL_LINE_SLOPE_RATIO: Any
    MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS: Any
    MIN_DIRECTION_TEXT_CHARS: Any
    MIN_IMAGE_AREA_POINTS_SQUARED: Any
    MIN_IMAGE_LINE_ASPECT_RATIO: Any
    MIN_IMAGE_THIN_SIDE_POINTS: Any
    OBJECT_PEER_OVERLAP_RATIO: Any
    OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS: Any
    OVERLAY_INTERSECTION_RATIO: Any
    PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO: Any
    PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS: Any
    ROTATION_DOMINANCE_THRESHOLD: Any
    SORT_BBOX_TOP_HEIGHT_POINTS: Any
    TABLE_IMAGE_DPI: Any
    TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO: Any
    VECTOR_ENDPOINT_GRID_CELL_POINTS: Any
    VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS: Any
    VISUAL_MASK_CLOSE_GAP_PAGE_RATIO: Any
    VISUAL_MASK_PIXELS_PER_POINT: Any
    VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER: Any
    VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO: Any
    VISUAL_REGION_DPI: Any
    VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO: Any
    VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO: Any
    VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO: Any
    VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO: Any
    artifact_emitter: Any
    claim_resolution_actions_output: Any
    claim_resolution_output: Any
    composite_output_groups_output: Any
    detector_claims_output: Any
    document_run_state: Any
    document_toc_entries: Any
    figure_caption_records_output: Any
    formula_regions_output: Any
    header_footer_pages_output: Any
    heading_records_output: Any
    logical_horizontal_rule_audit_output: Any
    page_indices: Any
    pages_output: Any
    parse_result: Any
    processed_pages: Any
    run_debug_output: Any
    run_enable_font_heading_inference: Any
    run_enable_formula_detection: Any
    run_formula_output_mode: Any
    run_enable_heading_detection: Any
    run_enable_pdf_outline_heading_assignment: Any
    run_force_pure_text_table_to_markdown: Any
    run_keep_header_text: Any
    run_metrics: Any
    run_output_dir: Any
    run_source_pdf: Any
    serialize_header_footer_templates: Any
    source_images_output: Any
    source_page_count: Any
    started: Any
    tables_output: Any
    visual_regions_output: Any
    write_claim_conflict_report: Any
    write_figure_caption_report: Any
    write_formula_region_report: Any
    write_gaptree_report: Any
    write_header_footer_report: Any
    write_heading_detection_report: Any
    write_logical_horizontal_rule_report: Any
    write_source_image_report: Any
    write_visual_region_report: Any


def _collect_document_finalize_statistics(
    dependencies: DocumentFinalizeDependencies,
) -> dict[str, Any]:
    """Collect document-wide counters reused by the summary and reports."""

    embedded_tables = [
        item
        for item in dependencies.tables_output
        if item.get("rendered_as_image")
    ]
    markdown_tables = [
        item
        for item in dependencies.tables_output
        if item.get("force_pure_text_markdown")
    ]
    auto_rotated_tables = [
        item
        for item in embedded_tables
        if item["csv_rotation_degrees"] != 0
    ]
    return {
        "end_to_end_seconds": time.perf_counter() - dependencies.started,
        "embedded_tables": embedded_tables,
        "markdown_tables": markdown_tables,
        "auto_rotated_tables": auto_rotated_tables,
        "page_counts": Counter(
            item["page_number"]
            for item in embedded_tables
        ),
        "source_image_mode_counts": Counter(
            item["mode"] for item in dependencies.source_images_output
        ),
        "source_image_reason_counts": Counter(
            reason
            for item in dependencies.source_images_output
            for reason in item["reason"]
        ),
        "visual_region_page_counts": Counter(
            item["page_number"]
            for item in dependencies.visual_regions_output
        ),
        "formula_region_page_counts": Counter(
            item["page_number"]
            for item in dependencies.formula_regions_output
        ),
        "header_band_pages": [
            record["page_number"]
            for record in dependencies.header_footer_pages_output
            if record["header"]["band"] is not None
        ],
        "footer_band_pages": [
            record["page_number"]
            for record in dependencies.header_footer_pages_output
            if record["footer"]["band"] is not None
        ],
        "serialized_header_footer_templates": (
            dependencies.serialize_header_footer_templates(
                dependencies.document_run_state.header_footer_templates
            )
        ),
    }


def finalize_document_run(*, dependencies: DocumentFinalizeDependencies) -> Any:
    CLAIM_DETECTOR_PRIORITY = dependencies.CLAIM_DETECTOR_PRIORITY
    CLUSTER_TABLE_CLOSE_GAP_POINTS = dependencies.CLUSTER_TABLE_CLOSE_GAP_POINTS
    CLUSTER_TABLE_DRAWING_BUDGET = dependencies.CLUSTER_TABLE_DRAWING_BUDGET
    COMPOSITE_IMAGE_DPI = dependencies.COMPOSITE_IMAGE_DPI
    FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO = dependencies.FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO
    FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO = dependencies.FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO
    FORMULA_IMAGE_DPI = dependencies.FORMULA_IMAGE_DPI
    HEADER_FOOTER_CANDIDATE_COUNT = dependencies.HEADER_FOOTER_CANDIDATE_COUNT
    HEADER_FOOTER_DETECT_FOOTER = dependencies.HEADER_FOOTER_DETECT_FOOTER
    HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE = dependencies.HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE
    HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS = dependencies.HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS
    HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE = dependencies.HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE
    HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE = dependencies.HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE
    HEADER_FOOTER_SAMPLE_MAX_PAGE = dependencies.HEADER_FOOTER_SAMPLE_MAX_PAGE
    HEADER_FOOTER_SAMPLE_START_PAGE = dependencies.HEADER_FOOTER_SAMPLE_START_PAGE
    HEADER_FOOTER_X_TOLERANCE_POINTS = dependencies.HEADER_FOOTER_X_TOLERANCE_POINTS
    HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS = dependencies.HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS
    HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO = dependencies.HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO
    HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS = dependencies.HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS
    HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS = dependencies.HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS
    HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS = dependencies.HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS
    HEADING_MAX_MARKDOWN_LEVEL = dependencies.HEADING_MAX_MARKDOWN_LEVEL
    LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES = dependencies.LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES
    MAX_HORIZONTAL_LINE_SLOPE_RATIO = dependencies.MAX_HORIZONTAL_LINE_SLOPE_RATIO
    MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS = dependencies.MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS
    MIN_DIRECTION_TEXT_CHARS = dependencies.MIN_DIRECTION_TEXT_CHARS
    MIN_IMAGE_AREA_POINTS_SQUARED = dependencies.MIN_IMAGE_AREA_POINTS_SQUARED
    MIN_IMAGE_LINE_ASPECT_RATIO = dependencies.MIN_IMAGE_LINE_ASPECT_RATIO
    MIN_IMAGE_THIN_SIDE_POINTS = dependencies.MIN_IMAGE_THIN_SIDE_POINTS
    OBJECT_PEER_OVERLAP_RATIO = dependencies.OBJECT_PEER_OVERLAP_RATIO
    OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS = dependencies.OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
    OVERLAY_INTERSECTION_RATIO = dependencies.OVERLAY_INTERSECTION_RATIO
    PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO = dependencies.PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
    PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS = dependencies.PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
    ROTATION_DOMINANCE_THRESHOLD = dependencies.ROTATION_DOMINANCE_THRESHOLD
    SORT_BBOX_TOP_HEIGHT_POINTS = dependencies.SORT_BBOX_TOP_HEIGHT_POINTS
    TABLE_IMAGE_DPI = dependencies.TABLE_IMAGE_DPI
    TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO = dependencies.TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO
    VECTOR_ENDPOINT_GRID_CELL_POINTS = dependencies.VECTOR_ENDPOINT_GRID_CELL_POINTS
    VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS = dependencies.VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS
    VISUAL_MASK_CLOSE_GAP_PAGE_RATIO = dependencies.VISUAL_MASK_CLOSE_GAP_PAGE_RATIO
    VISUAL_MASK_PIXELS_PER_POINT = dependencies.VISUAL_MASK_PIXELS_PER_POINT
    VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER = dependencies.VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER
    VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO = dependencies.VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO
    VISUAL_REGION_DPI = dependencies.VISUAL_REGION_DPI
    VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO = dependencies.VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO
    VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO = dependencies.VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO
    VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO = dependencies.VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO
    VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO = dependencies.VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO
    artifact_emitter = dependencies.artifact_emitter
    claim_resolution_actions_output = dependencies.claim_resolution_actions_output
    claim_resolution_output = dependencies.claim_resolution_output
    composite_output_groups_output = dependencies.composite_output_groups_output
    detector_claims_output = dependencies.detector_claims_output
    document_run_state = dependencies.document_run_state
    document_toc_entries = dependencies.document_toc_entries
    figure_caption_records_output = dependencies.figure_caption_records_output
    formula_regions_output = dependencies.formula_regions_output
    header_footer_pages_output = dependencies.header_footer_pages_output
    heading_records_output = dependencies.heading_records_output
    logical_horizontal_rule_audit_output = dependencies.logical_horizontal_rule_audit_output
    page_indices = dependencies.page_indices
    pages_output = dependencies.pages_output
    parse_result = dependencies.parse_result
    processed_pages = dependencies.processed_pages
    run_debug_output = dependencies.run_debug_output
    run_enable_font_heading_inference = dependencies.run_enable_font_heading_inference
    run_enable_formula_detection = dependencies.run_enable_formula_detection
    run_formula_output_mode = dependencies.run_formula_output_mode
    run_enable_heading_detection = dependencies.run_enable_heading_detection
    run_enable_pdf_outline_heading_assignment = dependencies.run_enable_pdf_outline_heading_assignment
    run_force_pure_text_table_to_markdown = dependencies.run_force_pure_text_table_to_markdown
    run_keep_header_text = dependencies.run_keep_header_text
    run_metrics = dependencies.run_metrics
    run_output_dir = dependencies.run_output_dir
    run_source_pdf = dependencies.run_source_pdf
    serialize_header_footer_templates = dependencies.serialize_header_footer_templates
    source_images_output = dependencies.source_images_output
    source_page_count = dependencies.source_page_count
    tables_output = dependencies.tables_output
    visual_regions_output = dependencies.visual_regions_output
    write_claim_conflict_report = dependencies.write_claim_conflict_report
    write_figure_caption_report = dependencies.write_figure_caption_report
    write_formula_region_report = dependencies.write_formula_region_report
    write_gaptree_report = dependencies.write_gaptree_report
    write_header_footer_report = dependencies.write_header_footer_report
    write_heading_detection_report = dependencies.write_heading_detection_report
    write_logical_horizontal_rule_report = dependencies.write_logical_horizontal_rule_report
    write_source_image_report = dependencies.write_source_image_report
    write_visual_region_report = dependencies.write_visual_region_report

    finalize_statistics = _collect_document_finalize_statistics(
        dependencies
    )
    end_to_end_seconds = finalize_statistics["end_to_end_seconds"]
    embedded_tables = finalize_statistics["embedded_tables"]
    markdown_tables = finalize_statistics["markdown_tables"]
    auto_rotated_tables = finalize_statistics["auto_rotated_tables"]
    page_counts = finalize_statistics["page_counts"]
    source_image_mode_counts = finalize_statistics[
        "source_image_mode_counts"
    ]
    source_image_reason_counts = finalize_statistics[
        "source_image_reason_counts"
    ]
    visual_region_page_counts = finalize_statistics[
        "visual_region_page_counts"
    ]
    formula_region_page_counts = finalize_statistics[
        "formula_region_page_counts"
    ]
    header_band_pages = finalize_statistics["header_band_pages"]
    footer_band_pages = finalize_statistics["footer_band_pages"]
    serialized_header_footer_templates = finalize_statistics[
        "serialized_header_footer_templates"
    ]

    summary = {
        "tool": "PyMuPDF + Pillow",
        "pymupdf_version": fitz.VersionBind,
        "source_pdf": str(run_source_pdf),
        "source_page_count": source_page_count,
        "processed_pages": processed_pages,
        "processed_page_numbers": [
            page_index + 1 for page_index in page_indices
        ],
        "markdown_page_naming_policy": {
            "source": "Document.get_toc(simple=False)",
            "toc_entry_count": len(document_toc_entries),
            "exact_page_policy": (
                "use the first outline entry targeting the current page"
            ),
            "continuation_page_policy": (
                "inherit the last active outline entry from the nearest "
                "preceding page"
            ),
            "multiple_entries_same_page_policy": (
                "display the first entry as the page name; use the last "
                "entry as the inherited section for following pages"
            ),
            "markdown_format": "## {outline title}",
            "fallback": "## Page N when no outline is available",
            "full_outline_path_storage": (
                "pages_with_text_reflow.json:outline_page_name.outline_path"
            ),
            "named_processed_page_count": sum(
                (page_index + 1)
                in document_run_state.outline_page_name_index
                for page_index in page_indices
            ),
            "fallback_processed_page_count": sum(
                (page_index + 1)
                not in document_run_state.outline_page_name_index
                for page_index in page_indices
            ),
        },
        "heading_detection_policy": {
            "enabled": run_enable_heading_detection,
            "font_heading_inference_enabled": (
                run_enable_font_heading_inference
            ),
            "font_heading_inference_environment_variable": (
                "PYMUPDF_ENABLE_FONT_HEADING_INFERENCE"
            ),
            "body_font_size": document_run_state.typography_profile.get(
                "body_font_size"
            ),
            "font_size_cluster_tolerance_points": (
                HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS
            ),
            "heading_size_clusters": (
                document_run_state.typography_profile.get(
                    "heading_size_clusters", []
                )
            ),
            "maximum_markdown_level": (
                HEADING_MAX_MARKDOWN_LEVEL
            ),
            "small_font_policy": (
                "font sizes below the document body cluster never "
                "become headings"
            ),
            "same_body_size_policy": (
                "requires majority-bold text plus a valid chapter "
                "number; page-number rule labels are excluded"
            ),
            "page_local_body_policy": {
                "minimum_visible_characters": (
                    HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS
                ),
                "frequent_cluster_ratio": (
                    HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO
                ),
                "minimum_cluster_characters": (
                    HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS
                ),
                "reference": (
                    "max(document body font, highest frequent "
                    "page-local body font cluster)"
                ),
                "minimum_heading_size_delta_points": (
                    HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS
                ),
            },
            "pdf_outline_policy": {
                "enabled": run_enable_pdf_outline_heading_assignment,
                "available_entry_count": sum(
                    len(entries)
                    for entries in (
                        document_run_state.outline_heading_index.values()
                    )
                ),
                "assignment": (
                    "strict normalized text match on the current target "
                    "page; duplicate matches are resolved by outline "
                    "target geometry"
                ),
                "text_policy": (
                    "exact normalized text or exact text after removing "
                    "a structural chapter-number prefix creates a heading; "
                    "no fuzzy or keyword matching"
                ),
                "page_title_deduplication": (
                    "the first exact-page outline heading is emitted by "
                    "the page H2 and its native text line is suppressed "
                    "once from the body"
                ),
                "geometry_fallback": (
                    "available only when font heading inference is enabled"
                ),
                "priority": (
                    "outline strict-text match before optional font heuristic"
                ),
            },
            "excluded_regions": [
                "header/footer bands",
                "all valid table candidates",
                "owned visual-region text",
                "owned source-image text",
                "rendered formula text",
            ],
            "heading_count": len(heading_records_output),
            "profile": document_run_state.typography_profile,
            "audit_output": "heading_detection_report.md",
        },
        "figure_caption_policy": {
            "enabled": True,
            "stage": (
                "after final visual/source-image ownership and before "
                "heading classification"
            ),
            "candidate_scope": (
                "unclaimed horizontal single-line native text"
            ),
            "required_evidence": [
                "generic figure-number label structure",
                "vertical adjacency to a surviving visual region",
                "no unclaimed text line between caption and region",
                "one caption line assigned to at most one region",
            ],
            "x_overlap_required": False,
            "markdown_output": "standalone bold paragraph",
            "figure_caption_count": len(
                figure_caption_records_output
            ),
            "audit_output": "figure_caption_detection_report.md",
        },
        "threshold_rule": "row_count * column_count > 6",
        "pure_text_table_mode": {
            "force_pure_text_table_to_markdown": (
                run_force_pure_text_table_to_markdown
            ),
            "complex_visual_tables_always_rendered": True,
            "complexity_signals": [
                "image block overlaps table",
                "non-axis-rule vector overlaps table",
                "math-font text line overlaps table",
            ],
        },
        "text_separator_policy": {
            "enabled": True,
            "source": "unclaimed same-span adjacent horizontal rules",
            "content_condition": "native PDF text only",
            "output": "Markdown thematic break `---`",
            "render_as_image": False,
            "skip_when": [
                "image block overlaps the band",
                "table candidate overlaps the band",
                "visual region overlaps the band",
                "formula region overlaps the band",
                "substantive non-rule vector overlaps the band",
            ],
        },
        "logical_horizontal_rule_premerge_policy": {
            "enabled": True,
            "stage": "before table and visual detection",
            "input": "one-pass axis geometry index from raw drawing items",
            "axis_views": [
                "all horizontal and vertical logical rules for table skeletons",
                "long horizontal rules for rule-band and visual routing",
            ],
            "same_y_grouping": (
                "standardized Y coordinate bucket; no table "
                "connection tolerance"
            ),
            "coordinate_decimal_places": (
                LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES
            ),
            "x_connection": (
                "normalized_endpoint_touch_or_overlap_only"
            ),
            "table_connection_tolerance_points": (
                PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
            ),
            "minimum_width_page_ratio_after_merge": (
                FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO
            ),
            "preserve_source_drawing_ids": True,
            "visual_stage_filter": (
                "discard a logical rule when any source drawing ID "
                "was already assigned to a table skeleton or "
                "preclassified rule band"
            ),
            "audit_output": {
                "json": "logical_horizontal_rules.json",
                "markdown": "logical_horizontal_rules_report.md",
                "record_scope": (
                    "logical rules built from at least two source drawings"
                ),
                "record_count": len(
                    logical_horizontal_rule_audit_output
                ),
            },
        },
        "preclassified_rule_band_policy": {
            "enabled": True,
            "stage": "before page.find_tables()",
            "minimum_horizontal_rule_x_overlap_ratio": (
                PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
            ),
            "table_structure_rejection": {
                "repeated_horizontal_pair": (
                    "same top/bottom Y positions repeated in another X span"
                ),
                "vertical_connection": (
                    "center-line X proximity plus Y coverage; zero-area lines supported"
                ),
                "vertical_connection_tolerance_points": (
                    PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS
                ),
                "minimum_vertical_coverage_ratio": (
                    FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO
                ),
                "affects_render_bbox": False,
            },
            "pure_text_output": "Markdown thematic break `---`",
            "image_or_complex_vector_output": (
                "single visual-region PNG with native text channel"
            ),
            "adjacent_empty_band_table_attachment": {
                "enabled": True,
                "empty_definition": (
                    "no source image, no text outside confirmed tables, "
                    "and content drawings are rectangular fills only"
                ),
                "edge_tolerance_points": (
                    OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
                ),
                "minimum_x_overlap_ratio": (
                    PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO
                ),
                "changes_detection_bbox": False,
                "changes_render_bbox": True,
                "changes_sort_bbox": False,
            },
            "table_detector_input": (
                "remaining drawings after preclassified bands are routed"
            ),
            "preserve_as_table_when": [
                "a vertical separator crosses most of the band",
                "a regular multi-row or multi-column grid is present",
            ],
        },
        "table_detection_policy": {
            "native_detector": "page.find_tables()",
            "cluster_table_gate": {
                "drawing_budget_per_cluster": CLUSTER_TABLE_DRAWING_BUDGET,
                "precluster_close_gap_points": CLUSTER_TABLE_CLOSE_GAP_POINTS,
                "small_cluster_policy": "test all drawings",
                "large_cluster_policy": "select axis-rule skeleton source items from the complete cluster; use spatial 2000-object selection only when no axis structure is available",
                "purpose": "preserve outer table extremes while excluding complex internal vector paths",
            },
            "geometry_fallback_stage": (
                "after native detection and before vector separator "
                "filtering"
            ),
            "geometry_fallback_conditions": [
                "at least 4 aligned long horizontal rules",
                "at least one internal vertical separator covering 60%",
                "vertical breaks no larger than 18pt are mergeable",
                "at least 3 populated rows and 2 populated columns",
            ],
            "source_counts": dict(
                Counter(
                    item.get("detection_source", "pymupdf_native")
                    for item in tables_output
                )
            ),
            "precluster_object_overlap": {
                "enabled": True,
                "stage": "after table detection and before visual morphology",
                "peer_overlap_ratio": OBJECT_PEER_OVERLAP_RATIO,
                "drawing_containment_tolerance_points": (
                    OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS
                ),
                "protected_objects": (
                    "emittable tables and sufficiently large source-image "
                    "groups without external vector overlap or peer-object "
                    "overlap"
                ),
                "postcluster_fallback": (
                    "resolve_table_visual_candidate_ownership()"
                ),
            },
        },
        "table_image_dpi": TABLE_IMAGE_DPI,
        "output_policy": (
            "existing output directory is renamed to a timestamped "
            "backup before each run"
        ),
        "header_footer_policy": {
            "detect_footer": HEADER_FOOTER_DETECT_FOOTER,
            "keep_header_text_in_markdown": (
                run_keep_header_text
            ),
            "header_text_markdown_format": (
                "页眉:[页眉文本]"
                if run_keep_header_text
                else None
            ),
            "sample_pages": (
                f"{HEADER_FOOTER_SAMPLE_START_PAGE}-"
                f"{HEADER_FOOTER_SAMPLE_MAX_PAGE}"
            ),
            "candidate_count_per_edge": (
                HEADER_FOOTER_CANDIDATE_COUNT
            ),
            "candidate_edge_page_ratio_by_edge": (
                HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE
            ),
            "footer_bbox_must_be_fully_inside_candidate_band": True,
            "minimum_matched_objects_by_edge": (
                HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE
            ),
            "minimum_support_ratio_by_edge": (
                HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE
            ),
            "confirmed_object_action": (
                "remove matched drawing objects before every downstream "
                "vector-dependent stage"
            ),
            "header_footer_text_policy": (
                "preserve native header text, chapter labels, footer text, "
                "and page numbers unless header text retention is explicitly "
                "enabled for duplicate header output"
            ),
            "text_content_used_for_matching": False,
            "footer_composite_drawing_policy": {
                "enabled": True,
                "candidate_band_only": True,
                "maximum_horizontal_gap_points": (
                    HEADER_FOOTER_X_TOLERANCE_POINTS
                ),
                "baseline_tolerance_points": (
                    HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS
                ),
                "preserve_all_source_drawing_ids": True,
            },
            "footer_horizontal_position_policy": (
                "distance to nearest outer page edge; left/right mirror "
                "positions share one template"
            ),
            "hard_conditions": [
                "same object type",
                "height difference <= 1pt",
                "header absolute x0 difference <= 2pt; footer outer-edge distance and width difference <= 2pt",
                "edge offset difference <= 3pt",
                "identical normalized image/vector content digest",
            ],
            "templates": serialized_header_footer_templates,
            "header_band_page_count": len(header_band_pages),
            "header_band_pages": header_band_pages,
            "footer_band_page_count": len(footer_band_pages),
            "footer_band_pages": footer_band_pages,
        },
        "rotation_detection": {
            "source": "page.get_text('dict') line['dir']",
            "minimum_dominant_ratio": (
                ROTATION_DOMINANCE_THRESHOLD
            ),
            "minimum_text_characters": (
                MIN_DIRECTION_TEXT_CHARS
            ),
        },
        "csv_policy": {
            "default_csv_data": (
                "reflowed_data, rotated with the same angle "
                "as the corresponding PNG"
            ),
            "raw_data_storage": (
                "original table.extract() output remains "
                "in tables_with_rotation.json as raw_data"
            ),
        },
        "source_image_policy": {
            "decision_scope": "generic PDF object geometry; no page-specific rules",
            "direct_copy": (
                "single sufficiently large image with no overlapping "
                "text/vector/annotation/widget, mask, rotation, or "
                "overlapping image"
            ),
            "rendered_composite": (
                "render union bbox when generic overlay or complex "
                "image conditions are detected"
            ),
            "covered_by_table": (
                "do not duplicate an image whose center is inside an "
                "embedded large-table bbox"
            ),
            "covered_by_visual_region": (
                "do not duplicate an image whose center is inside a "
                "rendered coarse visual region"
            ),
            "filter_rule": (
                "(minor_side < 12pt and aspect_ratio > 5) or "
                "bbox_area < 160pt^2"
            ),
            "thin_side_points": MIN_IMAGE_THIN_SIDE_POINTS,
            "line_aspect_ratio": MIN_IMAGE_LINE_ASPECT_RATIO,
            "minimum_area_points_squared": (
                MIN_IMAGE_AREA_POINTS_SQUARED
            ),
            "overlay_intersection_ratio": OVERLAY_INTERSECTION_RATIO,
            "composite_image_dpi": COMPOSITE_IMAGE_DPI,
            "bbox_policy": {
                "render_bbox": "full copied or composite image region",
                "sort_bbox": "top anchor strip only",
                "sort_bbox_height_points": SORT_BBOX_TOP_HEIGHT_POINTS,
            },
            "mode_counts": dict(source_image_mode_counts),
            "reason_counts": dict(source_image_reason_counts),
        },
        "visual_region_policy": {
            "method": (
                "exclude tables/background, remove high-confidence filled "
                "horizontal bars and unmarked single horizontal strokes "
                "before morphology, form initial "
                "connected components, apply the shared image size filter, "
                "spatially merge surviving vector clusters, attach text/"
                "images once, then render without post-attachment distance "
                "merging"
            ),
            "region_count": len(visual_regions_output),
            "pages": sorted(visual_region_page_counts),
            "regions_per_page": dict(
                sorted(visual_region_page_counts.items())
            ),
            "mask_pixels_per_point": VISUAL_MASK_PIXELS_PER_POINT,
            "close_gap_page_ratio": (
                VISUAL_MASK_CLOSE_GAP_PAGE_RATIO
            ),
            "vertical_merge_page_ratio": (
                VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO
            ),
            "horizontal_merge_page_ratio": (
                VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO
            ),
            "pre_morphology_separator_filter": {
                "filled_horizontal_bar_rule": (
                    "single filled rectangle with height < 12pt and "
                    "width/height > 5 is removed unless an endpoint "
                    "has a compact vector marker; collinear same-Y "
                    "end-to-end filled bars never protect each other"
                ),
                "single_stroke_rule": (
                    "single near-horizontal stroke with effective minor "
                    "< 12pt and aspect ratio > 5 is removed unless an "
                    "endpoint has a compact vector marker"
                ),
                "endpoint_search_radius_points": (
                    VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS
                ),
                "endpoint_grid_cell_points": (
                    VECTOR_ENDPOINT_GRID_CELL_POINTS
                ),
                "maximum_marker_major_points": (
                    MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS
                ),
                "maximum_horizontal_slope_ratio": (
                    MAX_HORIZONTAL_LINE_SLOPE_RATIO
                ),
            },
            "pre_merge_component_filter_enabled": True,
            "pre_merge_component_filter_rule": (
                "(minor_side < 12pt and aspect_ratio > 5) or "
                "bbox_area < 160pt^2"
            ),
            "pure_vector_same_layer_merge": {
                "enabled": True,
                "passes": 1,
                "vertical_overlap_ratio": (
                    VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO
                ),
                "horizontal_gap_page_ratio": (
                    VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO
                ),
                "uses_post_merge_bbox_recursively": False,
            },
            "horizontal_relation_merge": {
                "enabled": True,
                "horizontal_connector": (
                    "merge same-row regions when one horizontal vector "
                    "line spans their gap"
                ),
                "shared_frame": (
                    "merge regions whose centers lie in the same band "
                    "between adjacent aligned horizontal frame rules"
                ),
                "vertical_connector_merge": False,
                "distance_threshold_added": False,
            },
            "post_merge_filter_enabled": False,
            "post_attachment_distance_merge_enabled": False,
            "shared_source_image_merge_enabled": True,
            "shared_source_image_merge_rule": (
                "merge whenever two visual regions reference at least "
                "one identical source image block; no Y or distance "
                "threshold is applied"
            ),
            "crop_text_bbox_policy": {
                "enabled": True,
                "association_geometry": (
                    "text bbox overlap with the expanded union of the "
                    "visual cluster and attached source image, or "
                    "single-direction gap with projection overlap"
                ),
                "minimum_text_bbox_overlap_ratio": (
                    VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO
                ),
                "maximum_gap_multiplier": (
                    VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER
                ),
                "endpoint_text_promotion": (
                    "after Liang-Barsky intersection, classify vector "
                    "segments by geometry: axis-aligned segments bridging "
                    "opposite visual-region sides are structural rules and "
                    "cannot promote outside text; remaining boundary/outer "
                    "endpoints use one-hop direction-agnostic attachment"
                ),
                "affects_text_ownership": "conditional",
                "affects_global_text_deduplication": "conditional",
            },
            "render_dpi": VISUAL_REGION_DPI,
            "bbox_policy": {
                "render_bbox": "full visual crop",
                "sort_bbox_with_owned_text": "top anchor strip",
                "sort_bbox_without_owned_text": "full render bbox",
                "sort_bbox_height_points": SORT_BBOX_TOP_HEIGHT_POINTS,
            },
        },
        "formula_render_policy": {
            "enabled": run_enable_formula_detection,
            # "default_enabled": True,
            "default_enabled": run_enable_formula_detection,
            "mode": (
                (
                    run_formula_output_mode
                    if run_enable_formula_detection
                    else "disabled"
                )
            ),
            "ocr": False,
            "latex_reconstruction": run_formula_output_mode in {
                "latex",
                "hybrid",
            },
            "detection": (
                "PDF native font/symbol structure, same-column vertical "
                "clustering, and equation-number anchors"
                if run_enable_formula_detection
                else "disabled"
            ),
            "text_deduplication": (
                "remove only native PDF line boxes owned by the formula "
                "output region"
                if run_enable_formula_detection
                else "disabled"
            ),
            "render_dpi": FORMULA_IMAGE_DPI,
            "region_count": len(formula_regions_output),
            "pages": sorted(formula_region_page_counts),
            "regions_per_page": dict(
                sorted(formula_region_page_counts.items())
            ),
        },
        "text_reflow": {
            "backend": "gaptree",
            "method": (
                "detect and atomize tables/vector visual regions/source "
                "images first; group final elements by line direction; "
                "map each group into virtual upright coordinates; then "
                "sort with the configured spatial backend"
            ),
            "non_text_bbox_policy": (
                "render with render_bbox; sort with sort_bbox; "
                "text falls back to its original bbox"
            ),
            "table_text_policy": (
                "exclude text lines whose centers are inside "
                "embedded large-table bounding boxes"
            ),
            "visual_region_text_policy": (
                "merge vector/framework/source-image structure first; "
                "assign native PDF text once against the final content "
                "bbox; emit owned text below the PNG in the same Markdown "
                "atom; remove only owned line bboxes from global text; "
                "nearby crop-only text is audited but does not expand the "
                "render bbox unless it is close to a terminal vector "
                "endpoint; endpoint attachment is one-hop and direction "
                "agnostic"
            ),
            "formula_text_policy": (
                "render accepted display formulas as PNG and remove only "
                "the exact native text line boxes assigned to each formula"
            ),
        },
        "claim_resolution_policy": {
            "candidate_survival_applied": True,
            "render_bbox_rebuilt_from_owned_elements": True,
            "owned_feature_bbox_preview": True,
            "stage_f_text_ownership_preview": True,
            "stage_f_applied_to_markdown": True,
            "detector_priority": CLAIM_DETECTOR_PRIORITY,
            "suppressed_output_count": sum(
                record["suppressed_output_count"]
                for record in claim_resolution_actions_output
            ),
            "composite_output_groups": {
                "overlap_threshold": (
                    TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO
                ),
                "proposed_group_count": len(
                    composite_output_groups_output
                ),
                "active_group_count": sum(
                    group.get("active", False)
                    for group in composite_output_groups_output
                ),
                "inactive_group_count": sum(
                    not group.get("active", False)
                    for group in composite_output_groups_output
                ),
                "element_ownership_changed": False,
                "renderer_bbox_policy": (
                    "union renderer-owned content with active member table "
                    "and member visual bboxes; keep sort bbox on "
                    "renderer-owned features"
                ),
            },
        },
        "detected_tables": len(tables_output),
        "embedded_table_images": len(embedded_tables),
        "markdown_tables": len(markdown_tables),
        "auto_rotated_table_images": len(auto_rotated_tables),
        "embedded_pages": sorted(page_counts),
        "embedded_images_per_page": dict(
            sorted(page_counts.items())
        ),
        "timing_seconds": {
            "table_detection_total": round(
                run_metrics.table_detection_seconds,
                6,
            ),
            "table_detection_gate_total": round(
                run_metrics.table_detection_gate_seconds,
                6,
            ),
            "table_screenshot_total": round(
                run_metrics.table_screenshot_seconds,
                6,
            ),
            "source_image_processing_total": round(
                run_metrics.source_image_processing_seconds,
                6,
            ),
            "visual_region_processing_total": round(
                run_metrics.visual_region_processing_seconds,
                6,
            ),
            "formula_processing_total": round(
                run_metrics.formula_processing_seconds,
                6,
            ),
            "heading_profile_total": round(
                run_metrics.heading_profile_seconds,
                6,
            ),
            "heading_detection_total": round(
                run_metrics.heading_detection_seconds,
                6,
            ),
            "text_separator_detection_total": round(
                run_metrics.text_separator_detection_seconds,
                6,
            ),
            "text_separator_output_count": (
                run_metrics.text_separator_output_count
            ),
            "preclassified_rule_band_detection_total": round(
                run_metrics.preclassified_rule_band_detection_seconds,
                6,
            ),
            "preclassified_rule_band_output_count": (
                run_metrics.preclassified_rule_band_output_count
            ),
            "logical_horizontal_rule_premerge_total": round(
                run_metrics.logical_horizontal_rule_seconds_total,
                6,
            ),
            "logical_horizontal_rule_audit_finalize_total": round(
                run_metrics.logical_horizontal_rule_audit_finalize_seconds_total,
                6,
            ),
            "axis_geometry_index_total": round(
                run_metrics.axis_geometry_index_seconds_total,
                6,
            ),
            "header_footer_template_total": round(
                run_metrics.header_footer_template_seconds,
                6,
            ),
            "header_footer_page_detection_total": round(
                run_metrics.header_footer_page_detection_seconds,
                6,
            ),
            "inventory_build_total": round(
                run_metrics.inventory_build_seconds,
                6,
            ),
            "end_to_end": round(end_to_end_seconds, 6),
        },
    }
    parse_result.summary = summary

    artifact_emitter.write_parse_result_markdown(parse_result)

    if not run_debug_output:
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )
        # return
        return parse_result

    write_source_image_report(
        path=run_output_dir / "source_images_report.md",
        records=source_images_output,
    )

    write_visual_region_report(
        path=run_output_dir / "visual_regions_report.md",
        records=visual_regions_output,
    )

    write_formula_region_report(
        path=run_output_dir / "formula_regions_report.md",
        records=formula_regions_output,
    )

    write_heading_detection_report(
        path=run_output_dir / "heading_detection_report.md",
        typography_profile=document_run_state.typography_profile,
        records=heading_records_output,
        enable_heading_detection=run_enable_heading_detection,
        enable_pdf_outline_heading_assignment=(
            run_enable_pdf_outline_heading_assignment
        ),
        enable_font_heading_inference=run_enable_font_heading_inference,
    )

    write_figure_caption_report(
        path=run_output_dir / "figure_caption_detection_report.md",
        records=figure_caption_records_output,
    )

    header_footer_output = {
        "templates": serialize_header_footer_templates(
            document_run_state.header_footer_templates
        ),
        "pages": header_footer_pages_output,
    }
    write_header_footer_report(
        path=run_output_dir / "header_footer_report.md",
        templates=document_run_state.header_footer_templates,
        pages=header_footer_pages_output,
        detect_footer=HEADER_FOOTER_DETECT_FOOTER,
        keep_header_text_in_markdown=run_keep_header_text,
    )

    write_gaptree_report(
        path=run_output_dir / "gaptree_reading_order_report.md",
        pages=pages_output,
    )

    write_logical_horizontal_rule_report(
        path=(
            run_output_dir / "logical_horizontal_rules_report.md"
        ),
        records=logical_horizontal_rule_audit_output,
    )

    write_claim_conflict_report(
        path=run_output_dir / "claim_conflicts_report.md",
        page_resolutions=claim_resolution_output,
    )

    from nativePDF_structurer.ir import build_document_ir

    document_ir = build_document_ir(
        source_pdf=run_source_pdf,
        output_dir=run_output_dir,
        source_page_count=source_page_count,
        summary=summary,
        pages=pages_output,
        tables=tables_output,
        source_images=source_images_output,
        visual_regions=visual_regions_output,
        formula_regions=formula_regions_output,
        detector_claims=detector_claims_output,
    )
    parse_result.document_ir = document_ir
    artifact_emitter.write_parse_result_debug_json(
        parse_result,
        header_footer_detection=header_footer_output,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    return parse_result
from functools import partial
import os
import re
from pathlib import Path
from nativePDF_structurer.result import ParseResult
from nativePDF_structurer.config import ParseRunContext, RunConfig
from nativePDF_structurer.page_commit import commit_page_context
from nativePDF_structurer.claims import CLAIM_DETECTOR_PRIORITY, create_claim_manifest
from nativePDF_structurer.inventory import build_page_inventory
from nativePDF_structurer.output_ownership import apply_claim_resolution_to_outputs
from nativePDF_structurer.text_ownership import build_stage_f_text_ownership_preview
from nativePDF_structurer.reading_order_backend import sort_elements_with_gap_tree
from nativePDF_structurer.page_reflow import build_reflowed_page_text, append_reflowed_text_to_markdown
from nativePDF_structurer.rule_geometry import build_axis_geometry_index, build_logical_horizontal_rules
from nativePDF_structurer.rule_preclassification import detect_preclassified_rule_bands, build_preclassified_text_separator_outputs
from nativePDF_structurer.debug_reports import write_logical_horizontal_rule_report, write_visual_region_report, write_claim_conflict_report, write_formula_region_report, write_source_image_report, serialize_header_footer_templates, write_header_footer_report, write_heading_detection_report, write_figure_caption_report, render_gaptree_debug_image, write_gaptree_report
from nativePDF_structurer.table_output import emit_table_candidate
from nativePDF_structurer.table_output import padded_table_candidate_clip, save_rotated_table_image, table_to_markdown, write_table_csv
from nativePDF_structurer.header_footer import build_header_footer_templates, detect_page_header_footer_bands, extract_header_text_for_markdown, filter_confirmed_header_footer_drawings
from nativePDF_structurer.formula_regions import MATH_FONT_HINTS, process_page_formula_regions
from nativePDF_structurer.figure_captions import classify_figure_caption_roles
from nativePDF_structurer.heading_detection import annotate_heading_levels, build_document_typography_profile
from nativePDF_structurer.visual_assets import MIN_IMAGE_AREA_POINTS_SQUARED, MIN_IMAGE_LINE_ASPECT_RATIO, MIN_IMAGE_THIN_SIDE_POINTS, MAX_HORIZONTAL_LINE_SLOPE_RATIO, VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO, VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER, VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO, VISUAL_MASK_CLOSE_GAP_PAGE_RATIO, VISUAL_MASK_PIXELS_PER_POINT, extract_source_image_blocks, image_content_signature, image_is_large_enough, SORT_BBOX_TOP_HEIGHT_POINTS
from nativePDF_structurer.visual_assets import VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO, VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO, VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO, VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO
from nativePDF_structurer.table_detection import TablePipelineDependencies, TablePreclusterDependencies, build_page_table_candidates, detect_page_tables, finalize_page_table_candidates
from nativePDF_structurer.table_structure import TableStructureDependencies, unique_rule_positions
from nativePDF_structurer.cross_detector_ownership import CrossDetectorOwnershipDependencies, build_table_visual_composite_output_groups, finalize_table_visual_composite_output_groups, resolve_table_visual_candidate_ownership, restore_inactive_composite_table_outputs
from nativePDF_structurer.layout_geometry import attach_empty_adjacent_rule_bands_to_tables, build_pdf_outline_heading_index, build_pdf_outline_page_name_index, detect_text_separator_markdown_elements, drawing_axis_rule, drawing_content_signature, drawing_is_stroked_preclassified_horizontal_rule, extract_text_lines, finalize_logical_horizontal_rule_audit, format_markdown_pdf_page_title, group_horizontal_rules_by_span, horizontal_pair_has_boundary_grid_evidence, horizontal_pair_repeats_in_another_span, line_is_inside_any_table, line_matches_running_header_layout, meaningful_overlap, merge_collinear_vertical_rules, prepare_precluster_object_exclusions, rect_is_mostly_inside_any, rule_element_ids, vertical_rule_connects_horizontal_pair
from nativePDF_structurer.visual_processing import annotate_inventory_labels, apply_stage_e_f_to_visual_outputs, apply_stage_e_to_source_outputs, drawing_is_filled_horizontal_bar, drawing_is_long_horizontal_single_line, process_page_source_images, process_page_visual_regions, render_preclassified_rule_band_visuals
from nativePDF_structurer.page_finalize import finalize_page_context
from nativePDF_structurer.run_state import DocumentRunMetrics, DocumentRunState

def environment_boolean(name: str, default: bool) -> bool:
    """从环境变量读取布尔开关。"""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default

def reset_output_dir(path: Path) -> None:
    """Back up an existing output directory and create a clean one.

    将已有输出目录重命名为带时间戳的备份，再创建干净目录，避免残留文件污染。
    """
    if path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(
            f"{path.name}.backup_{timestamp}"
        )
        path.rename(backup_path)

    path.mkdir(parents=True, exist_ok=True)

def _apply_parse_run_context(context: ParseRunContext) -> None:
    """Apply a public run context to the module-level execution settings.

    将 ``RunConfig`` 中公开的配置集中映射为本次运行设置。该函数不访问PDF，
    也不产生文件系统副作用；未暴露给CLI的算法常量继续使用模块默认值。
    """

    global DEBUG_OUTPUT
    global SOURCE_PDF
    global OUTPUT_DIR
    global TEST_PAGE_NUMBERS
    global ENABLE_HEADING_DETECTION
    global ENABLE_FONT_HEADING_INFERENCE
    global ENABLE_TABLE_HEADER_DETECTION
    global KEEP_HEADER_TEXT_IN_MARKDOWN
    global ENABLE_FORMULA_DETECTION
    global FORMULA_OUTPUT_MODE
    global ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT
    global FORCE_PURE_TEXT_TABLE_TO_MARKDOWN

    config = context.config
    DEBUG_OUTPUT = config.debug_output
    SOURCE_PDF = config.source_pdf
    OUTPUT_DIR = config.output_dir
    TEST_PAGE_NUMBERS = context.page_numbers
    ENABLE_HEADING_DETECTION = config.enable_heading_detection
    ENABLE_FONT_HEADING_INFERENCE = config.enable_font_heading_inference
    ENABLE_TABLE_HEADER_DETECTION = config.enable_table_header_detection
    KEEP_HEADER_TEXT_IN_MARKDOWN = config.keep_header_text
    ENABLE_FORMULA_DETECTION = config.enable_formula_detection
    FORMULA_OUTPUT_MODE = config.formula_output_mode
    ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT = (
        config.enable_pdf_outline_heading_assignment
    )
    FORCE_PURE_TEXT_TABLE_TO_MARKDOWN = (
        config.force_pure_text_table_to_markdown
    )

DEBUG_OUTPUT = environment_boolean(
    "PYMUPDF_DEBUG_OUTPUT",
    False,
)

SOURCE_PDF = Path(
    os.environ.get(
        "PYMUPDF_SOURCE_PDF",
        "input.pdf",
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PYMUPDF_OUTPUT_DIR",
        "output",
    )
)

PAGE_LIMIT = 60

_TEST_PAGE_SPEC = os.environ.get("PYMUPDF_TEST_PAGES", "").strip()

TEST_PAGE_NUMBERS = (
    tuple(
        int(value.strip())
        for value in _TEST_PAGE_SPEC.split(",")
        if value.strip()
    )
    if _TEST_PAGE_SPEC
    else tuple(range(1, PAGE_LIMIT + 1))
)

MAX_SMALL_TABLE_CELLS = 6

FORCE_PURE_TEXT_TABLE_TO_MARKDOWN = False

PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO = 0.80

PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS = 1.0

LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES = 3

FALLBACK_TABLE_MIN_HORIZONTAL_RULES = 4

FALLBACK_TABLE_MIN_POPULATED_ROWS = 3

FALLBACK_TABLE_MIN_POPULATED_COLUMNS = 2

FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO = 0.15

FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS = 3.0

FALLBACK_TABLE_MIN_VERTICAL_SEGMENT_POINTS = 8.0

FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS = 3.0

TABLE_SKELETON_HORIZONTAL_GAP_POINTS = 1.0

FALLBACK_TABLE_MAX_VERTICAL_BREAK_POINTS = 18.0

FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO = 0.60

FALLBACK_TABLE_NATIVE_REPLACEMENT_OVERLAP_RATIO = 0.80

FALLBACK_TABLE_NATIVE_EXPANSION_AREA_RATIO = 1.05

CLUSTER_TABLE_DRAWING_BUDGET = 2000

CLUSTER_TABLE_CLOSE_GAP_POINTS = 4.0

CLUSTER_TABLE_ATTACH_MARGIN_POINTS = 2.0

CLUSTER_TABLE_DRAWING_OVERLAP_RATIO = 0.10

TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO = 0.65

TABLE_VISUAL_SOURCE_CLUSTER_EXPLANATION_RATIO = 0.80

OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS = 1.5

OBJECT_PEER_OVERLAP_RATIO = 0.10

PRECLASSIFIED_RULE_BAND_MIN_BOUNDARY_VERTICAL_RULE_COUNT = 2

OBJECT_VECTOR_SPATIAL_GRID_CELL_POINTS = 64.0

TABLE_IMAGE_DPI = 200

TABLE_PADDING_POINTS = 3.0

COMPOSITE_IMAGE_DPI = 200

OVERLAY_INTERSECTION_RATIO = 0.10

VISUAL_REGION_DPI = 200

MAX_BACKGROUND_AREA_PAGE_RATIO = 0.80

VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS = 8.0

VECTOR_ENDPOINT_GRID_CELL_POINTS = 12.0

MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS = 24.0

ROTATION_DOMINANCE_THRESHOLD = 0.70

MIN_DIRECTION_TEXT_CHARS = 10

TEXT_ROW_TOLERANCE_POINTS = 3.0

ENABLE_HEADING_DETECTION = environment_boolean(
    "PYMUPDF_ENABLE_HEADING_DETECTION",
    True,
)

ENABLE_FONT_HEADING_INFERENCE = environment_boolean(
    "PYMUPDF_ENABLE_FONT_HEADING_INFERENCE",
    False,
)

ENABLE_TABLE_HEADER_DETECTION = environment_boolean(
    "PYMUPDF_ENABLE_TABLE_HEADER_DETECTION",
    False,
)

ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT = True

HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS = 0.35

HEADING_MIN_PLAUSIBLE_FONT_SIZE_POINTS = 2.0

HEADING_MAX_TEXT_CHARACTERS = 180

HEADING_SAME_BODY_REQUIRES_BOLD = True

HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS = 4.0

HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO = 0.50

FIGURE_CAPTION_MAX_TEXT_CHARACTERS = 120

FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER = 4.0

FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS = 24.0

FIGURE_CAPTION_LABEL_PATTERN = re.compile(
    r"^(?:图|图示|fig(?:ure)?\.?)[\s]*\d+[\s]*[-‐‑‒–—:：.)]",
    re.IGNORECASE,
)

HEADING_MAX_MARKDOWN_LEVEL = 3

HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS = 80

HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO = 0.35

HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS = 24

HEADING_BODY_STYLE_MIN_CHARACTERS = 120

HEADING_BODY_STYLE_MIN_MULTILINE_BLOCK_RATIO = 0.20

HEADING_BODY_STYLE_MIN_LONG_LINE_RATIO = 0.20

HEADING_BODY_STYLE_HIGH_CHARACTER_SHARE = 0.05

HEADING_STYLE_MIN_SHORT_LINE_RATIO = 0.65

HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT = 3

HEADING_RUNNING_HEADER_TOP_PAGE_RATIO = 0.08

HEADING_RUNNING_HEADER_MIN_PAGE_SUPPORT_RATIO = 0.25

HEADING_RUNNING_HEADER_MIN_PAGE_SUPPORT = 3

HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS = 72.0

HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO = 0.35

HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS = 2.0

GAPTREE_DEBUG_DPI = 120

HEADER_FOOTER_SAMPLE_START_PAGE = 2

HEADER_FOOTER_SAMPLE_MAX_PAGE = 20

HEADER_FOOTER_CANDIDATE_COUNT = 3

HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE = {
    "header": 0.18,
    "footer": 0.08,
}

HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS = 1.0

HEADER_FOOTER_X_TOLERANCE_POINTS = 2.0

HEADER_FOOTER_EDGE_TOLERANCE_POINTS = 3.0

HEADER_FOOTER_BAND_PADDING_POINTS = 2.0

HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE = {
    "header": 2,
    "footer": 1,
}

HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE = {
    "header": 0.60,
    "footer": 0.45,
}

HEADER_FOOTER_DETECT_FOOTER = True

KEEP_HEADER_TEXT_IN_MARKDOWN = False

HEADER_TEXT_OVERLAP_THRESHOLD = 0.50

ENABLE_FORMULA_DETECTION = True

FORMULA_OUTPUT_MODE = "image"


FORMULA_IMAGE_DPI = 200

FORMULA_PADDING_POINTS = 3.0

FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO = 0.28

unique_rule_positions = partial(
    unique_rule_positions,
    tolerance=FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS,
)

def main(context: ParseRunContext | None = None) -> ParseResult:
    """Parse the configured PDF, emit artifacts, and return ``ParseResult``.

    解析配置指定的PDF，写出Markdown及可选调试制品，并返回文档级结果对象。
    公共API传入不可变的 ``ParseRunContext``；直接调用时可省略context并使用
    环境变量和模块默认值。

    ``ParseResult`` owns Markdown sections, page records, detector outputs,
    provenance, the final summary, and the optional compact document IR.
    ``ParseResult`` 统一持有Markdown、页面记录、检测结果、来源信息、摘要，
    以及调试模式下构建的结构化IR。
    """
    if context is not None:
        _apply_parse_run_context(context)
    run_source_pdf = SOURCE_PDF
    run_output_dir = OUTPUT_DIR
    run_page_numbers = TEST_PAGE_NUMBERS
    run_debug_output = DEBUG_OUTPUT
    run_enable_heading_detection = ENABLE_HEADING_DETECTION
    run_enable_font_heading_inference = ENABLE_FONT_HEADING_INFERENCE
    run_keep_header_text = KEEP_HEADER_TEXT_IN_MARKDOWN
    run_enable_formula_detection = ENABLE_FORMULA_DETECTION
    run_formula_output_mode = FORMULA_OUTPUT_MODE
    run_enable_pdf_outline_heading_assignment = (
        ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT
    )
    run_force_pure_text_table_to_markdown = (
        FORCE_PURE_TEXT_TABLE_TO_MARKDOWN
    )
    from nativePDF_structurer.artifacts import prepare_output_layout
    from nativePDF_structurer.emitter import ArtifactEmitter

    output_layout = prepare_output_layout(
        run_output_dir,
        reset=reset_output_dir,
        formula_enabled=run_enable_formula_detection,
        debug_enabled=run_debug_output,
    )
    artifact_emitter = ArtifactEmitter(output_layout)

    started = time.perf_counter()

    parse_result = ParseResult()
    pages_output = parse_result.pages
    tables_output = parse_result.tables
    source_images_output = parse_result.source_images
    visual_regions_output = parse_result.visual_regions
    formula_regions_output = parse_result.formula_regions
    heading_records_output = parse_result.headings
    figure_caption_records_output = parse_result.figure_captions
    header_footer_pages_output = parse_result.header_footer_pages
    detector_claims_output = parse_result.detector_claims
    claim_resolution_output = parse_result.claim_resolutions
    claim_resolution_actions_output = (
        parse_result.claim_resolution_actions
    )
    logical_horizontal_rule_audit_output = (
        parse_result.logical_horizontal_rules
    )
    composite_output_groups_output = (
        parse_result.composite_output_groups
    )

    run_metrics = DocumentRunMetrics()
    document = fitz.open(run_source_pdf)

    source_page_count = document.page_count
    document_toc_entries = document.get_toc(simple=False)
    document_outline_heading_index = (
        build_pdf_outline_heading_index(
            document,
            toc_entries=document_toc_entries,
        )
        if (
            run_enable_heading_detection
            and run_enable_pdf_outline_heading_assignment
        )
        else {}
    )
    document_outline_page_name_index = (
        build_pdf_outline_page_name_index(
            page_count=source_page_count,
            toc_entries=document_toc_entries,
        )
    )
    drawings_cache: dict[int, list[dict]] = {}
    header_footer_template_started = time.perf_counter()
    header_footer_templates = build_header_footer_templates(
        document=document,
        drawings_cache=drawings_cache,
        sample_start_page=HEADER_FOOTER_SAMPLE_START_PAGE,
        sample_max_page=HEADER_FOOTER_SAMPLE_MAX_PAGE,
        candidate_count=HEADER_FOOTER_CANDIDATE_COUNT,
        detect_footer=HEADER_FOOTER_DETECT_FOOTER,
        minimum_matched_objects_by_edge=(
            HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE
        ),
        minimum_support_ratio_by_edge=(
            HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE
        ),
        edge_page_ratio_by_edge=(
            HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE
        ),
        edge_tolerance_points=HEADER_FOOTER_EDGE_TOLERANCE_POINTS,
        height_tolerance_points=HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS,
        x_tolerance_points=HEADER_FOOTER_X_TOLERANCE_POINTS,
        max_background_area_page_ratio=MAX_BACKGROUND_AREA_PAGE_RATIO,
        image_content_signature=image_content_signature,
        drawing_content_signature=drawing_content_signature,
    )
    run_metrics.header_footer_template_seconds = (
        time.perf_counter() - header_footer_template_started
    )
    page_indices = [
        page_number - 1
        for page_number in run_page_numbers
        if 1 <= page_number <= source_page_count
    ]
    processed_pages = len(page_indices)

    heading_profile_started = time.perf_counter()
    if (
        run_enable_heading_detection
        and run_enable_font_heading_inference
    ):
        heading_profile_line_sets: list[list[dict]] = []
        heading_profile_page_rects: list[fitz.Rect] = []
        for profile_page_index in range(source_page_count):
            profile_page = document.load_page(profile_page_index)
            profile_page_dict = profile_page.get_text(
                "dict",
                flags=fitz.TEXTFLAGS_TEXT,
                sort=False,
            )
            heading_profile_line_sets.append(
                extract_text_lines(profile_page_dict)
            )
            heading_profile_page_rects.append(
                fitz.Rect(profile_page.rect)
            )
        document_typography_profile = (
            build_document_typography_profile(
                heading_profile_line_sets,
                page_rects=heading_profile_page_rects,
                enable_heading_detection=ENABLE_HEADING_DETECTION,
                font_size_cluster_tolerance_points=(
                    HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS
                ),
                body_style_min_characters=(
                    HEADING_BODY_STYLE_MIN_CHARACTERS
                ),
                body_style_min_long_line_ratio=(
                    HEADING_BODY_STYLE_MIN_LONG_LINE_RATIO
                ),
                body_style_min_multiline_block_ratio=(
                    HEADING_BODY_STYLE_MIN_MULTILINE_BLOCK_RATIO
                ),
                body_style_high_character_share=(
                    HEADING_BODY_STYLE_HIGH_CHARACTER_SHARE
                ),
                style_min_short_line_ratio=(
                    HEADING_STYLE_MIN_SHORT_LINE_RATIO
                ),
                running_header_top_page_ratio=(
                    HEADING_RUNNING_HEADER_TOP_PAGE_RATIO
                ),
                running_header_min_page_support_ratio=(
                    HEADING_RUNNING_HEADER_MIN_PAGE_SUPPORT_RATIO
                ),
                running_header_min_page_support=(
                    HEADING_RUNNING_HEADER_MIN_PAGE_SUPPORT
                ),
            )
        )
    else:
        document_typography_profile = {
            "enabled": False,
            "body_font_size": None,
            "clusters": [],
            "heading_size_clusters": [],
            "body_style_keys": [],
            "heading_style_keys": [],
            "style_roles": [],
            "running_header_layout_templates": [],
            "reason": (
                "font_heading_inference_disabled"
                if run_enable_heading_detection
                else "heading_detection_disabled"
            ),
        }
    run_metrics.heading_profile_seconds = (
        time.perf_counter() - heading_profile_started
    )

    document_run_state = DocumentRunState(
        document=document,
        output_layout=output_layout,
        source_page_count=source_page_count,
        page_indices=page_indices,
        drawings_cache=drawings_cache,
        header_footer_templates=header_footer_templates,
        outline_heading_index=document_outline_heading_index,
        outline_page_name_index=document_outline_page_name_index,
        typography_profile=document_typography_profile,
        metrics=run_metrics,
    )

    table_pipeline_dependencies = TablePipelineDependencies(
        metrics=run_metrics,
        tables_dir=output_layout.tables,
        cluster_table_drawing_budget=CLUSTER_TABLE_DRAWING_BUDGET,
        fallback_table_max_vertical_break_points=(
            FALLBACK_TABLE_MAX_VERTICAL_BREAK_POINTS
        ),
        fallback_table_min_horizontal_rules=(
            FALLBACK_TABLE_MIN_HORIZONTAL_RULES
        ),
        fallback_table_min_populated_columns=(
            FALLBACK_TABLE_MIN_POPULATED_COLUMNS
        ),
        fallback_table_min_populated_rows=(
            FALLBACK_TABLE_MIN_POPULATED_ROWS
        ),
        fallback_table_min_vertical_coverage_ratio=(
            FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO
        ),
        fallback_table_min_vertical_segment_points=(
            FALLBACK_TABLE_MIN_VERTICAL_SEGMENT_POINTS
        ),
        fallback_table_native_expansion_area_ratio=(
            FALLBACK_TABLE_NATIVE_EXPANSION_AREA_RATIO
        ),
        fallback_table_native_replacement_overlap_ratio=(
            FALLBACK_TABLE_NATIVE_REPLACEMENT_OVERLAP_RATIO
        ),
        fallback_table_rule_alignment_tolerance_points=(
            FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
        ),
        logical_horizontal_rule_coordinate_decimal_places=(
            LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES
        ),
        max_small_table_cells=MAX_SMALL_TABLE_CELLS,
        min_direction_text_chars=MIN_DIRECTION_TEXT_CHARS,
        rotation_dominance_threshold=ROTATION_DOMINANCE_THRESHOLD,
        math_font_hints=tuple(MATH_FONT_HINTS),
        drawing_axis_rule=drawing_axis_rule,
        table_precluster_dependencies=TablePreclusterDependencies(
            mask_pixels_per_point=VISUAL_MASK_PIXELS_PER_POINT,
            close_gap_points=CLUSTER_TABLE_CLOSE_GAP_POINTS,
            attach_margin_points=CLUSTER_TABLE_ATTACH_MARGIN_POINTS,
            drawing_overlap_ratio=CLUSTER_TABLE_DRAWING_OVERLAP_RATIO,
            max_background_area_page_ratio=(
                MAX_BACKGROUND_AREA_PAGE_RATIO
            ),
            drawing_is_filled_horizontal_bar=(
                drawing_is_filled_horizontal_bar
            ),
            drawing_is_long_horizontal_single_line=(
                drawing_is_long_horizontal_single_line
            ),
        ),
        table_structure_dependencies=TableStructureDependencies(
            alignment_tolerance_points=(
                FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS
            ),
            skeleton_horizontal_gap_points=(
                TABLE_SKELETON_HORIZONTAL_GAP_POINTS
            ),
            drawing_axis_rule=drawing_axis_rule,
        ),
        group_horizontal_rules_by_span=group_horizontal_rules_by_span,
        merge_collinear_vertical_rules=merge_collinear_vertical_rules,
        meaningful_overlap=meaningful_overlap,
        rect_is_mostly_inside_any=rect_is_mostly_inside_any,
        attach_empty_adjacent_rule_bands_to_tables=(
            attach_empty_adjacent_rule_bands_to_tables
        ),
    )

    cross_detector_ownership_dependencies = (
        CrossDetectorOwnershipDependencies(
            ownership_overlap_threshold=(
                TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO
            ),
            source_cluster_explanation_ratio=(
                TABLE_VISUAL_SOURCE_CLUSTER_EXPLANATION_RATIO
            ),
            object_peer_overlap_ratio=OBJECT_PEER_OVERLAP_RATIO,
            table_image_dpi=TABLE_IMAGE_DPI,
            rect_is_mostly_inside_any=rect_is_mostly_inside_any,
            emit_table_candidate=emit_table_candidate,
            write_table_csv=write_table_csv,
            table_to_markdown=table_to_markdown,
            padded_table_candidate_clip=padded_table_candidate_clip,
            save_rotated_table_image=save_rotated_table_image,
        )
    )

    page_stage_adapter_dependencies = PageStageAdapterDependencies(
        ENABLE_FONT_HEADING_INFERENCE=ENABLE_FONT_HEADING_INFERENCE,
        ENABLE_HEADING_DETECTION=ENABLE_HEADING_DETECTION,
        ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT=ENABLE_PDF_OUTLINE_HEADING_ASSIGNMENT,
        FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS=FALLBACK_TABLE_MAX_RULE_THICKNESS_POINTS,
        FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO=FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO,
        FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO=FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO,
        FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS=FALLBACK_TABLE_RULE_ALIGNMENT_TOLERANCE_POINTS,
        FIGURE_CAPTION_LABEL_PATTERN=FIGURE_CAPTION_LABEL_PATTERN,
        FIGURE_CAPTION_MAX_TEXT_CHARACTERS=FIGURE_CAPTION_MAX_TEXT_CHARACTERS,
        FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER=FIGURE_CAPTION_MAX_VERTICAL_GAP_MULTIPLIER,
        FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS=FIGURE_CAPTION_MAX_VERTICAL_GAP_POINTS,
        FORMULA_IMAGE_DPI=FORMULA_IMAGE_DPI,
        FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO=FORMULA_MIN_UNNUMBERED_WIDTH_PAGE_RATIO,
        FORMULA_PADDING_POINTS=FORMULA_PADDING_POINTS,
        FORMULA_OUTPUT_MODE=run_formula_output_mode,
        GAPTREE_DEBUG_DPI=GAPTREE_DEBUG_DPI,
        HEADER_FOOTER_BAND_PADDING_POINTS=HEADER_FOOTER_BAND_PADDING_POINTS,
        HEADER_FOOTER_CANDIDATE_COUNT=HEADER_FOOTER_CANDIDATE_COUNT,
        HEADER_FOOTER_DETECT_FOOTER=HEADER_FOOTER_DETECT_FOOTER,
        HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE=HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE,
        HEADER_FOOTER_EDGE_TOLERANCE_POINTS=HEADER_FOOTER_EDGE_TOLERANCE_POINTS,
        HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS=HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS,
        HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE=HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE,
        HEADER_FOOTER_X_TOLERANCE_POINTS=HEADER_FOOTER_X_TOLERANCE_POINTS,
        HEADER_TEXT_OVERLAP_THRESHOLD=HEADER_TEXT_OVERLAP_THRESHOLD,
        HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT=HEADING_CANDIDATE_MAX_BLOCK_LINE_COUNT,
        HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS=HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS,
        HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO=HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO,
        HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS=HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS,
        HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS=HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS,
        HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS=HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS,
        HEADING_MAX_MARKDOWN_LEVEL=HEADING_MAX_MARKDOWN_LEVEL,
        HEADING_MAX_TEXT_CHARACTERS=HEADING_MAX_TEXT_CHARACTERS,
        HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS=HEADING_OUTLINE_SHARED_TARGET_TOLERANCE_POINTS,
        HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS=HEADING_OUTLINE_TARGET_MAX_VERTICAL_DISTANCE_POINTS,
        HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO=HEADING_OUTLINE_TARGET_MAX_X_DISTANCE_PAGE_RATIO,
        HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO=HEADING_SAME_BODY_MIN_BOLD_CHARACTER_RATIO,
        HEADING_SAME_BODY_REQUIRES_BOLD=HEADING_SAME_BODY_REQUIRES_BOLD,
        LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES=LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES,
        MAX_BACKGROUND_AREA_PAGE_RATIO=MAX_BACKGROUND_AREA_PAGE_RATIO,
        MIN_IMAGE_AREA_POINTS_SQUARED=MIN_IMAGE_AREA_POINTS_SQUARED,
        PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO=PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO,
        PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS=PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS,
        TABLE_IMAGE_DPI=TABLE_IMAGE_DPI,
        VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO=VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO,
        annotate_heading_levels=annotate_heading_levels,
        annotate_inventory_labels=annotate_inventory_labels,
        append_reflowed_text_to_markdown=append_reflowed_text_to_markdown,
        apply_claim_resolution_to_outputs=apply_claim_resolution_to_outputs,
        apply_stage_e_f_to_visual_outputs=apply_stage_e_f_to_visual_outputs,
        apply_stage_e_to_source_outputs=apply_stage_e_to_source_outputs,
        build_axis_geometry_index=build_axis_geometry_index,
        build_logical_horizontal_rules=build_logical_horizontal_rules,
        build_page_inventory=build_page_inventory,
        build_preclassified_text_separator_outputs=build_preclassified_text_separator_outputs,
        build_reflowed_page_text=build_reflowed_page_text,
        build_stage_f_text_ownership_preview=build_stage_f_text_ownership_preview,
        build_table_visual_composite_output_groups=build_table_visual_composite_output_groups,
        classify_figure_caption_roles=classify_figure_caption_roles,
        create_claim_manifest=create_claim_manifest,
        cross_detector_ownership_dependencies=cross_detector_ownership_dependencies,
        detect_page_header_footer_bands=detect_page_header_footer_bands,
        detect_preclassified_rule_bands=detect_preclassified_rule_bands,
        detect_text_separator_markdown_elements=detect_text_separator_markdown_elements,
        document_run_state=document_run_state,
        drawing_axis_rule=drawing_axis_rule,
        drawing_content_signature=drawing_content_signature,
        drawing_is_stroked_preclassified_horizontal_rule=drawing_is_stroked_preclassified_horizontal_rule,
        emit_table_candidate=emit_table_candidate,
        extract_header_text_for_markdown=extract_header_text_for_markdown,
        extract_source_image_blocks=extract_source_image_blocks,
        extract_text_lines=extract_text_lines,
        filter_confirmed_header_footer_drawings=filter_confirmed_header_footer_drawings,
        finalize_logical_horizontal_rule_audit=finalize_logical_horizontal_rule_audit,
        finalize_table_visual_composite_output_groups=finalize_table_visual_composite_output_groups,
        group_horizontal_rules_by_span=group_horizontal_rules_by_span,
        horizontal_pair_has_boundary_grid_evidence=horizontal_pair_has_boundary_grid_evidence,
        horizontal_pair_repeats_in_another_span=horizontal_pair_repeats_in_another_span,
        image_content_signature=image_content_signature,
        image_is_large_enough=image_is_large_enough,
        line_is_inside_any_table=line_is_inside_any_table,
        line_matches_running_header_layout=line_matches_running_header_layout,
        meaningful_overlap=meaningful_overlap,
        merge_collinear_vertical_rules=merge_collinear_vertical_rules,
        padded_table_candidate_clip=padded_table_candidate_clip,
        prepare_precluster_object_exclusions=prepare_precluster_object_exclusions,
        process_page_formula_regions=process_page_formula_regions,
        process_page_source_images=process_page_source_images,
        process_page_visual_regions=process_page_visual_regions,
        rect_is_mostly_inside_any=rect_is_mostly_inside_any,
        render_gaptree_debug_image=render_gaptree_debug_image,
        render_preclassified_rule_band_visuals=render_preclassified_rule_band_visuals,
        resolve_table_visual_candidate_ownership=resolve_table_visual_candidate_ownership,
        restore_inactive_composite_table_outputs=restore_inactive_composite_table_outputs,
        rule_element_ids=rule_element_ids,
        run_metrics=run_metrics,
        save_rotated_table_image=save_rotated_table_image,
        sort_elements_with_gap_tree=sort_elements_with_gap_tree,
        unique_rule_positions=unique_rule_positions,
        vertical_rule_connects_horizontal_pair=vertical_rule_connects_horizontal_pair,
    )

    pipeline_page_dependencies = PagePipelineDependencies(
        document_run_state=document_run_state,
        table_pipeline_dependencies=table_pipeline_dependencies,
        initialize_page_context=partial(
            initialize_page_context,
            dependencies=page_stage_adapter_dependencies,
        ),
        process_page_header_footer=partial(
            process_page_header_footer,
            dependencies=page_stage_adapter_dependencies,
        ),
        prepare_page_geometry=partial(
            prepare_page_geometry,
            dependencies=page_stage_adapter_dependencies,
        ),
        prepare_page_rule_bands=partial(
            prepare_page_rule_bands,
            dependencies=page_stage_adapter_dependencies,
        ),
        detect_page_tables=detect_page_tables,
        build_page_table_candidates=build_page_table_candidates,
        finalize_page_table_candidates=finalize_page_table_candidates,
        prepare_page_visual_protection=partial(
            prepare_page_visual_protection,
            dependencies=page_stage_adapter_dependencies,
        ),
        detect_page_visual_regions=partial(
            detect_page_visual_regions,
            dependencies=page_stage_adapter_dependencies,
        ),
        propose_page_output_candidates=partial(
            propose_page_output_candidates,
            dependencies=page_stage_adapter_dependencies,
        ),
        detect_page_source_images=partial(
            detect_page_source_images,
            dependencies=page_stage_adapter_dependencies,
        ),
        resolve_page_claims=partial(
            resolve_page_claims,
            dependencies=page_stage_adapter_dependencies,
        ),
        apply_page_boundary_ownership=partial(
            apply_page_boundary_ownership,
            dependencies=page_stage_adapter_dependencies,
        ),
        process_page_semantics=partial(
            process_page_semantics,
            dependencies=page_stage_adapter_dependencies,
        ),
        process_page_reading_order=partial(
            process_page_reading_order,
            dependencies=page_stage_adapter_dependencies,
        ),
        finalize_page_context=finalize_page_context,
        format_markdown_pdf_page_title=format_markdown_pdf_page_title,
        keep_header_text=run_keep_header_text,
        formula_detection_enabled=run_enable_formula_detection,
        debug_output_enabled=run_debug_output,
        force_pure_text_table_to_markdown=(
            run_force_pure_text_table_to_markdown
        ),
    )

    process_document_pages(
        dependencies=DocumentPipelineDependencies(
            document_run_state=document_run_state,
            parse_result=parse_result,
            page_pipeline_dependencies=pipeline_page_dependencies,
            process_page=process_page,
            commit_page_context=commit_page_context,
            validate_page_contracts=run_debug_output,
        )
    )

    return finalize_document_run(
        dependencies=DocumentFinalizeDependencies(
            CLAIM_DETECTOR_PRIORITY=CLAIM_DETECTOR_PRIORITY,
            CLUSTER_TABLE_CLOSE_GAP_POINTS=CLUSTER_TABLE_CLOSE_GAP_POINTS,
            CLUSTER_TABLE_DRAWING_BUDGET=CLUSTER_TABLE_DRAWING_BUDGET,
            COMPOSITE_IMAGE_DPI=COMPOSITE_IMAGE_DPI,
            FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO=FALLBACK_TABLE_MIN_HORIZONTAL_LENGTH_PAGE_RATIO,
            FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO=FALLBACK_TABLE_MIN_VERTICAL_COVERAGE_RATIO,
            FORMULA_IMAGE_DPI=FORMULA_IMAGE_DPI,
            HEADER_FOOTER_CANDIDATE_COUNT=HEADER_FOOTER_CANDIDATE_COUNT,
            HEADER_FOOTER_DETECT_FOOTER=HEADER_FOOTER_DETECT_FOOTER,
            HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE=HEADER_FOOTER_EDGE_PAGE_RATIO_BY_EDGE,
            HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS=HEADER_FOOTER_HEIGHT_TOLERANCE_POINTS,
            HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE=HEADER_FOOTER_MIN_MATCHED_OBJECTS_BY_EDGE,
            HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE=HEADER_FOOTER_MIN_SUPPORT_RATIO_BY_EDGE,
            HEADER_FOOTER_SAMPLE_MAX_PAGE=HEADER_FOOTER_SAMPLE_MAX_PAGE,
            HEADER_FOOTER_SAMPLE_START_PAGE=HEADER_FOOTER_SAMPLE_START_PAGE,
            HEADER_FOOTER_X_TOLERANCE_POINTS=HEADER_FOOTER_X_TOLERANCE_POINTS,
            HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS=HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS,
            HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO=HEADING_LOCAL_BODY_CLUSTER_WEIGHT_RATIO,
            HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS=HEADING_LOCAL_BODY_MIN_CLUSTER_CHARACTERS,
            HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS=HEADING_LOCAL_BODY_MIN_SIZE_DELTA_POINTS,
            HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS=HEADING_LOCAL_BODY_MIN_VISIBLE_CHARACTERS,
            HEADING_MAX_MARKDOWN_LEVEL=HEADING_MAX_MARKDOWN_LEVEL,
            LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES=LOGICAL_HORIZONTAL_RULE_COORDINATE_DECIMAL_PLACES,
            MAX_HORIZONTAL_LINE_SLOPE_RATIO=MAX_HORIZONTAL_LINE_SLOPE_RATIO,
            MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS=MAX_VECTOR_ENDPOINT_MARKER_MAJOR_POINTS,
            MIN_DIRECTION_TEXT_CHARS=MIN_DIRECTION_TEXT_CHARS,
            MIN_IMAGE_AREA_POINTS_SQUARED=MIN_IMAGE_AREA_POINTS_SQUARED,
            MIN_IMAGE_LINE_ASPECT_RATIO=MIN_IMAGE_LINE_ASPECT_RATIO,
            MIN_IMAGE_THIN_SIDE_POINTS=MIN_IMAGE_THIN_SIDE_POINTS,
            OBJECT_PEER_OVERLAP_RATIO=OBJECT_PEER_OVERLAP_RATIO,
            OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS=OBJECT_PRECLUSTER_CONTAINMENT_TOLERANCE_POINTS,
            OVERLAY_INTERSECTION_RATIO=OVERLAY_INTERSECTION_RATIO,
            PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO=PRECLASSIFIED_RULE_BAND_MIN_X_OVERLAP_RATIO,
            PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS=PRECLASSIFIED_RULE_BAND_VECTOR_CONNECTION_TOLERANCE_POINTS,
            ROTATION_DOMINANCE_THRESHOLD=ROTATION_DOMINANCE_THRESHOLD,
            SORT_BBOX_TOP_HEIGHT_POINTS=SORT_BBOX_TOP_HEIGHT_POINTS,
            TABLE_IMAGE_DPI=TABLE_IMAGE_DPI,
            TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO=TABLE_VISUAL_OWNERSHIP_OVERLAP_RATIO,
            VECTOR_ENDPOINT_GRID_CELL_POINTS=VECTOR_ENDPOINT_GRID_CELL_POINTS,
            VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS=VECTOR_ENDPOINT_SEARCH_RADIUS_POINTS,
            VISUAL_MASK_CLOSE_GAP_PAGE_RATIO=VISUAL_MASK_CLOSE_GAP_PAGE_RATIO,
            VISUAL_MASK_PIXELS_PER_POINT=VISUAL_MASK_PIXELS_PER_POINT,
            VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER=VISUAL_REGION_CROP_TEXT_MAX_GAP_MULTIPLIER,
            VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO=VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO,
            VISUAL_REGION_DPI=VISUAL_REGION_DPI,
            VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO=VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO,
            VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO=VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO,
            VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO=VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO,
            VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO=VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO,
            artifact_emitter=artifact_emitter,
            claim_resolution_actions_output=claim_resolution_actions_output,
            claim_resolution_output=claim_resolution_output,
            composite_output_groups_output=composite_output_groups_output,
            detector_claims_output=detector_claims_output,
            document_run_state=document_run_state,
            document_toc_entries=document_toc_entries,
            figure_caption_records_output=figure_caption_records_output,
            formula_regions_output=formula_regions_output,
            header_footer_pages_output=header_footer_pages_output,
            heading_records_output=heading_records_output,
            logical_horizontal_rule_audit_output=logical_horizontal_rule_audit_output,
            page_indices=page_indices,
            pages_output=pages_output,
            parse_result=parse_result,
            processed_pages=processed_pages,
            run_debug_output=run_debug_output,
            run_enable_font_heading_inference=run_enable_font_heading_inference,
            run_enable_formula_detection=run_enable_formula_detection,
            run_formula_output_mode=run_formula_output_mode,
            run_enable_heading_detection=run_enable_heading_detection,
            run_enable_pdf_outline_heading_assignment=run_enable_pdf_outline_heading_assignment,
            run_force_pure_text_table_to_markdown=run_force_pure_text_table_to_markdown,
            run_keep_header_text=run_keep_header_text,
            run_metrics=run_metrics,
            run_output_dir=run_output_dir,
            run_source_pdf=run_source_pdf,
            serialize_header_footer_templates=serialize_header_footer_templates,
            source_images_output=source_images_output,
            source_page_count=source_page_count,
            started=started,
            tables_output=tables_output,
            visual_regions_output=visual_regions_output,
            write_claim_conflict_report=write_claim_conflict_report,
            write_figure_caption_report=write_figure_caption_report,
            write_formula_region_report=write_formula_region_report,
            write_gaptree_report=write_gaptree_report,
            write_header_footer_report=write_header_footer_report,
            write_heading_detection_report=write_heading_detection_report,
            write_logical_horizontal_rule_report=write_logical_horizontal_rule_report,
            write_source_image_report=write_source_image_report,
            write_visual_region_report=write_visual_region_report,
        )
    )

def _page_numbers_for_config(config: RunConfig, page_count: int) -> tuple[int, ...]:
    """Resolve one-based page selection against the opened PDF."""
    if config.pages is None:
        return tuple(range(1, page_count + 1))
    invalid = [page for page in config.pages if page > page_count]
    if invalid:
        raise ValueError(f"page number exceeds PDF page count {page_count}: {invalid}")
    return config.pages


def run(config: RunConfig) -> ParseResult:
    """Run one configured PDF structuring job and return its result.

    执行一次已配置的PDF结构化任务，并返回 ``ParseResult``。函数会根据实际
    文档页数校验所选页码，再通过不可变的 ``ParseRunContext`` 启动流水线。
    """
    with fitz.open(config.source_pdf) as document:
        pages = _page_numbers_for_config(config, document.page_count)
    context = ParseRunContext(config=config, page_numbers=pages)
    return main(context)
