"""定义文档运行期资源和带类型的页面阶段结果。

模块提供跨选定页面共享的文档
状态、累计计时指标以及页面阶段返回的轻量数据类；最终Markdown和来源记录
仍由 ``ParseResult`` 持有。

Define document-run resources and typed page-stage results.

Provides the mutable document-level state shared across selected pages,
accumulated timing metrics, and lightweight dataclasses returned by page
stages. Final Markdown and provenance collections remain owned by
``ParseResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentRunMetrics:
    table_detection_seconds: float = 0.0
    table_detection_gate_seconds: float = 0.0
    table_screenshot_seconds: float = 0.0
    source_image_processing_seconds: float = 0.0
    visual_region_processing_seconds: float = 0.0
    formula_processing_seconds: float = 0.0
    heading_profile_seconds: float = 0.0
    heading_detection_seconds: float = 0.0
    text_separator_detection_seconds: float = 0.0
    text_separator_output_count: int = 0
    preclassified_rule_band_detection_seconds: float = 0.0
    preclassified_rule_band_output_count: int = 0
    logical_horizontal_rule_seconds_total: float = 0.0
    logical_horizontal_rule_audit_finalize_seconds_total: float = 0.0
    axis_geometry_index_seconds_total: float = 0.0
    header_footer_template_seconds: float = 0.0
    header_footer_page_detection_seconds: float = 0.0
    drawing_extraction_seconds: float = 0.0
    inventory_build_seconds: float = 0.0


@dataclass
class DocumentRunState:
    document: Any
    output_layout: Any
    source_page_count: int
    page_indices: list[int]
    drawings_cache: dict[int, list[dict[str, Any]]]
    header_footer_templates: list[dict[str, Any]]
    outline_heading_index: dict[int, list[dict[str, Any]]]
    outline_page_name_index: dict[int, dict[str, Any]]
    typography_profile: dict[str, Any]
    metrics: DocumentRunMetrics = field(default_factory=DocumentRunMetrics)


@dataclass
class PageInitialization:
    page_number: int
    page: Any
    page_started: float
    page_drawings: list[dict[str, Any]]
    page_dict: dict[str, Any]
    text_lines: list[dict[str, Any]]
    source_images: list[dict[str, Any]]
    page_context: Any
    page_inventory: Any
    page_drawing_seconds: float
    page_inventory_seconds: float


@dataclass
class PageHeaderFooterResult:
    page_drawings: list[dict[str, Any]]
    header_footer_rects: list[Any]
    header_footer_record: dict[str, Any]
    vector_exclusion_rects: list[Any]
    header_text: str
    header_text_lines: list[dict[str, Any]]
    detection_seconds: float


@dataclass
class PageGeometryPreparation:
    axis_geometry_index: dict[str, Any]
    logical_horizontal_rules: list[dict[str, Any]]
    logical_horizontal_rule_audit: list[dict[str, Any]]
    axis_seconds: float
    logical_rule_seconds: float


@dataclass
class PageRuleBandPreparation:
    rule_band_result: dict[str, Any]
    bands: list[dict[str, Any]]
    drawing_ids: set[str]
    detection_seconds: float
    output_count: int


@dataclass
class PageTableDetectionResult:
    detected_tables: list[Any]
    metadata: dict[str, Any]
    table_claims: list[dict[str, Any]]
    gate_seconds: float
    detection_seconds: float
    dense_vector_page: bool
    detection_skipped: bool
    skip_reason: str | None


@dataclass
class PageTableCandidateBuildResult:
    candidates: list[dict[str, Any]]
    detected_table_count: int


@dataclass
class PageTableCandidateFinalization:
    candidates: list[dict[str, Any]]
    claims_by_id: dict[str, dict[str, Any]]
    skeleton_drawing_ids: set[str]
    full_drawing_ids: set[str]
    adjacent_rule_band_attachment: dict[str, Any]


@dataclass
class PageVisualProtectionPreparation:
    object_exclusions: dict[str, Any]
    protected_rects: list[Any]
    table_visual_protection_rects: list[Any]


@dataclass
class PageVisualDetectionResult:
    records: list[dict[str, Any]]
    markdown_elements: list[dict[str, Any]]
    rects: list[Any]
    text_rects: list[Any]
    claims: list[dict[str, Any]]
    base_seconds: float
    preclassified_seconds: float
    embedded_child_visual_count: int


@dataclass
class PageOutputCandidateProposal:
    table_ownership_decisions: list[dict[str, Any]]
    suppressed_visual_indices: set[int]
    decisions_by_table_index: dict[int, dict[str, Any]]
    composite_output_groups: list[dict[str, Any]]


@dataclass
class PageSourceImageDetectionResult:
    records: list[dict[str, Any]]
    markdown_elements: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    detection_seconds: float


@dataclass
class PageClaimResolutionResult:
    resolution: dict[str, Any]
    suppressed_actions: list[dict[str, Any]]
    output_group_finalization: dict[str, Any]
    restore_actions: list[dict[str, Any]]
    restored_table_screenshot_seconds: float


@dataclass
class PageBoundaryOwnershipResult:
    excluded_table_rects: list[Any]
    visual_region_rects: list[Any]
    visual_region_text_rects: list[Any]
    source_image_text_rects: list[Any]
    emitted_claim_ids: set[str]
    active_composite_table_claim_ids: set[str]
    visual_application: dict[str, Any]
    source_application: dict[str, Any]


@dataclass
class PageSemanticProcessingResult:
    figure_caption_records: list[dict[str, Any]]
    formula_records: list[dict[str, Any]]
    formula_markdown_elements: list[dict[str, Any]]
    formula_text_rects: list[Any]
    formula_seconds: float
    text_separator_markdown_elements: list[dict[str, Any]]
    text_separator_records: list[dict[str, Any]]
    text_separator_seconds: float
    heading_records: list[dict[str, Any]]
    heading_seconds: float


@dataclass
class PageReadingOrderProcessingResult:
    metadata: dict[str, Any]
    grouped_markdown: dict[str, str]
    normal_direction_metadata: dict[str, Any]
    debug_path: Any | None
    markdown_lines: list[str]
