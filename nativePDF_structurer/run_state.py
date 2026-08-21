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


#@GPT 文档累计指标此前由main()中的大量浮点局部变量持有；集中成独立对象后，未来process_page()只需更新一个显式依赖，不必通过十余个返回值或nonlocal变量回传。
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


#@GPT DocumentRunState只保存打开文档期间共享的运行期资源，不接管用户配置或最终ParseResult；三层所有权因此保持为ParseRunContext→DocumentRunState→PageRunContext。
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


#@GPT 页面初始化结果把本页原始对象、缓存耗时和PageRunContext一起返回；这是抽取process_page()前的第一条稳定页面输入边界。
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


#@GPT 页眉页尾阶段只返回下游真正需要的完成态，避免主循环继续依赖模板匹配、drawing过滤和页眉文字提取的内部变量。
@dataclass
class PageHeaderFooterResult:
    page_drawings: list[dict[str, Any]]
    header_footer_rects: list[Any]
    header_footer_record: dict[str, Any]
    vector_exclusion_rects: list[Any]
    header_text: str
    header_text_lines: list[dict[str, Any]]
    detection_seconds: float


#@GPT 几何预处理结果同时承载轴向索引、逻辑横线和两段耗时；表格检测阶段只消费这个完成态，不需要重新扫描页面drawing。
@dataclass
class PageGeometryPreparation:
    axis_geometry_index: dict[str, Any]
    logical_horizontal_rules: list[dict[str, Any]]
    logical_horizontal_rule_audit: list[dict[str, Any]]
    axis_seconds: float
    logical_rule_seconds: float


#@GPT 规则带预分类结果明确区分完整检测结果、下游band列表和被消费drawing ID；表格检测入口因此不再从内部metadata重复拆解这些值。
@dataclass
class PageRuleBandPreparation:
    rule_band_result: dict[str, Any]
    bands: list[dict[str, Any]]
    drawing_ids: set[str]
    detection_seconds: float
    output_count: int


#@GPT 表格检测入口返回检测对象、元数据、Claim和预算状态；候选构建阶段不再自行推导“是否截断大簇”等检测器内部结论。
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


#@GPT 候选构建阶段将PyMuPDF表格转换为项目内部候选字典；返回对象记录候选列表和原始检测数量，后续Claim筛选不再接触table.extract()细节。
@dataclass
class PageTableCandidateBuildResult:
    candidates: list[dict[str, Any]]
    detected_table_count: int


#@GPT 候选筛选结果保存阶段C最终存活表格、Claim索引和drawing所有权集合；视觉阶段只消费这些完成态，不再重复判断表格结构有效性。
@dataclass
class PageTableCandidateFinalization:
    candidates: list[dict[str, Any]]
    claims_by_id: dict[str, dict[str, Any]]
    skeleton_drawing_ids: set[str]
    full_drawing_ids: set[str]
    adjacent_rule_band_attachment: dict[str, Any]


#@GPT 视觉预聚簇保护结果分别保存完整对象决策、通用保护矩形和表格视觉空洞，供视觉检测与页面审计复用。
@dataclass
class PageVisualProtectionPreparation:
    object_exclusions: dict[str, Any]
    protected_rects: list[Any]
    table_visual_protection_rects: list[Any]


#@GPT 视觉检测结果包含普通视觉候选和预分类视觉带合并后的完成态，并分别保留两段耗时以维持现有页面计时口径。
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


#@GPT 输出候选提议结果保存兼容bbox审计、neutral表格decision和组合输出组；阶段D随后只处理已原子化的表格/视觉/原图候选。
@dataclass
class PageOutputCandidateProposal:
    table_ownership_decisions: list[dict[str, Any]]
    suppressed_visual_indices: set[int]
    decisions_by_table_index: dict[int, dict[str, Any]]
    composite_output_groups: list[dict[str, Any]]


#@GPT 原图检测结果把记录、Markdown原子、Claim和耗时作为同一完成态返回；阶段D不再依赖包装层内部的局部变量。
@dataclass
class PageSourceImageDetectionResult:
    records: list[dict[str, Any]]
    markdown_elements: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    detection_seconds: float


#@GPT 阶段D结果保存唯一归属字典、组合组最终化、恢复动作和三类候选淘汰动作；阶段E/F只消费存活完成态。
@dataclass
class PageClaimResolutionResult:
    resolution: dict[str, Any]
    suppressed_actions: list[dict[str, Any]]
    output_group_finalization: dict[str, Any]
    restore_actions: list[dict[str, Any]]
    restored_table_screenshot_seconds: float


#@GPT 阶段E/F结果保存最终结构边界、精确文字排除框和活动Claim集合；后续语义阶段只消费这些完成态。
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


#@GPT 页面语义结果按既有顺序汇总图题、公式、分隔符和标题完成态；阅读顺序阶段只消费这些最终原子和精确排除框。
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


#@GPT 阅读顺序结果保存完整排序元数据、分方向Markdown和调试图路径；页面record阶段不再重新派生GapTree正常方向节点。
@dataclass
class PageReadingOrderProcessingResult:
    metadata: dict[str, Any]
    grouped_markdown: dict[str, str]
    normal_direction_metadata: dict[str, Any]
    debug_path: Any | None
    markdown_lines: list[str]
