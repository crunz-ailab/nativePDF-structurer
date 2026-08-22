"""定义单页解析流水线使用的可变状态。

``PageRunContext`` 持有原始页面对象、
工作视图、页面缓存、检测候选、归属结果、阅读顺序元数据和最终页面制品；
文档配置和跨页结果列表不属于该上下文。

关键不变量:
    * ``raw_drawings`` 永不被过滤或重排。
    * ``drawings`` 表示后续阶段使用的当前工作视图。
    * 所有阶段更新同一个权威上下文，不维护竞争状态副本。
    * 页面完成态提交后即可释放该上下文。

Define mutable state for one page's parsing pipeline.

``PageRunContext`` owns raw page objects, working views, shared per-page
caches, detector proposals, ownership results, reading-order metadata, and
final page artifacts. Document configuration and document-wide result lists
remain outside this context.

Key invariants:
    * ``raw_drawings`` is never filtered or reordered.
    * ``drawings`` is the current working view for downstream stages.
    * Each stage updates one authoritative context instead of local copies.
    * A page context is discarded after its completed state is committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageRunContext:
    """Own one page's inputs and intermediate structures during parsing.

    管理单页解析期间的输入对象、共享缓存、阶段结果与最终页面制品。

    ``raw_drawings`` is the untouched result of ``page.get_drawings()``.
    ``drawings`` is the working list after confirmed page-header/footer
    vectors have been removed.  Keeping both references prevents a later
    detector from accidentally treating a filtered list as the raw inventory.

    The context is the authoritative owner of shared page objects and can be
    passed directly to page stages without creating competing state copies.
    该对象是页面共享状态的唯一所有者，各阶段直接更新同一实例，避免局部副本失步。
    """

    page_index: int
    page_number: int
    page: Any
    page_rect: Any
    page_dict: dict[str, Any]
    text_lines: list[dict[str, Any]]
    source_images: list[dict[str, Any]]
    raw_drawings: list[dict[str, Any]]
    drawings: list[dict[str, Any]] | None = None

    page_inventory: Any | None = None
    inventory_audit_record: dict[str, Any] = field(default_factory=dict)
    header_footer_rects: list[Any] = field(default_factory=list)
    header_footer_record: dict[str, Any] = field(default_factory=dict)
    vector_exclusion_rects: list[Any] = field(default_factory=list)
    axis_geometry_index: dict[str, Any] | None = None
    logical_horizontal_rules: list[dict[str, Any]] = field(
        default_factory=list
    )
    logical_horizontal_rule_audit: list[dict[str, Any]] = field(
        default_factory=list
    )
    preclassified_rule_band_result: dict[str, Any] | None = None
    detected_tables: list[Any] = field(default_factory=list)
    table_detection_metadata: dict[str, Any] = field(default_factory=dict)
    table_claims: list[dict[str, Any]] = field(default_factory=list)
    dense_vector_page: bool = False
    table_detection_skipped: bool = False
    table_detection_skip_reason: str | None = None
    table_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )
    table_skeleton_drawing_ids: set[str] = field(
        default_factory=set
    )
    table_full_drawing_ids: set[str] = field(default_factory=set)
    precluster_object_exclusions: dict[str, Any] = field(
        default_factory=dict
    )
    precluster_protected_rects: list[Any] = field(default_factory=list)
    table_visual_protection_rects: list[Any] = field(
        default_factory=list
    )
    table_records: list[dict[str, Any]] = field(default_factory=list)
    table_markdown_elements: list[dict[str, Any]] = field(
        default_factory=list
    )
    excluded_table_rects: list[Any] = field(default_factory=list)
    embedded_table_count: int = 0
    auto_rotated_table_count: int = 0
    anomalous_table_claimed_count: int = 0
    anomalous_table_unclaimed_count: int = 0
    visual_region_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    visual_region_markdown_elements: list[dict[str, Any]] = field(
        default_factory=list
    )
    visual_region_rects: list[Any] = field(default_factory=list)
    visual_region_text_rects: list[Any] = field(default_factory=list)
    visual_region_claims: list[dict[str, Any]] = field(
        default_factory=list
    )
    visual_region_seconds: float = 0.0
    embedded_child_visual_count: int = 0
    source_image_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    source_image_markdown_elements: list[dict[str, Any]] = field(
        default_factory=list
    )
    source_image_text_rects: list[Any] = field(default_factory=list)
    source_image_claims: list[dict[str, Any]] = field(
        default_factory=list
    )
    source_image_seconds: float = 0.0
    formula_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    formula_markdown_elements: list[dict[str, Any]] = field(
        default_factory=list
    )
    formula_text_rects: list[Any] = field(default_factory=list)
    formula_seconds: float = 0.0
    preclassified_text_separator_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    preclassified_text_separator_markdown_elements: list[
        dict[str, Any]
    ] = field(default_factory=list)
    preclassified_text_separator_claims: list[
        dict[str, Any]
    ] = field(default_factory=list)
    text_separator_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    text_separator_markdown_elements: list[dict[str, Any]] = field(
        default_factory=list
    )
    text_separator_seconds: float = 0.0
    figure_caption_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    heading_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    heading_seconds: float = 0.0
    detector_claims: list[dict[str, Any]] = field(
        default_factory=list
    )
    composite_output_groups: list[dict[str, Any]] = field(
        default_factory=list
    )
    claim_resolution: dict[str, Any] = field(default_factory=dict)
    suppressed_claim_actions: list[dict[str, Any]] = field(
        default_factory=list
    )
    table_ownership_decisions: list[dict[str, Any]] = field(
        default_factory=list
    )
    suppressed_visual_indices: set[int] = field(default_factory=set)
    reading_order_metadata: dict[str, Any] = field(default_factory=dict)
    grouped_markdown: dict[str, str] = field(default_factory=dict)
    reading_order_debug_path: Any | None = None
    outline_page_name: dict[str, Any] | None = None
    markdown_page_title: str = ""
    markdown_lines: list[str] = field(default_factory=list)
    markdown_section: str = ""
    page_record: dict[str, Any] = field(default_factory=dict)
    page_total_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    stage_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.drawings is None:
            self.drawings = self.raw_drawings

    def record_timing(self, name: str, seconds: float) -> None:
        """Record one named stage duration without altering parser decisions."""

        self.timings[name] = float(seconds)

    def record_stage(self, name: str) -> None:
        """Append one completed page-pipeline stage for debug auditing."""

        self.stage_history.append(str(name))
