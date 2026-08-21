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


#@GPT 页面上下文只保存本页输入、共享缓存和阶段性检测结果；不持有跨页配置或最终文档列表，避免把页面状态重新写回全局变量。
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

    #@GPT 以下字段按阶段逐步填充；默认值保证空白页也能完整走完流水线，不需要检测器自己创建隐式缓存。
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
    #@GPT 表格候选、截图原子和正文去重框在同一页内会经历“候选→仲裁→重建”三次变更，因此由页面上下文统一持有，避免局部变量之间出现过期副本。
    table_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )
    table_skeleton_drawing_ids: set[str] = field(
        default_factory=set
    )
    table_full_drawing_ids: set[str] = field(default_factory=set)
    #@GPT 表格候选完成结构确认后，视觉阶段共享同一份前置对象排除决策和保护矩形；页面上下文持有它们以避免主循环重新计算bbox。
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
    #@GPT 视觉区域跨越预分类、归属仲裁和边界重建阶段；相关记录与Markdown原子由同一页面上下文持有，避免状态副本失步。
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
    #@GPT 原始图片的候选、归属、边界和文字框使用同一状态容器，保证仲裁后的记录与Markdown引用一致。
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
    #@GPT 公式阶段的记录、Markdown原子和精确文字框会共同进入分隔线检测、标题排除与阅读顺序；放入页面上下文后，开启和关闭公式检测都提供同一份稳定空值契约。
    formula_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    formula_markdown_elements: list[dict[str, Any]] = field(
        default_factory=list
    )
    formula_text_rects: list[Any] = field(default_factory=list)
    formula_seconds: float = 0.0
    #@GPT 纯文字分隔符在表格前预分类和阶段F后补检两处生成；上下文只保存跨阶段声明及最终输出，不保存函数内部临时候选。
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
    #@GPT 图题分类会原地标记PageRunContext.text_lines，并把精简关系记录同时交给文档级provenance和页面摘要；上下文只保存最终记录，不复制已由本对象持有的文字行。
    figure_caption_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    #@GPT 标题分类会原地更新本页text_lines，并把精简审计记录和阶段耗时交给页面摘要、文档provenance与性能报告；由PageRunContext统一保存可避免标记结果和审计结果来自不同调用。
    heading_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    heading_seconds: float = 0.0
    #@GPT 阶段C/D的权威状态只保存跨越候选生成、唯一归属、边界重建和最终审计的对象；各输出类型独立返回的临时suppression列表仍留在阶段内部，避免上下文同时保存分支态与合并态。
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
    #@GPT bbox关系审计属于页面完成态，由页面上下文持有并统一写入最终record。
    table_ownership_decisions: list[dict[str, Any]] = field(
        default_factory=list
    )
    suppressed_visual_indices: set[int] = field(default_factory=set)
    #@GPT 阅读顺序阶段的输入原子列表只在一次函数调用中使用，不进入上下文；跨越排序、Markdown拼接、页面摘要和调试输出的结果才由PageRunContext持有。
    reading_order_metadata: dict[str, Any] = field(default_factory=dict)
    grouped_markdown: dict[str, str] = field(default_factory=dict)
    reading_order_debug_path: Any | None = None
    #@GPT 页面输出从目录标题初始化后会持续追加页眉、提示和排序正文；最终Markdown section、页面provenance和总耗时必须与这些行保持同一生命周期，因此由PageRunContext统一持有，文档结果只接收完成态。
    outline_page_name: dict[str, Any] | None = None
    markdown_page_title: str = ""
    markdown_lines: list[str] = field(default_factory=list)
    markdown_section: str = ""
    page_record: dict[str, Any] = field(default_factory=dict)
    page_total_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    #@GPT 只记录页面阶段完成顺序，供debug契约检查确认阶段没有被跳过或交换；该台账不参与任何检测、排序或Markdown输出。
    stage_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        #@GPT 未传入working drawings时直接沿用原始列表，避免复制大型drawing集合并保持对象身份与顺序。
        if self.drawings is None:
            self.drawings = self.raw_drawings

    def record_timing(self, name: str, seconds: float) -> None:
        """Record one named stage duration without altering parser decisions."""

        self.timings[name] = float(seconds)

    #@GPT 阶段入口由主流程集中登记，避免每个检测器自行维护一套顺序状态；重复名称保留在列表中，由契约校验报告而不是静默覆盖。
    def record_stage(self, name: str) -> None:
        """Append one completed page-pipeline stage for debug auditing."""

        self.stage_history.append(str(name))
