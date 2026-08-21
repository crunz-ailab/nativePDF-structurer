"""定义单个已解析文档的内存结果模型。

``ParseResult`` 统一持有Markdown、
页面记录、检测输出、来源信息、摘要和紧凑IR；打开的PDF、缓存和空间索引
等运行期资源不属于结果对象。

Define the in-memory result model for one parsed document.

Gives Markdown sections, page records, detector outputs, provenance, summary,
and compact IR a single document-level owner. Runtime resources such as open
documents, caches, and spatial indexes are excluded from the result model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Record = dict[str, Any]


@dataclass
class ParseResult:
    """Accumulate all persistent outputs produced while parsing one document.

    汇总单个文档解析过程中产生的全部持久化结果。

    A new instance must be created for every input PDF.  Every list uses its
    own ``default_factory`` so two parser runs cannot accidentally share
    mutable records.  Runtime-only state such as open documents, drawing
    caches, timing counters, spatial indexes, and output directories is
    intentionally excluded.

    每个输入PDF都必须创建独立实例；所有可变集合使用独立的
    ``default_factory``，防止不同解析任务共享状态。打开的文档、空间索引和
    输出目录等运行期对象不属于该结果模型。

    The fields fall into three groups:

    - ``markdown_sections`` and ``pages`` describe the final reading stream;
    - table, image, formula, heading, and layout lists describe emitted
      document structures;
    - inventory, claim, resolution, and logical-rule lists preserve
      provenance needed for debugging or later VLM escalation.

    ``summary`` and ``document_ir`` are assigned only after all pages have
    completed.  They remain dictionaries because the summary is an internal
    report and the IR has its own explicit schema version.

    ``summary`` 与 ``document_ir`` 在全部页面完成后写入；IR通过独立的schema
    版本管理其公开结构。
    """

    markdown_sections: list[str] = field(default_factory=list)
    pages: list[Record] = field(default_factory=list)
    tables: list[Record] = field(default_factory=list)
    source_images: list[Record] = field(default_factory=list)
    visual_regions: list[Record] = field(default_factory=list)
    formula_regions: list[Record] = field(default_factory=list)
    headings: list[Record] = field(default_factory=list)
    figure_captions: list[Record] = field(default_factory=list)
    header_footer_pages: list[Record] = field(default_factory=list)
    inventory_audit: list[Record] = field(default_factory=list)
    detector_claims: list[Record] = field(default_factory=list)
    claim_resolutions: list[Record] = field(default_factory=list)
    claim_resolution_actions: list[Record] = field(default_factory=list)
    logical_horizontal_rules: list[Record] = field(default_factory=list)
    composite_output_groups: list[Record] = field(default_factory=list)
    summary: Record = field(default_factory=dict)
    document_ir: Record = field(default_factory=dict)
