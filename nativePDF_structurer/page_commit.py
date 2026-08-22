"""将已完成页面提交到文档级结果。

本模块把Markdown、来源信息、检测记录和可选契约诊断从
``PageRunContext`` 提交到 ``ParseResult``，不重新运行检测器，也不推导
新的页面几何。

Commit one completed page context to the document result.

Transfers finalized Markdown, provenance, detector records, and optional
contract diagnostics from ``PageRunContext`` into ``ParseResult``. The commit
step does not rerun detectors or derive new page geometry.
"""

from __future__ import annotations

from .page_context import PageRunContext
from .page_contract import validate_page_context
from .result import ParseResult


def commit_page_context(
    result: ParseResult,
    context: PageRunContext,
    *,
    validate: bool = False,
) -> None:
    """Append every persistent page artifact without changing detector data."""

    if validate:
        context.page_record["page_contract_warnings"] = (
            validate_page_context(context)
        )
        context.page_record["page_contract_stage_history"] = list(
            context.stage_history
        )

    result.inventory_audit.append(context.inventory_audit_record)
    result.header_footer_pages.append(
        {
            "page_number": context.page_number,
            **context.header_footer_record,
            "detection_seconds": round(
                context.timings.get("header_footer_detection", 0.0),
                6,
            ),
        }
    )
    result.claim_resolution_actions.append(
        {
            "page_number": context.page_number,
            "suppressed_output_count": len(
                context.suppressed_claim_actions
            ),
            "suppressed_outputs": context.suppressed_claim_actions,
        }
    )
    result.logical_horizontal_rules.extend(
        context.logical_horizontal_rule_audit
    )
    result.figure_captions.extend(
        {
            "page_number": context.page_number,
            **record,
        }
        for record in context.figure_caption_records
    )

    result.tables.extend(context.table_records)
    result.visual_regions.extend(context.visual_region_records)
    result.source_images.extend(context.source_image_records)
    result.detector_claims.append(
        {
            "page_number": context.page_number,
            "claims": context.detector_claims,
            "table_detection_claims": context.table_claims,
            "visual_region_claims": context.visual_region_claims,
            "source_image_claims": context.source_image_claims,
            "output_groups": context.composite_output_groups,
            "resolution_preview": context.claim_resolution,
            "resolution_actions": context.suppressed_claim_actions,
        }
    )
    result.claim_resolutions.append(
        {
            "page_number": context.page_number,
            "claims": context.detector_claims,
            "output_groups": context.composite_output_groups,
            "resolution_preview": context.claim_resolution,
            "resolution_actions": context.suppressed_claim_actions,
        }
    )
    result.composite_output_groups.extend(
        context.composite_output_groups
    )
    result.formula_regions.extend(context.formula_records)
    result.headings.extend(context.heading_records)
    result.pages.append(context.page_record)
    result.markdown_sections.append(context.markdown_section)
