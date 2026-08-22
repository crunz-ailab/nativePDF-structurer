"""在提交前构建页面完成态。

本模块生成稳定的页面provenance记录，拼接页面Markdown，并记录最终耗时和
输出计数；它不修改检测归属或阅读顺序。

Finalize a processed page before committing it to ``ParseResult``.

Builds the stable page provenance record, joins the page Markdown section,
and records final timing and output counts from ``PageRunContext``. It does not
modify detector ownership or reading order.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .page_context import PageRunContext


@dataclass
class PageFinalizationResult:
    """The persistent page artifacts produced by finalization."""

    page_record: dict[str, Any]
    markdown_section: str
    total_seconds: float


def finalize_page_context(
    context: PageRunContext,
    *,
    page_started: float,
) -> PageFinalizationResult:
    """Build page provenance and the final Markdown section."""

    context.page_total_seconds = time.perf_counter() - page_started
    precluster = context.precluster_object_exclusions
    table_candidates = context.table_candidates
    table_decisions = context.table_ownership_decisions
    context.page_record = {
        "page_number": context.page_number,
        "markdown_page_title": context.markdown_page_title,
        "outline_page_name": context.outline_page_name,
        "page_rotation_metadata": context.page.rotation,
        "header_footer_detection": context.header_footer_record,
        "detected_table_count": len(context.detected_tables),
        "table_candidate_count": len(table_candidates),
        "logical_table_container_count": sum(
            candidate.get("logical_table_container", False)
            for candidate in table_candidates
        ),
        "table_skeleton_drawing_count": len(
            context.table_skeleton_drawing_ids
        ),
        "embedded_child_visual_count": context.embedded_child_visual_count,
        "composite_output_group_count": len(
            context.composite_output_groups
        ),
        "active_composite_output_group_count": sum(
            group.get("active", False)
            for group in context.composite_output_groups
        ),
        "precluster_object_overlap": {
            "protected_rect_count": len(context.precluster_protected_rects),
            "protected_table_numbers": precluster[
                "protected_table_numbers"
            ],
            "protected_image_group_indices": precluster[
                "protected_image_group_indices"
            ],
            "table_decisions": [
                {
                    "table_number": record["table_number"],
                    "protected": record["precluster_protected"],
                    "reason": record["precluster_reason"],
                    "external_drawing_count": record[
                        "vector_overlap"
                    ]["external_drawing_count"],
                    "contained_by_source_cluster": record[
                        "contained_by_source_cluster"
                    ],
                    "source_cluster_bbox": record["source_cluster_bbox"],
                    "overlaps_table_numbers": record[
                        "overlaps_table_numbers"
                    ],
                    "overlaps_image_groups": record[
                        "overlaps_image_groups"
                    ],
                }
                for record in precluster["tables"]
            ],
        },
        "table_candidate_claimed_by_visual_count": sum(
            decision["claimed_by_visual_region"]
            for decision in table_decisions
        ),
        "visual_candidate_suppressed_by_table_count": len(
            context.suppressed_visual_indices
        ),
        "table_visual_ownership": [
            {
                "table_number": table_candidates[
                    decision["table_index"]
                ]["table_number"],
                "claimed_by_visual_region": decision[
                    "claimed_by_visual_region"
                ],
                "visual_region_number": (
                    decision["visual_region_index"] + 1
                    if decision["visual_region_index"] is not None
                    else None
                ),
                "table_overlap_ratio": round(
                    decision["table_overlap_ratio"], 6
                ),
                "visual_overlap_ratio": round(
                    decision["visual_overlap_ratio"], 6
                ),
            }
            for decision in table_decisions
        ],
        "anomalous_single_value_table_count": len(
            [
                candidate
                for candidate in table_candidates
                if candidate["is_anomalous_single_value_table"]
            ]
        ),
        "anomalous_single_value_table_claimed_count": (
            context.anomalous_table_claimed_count
        ),
        "anomalous_single_value_table_unclaimed_count": (
            context.anomalous_table_unclaimed_count
        ),
        "table_detection_gate": {
            "dense_vector_page": context.dense_vector_page,
            "drawing_count": len(
                context.preclassified_rule_band_result[
                    "table_detection_drawings"
                ]
            ),
            "skipped_native_detection": context.table_detection_skipped,
            "skip_reason": context.table_detection_skip_reason,
            "geometry_fallback_candidate_count": sum(
                getattr(table, "detection_source", "pymupdf_native")
                != "pymupdf_native"
                for table in context.detected_tables
            ),
            **context.table_detection_metadata,
            "axis_geometry_index": context.axis_geometry_index["metadata"],
            "protective_table_rect_count": 0,
        },
        "preclassified_rule_bands": (
            context.preclassified_rule_band_result["metadata"]
        ),
        "embedded_table_image_count": context.embedded_table_count,
        "auto_rotated_table_image_count": (
            context.auto_rotated_table_count
        ),
        "visual_region_count": len(context.visual_region_records),
        "formula_region_count": len(context.formula_records),
        "text_separator_count": len(
            context.text_separator_markdown_elements
        ),
        "text_separator_records": context.text_separator_records,
        "source_image_object_count": len(context.source_image_records),
        "embedded_source_image_count": len(
            context.source_image_markdown_elements
        ),
        "source_image_modes": dict(
            Counter(record["mode"] for record in context.source_image_records)
        ),
        "heading_detection": {
            "heading_count": len(context.heading_records),
            "headings": context.heading_records,
        },
        "figure_caption_detection": {
            "figure_caption_count": len(context.figure_caption_records),
            "figure_captions": context.figure_caption_records,
        },
        "text_reflow": context.reading_order_metadata,
        "gaptree_debug_image": str(context.reading_order_debug_path),
        "timing_seconds": {
            "drawing_extraction": round(
                context.timings.get("drawing_extraction", 0.0), 6
            ),
            "table_detection": round(
                context.timings.get("table_detection", 0.0), 6
            ),
            "table_detection_gate": round(
                context.timings.get("table_detection_gate", 0.0), 6
            ),
            "page_total": round(context.page_total_seconds, 6),
            "source_images": round(context.source_image_seconds, 6),
            "visual_regions": round(context.visual_region_seconds, 6),
            "formula_regions": round(context.formula_seconds, 6),
            "heading_detection": round(context.heading_seconds, 6),
            "text_separators": round(context.text_separator_seconds, 6),
            "preclassified_rule_bands": round(
                context.timings.get("preclassified_rule_bands", 0.0), 6
            ),
            "header_footer_detection": round(
                context.timings.get("header_footer_detection", 0.0), 6
            ),
        },
    }
    context.markdown_section = "\n".join(context.markdown_lines)
    return PageFinalizationResult(
        page_record=context.page_record,
        markdown_section=context.markdown_section,
        total_seconds=context.page_total_seconds,
    )
