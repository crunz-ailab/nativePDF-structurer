"""校验已完成页面的流水线一致性契约。

本模块检查阶段顺序、Claim引用、输出边界、制品路径和来源关联。校验只报告
警告，不改变检测器决策，也不自动修复状态。

Validate consistency contracts for a completed page pipeline.

Checks stage order, Claim references, output boundaries, artifact paths, and
provenance links after a page is processed. Validation reports warnings but
never changes detector decisions or repairs page state.
"""

from __future__ import annotations

import math
from pathlib import Path

from .page_context import PageRunContext


class PageStage:
    """Stable names for the externally auditable page-pipeline stages."""

    INITIALIZATION = "initialization"
    HEADER_FOOTER = "header_footer"
    GEOMETRY = "geometry"
    RULE_BANDS = "rule_bands"
    TABLE_DETECTION = "table_detection"
    TABLE_CANDIDATES = "table_candidates"
    TABLE_FINALIZATION = "table_finalization"
    VISUAL_PROTECTION = "visual_protection"
    VISUAL_DETECTION = "visual_detection"
    OUTPUT_PROPOSAL = "output_proposal"
    SOURCE_IMAGES = "source_images"
    CLAIM_RESOLUTION = "claim_resolution"
    BOUNDARY_OWNERSHIP = "boundary_ownership"
    SEMANTICS = "semantics"
    READING_ORDER = "reading_order"
    FINALIZATION = "finalization"


EXPECTED_PAGE_STAGE_HISTORY = (
    PageStage.INITIALIZATION,
    PageStage.HEADER_FOOTER,
    PageStage.GEOMETRY,
    PageStage.RULE_BANDS,
    PageStage.TABLE_DETECTION,
    PageStage.TABLE_CANDIDATES,
    PageStage.TABLE_FINALIZATION,
    PageStage.VISUAL_PROTECTION,
    PageStage.VISUAL_DETECTION,
    PageStage.OUTPUT_PROPOSAL,
    PageStage.SOURCE_IMAGES,
    PageStage.CLAIM_RESOLUTION,
    PageStage.BOUNDARY_OWNERSHIP,
    PageStage.SEMANTICS,
    PageStage.READING_ORDER,
    PageStage.FINALIZATION,
)


def _claim_reference_warnings(
    context: PageRunContext,
    claim_ids: set[str],
) -> list[str]:
    """Return warnings for invalid Claim references and ownership links."""

    warnings: list[str] = []
    seen_claim_ids: set[str] = set()
    inventory = context.page_inventory
    inventory_ids = (
        set(inventory.entries)
        if inventory is not None and hasattr(inventory, "entries")
        else None
    )

    for claim in context.detector_claims:
        claim_id = claim.get("claim_id")
        if claim_id is None:
            warnings.append("claim.missing_claim_id")
            continue
        if claim_id in seen_claim_ids:
            warnings.append(f"claim.duplicate_claim_id:{claim_id}")
        seen_claim_ids.add(claim_id)
        if claim.get("output_element_id") not in (None, claim_id):
            warnings.append(f"claim.output_id_mismatch:{claim_id}")

        claimed_ids = set(claim.get("claimed_element_ids") or [])
        feature_ids = set(claim.get("feature_element_ids") or [])
        for field_name in (
            "input_element_ids",
            "claimed_element_ids",
            "feature_element_ids",
        ):
            for element_id in claim.get(field_name) or []:
                if inventory_ids is not None and element_id not in inventory_ids:
                    warnings.append(
                        f"claim.{field_name}_missing:{claim_id}:{element_id}"
                    )
        for element_id in sorted(feature_ids - claimed_ids):
            warnings.append(
                f"claim.feature_not_claimed:{claim_id}:{element_id}"
            )

    if inventory_ids is not None:
        for element_id, entry in inventory.entries.items():
            owner = entry.get("owner")
            if owner is not None and owner not in claim_ids:
                warnings.append(
                    f"inventory.owner_unknown_claim:{element_id}:{owner}"
                )

    owner_by_element_id = context.claim_resolution.get(
        "owner_by_element_id", {}
    )
    for element_id, owner in owner_by_element_id.items():
        if inventory_ids is not None and element_id not in inventory_ids:
            warnings.append(
                f"resolution.element_missing:{element_id}"
            )
        if owner is not None and owner not in claim_ids:
            warnings.append(
                f"resolution.owner_unknown_claim:{element_id}:{owner}"
            )
    return warnings


def _coerce_bbox(value: object) -> tuple[float, float, float, float] | None:
    """Convert a Rect-like value to finite numeric coordinates."""

    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        return None
    return values  # type: ignore[return-value]


def _output_boundary_warnings(
    context: PageRunContext,
) -> list[str]:
    """Return warnings for malformed or out-of-page output atom bounds."""

    warnings: list[str] = []
    page_bbox = _coerce_bbox(context.page_rect)
    output_groups = (
        ("table", context.table_markdown_elements),
        ("visual", context.visual_region_markdown_elements),
        ("source_image", context.source_image_markdown_elements),
        ("formula", context.formula_markdown_elements),
        ("separator", context.text_separator_markdown_elements),
    )
    for output_name, elements in output_groups:
        for index, element in enumerate(elements):
            for field_name in ("bbox", "render_bbox", "sort_bbox"):
                if field_name not in element:
                    warnings.append(
                        f"{output_name}.element_bbox_missing:{index}:{field_name}"
                    )
                    continue
                bbox = _coerce_bbox(element[field_name])
                if bbox is None:
                    warnings.append(
                        f"{output_name}.element_bbox_invalid:{index}:{field_name}"
                    )
                    continue
                x0, y0, x1, y1 = bbox
                if x1 < x0 or y1 < y0:
                    warnings.append(
                        f"{output_name}.element_bbox_reversed:{index}:{field_name}"
                    )
                    continue
                if page_bbox is None:
                    continue
                page_x0, page_y0, page_x1, page_y1 = page_bbox
                if (
                    x0 < page_x0 - 1e-3
                    or y0 < page_y0 - 1e-3
                    or x1 > page_x1 + 1e-3
                    or y1 > page_y1 + 1e-3
                ):
                    warnings.append(
                        f"{output_name}.element_bbox_outside_page:{index}:{field_name}"
                    )
    return warnings


def _output_artifact_warnings(
    context: PageRunContext,
) -> list[str]:
    """Return warnings for missing, duplicate, or untracked output files."""

    warnings: list[str] = []
    seen_paths: dict[str, tuple[str, int]] = {}
    output_groups = (
        (
            "table",
            context.table_markdown_elements,
            context.table_records,
            "image_path",
        ),
        (
            "visual",
            context.visual_region_markdown_elements,
            context.visual_region_records,
            "output_path",
        ),
        (
            "source_image",
            context.source_image_markdown_elements,
            context.source_image_records,
            "output_path",
        ),
        (
            "formula",
            context.formula_markdown_elements,
            context.formula_records,
            "output_path",
        ),
    )
    for output_name, elements, records, record_path_field in output_groups:
        record_claim_ids = {
            record.get("claim_id")
            for record in records
            if record.get("claim_id") is not None
        }
        record_paths = {
            str(record.get(record_path_field))
            for record in records
            if record.get(record_path_field)
        }
        for index, element in enumerate(elements):
            claim_id = element.get("claim_id")
            if claim_id is not None and claim_id not in record_claim_ids:
                warnings.append(
                    f"{output_name}.markdown_record_claim_missing:{claim_id}"
                )
            output_path = element.get("output_path")
            if output_path is None:
                continue
            try:
                path = Path(output_path)
            except TypeError:
                warnings.append(
                    f"{output_name}.artifact_path_invalid:{index}"
                )
                continue
            path_key = str(path)
            if path_key in seen_paths:
                first_name, first_index = seen_paths[path_key]
                warnings.append(
                    f"{output_name}.artifact_path_duplicate:{index}:"
                    f"{first_name}:{first_index}"
                )
            else:
                seen_paths[path_key] = (output_name, index)
            if path_key not in record_paths:
                warnings.append(
                    f"{output_name}.artifact_record_missing:{index}:{path_key}"
                )
            if not path.is_file():
                warnings.append(
                    f"{output_name}.artifact_file_missing:{index}:{path_key}"
                )
    return warnings


def validate_page_context(context: PageRunContext) -> list[str]:
    """Return warnings for inconsistent completed page state."""

    warnings: list[str] = []
    if context.stage_history and tuple(context.stage_history) != EXPECTED_PAGE_STAGE_HISTORY:
        warnings.append("stage_history.mismatch")
    if context.page_record:
        if context.page_record.get("page_number") != context.page_number:
            warnings.append("page_record.page_number_mismatch")
    if context.markdown_section != "\n".join(context.markdown_lines):
        warnings.append("markdown_section_mismatch")

    claim_ids = {
        claim.get("claim_id")
        for claim in context.detector_claims
        if claim.get("claim_id") is not None
    }
    warnings.extend(_claim_reference_warnings(context, claim_ids))
    warnings.extend(_output_boundary_warnings(context))
    warnings.extend(_output_artifact_warnings(context))
    for output_name, elements in (
        ("table", context.table_markdown_elements),
        ("visual", context.visual_region_markdown_elements),
        ("source_image", context.source_image_markdown_elements),
        ("formula", context.formula_markdown_elements),
        ("separator", context.text_separator_markdown_elements),
    ):
        for element in elements:
            claim_id = element.get("claim_id")
            if claim_id is not None and claim_id not in claim_ids:
                warnings.append(
                    f"{output_name}.markdown_claim_missing:{claim_id}"
                )

    candidate_count = len(context.table_candidates)
    for decision in context.table_ownership_decisions:
        table_index = decision.get("table_index")
        if not isinstance(table_index, int) or not (
            0 <= table_index < candidate_count
        ):
            warnings.append("table_ownership.table_index_out_of_range")
    return warnings
