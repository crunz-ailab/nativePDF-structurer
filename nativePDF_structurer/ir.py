"""构建面向检索和来源追踪的紧凑中间表示。

模块从 ``ParseResult`` 中选择稳定
的页面、资源、Claim和provenance字段，并将生成资源路径转换为运行目录
相对路径；检测器临时状态和原始PDF载荷不会进入IR。

Build the compact intermediate representation for retrieval.

Selects stable page, asset, Claim, and provenance fields from ``ParseResult``
and converts generated asset paths to run-relative form. Detector scratch
state and raw PDF payloads are intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


IR_SCHEMA_VERSION = "0.1"


def _bbox(value: Any) -> list[float] | None:
    """Normalize a rectangle-like value without importing a PDF geometry type."""

    if value is None:
        return None
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) != 4:
        return None
    try:
        return [round(float(item), 3) for item in values]
    except (TypeError, ValueError):
        return None


def _relative_asset_path(value: Any, output_dir: Path) -> str | None:
    """Store generated asset paths relative to the run directory when possible."""

    if not value:
        return None
    raw = Path(str(value))
    try:
        return raw.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(value)


def _first_path(record: dict[str, Any]) -> Any:
    """Find the primary generated asset path across detector record variants."""

    for key in ("output_path", "image_path", "path"):
        if record.get(key):
            return record[key]
    artifacts = record.get("output_artifacts") or []
    if artifacts and isinstance(artifacts[0], dict):
        return artifacts[0].get("path")
    return None


def _asset_record(
    record: dict[str, Any],
    *,
    kind: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Convert one image/table/formula record to a stable asset entry."""

    claim_id = record.get("claim_id") or record.get("output_element_id")
    return {
        "asset_id": claim_id,
        "kind": kind,
        "page_number": record.get("page_number"),
        "path": _relative_asset_path(_first_path(record), output_dir),
        "csv_path": _relative_asset_path(
            record.get("csv_path"), output_dir
        ),
        "bbox": _bbox(
            record.get("output_bbox")
            or record.get("render_bbox")
            or record.get("bbox")
        ),
        "render_bbox": _bbox(record.get("render_bbox")),
        "sort_bbox": _bbox(record.get("sort_bbox")),
        "mode": record.get("mode"),
        "reason": record.get("reason") or record.get("reason_codes") or [],
        "source_element_ids": list(
            record.get("source_element_ids")
            or record.get("feature_element_ids")
            or []
        ),
        "text_element_ids": list(
            record.get("stage_f_text_element_ids")
            or record.get("native_text_element_ids")
            or []
        ),
    }


def _layout_record(page: dict[str, Any]) -> dict[str, Any]:
    """Keep layout facts useful for ordering consumers and omit debug geometry noise."""

    groups = []
    text_reflow = page.get("text_reflow") or {}
    for direction in text_reflow.get("direction_groups") or []:
        gaptree = direction.get("gaptree") or {}
        groups.append(
            {
                "angle": direction.get("angle"),
                "label": direction.get("label"),
                "text_line_count": direction.get("text_line_count", 0),
                "image_element_count": direction.get(
                    "image_element_count", 0
                ),
                "character_count": direction.get("character_count", 0),
                "gaptree": {
                    "backend": gaptree.get("backend", "gaptree"),
                    "row_count": gaptree.get("row_count", 0),
                    "cut_count": gaptree.get("cut_count", 0),
                    "node_count": gaptree.get("node_count", 0),
                    "cuts": gaptree.get("cuts", []),
                    "nodes": gaptree.get("nodes", []),
                },
            }
        )
    return {
        "direction_groups": groups,
        "heading_count": page.get("heading_detection", {}).get(
            "heading_count", 0
        ),
        "text_separator_count": page.get("text_separator_count", 0),
    }


def _compact_claim(
    claim: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Retain claim ownership facts without detector-specific metadata blobs."""

    return {
        "claim_id": claim.get("claim_id"),
        "detector": claim.get("detector"),
        "output_element_id": claim.get("output_element_id"),
        "confidence": claim.get("confidence"),
        "input_element_ids": list(claim.get("input_element_ids") or []),
        "claimed_element_ids": list(
            claim.get("claimed_element_ids") or []
        ),
        "feature_element_ids": list(
            claim.get("feature_element_ids") or []
        ),
        "bbox": _bbox(claim.get("bbox")),
        "reason_codes": list(claim.get("reason_codes") or []),
        "output_artifacts": [
            {
                "type": artifact.get("type"),
                "path": _relative_asset_path(
                    artifact.get("path"), output_dir
                ),
                "bbox": _bbox(artifact.get("bbox")),
            }
            for artifact in claim.get("output_artifacts") or []
            if isinstance(artifact, dict)
        ],
    }


def _page_provenance(
    record: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one compact page-level ownership and conflict record."""

    preview = record.get("resolution_preview") or {}
    return {
        "claims": [
            _compact_claim(claim, output_dir=output_dir)
            for claim in record.get("claims") or []
            if isinstance(claim, dict)
        ],
        "owner_by_element_id": dict(
            preview.get("owner_by_element_id") or {}
        ),
        "conflict_element_count": preview.get(
            "conflict_element_count", 0
        ),
        "invalid_element_reference_count": preview.get(
            "invalid_element_reference_count", 0
        ),
        "conflicts": preview.get("conflicts") or [],
    }


def build_document_ir(
    *,
    source_pdf: Path,
    output_dir: Path,
    source_page_count: int,
    summary: dict[str, Any],
    pages: Iterable[dict[str, Any]],
    tables: Iterable[dict[str, Any]],
    source_images: Iterable[dict[str, Any]],
    visual_regions: Iterable[dict[str, Any]],
    formula_regions: Iterable[dict[str, Any]],
    detector_claims: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Create the stable IR document emitted in debug mode."""

    page_records = list(pages)
    table_records = list(tables)
    source_image_records = list(source_images)
    visual_records = list(visual_regions)
    formula_records = list(formula_regions)
    claim_records = list(detector_claims)

    assets = [
        _asset_record(record, kind="table", output_dir=output_dir)
        for record in table_records
    ]
    assets.extend(
        _asset_record(record, kind="source_image", output_dir=output_dir)
        for record in source_image_records
    )
    assets.extend(
        _asset_record(record, kind="visual_region", output_dir=output_dir)
        for record in visual_records
    )
    assets.extend(
        _asset_record(record, kind="formula", output_dir=output_dir)
        for record in formula_records
    )

    provenance_by_page = {
        record.get("page_number"): _page_provenance(
            record,
            output_dir=output_dir,
        )
        for record in claim_records
        if record.get("page_number") is not None
    }

    ir_pages = []
    for page in page_records:
        page_number = page.get("page_number")
        ir_pages.append(
            {
                "page_number": page_number,
                "title": page.get("markdown_page_title"),
                "outline": page.get("outline_page_name"),
                "rotation": page.get("page_rotation_metadata", 0),
                "layout": _layout_record(page),
                "asset_ids": [
                    asset["asset_id"]
                    for asset in assets
                    if asset.get("page_number") == page_number
                    and asset.get("asset_id")
                ],
                "provenance": provenance_by_page.get(
                    page_number,
                    {
                        "claims": [],
                        "owner_by_element_id": {},
                        "conflict_element_count": 0,
                        "invalid_element_reference_count": 0,
                        "conflicts": [],
                    },
                ),
            }
        )

    return {
        "schema": "nativePDF-structurer/document-ir",
        "schema_version": IR_SCHEMA_VERSION,
        "source": {
            "pdf": str(source_pdf),
            "page_count": source_page_count,
        },
        "markdown": {
            "path": "document_with_reflowed_text_and_tables.md",
            "page_count": len(ir_pages),
        },
        "summary": {
            "processed_pages": summary.get("processed_pages", 0),
            "processed_page_numbers": summary.get(
                "processed_page_numbers", []
            ),
            "detected_tables": summary.get("detected_tables", 0),
            "visual_regions": summary.get(
                "visual_region_policy", {}
            ).get("region_count", 0),
            "source_images": len(source_image_records),
            "formula_regions": len(formula_records),
        },
        "pages": ir_pages,
        "assets": assets,
    }
