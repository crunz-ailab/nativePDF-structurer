"""在表格检测前预分类成对水平规则线。

模块将相邻且跨度相似的横线分流为纯
文字分隔符、视觉说明带，或继续保留给表格检测的结构；竖线覆盖、重复跨度、
图片、文字和非规则矢量共同提供判定证据。

关键不变量:
    预分类只路由候选，不修改原始drawing台账；确认属于表格骨架的规则线不会
    被再次作为视觉说明带输出。

Pre-route paired horizontal-rule bands before table detection.

Classifies adjacent, similarly spanned horizontal rules as text separators,
visual instruction bands, or structures that must remain available to table
detection. Vertical coverage, repeated spans, images, text, and non-rule
drawings provide the structural evidence.
"""

from __future__ import annotations

from collections.abc import Callable

import fitz

from .geometry import rect_area, rect_center
from .inventory import PageInventory

def detect_preclassified_rule_bands(
    page_rect: fitz.Rect,
    drawings: list[dict],
    text_lines: list[dict],
    source_images: list[dict],
    page_inventory: PageInventory,
    logical_horizontal_rules: list[dict],
    axis_geometry_index: dict,
    *,
    drawing_is_stroked_preclassified_horizontal_rule: Callable[..., bool],
    rule_element_ids: Callable[[dict], set[str]],
    merge_collinear_vertical_rules: Callable[[list[dict]], list[dict]],
    group_horizontal_rules_by_span: Callable[[list[dict]], list[list[dict]]],
    unique_rule_positions: Callable[[list[dict]], list[float]],
    horizontal_pair_repeats_in_another_span: Callable[..., bool],
    image_is_large_enough: Callable[..., bool],
    meaningful_overlap: Callable[[fitz.Rect, fitz.Rect], bool],
    vertical_rule_connects_horizontal_pair: Callable[..., bool],
    horizontal_pair_has_boundary_grid_evidence: Callable[..., dict],
    minimum_x_overlap_ratio: float,
    alignment_tolerance_points: float,
    minimum_vertical_coverage_ratio: float,
    minimum_image_area_points_squared: float,
    vector_connection_tolerance_points: float,
) -> dict:
    """在表格检测前分流双水平规则说明带。"""
    horizontal_rules: list[dict] = []
    rule_by_element_id: dict[str, dict] = {}
    rejected_filled_rule_count = 0

    for rule in logical_horizontal_rules:
        source_ids = sorted(rule_element_ids(rule))
        source_drawings = [
            page_inventory.payload_of(element_id)
            for element_id in source_ids
            if element_id in page_inventory.entries
            and page_inventory.entry(element_id)["kind"] == "drawing"
        ]
        if not source_drawings or not all(
            drawing_is_stroked_preclassified_horizontal_rule(
                drawing,
                page_rect,
            )
            for drawing in source_drawings
        ):
            rejected_filled_rule_count += 1
            continue
        rule_record = {
            **rule,
            "drawing": None,
            "element_ids": sorted(rule_element_ids(rule)),
        }
        horizontal_rules.append(rule_record)
        for element_id in rule_record["element_ids"]:
            rule_by_element_id[element_id] = rule_record

    if len(horizontal_rules) < 2:
        return {
            "bands": [],
            "table_detection_drawings": list(drawings),
            "preclassified_rule_drawing_ids": [],
            "metadata": {
                "candidate_group_count": 0,
                "accepted_band_count": 0,
                "table_detection_drawing_count": len(drawings),
                "rejected_filled_rule_count": rejected_filled_rule_count,
                "reason": "fewer_than_two_horizontal_rules",
            },
        }

    allowed_drawing_ids = {
        element_id
        for drawing in drawings
        for element_id in [
            page_inventory.id_for_payload(drawing, "drawing")
        ]
        if element_id is not None
    }
    vertical_rules = [
        rule
        for element_id, rule in axis_geometry_index[
            "legacy_axis_rule_by_element_id"
        ].items()
        if element_id in allowed_drawing_ids
        and rule is not None
        and rule["orientation"] == "v"
    ]
    merged_vertical_rules = merge_collinear_vertical_rules(
        vertical_rules
    )
    horizontal_rule_groups = group_horizontal_rules_by_span(
        horizontal_rules
    )
    bands: list[dict] = []
    preclassified_drawing_ids: set[str] = set()
    tolerance = alignment_tolerance_points
    repeated_pair_rejection_count = 0
    connected_vertical_rejection_count = 0
    boundary_grid_rejection_count = 0

    for group in horizontal_rule_groups:
        positions = unique_rule_positions(group)
        if len(positions) < 2:
            continue

        for top, bottom in zip(positions, positions[1:]):
            if bottom - top <= tolerance * 2:
                continue

            top_rules = [
                rule
                for rule in group
                if abs(rule["position"] - top) <= tolerance
            ]
            bottom_rules = [
                rule
                for rule in group
                if abs(rule["position"] - bottom) <= tolerance
            ]
            pair_rules = [*top_rules, *bottom_rules]
            if not pair_rules:
                continue

            band_x0 = min(rule["rect"].x0 for rule in pair_rules)
            band_x1 = max(rule["rect"].x1 for rule in pair_rules)
            overlap_x0 = max(rule["rect"].x0 for rule in pair_rules)
            overlap_x1 = min(rule["rect"].x1 for rule in pair_rules)
            overlap_width = max(0.0, overlap_x1 - overlap_x0)
            shorter_width = min(
                max(0.0, rule["rect"].width)
                for rule in pair_rules
            )
            if (
                shorter_width <= 0
                or overlap_width / shorter_width
                < minimum_x_overlap_ratio
            ):
                continue

            band_rect = fitz.Rect(
                band_x0,
                top,
                band_x1,
                bottom,
            ).normalize()
            if rect_area(band_rect) <= 0:
                continue

            if horizontal_pair_repeats_in_another_span(
                horizontal_rule_groups=horizontal_rule_groups,
                current_group=group,
                top=top,
                bottom=bottom,
                tolerance=tolerance,
            ):
                repeated_pair_rejection_count += 1
                continue

            pair_rule_ids = set().union(
                *(rule_element_ids(rule) for rule in pair_rules)
            )

            band_text_lines = [
                line
                for line in text_lines
                if line.get("kind", "text") == "text"
                and line.get("angle", 0) == 0
                and band_rect.contains(
                    rect_center(fitz.Rect(line["bbox"]))
                )
            ]

            band_images = [
                image
                for image in source_images
                if image_is_large_enough(
                    image_rect=fitz.Rect(image["bbox"]),
                    page_rect=page_rect,
                )
                and meaningful_overlap(
                    band_rect,
                    fitz.Rect(image["bbox"]),
                )
            ]

            non_rule_drawings: list[dict] = []
            has_vertical_separator = (
                vertical_rule_connects_horizontal_pair(
                    merged_vertical_rules=merged_vertical_rules,
                    band_rect=band_rect,
                    top=top,
                    bottom=bottom,
                )
            )
            for drawing in drawings:
                drawing_id = page_inventory.id_for_payload(
                    drawing,
                    "drawing",
                )
                if drawing_id is None or drawing_id in pair_rule_ids:
                    continue

                drawing_rect = fitz.Rect(drawing["rect"]).normalize()
                if not meaningful_overlap(band_rect, drawing_rect):
                    continue

                drawing_rule = axis_geometry_index[
                    "legacy_axis_rule_by_element_id"
                ].get(drawing_id)
                if (
                    drawing_rule is not None
                    and drawing_rule["orientation"] == "v"
                ):
                    vertical_overlap = band_rect & drawing_rect
                    vertical_coverage = (
                        vertical_overlap.height / band_rect.height
                        if band_rect.height > 0
                        else 0.0
                    )
                    if (
                        vertical_coverage
                        >= minimum_vertical_coverage_ratio
                    ):
                        has_vertical_separator = True
                    continue

                if rect_area(drawing_rect) >= minimum_image_area_points_squared:
                    non_rule_drawings.append(drawing)

            if has_vertical_separator:
                connected_vertical_rejection_count += 1
                continue

            boundary_grid_evidence = (
                horizontal_pair_has_boundary_grid_evidence(
                    merged_vertical_rules=merged_vertical_rules,
                    top=top,
                    bottom=bottom,
                    band_x0=band_x0,
                    band_x1=band_x1,
                    tolerance=vector_connection_tolerance_points,
                )
            )
            if boundary_grid_evidence["has_boundary_grid_evidence"]:
                boundary_grid_rejection_count += 1
                continue
            if (
                not band_text_lines
                and not band_images
                and not non_rule_drawings
            ):
                continue

            mode = (
                "visual_rule_band"
                if band_images or non_rule_drawings
                else "text_separator"
            )
            rule_ids = sorted(pair_rule_ids)
            content_drawing_ids = [
                page_inventory.id_for_payload(
                    drawing,
                    "drawing",
                )
                for drawing in non_rule_drawings
            ]
            content_drawing_ids = [
                element_id
                for element_id in content_drawing_ids
                if element_id is not None
            ]
            image_ids = page_inventory.ids_for_payloads(
                band_images,
                "image",
            )
            text_ids = page_inventory.ids_for_payloads(
                band_text_lines,
                "text",
            )
            consumed_drawing_ids = sorted(
                set(rule_ids).union(content_drawing_ids)
            )

            all_band_drawing_ids = [
                drawing_id
                for drawing in drawings
                for drawing_id in [
                    page_inventory.id_for_payload(
                        drawing,
                        "drawing",
                    )
                ]
                if drawing_id is not None
                and (
                    meaningful_overlap(
                        band_rect,
                        fitz.Rect(drawing["rect"]),
                    )
                    or band_rect.contains(
                        rect_center(fitz.Rect(drawing["rect"]))
                    )
                )
            ]
            all_band_drawing_ids = sorted(
                set(all_band_drawing_ids).union(pair_rule_ids)
            )

            band = {
                "bbox": band_rect,
                "mode": mode,
                "rule_drawing_ids": rule_ids,
                "content_drawing_ids": sorted(content_drawing_ids),
                "drawing_element_ids": sorted(
                    set(all_band_drawing_ids)
                ),
                "image_element_ids": sorted(image_ids),
                "text_element_ids": sorted(text_ids),
                "text_lines": band_text_lines,
                "reason": (
                    "two_aligned_horizontal_rules_with_native_text"
                    if mode == "text_separator"
                    else "two_aligned_horizontal_rules_with_image_or_vector_content"
                ),
            }
            bands.append(band)
            preclassified_drawing_ids.update(
                consumed_drawing_ids
            )

    accepted_bands: list[dict] = []
    accepted_ids: set[str] = set()
    for band in sorted(
        bands,
        key=lambda item: (
            item["bbox"].y0,
            item["bbox"].x0,
        ),
    ):
        band_rule_ids = set(band["rule_drawing_ids"])
        if (
            band["mode"] != "text_separator"
            and band_rule_ids & accepted_ids
        ):
            continue
        accepted_bands.append(band)
        if band["mode"] != "text_separator":
            accepted_ids.update(band_rule_ids)

    accepted_drawing_ids = {
        element_id
        for band in accepted_bands
        for element_id in band["drawing_element_ids"]
    }
    table_detection_drawings = [
        drawing
        for drawing in drawings
        if page_inventory.id_for_payload(
            drawing,
            "drawing",
        ) not in accepted_drawing_ids
    ]
    return {
        "bands": accepted_bands,
        "table_detection_drawings": table_detection_drawings,
        "preclassified_rule_drawing_ids": sorted(
            accepted_drawing_ids
        ),
        "metadata": {
            "candidate_group_count": len(
                horizontal_rule_groups
            ),
            "accepted_band_count": len(accepted_bands),
            "rejected_by_repeated_y_pair_count": (
                repeated_pair_rejection_count
            ),
            "rejected_by_connected_vertical_count": (
                connected_vertical_rejection_count
            ),
            "rejected_by_boundary_grid_count": (
                boundary_grid_rejection_count
            ),
            "visual_rule_band_count": sum(
                band["mode"] == "visual_rule_band"
                for band in accepted_bands
            ),
            "text_separator_band_count": sum(
                band["mode"] == "text_separator"
                for band in accepted_bands
            ),
            "table_detection_drawing_count": len(
                table_detection_drawings
            ),
            "preclassified_rule_drawing_ids": sorted(
                accepted_drawing_ids
            ),
            "rejected_filled_rule_count": rejected_filled_rule_count,
            "bands": [
                {
                    "mode": band["mode"],
                    "bbox": [
                        round(value, 3)
                        for value in band["bbox"]
                    ],
                    "rule_drawing_ids": band[
                        "rule_drawing_ids"
                    ],
                    "drawing_element_ids": band[
                        "drawing_element_ids"
                    ],
                    "image_element_ids": band[
                        "image_element_ids"
                    ],
                    "text_element_ids": band[
                        "text_element_ids"
                    ],
                    "reason": band["reason"],
                }
                for band in accepted_bands
            ],
        },
    }

def build_preclassified_text_separator_outputs(
    page_number: int,
    bands: list[dict],
    *,
    create_claim_manifest: Callable[..., dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """生成预分类纯文字分隔带的Markdown元素、记录和声明。"""
    markdown_elements: list[dict] = []
    records: list[dict] = []
    claims: list[dict] = []

    separator_number = 0
    for band in bands:
        if band["mode"] != "text_separator":
            continue
        separator_number += 1
        separator_rect = fitz.Rect(band["bbox"])
        claim_id = (
            f"page#{page_number}:text_separator#{separator_number}"
        )
        claimed_ids = sorted(
            set(band["rule_drawing_ids"]).union(
                band["text_element_ids"]
            )
        )
        markdown_elements.append(
            {
                "bbox": separator_rect,
                "render_bbox": separator_rect,
                "sort_bbox": separator_rect,
                "text": "---",
                "direction": (1.0, 0.0),
                "angle": 0,
                "kind": "text_separator",
                "markdown": "---",
                "claim_id": claim_id,
                "mode": "preclassified_text_separator",
            }
        )
        records.append(
            {
                "page_number": page_number,
                "bbox": [round(value, 3) for value in separator_rect],
                "source_drawing_element_ids": sorted(
                    band["rule_drawing_ids"]
                ),
                "source_text_element_ids": sorted(
                    band["text_element_ids"]
                ),
                "reason": band["reason"],
                "claim_id": claim_id,
            }
        )
        claims.append(create_claim_manifest(
            detector_name="text_separator",
            output_element_id=claim_id,
            input_element_ids=claimed_ids,
            claimed_element_ids=claimed_ids,
            feature_element_ids=sorted(
                band["rule_drawing_ids"]
            ),
            bbox=separator_rect,
            confidence=0.95,
            reason_codes=[
                "preclassified_two_rule_band",
                "native_text_only",
                "markdown_thematic_break",
            ],
            output_artifacts=[
                {
                    "type": "markdown_text_separator",
                    "bbox": [
                        round(value, 3)
                        for value in separator_rect
                    ],
                }
            ],
            metadata={
                "mode": "preclassified_text_separator",
                "rule_drawing_ids": sorted(
                    band["rule_drawing_ids"]
                ),
                "text_element_ids": sorted(
                    band["text_element_ids"]
                ),
            },
        ))

    return markdown_elements, records, claims
