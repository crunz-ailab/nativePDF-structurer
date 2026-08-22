"""协调不同检测器之间的归属关系和复合输出。

模块在独立检测器完成后关联表格
候选、视觉区域和已提议输出，根据overlap、局部来源簇桥接、共享文字块、
顶部对齐和包含关系构建复合渲染组，但不转移原始图元owner。

处理流程:
    1. 为存活表格和视觉候选建立文字上下文键。
    2. 评估嵌套、局部来源簇桥接和对齐证据。
    3. 选择唯一renderer并登记表格或视觉成员。
    4. Claim仲裁后激活有效组合，失败时恢复独立成员。

关键不变量:
    * 复合分组不改变 ``owner_by_element_id``。
    * 只有renderer存活时才抑制成员输出。
    * 整页占位来源簇不能作为局部桥接证据。
    * 组合截图边界和renderer排序边界必须分离。

Coordinate ownership and composite outputs across detectors.

Relates table candidates, visual regions, and proposed output atoms after
independent detectors finish. It evaluates overlap, source-cluster bridges,
shared text blocks, top alignment, and containment to build composite render
groups without transferring raw element ownership.

Processing flow:
    1. Build text-context keys for surviving table and visual candidates.
    2. Evaluate nesting, local source-cluster bridges, and alignment evidence.
    3. Select one renderer and register table or visual members.
    4. Activate valid groups after Claim resolution or restore their members.

Key invariants:
    * Composite grouping does not change ``owner_by_element_id``.
    * A member output is suppressed only while its renderer survives.
    * Page-sized placeholder clusters cannot establish a local bridge.
    * Group render bounds and renderer sort bounds remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Any, Callable

import fitz

from .geometry import rect_area
from .inventory import PageInventory


@dataclass(frozen=True)
class CrossDetectorOwnershipDependencies:
    """Thresholds and output callbacks used by cross-detector policies."""

    ownership_overlap_threshold: float = 0.65
    source_cluster_explanation_ratio: float = 0.80
    object_peer_overlap_ratio: float = 0.10
    table_image_dpi: int = 200
    rect_is_mostly_inside_any: Callable[..., bool] | None = None
    emit_table_candidate: Callable[..., dict] | None = None
    write_table_csv: Callable[..., Any] | None = None
    table_to_markdown: Callable[..., str] | None = None
    padded_table_candidate_clip: Callable[..., fitz.Rect] | None = None
    save_rotated_table_image: Callable[..., Any] | None = None


def resolve_table_visual_candidate_ownership(
    table_candidates: list[dict],
    visual_region_rects: list[fitz.Rect],
    overlap_threshold: float | None = None,
    *,
    dependencies: CrossDetectorOwnershipDependencies,
) -> tuple[list[dict], set[int]]:
    if overlap_threshold is None:
        overlap_threshold = dependencies.ownership_overlap_threshold

    """用双向覆盖率决定表格候选与视觉候选的唯一输出归属。

    - 表格大部分落入视觉区，而视觉区只有小部分落入表格：视觉区是更大的完整对象。
    - 视觉区大部分落入表格：表格是容器，不再输出重复视觉图。
    - 两个方向都达到门限时表示两者范围近似，有可输出表格结构时优先保留表格。
    """
    decisions: list[dict] = []
    visual_regions_claiming_tables: set[int] = set()

    for table_index, candidate in enumerate(table_candidates):
        table_rect = fitz.Rect(candidate["bbox"])
        table_area = rect_area(table_rect)
        best_visual_index: int | None = None
        best_table_overlap = 0.0
        best_visual_overlap = 0.0

        if table_area > 0:
            for visual_index, visual_rect in enumerate(
                visual_region_rects
            ):
                visual_area = rect_area(visual_rect)
                if visual_area <= 0:
                    continue
                intersection = table_rect & visual_rect
                if intersection.is_empty:
                    continue
                intersection_area = rect_area(intersection)
                table_overlap = intersection_area / table_area
                visual_overlap = intersection_area / visual_area
                if (
                    table_overlap,
                    visual_overlap,
                ) > (
                    best_table_overlap,
                    best_visual_overlap,
                ):
                    best_visual_index = visual_index
                    best_table_overlap = table_overlap
                    best_visual_overlap = visual_overlap

        best_visual_area = (
            rect_area(visual_region_rects[best_visual_index])
            if best_visual_index is not None
            else 0.0
        )
        visual_claims_table = (
            best_visual_index is not None
            and best_table_overlap >= overlap_threshold
            and (
                best_visual_overlap < overlap_threshold
                or best_visual_area > table_area * 1.20
            )
        )
        if visual_claims_table and best_visual_index is not None:
            visual_regions_claiming_tables.add(best_visual_index)

        decisions.append(
            {
                "table_index": table_index,
                "claimed_by_visual_region": visual_claims_table,
                "visual_region_index": best_visual_index,
                "table_overlap_ratio": best_table_overlap,
                "visual_overlap_ratio": best_visual_overlap,
                "table_claims_visual_region": (
                    best_visual_index is not None
                    and not visual_claims_table
                    and best_visual_overlap >= overlap_threshold
                ),
            }
        )

    suppressed_visual_indices: set[int] = set()
    for visual_index, visual_rect in enumerate(visual_region_rects):
        if visual_index in visual_regions_claiming_tables:
            continue

        visual_area = rect_area(visual_rect)
        if visual_area <= 0:
            continue

        for candidate, decision in zip(table_candidates, decisions):
            if decision["claimed_by_visual_region"]:
                continue
            if not candidate["will_emit_if_unclaimed"]:
                continue
            if (
                decision["visual_region_index"] == visual_index
                and decision["table_claims_visual_region"]
            ):
                suppressed_visual_indices.add(visual_index)
                break

    return decisions, suppressed_visual_indices

def build_visual_text_block_keys(
    page_inventory: PageInventory,
    text_lines: list[dict],
    table_candidates: list[dict],
    visual_records: list[dict],
    excluded_header_footer_rects: list[fitz.Rect] | None = None,
    *,
    dependencies: CrossDetectorOwnershipDependencies,
) -> dict[str, tuple[str | None, str | None]]:
    """为表格/视觉候选建立正文边界键，而不是用固定页码或单纯距离分组。

    每个候选使用其上方最近正文行和下方最近正文行作为边界。
    候选自身内部的文字、页眉页尾文字以及已被其他候选结构覆盖的文字
    不参与边界检测。两个候选的边界键相同，表示它们位于同一个视觉块。
    """
    excluded_header_footer_rects = list(
        excluded_header_footer_rects or []
    )

    candidate_rects: list[fitz.Rect] = []
    candidate_claim_ids: list[str] = []
    for candidate in table_candidates:
        if not candidate.get("structurally_valid_table", False):
            continue
        if not candidate.get("will_emit_if_unclaimed", False):
            continue
        candidate_rects.append(
            fitz.Rect(candidate.get("bbox", (0, 0, 0, 0))).normalize()
        )
        candidate_claim_ids.append(
            f"page#{page_inventory.page_number}:table#"
            f"{candidate['table_number']}"
        )
    for record in visual_records:
        claim_id = record.get("claim_id")
        if not claim_id:
            continue
        candidate_rects.append(
            fitz.Rect(
                record.get("content_bbox")
                or record.get("sort_source_bbox")
                or record.get("bbox", (0, 0, 0, 0))
            ).normalize()
        )
        candidate_claim_ids.append(claim_id)

    def line_is_inside_candidate(line_rect: fitz.Rect) -> bool:
        for candidate_rect in candidate_rects:
            line_area = rect_area(line_rect)
            if line_area <= 0:
                continue
            intersection = line_rect & candidate_rect
            if (
                not intersection.is_empty
                and rect_area(intersection) / line_area >= 0.50
            ):
                return True
            vertical_overlap = max(
                0.0,
                min(line_rect.y1, candidate_rect.y1)
                - max(line_rect.y0, candidate_rect.y0),
            )
            horizontal_overlap = max(
                0.0,
                min(line_rect.x1, candidate_rect.x1)
                - max(line_rect.x0, candidate_rect.x0),
            )
            if (
                line_rect.height > 0
                and vertical_overlap / line_rect.height >= 0.50
                and horizontal_overlap > 0
            ):
                return True
        return False

    body_lines: list[tuple[str, fitz.Rect]] = []
    text_heights: list[float] = []
    for line_index, line in enumerate(text_lines):
        line_rect = fitz.Rect(line["bbox"]).normalize()
        if line_rect.is_empty:
            continue
        if line_is_inside_candidate(line_rect):
            continue
        if dependencies.rect_is_mostly_inside_any(
            rect=line_rect,
            container_rects=excluded_header_footer_rects,
            minimum_ratio=0.50,
        ):
            continue
        text_id = page_inventory.id_for_payload(line, "text")
        if text_id is None:
            text_id = f"text_index:{line_index}"
        body_lines.append((text_id, line_rect))
        if line_rect.height > 0:
            text_heights.append(line_rect.height)

    top_band_tolerance = (
        max(1.0, statistics.median(text_heights))
        if text_heights
        else 2.0
    )

    block_keys: dict[str, tuple[str | None, str | None]] = {}
    for claim_id, candidate_rect in zip(
        candidate_claim_ids,
        candidate_rects,
    ):
        previous_line_id: str | None = None
        next_line_id: str | None = None
        previous_y = -float("inf")
        next_y = float("inf")
        for line_id, line_rect in body_lines:
            if line_rect.y1 <= candidate_rect.y0 and line_rect.y1 > previous_y:
                previous_y = line_rect.y1
                previous_line_id = line_id
            if line_rect.y0 >= candidate_rect.y1 and line_rect.y0 < next_y:
                next_y = line_rect.y0
                next_line_id = line_id
        block_keys[claim_id] = (
            previous_line_id,
            next_line_id,
        )

    block_keys["__top_band_tolerance__"] = (
        f"{top_band_tolerance:.6f}",
        None,
    )
    return block_keys

def evaluate_source_cluster_local_bridge(
    source_cluster_rect: fitz.Rect,
    table_rect: fitz.Rect,
    visual_rect: fitz.Rect,
    page_rect: fitz.Rect,
    explanation_threshold: float | None = None,
    support_threshold: float | None = None,
    *,
    dependencies: CrossDetectorOwnershipDependencies,
) -> dict:
    if explanation_threshold is None:
        explanation_threshold = dependencies.source_cluster_explanation_ratio
    if support_threshold is None:
        support_threshold = dependencies.object_peer_overlap_ratio

    """判断来源簇是否是表格与视觉候选之间的局部共同结构。"""
    source_cluster_rect = fitz.Rect(source_cluster_rect).normalize()
    table_rect = fitz.Rect(table_rect).normalize()
    visual_rect = fitz.Rect(visual_rect).normalize()
    page_rect = fitz.Rect(page_rect).normalize()

    source_cluster_is_specific = any(
        abs(float(left) - float(right)) > 0.01
        for left, right in zip(source_cluster_rect, page_rect)
    )
    source_cluster_area = rect_area(source_cluster_rect)
    table_area = rect_area(table_rect)
    visual_area = rect_area(visual_rect)

    group_rect = fitz.Rect(table_rect)
    group_rect.include_rect(visual_rect)

    source_explanation_ratio = 0.0
    table_support_ratio = 0.0
    visual_support_ratio = 0.0
    if source_cluster_area > 0:
        source_group_intersection = source_cluster_rect & group_rect
        if not source_group_intersection.is_empty:
            source_explanation_ratio = (
                rect_area(source_group_intersection) / source_cluster_area
            )
    table_support_denominator = min(source_cluster_area, table_area)
    if table_support_denominator > 0:
        source_table_intersection = source_cluster_rect & table_rect
        if not source_table_intersection.is_empty:
            table_support_ratio = (
                rect_area(source_table_intersection)
                / table_support_denominator
            )
    visual_support_denominator = min(source_cluster_area, visual_area)
    if visual_support_denominator > 0:
        source_visual_intersection = source_cluster_rect & visual_rect
        if not source_visual_intersection.is_empty:
            visual_support_ratio = (
                rect_area(source_visual_intersection)
                / visual_support_denominator
            )

    source_cluster_is_local_bridge = bool(
        source_cluster_is_specific
        and source_cluster_area > 0
        and source_explanation_ratio >= explanation_threshold
        and table_support_ratio >= support_threshold
        and visual_support_ratio >= support_threshold
    )
    return {
        "source_cluster_is_specific": source_cluster_is_specific,
        "source_cluster_explanation_ratio": source_explanation_ratio,
        "table_support_ratio": table_support_ratio,
        "visual_support_ratio": visual_support_ratio,
        "source_cluster_is_local_bridge": (
            source_cluster_is_local_bridge
        ),
    }

def build_table_visual_composite_output_groups(
    page_number: int,
    page_rect: fitz.Rect,
    page_inventory: PageInventory,
    text_lines: list[dict],
    excluded_header_footer_rects: list[fitz.Rect] | None,
    table_candidates: list[dict],
    visual_records: list[dict],
    visual_markdown_elements: list[dict],
    table_claims: list[dict],
    visual_claims: list[dict],
    overlap_threshold: float | None = None,
    *,
    dependencies: CrossDetectorOwnershipDependencies,
) -> list[dict]:
    if overlap_threshold is None:
        overlap_threshold = dependencies.ownership_overlap_threshold

    """把结构嵌套的表格与视觉Claim登记到同一复合输出组。"""
    table_claim_by_id = {
        claim["claim_id"]: claim for claim in table_claims
    }
    visual_claim_by_id = {
        claim["claim_id"]: claim for claim in visual_claims
    }
    visual_markdown_by_claim = {
        element.get("claim_id"): element
        for element in visual_markdown_elements
        if element.get("claim_id") is not None
    }
    groups_by_visual_claim: dict[str, dict] = {}

    for candidate in table_candidates:
        if not candidate.get("will_emit_if_unclaimed", False):
            continue
        if not candidate.get("structurally_valid_table", False):
            continue

        table_rect = fitz.Rect(candidate["bbox"]).normalize()
        table_area = rect_area(table_rect)
        if table_area <= 0:
            continue

        best_match: dict | None = None
        for visual_record in visual_records:
            visual_claim_id = visual_record.get("claim_id")
            if visual_claim_id not in visual_claim_by_id:
                continue

            visual_content_rect = fitz.Rect(
                visual_record.get("content_bbox")
                or visual_record.get("sort_source_bbox")
                or visual_record["bbox"]
            ).normalize()
            visual_area = rect_area(visual_content_rect)
            if visual_area <= table_area:
                continue

            intersection = table_rect & visual_content_rect
            if intersection.is_empty:
                continue
            intersection_area = rect_area(intersection)
            table_overlap_ratio = intersection_area / table_area
            visual_overlap_ratio = intersection_area / visual_area
            if table_overlap_ratio < overlap_threshold:
                continue
            if visual_overlap_ratio >= overlap_threshold:
                continue

            source_cluster_rect = fitz.Rect(
                candidate.get("source_cluster_bbox", table_rect)
            ).normalize()
            source_cluster_is_specific = any(
                abs(float(left) - float(right)) > 0.01
                for left, right in zip(
                    source_cluster_rect,
                    page_rect,
                )
            )
            source_cluster_intersects_visual = bool(
                source_cluster_is_specific
                and not (
                    source_cluster_rect & visual_content_rect
                ).is_empty
            )

            match = {
                "visual_record": visual_record,
                "visual_claim_id": visual_claim_id,
                "visual_content_bbox": visual_content_rect,
                "table_overlap_ratio": table_overlap_ratio,
                "visual_overlap_ratio": visual_overlap_ratio,
                "source_cluster_bbox": source_cluster_rect,
                "source_cluster_is_specific": (
                    source_cluster_is_specific
                ),
                "source_cluster_intersects_visual": (
                    source_cluster_intersects_visual
                ),
            }
            if best_match is None or (
                match["table_overlap_ratio"],
                -match["visual_overlap_ratio"],
                -rect_area(match["visual_content_bbox"]),
            ) > (
                best_match["table_overlap_ratio"],
                -best_match["visual_overlap_ratio"],
                -rect_area(best_match["visual_content_bbox"]),
            ):
                best_match = match

        if best_match is None:
            continue

        visual_record = best_match["visual_record"]
        visual_claim_id = best_match["visual_claim_id"]
        table_claim_id = (
            f"page#{page_number}:table#{candidate['table_number']}"
        )
        table_text_ids = sorted(
            page_inventory.ids_centred_in(
                table_rect,
                kinds=("text",),
            )
        )

        group = groups_by_visual_claim.get(visual_claim_id)
        if group is None:
            group = {
                "output_group_id": (
                    f"page#{page_number}:composite_visual#"
                    f"{len(groups_by_visual_claim) + 1}"
                ),
                "output_kind": "composite_visual",
                "renderer_claim_id": visual_claim_id,
                "member_claim_ids": [visual_claim_id],
                "member_table_claim_ids": [],
                "member_table_bboxes": [],
                "member_table_text_element_ids": [],
                "member_visual_claim_ids": [],
                "member_visual_bboxes": [],
                "member_visual_text_element_ids": [],
                "relations": [],
                "visual_relations": [],
                "active": None,
            }
            groups_by_visual_claim[visual_claim_id] = group

        output_group_id = group["output_group_id"]
        group["member_claim_ids"].append(table_claim_id)
        group["member_table_claim_ids"].append(table_claim_id)
        group["member_table_bboxes"].append(
            [round(value, 3) for value in table_rect]
        )
        group["member_table_text_element_ids"].extend(
            table_text_ids
        )
        group["relations"].append(
            {
                "table_claim_id": table_claim_id,
                "visual_claim_id": visual_claim_id,
                "table_bbox": [
                    round(value, 3) for value in table_rect
                ],
                "table_text_element_ids": table_text_ids,
                "visual_content_bbox": [
                    round(value, 3)
                    for value in best_match[
                        "visual_content_bbox"
                    ]
                ],
                "table_overlap_ratio": round(
                    best_match["table_overlap_ratio"], 6
                ),
                "visual_overlap_ratio": round(
                    best_match["visual_overlap_ratio"], 6
                ),
                "overlap_threshold": overlap_threshold,
                "source_cluster_bbox": [
                    round(value, 3)
                    for value in best_match[
                        "source_cluster_bbox"
                    ]
                ],
                "source_cluster_is_specific": best_match[
                    "source_cluster_is_specific"
                ],
                "source_cluster_intersects_visual": best_match[
                    "source_cluster_intersects_visual"
                ],
                "reason_codes": [
                    "table_mostly_inside_visual_content_bbox",
                    "visual_content_bbox_larger_than_table",
                    "directional_overlap_below_reverse_threshold",
                ],
            }
        )

        candidate["claimed_by_visual_region"] = True
        candidate["table_overlap_ratio"] = best_match[
            "table_overlap_ratio"
        ]
        candidate["visual_overlap_ratio"] = best_match[
            "visual_overlap_ratio"
        ]
        candidate["output_group_id"] = output_group_id
        candidate["output_group_parent_claim_id"] = visual_claim_id
        candidate["output_group_role"] = "member_table"

        table_claim = table_claim_by_id.get(table_claim_id)
        if table_claim is not None:
            table_claim["metadata"]["output_group_id"] = (
                output_group_id
            )
            table_claim["metadata"]["output_group_role"] = (
                "member_table"
            )
            table_claim["metadata"][
                "output_group_parent_claim_id"
            ] = visual_claim_id
            table_claim["reason_codes"] = sorted(
                set(table_claim.get("reason_codes", [])).union(
                    {"composite_output_member"}
                )
            )

        visual_record["output_group_id"] = output_group_id
        visual_record["output_group_role"] = "renderer"
        visual_record.setdefault(
            "composite_member_table_claim_ids", []
        ).append(table_claim_id)
        visual_record.setdefault(
            "composite_member_table_bboxes", []
        ).append([round(value, 3) for value in table_rect])
        visual_record.setdefault(
            "composite_member_table_text_element_ids", []
        ).extend(table_text_ids)

        visual_markdown = visual_markdown_by_claim.get(
            visual_claim_id
        )
        if visual_markdown is not None:
            visual_markdown["output_group_id"] = output_group_id
            visual_markdown["output_group_role"] = "renderer"
            visual_markdown.setdefault(
                "composite_member_table_claim_ids", []
            ).append(table_claim_id)
            visual_markdown.setdefault(
                "composite_member_table_bboxes", []
            ).append([round(value, 3) for value in table_rect])

        visual_claim = visual_claim_by_id.get(visual_claim_id)
        if visual_claim is not None:
            visual_claim["metadata"]["output_group_id"] = (
                output_group_id
            )
            visual_claim["metadata"]["output_group_role"] = (
                "renderer"
            )
            visual_claim["metadata"].setdefault(
                "composite_member_claim_ids", []
            ).append(table_claim_id)
            visual_claim["metadata"].setdefault(
                "composite_member_text_element_ids", []
            ).extend(table_text_ids)
            visual_claim["reason_codes"] = sorted(
                set(visual_claim.get("reason_codes", [])).union(
                    {"composite_output_renderer"}
                )
            )

    block_keys = build_visual_text_block_keys(
        page_inventory=page_inventory,
        text_lines=text_lines,
        table_candidates=table_candidates,
        visual_records=visual_records,
        excluded_header_footer_rects=excluded_header_footer_rects,
        dependencies=dependencies,
    )
    top_band_tolerance = float(
        block_keys.get("__top_band_tolerance__", ("2.0", None))[0]
    )
    visual_record_by_claim_id = {
        record.get("claim_id"): record
        for record in visual_records
        if record.get("claim_id") is not None
    }

    def ensure_renderer_group(
        renderer_record: dict,
    ) -> dict:
        renderer_claim_id = renderer_record["claim_id"]
        group = groups_by_visual_claim.get(renderer_claim_id)
        if group is None:
            group = {
                "output_group_id": (
                    f"page#{page_number}:composite_visual#"
                    f"{len(groups_by_visual_claim) + 1}"
                ),
                "output_kind": "composite_visual",
                "renderer_claim_id": renderer_claim_id,
                "member_claim_ids": [renderer_claim_id],
                "member_table_claim_ids": [],
                "member_table_bboxes": [],
                "member_table_text_element_ids": [],
                "member_visual_claim_ids": [],
                "member_visual_bboxes": [],
                "member_visual_text_element_ids": [],
                "relations": [],
                "visual_relations": [],
                "active": None,
            }
            groups_by_visual_claim[renderer_claim_id] = group
        group.setdefault("member_visual_claim_ids", [])
        group.setdefault("member_visual_bboxes", [])
        group.setdefault("member_visual_text_element_ids", [])
        group.setdefault("visual_relations", [])
        renderer_record["output_group_id"] = group["output_group_id"]
        renderer_record["output_group_role"] = "renderer"
        renderer_record.setdefault(
            "composite_member_visual_claim_ids", []
        )
        renderer_record.setdefault("composite_member_visual_bboxes", [])
        renderer_record.setdefault(
            "composite_member_visual_text_element_ids", []
        )
        renderer_claim = visual_claim_by_id.get(renderer_claim_id)
        if renderer_claim is not None:
            renderer_claim["metadata"]["output_group_id"] = (
                group["output_group_id"]
            )
            renderer_claim["metadata"]["output_group_role"] = (
                "renderer"
            )
            renderer_claim["metadata"].setdefault(
                "composite_member_claim_ids", []
            )
            renderer_claim["metadata"].setdefault(
                "composite_member_text_element_ids", []
            )
        renderer_markdown = visual_markdown_by_claim.get(renderer_claim_id)
        if renderer_markdown is not None:
            renderer_markdown["output_group_id"] = group[
                "output_group_id"
            ]
            renderer_markdown["output_group_role"] = "renderer"
            renderer_markdown.setdefault(
                "composite_member_visual_claim_ids", []
            )
            renderer_markdown.setdefault(
                "composite_member_visual_bboxes", []
            )
        return group

    def attach_visual_member(
        group: dict,
        member_record: dict,
        relation_reason_codes: list[str],
    ) -> bool:
        renderer_claim_id = group["renderer_claim_id"]
        member_claim_id = member_record.get("claim_id")
        if not member_claim_id or member_claim_id == renderer_claim_id:
            return False
        existing_group_id = member_record.get("output_group_id")
        if existing_group_id and existing_group_id != group[
            "output_group_id"
        ]:
            return False

        member_rect = fitz.Rect(
            member_record.get("content_bbox")
            or member_record.get("sort_source_bbox")
            or member_record["bbox"]
        ).normalize()
        member_claim = visual_claim_by_id.get(member_claim_id)
        member_text_ids = sorted(
            set(
                member_claim.get("metadata", {}).get(
                    "text_element_ids", []
                )
                if member_claim is not None
                else []
            )
        )
        if member_claim_id not in group["member_visual_claim_ids"]:
            group["member_visual_claim_ids"].append(member_claim_id)
            group["member_visual_bboxes"].append(
                [round(value, 3) for value in member_rect]
            )
            group["member_visual_text_element_ids"].extend(
                member_text_ids
            )
            group["member_claim_ids"].append(member_claim_id)
            group["visual_relations"].append(
                {
                    "renderer_claim_id": renderer_claim_id,
                    "member_visual_claim_id": member_claim_id,
                    "member_visual_bbox": [
                        round(value, 3) for value in member_rect
                    ],
                    "text_block_key": block_keys.get(
                        member_claim_id
                    ),
                    "reason_codes": sorted(set(relation_reason_codes)),
                }
            )

        member_record["output_group_id"] = group["output_group_id"]
        member_record["output_group_role"] = "member_visual"
        member_record["embedded_in_parent_visual"] = True
        member_record["parent_visual_claim_id"] = renderer_claim_id
        member_record.setdefault("composite_parent_claim_id", renderer_claim_id)
        member_record.setdefault("composite_member_text_element_ids", [])
        member_record["composite_member_text_element_ids"] = sorted(
            set(member_record["composite_member_text_element_ids"]).union(
                member_text_ids
            )
        )

        member_markdown = visual_markdown_by_claim.get(member_claim_id)
        if member_markdown is not None:
            member_markdown["output_group_id"] = group[
                "output_group_id"
            ]
            member_markdown["output_group_role"] = "member_visual"
            member_markdown["embedded_in_parent_visual"] = True
            member_markdown["parent_visual_claim_id"] = renderer_claim_id

        if member_claim is not None:
            member_claim["metadata"]["output_group_id"] = (
                group["output_group_id"]
            )
            member_claim["metadata"]["output_group_role"] = (
                "member_visual"
            )
            member_claim["metadata"]["parent_visual_claim_id"] = (
                renderer_claim_id
            )
            member_claim["reason_codes"] = sorted(
                set(member_claim.get("reason_codes", [])).union(
                    {"composite_output_member_visual"}
                )
            )
        return True

    def attach_table_member(
        group: dict,
        candidate: dict,
        relation_reason_codes: list[str],
    ) -> None:
        table_claim_id = (
            f"page#{page_number}:table#{candidate['table_number']}"
        )
        if table_claim_id in group["member_table_claim_ids"]:
            return
        table_rect = fitz.Rect(candidate["bbox"]).normalize()
        table_source_cluster_rect = fitz.Rect(
            candidate.get("source_cluster_bbox", table_rect)
        ).normalize()
        table_source_cluster_is_specific = any(
            abs(float(left) - float(right)) > 0.01
            for left, right in zip(
                table_source_cluster_rect,
                page_rect,
            )
        )
        table_text_ids = sorted(
            page_inventory.ids_centred_in(
                table_rect,
                kinds=("text",),
            )
        )
        group["member_claim_ids"].append(table_claim_id)
        group["member_table_claim_ids"].append(table_claim_id)
        group["member_table_bboxes"].append(
            [round(value, 3) for value in table_rect]
        )
        group["member_table_text_element_ids"].extend(table_text_ids)
        group["relations"].append(
            {
                "table_claim_id": table_claim_id,
                "visual_claim_id": group["renderer_claim_id"],
                "table_bbox": [
                    round(value, 3) for value in table_rect
                ],
                "table_text_element_ids": table_text_ids,
                "visual_content_bbox": [],
                "table_overlap_ratio": 0.0,
                "visual_overlap_ratio": 0.0,
                "overlap_threshold": overlap_threshold,
                "source_cluster_bbox": [
                    round(value, 3)
                    for value in table_source_cluster_rect
                ],
                "source_cluster_is_specific": (
                    table_source_cluster_is_specific
                ),
                "source_cluster_intersects_visual": False,
                "reason_codes": sorted(set(relation_reason_codes)),
            }
        )
        candidate["claimed_by_visual_region"] = True
        candidate["table_overlap_ratio"] = 0.0
        candidate["visual_overlap_ratio"] = 0.0
        candidate["output_group_id"] = group["output_group_id"]
        candidate["output_group_parent_claim_id"] = group[
            "renderer_claim_id"
        ]
        candidate["output_group_role"] = "member_table"

        table_claim = table_claim_by_id.get(table_claim_id)
        if table_claim is not None:
            table_claim["metadata"]["output_group_id"] = (
                group["output_group_id"]
            )
            table_claim["metadata"]["output_group_role"] = (
                "member_table"
            )
            table_claim["metadata"][
                "output_group_parent_claim_id"
            ] = group["renderer_claim_id"]
            table_claim["reason_codes"] = sorted(
                set(table_claim.get("reason_codes", [])).union(
                    {"composite_output_member"}
                )
            )

        renderer_record = visual_record_by_claim_id.get(
            group["renderer_claim_id"]
        )
        if renderer_record is not None:
            renderer_record.setdefault(
                "composite_member_table_claim_ids", []
            ).append(table_claim_id)
            renderer_record.setdefault(
                "composite_member_table_bboxes", []
            ).append([round(value, 3) for value in table_rect])
            renderer_record.setdefault(
                "composite_member_table_text_element_ids", []
            ).extend(table_text_ids)
        renderer_claim = visual_claim_by_id.get(
            group["renderer_claim_id"]
        )
        if renderer_claim is not None:
            renderer_claim["metadata"].setdefault(
                "composite_member_claim_ids", []
            ).append(table_claim_id)
            renderer_claim["metadata"].setdefault(
                "composite_member_text_element_ids", []
            ).extend(table_text_ids)

    for candidate in table_candidates:
        if not candidate.get("structurally_valid_table", False):
            continue
        if not candidate.get("will_emit_if_unclaimed", False):
            continue
        if not candidate.get("table_has_complex_visual_content", False):
            continue
        table_claim_id = (
            f"page#{page_number}:table#{candidate['table_number']}"
        )
        table_block_key = block_keys.get(table_claim_id)
        same_block_relation_audit: list[dict] = []
        table_claim_for_audit = table_claim_by_id.get(table_claim_id)
        if table_block_key is None:
            continue
        table_rect = fitz.Rect(candidate["bbox"]).normalize()
        source_cluster_rect = fitz.Rect(
            candidate.get("source_cluster_bbox", table_rect)
        ).normalize()
        related_visuals: list[tuple[dict, list[str]]] = []
        for visual_record in visual_records:
            visual_claim_id = visual_record.get("claim_id")
            if visual_claim_id not in visual_claim_by_id:
                continue
            visual_rect = fitz.Rect(
                visual_record.get("content_bbox")
                or visual_record.get("sort_source_bbox")
                or visual_record["bbox"]
            ).normalize()
            same_block = block_keys.get(visual_claim_id) == table_block_key
            source_cluster_relation = (
                evaluate_source_cluster_local_bridge(
                    source_cluster_rect=source_cluster_rect,
                    table_rect=table_rect,
                    visual_rect=visual_rect,
                    page_rect=page_rect,
                    dependencies=dependencies,
                )
            )
            source_cluster_is_local_bridge = source_cluster_relation[
                "source_cluster_is_local_bridge"
            ]
            top_aligned = abs(table_rect.y0 - visual_rect.y0) <= (
                top_band_tolerance
            )
            vertical_projection_overlap = max(
                0.0,
                min(table_rect.y1, visual_rect.y1)
                - max(table_rect.y0, visual_rect.y0),
            )
            source_cluster_bridges_with_vertical_projection = bool(
                source_cluster_is_local_bridge
                and vertical_projection_overlap > 0.0
            )
            same_block_relation_audit.append(
                {
                    "visual_claim_id": visual_claim_id,
                    "table_block_key": table_block_key,
                    "visual_block_key": block_keys.get(
                        visual_claim_id
                    ),
                    "same_text_block": same_block,
                    "source_cluster_is_specific": (
                        source_cluster_relation[
                            "source_cluster_is_specific"
                        ]
                    ),
                    "source_cluster_explanation_ratio": round(
                        source_cluster_relation[
                            "source_cluster_explanation_ratio"
                        ],
                        6,
                    ),
                    "table_support_ratio": round(
                        source_cluster_relation[
                            "table_support_ratio"
                        ],
                        6,
                    ),
                    "visual_support_ratio": round(
                        source_cluster_relation[
                            "visual_support_ratio"
                        ],
                        6,
                    ),
                    "source_cluster_support_normalization": (
                        "smaller_of_source_cluster_and_candidate"
                    ),
                    "source_cluster_is_local_bridge": (
                        source_cluster_is_local_bridge
                    ),
                    "vertical_projection_overlap_points": round(
                        vertical_projection_overlap,
                        6,
                    ),
                    "source_cluster_bridges_with_vertical_projection": (
                        source_cluster_bridges_with_vertical_projection
                    ),
                    "top_aligned": top_aligned,
                }
            )
            if not same_block:
                continue
            if (
                not source_cluster_bridges_with_vertical_projection
                and not top_aligned
            ):
                continue
            reasons = ["same_text_visual_block"]
            if source_cluster_is_local_bridge:
                reasons.append("table_source_cluster_locally_bridges_visual")
            if source_cluster_bridges_with_vertical_projection:
                reasons.append(
                    "table_visual_vertical_projection_overlap"
                )
            if top_aligned:
                reasons.append("table_visual_top_band_aligned")
            related_visuals.append((visual_record, reasons))

        if table_claim_for_audit is not None:
            table_claim_for_audit["metadata"][
                "same_block_relation_audit"
            ] = same_block_relation_audit
            table_claim_for_audit["metadata"][
                "visual_records_seen_for_same_block"
            ] = [
                record.get("claim_id")
                for record in visual_records
            ]
            table_claim_for_audit["metadata"][
                "visual_claim_ids_seen_for_same_block"
            ] = sorted(visual_claim_by_id)

        if not related_visuals:
            continue

        existing_group = groups_by_visual_claim.get(
            candidate.get("output_group_parent_claim_id")
        )
        if existing_group is not None:
            group = existing_group
        else:
            renderer_record = max(
                (record for record, _ in related_visuals),
                key=lambda record: rect_area(
                    fitz.Rect(
                        record.get("content_bbox")
                        or record.get("sort_source_bbox")
                        or record["bbox"]
                    )
                ),
            )
            group = ensure_renderer_group(renderer_record)

        renderer_relation_reasons = next(
            (
                reasons
                for record, reasons in related_visuals
                if record["claim_id"] == group["renderer_claim_id"]
            ),
            [],
        )
        attach_table_member(
            group=group,
            candidate=candidate,
            relation_reason_codes=[
                "same_text_visual_block",
                "complex_visual_table_candidate",
                *renderer_relation_reasons,
            ],
        )
        for visual_record, relation_reasons in related_visuals:
            if visual_record["claim_id"] == group["renderer_claim_id"]:
                continue
            attach_visual_member(
                group=group,
                member_record=visual_record,
                relation_reason_codes=relation_reasons,
            )

    groups = list(groups_by_visual_claim.values())
    for group in groups:
        group["member_claim_ids"] = sorted(
            set(group["member_claim_ids"])
        )
        group["member_table_claim_ids"] = sorted(
            set(group["member_table_claim_ids"])
        )
        group["member_table_text_element_ids"] = sorted(
            set(group["member_table_text_element_ids"])
        )
        group["member_visual_claim_ids"] = sorted(
            set(group.get("member_visual_claim_ids", []))
        )
        group["member_visual_text_element_ids"] = sorted(
            set(group.get("member_visual_text_element_ids", []))
        )
        group["member_claim_ids"] = sorted(
            set(group["member_claim_ids"])
        )
    return groups

def finalize_table_visual_composite_output_groups(
    output_groups: list[dict],
    resolution: dict,
    table_candidates: list[dict],
    visual_records: list[dict],
    visual_markdown_elements: list[dict],
    table_claims: list[dict],
    visual_claims: list[dict],
    *,
    dependencies: CrossDetectorOwnershipDependencies,
) -> dict:
    """按阶段D存活结果确认复合输出组，并生成表格回退清单。"""
    candidate_resolution_by_id = {
        candidate["claim_id"]: candidate
        for candidate in resolution.get("candidates", [])
    }
    table_candidate_by_claim_id = {
        claim["claim_id"]: candidate
        for candidate in table_candidates
        for claim in table_claims
        if claim["claim_id"].endswith(
            f":table#{candidate['table_number']}"
        )
    }
    table_claim_by_id = {
        claim["claim_id"]: claim for claim in table_claims
    }
    visual_claim_by_id = {
        claim["claim_id"]: claim for claim in visual_claims
    }
    visual_record_by_claim_id = {
        record.get("claim_id"): record
        for record in visual_records
        if record.get("claim_id") is not None
    }
    visual_markdown_by_claim_id = {
        element.get("claim_id"): element
        for element in visual_markdown_elements
        if element.get("claim_id") is not None
    }

    def claim_survives(claim_id: str) -> bool:
        candidate_resolution = candidate_resolution_by_id.get(
            claim_id
        )
        return bool(
            candidate_resolution
            and candidate_resolution.get("survives_preview", False)
        )

    restored_table_claim_ids: set[str] = set()
    active_group_ids: list[str] = []
    inactive_group_ids: list[str] = []

    for group in output_groups:
        renderer_claim_id = group["renderer_claim_id"]
        renderer_survives = claim_survives(renderer_claim_id)
        surviving_table_claim_ids = [
            claim_id
            for claim_id in group["member_table_claim_ids"]
            if claim_survives(claim_id)
        ]
        surviving_member_visual_claim_ids = [
            claim_id
            for claim_id in group.get("member_visual_claim_ids", [])
            if claim_survives(claim_id)
        ]
        active = bool(
            renderer_survives
            and (
                surviving_table_claim_ids
                or surviving_member_visual_claim_ids
            )
        )
        group["active"] = active
        group["renderer_survives"] = renderer_survives
        group["surviving_member_table_claim_ids"] = (
            surviving_table_claim_ids
        )
        group["surviving_member_visual_claim_ids"] = (
            surviving_member_visual_claim_ids
        )
        group["inactive_reason"] = (
            None
            if active
            else (
                "renderer_claim_did_not_survive"
                if not renderer_survives
                else "no_member_claim_survived"
            )
        )

        active_relations = [
            relation
            for relation in group["relations"]
            if relation["table_claim_id"]
            in surviving_table_claim_ids
        ]
        active_table_bboxes = [
            relation["table_bbox"]
            for relation in active_relations
        ]
        active_table_text_ids = sorted(
            {
                text_id
                for relation in active_relations
                for text_id in relation.get(
                    "table_text_element_ids", []
                )
            }
        )
        active_visual_relations = [
            relation
            for relation in group.get("visual_relations", [])
            if relation["member_visual_claim_id"]
            in surviving_member_visual_claim_ids
        ]
        active_visual_bboxes = [
            relation["member_visual_bbox"]
            for relation in active_visual_relations
        ]
        active_visual_text_ids = sorted(
            {
                text_id
                for claim_id in surviving_member_visual_claim_ids
                for text_id in visual_claim_by_id.get(
                    claim_id,
                    {},
                )
                .get("metadata", {})
                .get("text_element_ids", [])
            }
        )

        visual_record = visual_record_by_claim_id.get(
            renderer_claim_id
        )
        visual_markdown = visual_markdown_by_claim_id.get(
            renderer_claim_id
        )
        visual_claim = visual_claim_by_id.get(renderer_claim_id)

        if active:
            active_group_ids.append(group["output_group_id"])
            group["member_claim_ids"] = [
                renderer_claim_id,
                *surviving_table_claim_ids,
                *surviving_member_visual_claim_ids,
            ]
            group["member_table_claim_ids"] = (
                surviving_table_claim_ids
            )
            group["member_table_bboxes"] = active_table_bboxes
            group["member_table_text_element_ids"] = (
                active_table_text_ids
            )
            group["relations"] = active_relations
            group["member_visual_claim_ids"] = (
                surviving_member_visual_claim_ids
            )
            group["member_visual_bboxes"] = active_visual_bboxes
            group["member_visual_text_element_ids"] = (
                active_visual_text_ids
            )
            group["visual_relations"] = active_visual_relations

            if visual_record is not None:
                visual_record["output_group_active"] = True
                visual_record[
                    "composite_member_table_claim_ids"
                ] = surviving_table_claim_ids
                visual_record[
                    "composite_member_table_bboxes"
                ] = active_table_bboxes
                visual_record[
                    "composite_member_table_text_element_ids"
                ] = active_table_text_ids
                visual_record[
                    "composite_member_visual_claim_ids"
                ] = surviving_member_visual_claim_ids
                visual_record[
                    "composite_member_visual_bboxes"
                ] = active_visual_bboxes
                visual_record[
                    "composite_member_visual_text_element_ids"
                ] = active_visual_text_ids
            if visual_markdown is not None:
                visual_markdown["output_group_active"] = True
                visual_markdown[
                    "composite_member_table_claim_ids"
                ] = surviving_table_claim_ids
                visual_markdown[
                    "composite_member_table_bboxes"
                ] = active_table_bboxes
                visual_markdown[
                    "composite_member_visual_claim_ids"
                ] = surviving_member_visual_claim_ids
                visual_markdown[
                    "composite_member_visual_bboxes"
                ] = active_visual_bboxes
            if visual_claim is not None:
                visual_claim["metadata"][
                    "output_group_active"
                ] = True
                visual_claim["metadata"][
                    "composite_member_claim_ids"
                ] = sorted(
                    set(surviving_table_claim_ids).union(
                        surviving_member_visual_claim_ids
                    )
                )
                visual_claim["metadata"][
                    "composite_member_text_element_ids"
                ] = sorted(
                    set(active_table_text_ids).union(
                        active_visual_text_ids
                    )
                )
            for table_claim_id in surviving_table_claim_ids:
                table_candidate = table_candidate_by_claim_id.get(
                    table_claim_id
                )
                if table_candidate is not None:
                    table_candidate["output_group_active"] = True
                table_claim = table_claim_by_id.get(table_claim_id)
                if table_claim is not None:
                    table_claim["metadata"][
                        "output_group_active"
                    ] = True
            for member_visual_claim_id in surviving_member_visual_claim_ids:
                member_visual_record = visual_record_by_claim_id.get(
                    member_visual_claim_id
                )
                if member_visual_record is not None:
                    member_visual_record[
                        "output_group_active"
                    ] = True
                    member_visual_record[
                        "embedded_in_parent_visual"
                    ] = True
                    member_visual_record[
                        "parent_visual_claim_id"
                    ] = renderer_claim_id
                member_visual_markdown = (
                    visual_markdown_by_claim_id.get(
                        member_visual_claim_id
                    )
                )
                if member_visual_markdown is not None:
                    member_visual_markdown[
                        "output_group_active"
                    ] = True
                    member_visual_markdown[
                        "embedded_in_parent_visual"
                    ] = True
                    member_visual_markdown[
                        "parent_visual_claim_id"
                    ] = renderer_claim_id
                member_visual_claim = visual_claim_by_id.get(
                    member_visual_claim_id
                )
                if member_visual_claim is not None:
                    member_visual_claim["metadata"][
                        "output_group_active"
                    ] = True
                    member_visual_claim["metadata"][
                        "parent_visual_claim_id"
                    ] = renderer_claim_id
            continue

        inactive_group_ids.append(group["output_group_id"])
        restored_table_claim_ids.update(
            surviving_table_claim_ids
        )
        for table_claim_id in group["member_table_claim_ids"]:
            table_candidate = table_candidate_by_claim_id.get(
                table_claim_id
            )
            if table_candidate is not None:
                table_candidate["claimed_by_visual_region"] = False
                table_candidate[
                    "output_group_active"
                ] = False
            table_claim = table_claim_by_id.get(table_claim_id)
            if table_claim is not None:
                table_claim["metadata"][
                    "output_group_active"
                ] = False
        if visual_record is not None:
            visual_record["output_group_active"] = False
            visual_record["composite_member_table_claim_ids"] = []
            visual_record["composite_member_table_bboxes"] = []
            visual_record[
                "composite_member_table_text_element_ids"
            ] = []
            visual_record["composite_member_visual_claim_ids"] = []
            visual_record["composite_member_visual_bboxes"] = []
            visual_record[
                "composite_member_visual_text_element_ids"
            ] = []
        if visual_markdown is not None:
            visual_markdown["output_group_active"] = False
            visual_markdown["composite_member_table_claim_ids"] = []
            visual_markdown["composite_member_table_bboxes"] = []
            visual_markdown["composite_member_visual_claim_ids"] = []
            visual_markdown["composite_member_visual_bboxes"] = []
        if visual_claim is not None:
            visual_claim["metadata"][
                "output_group_active"
            ] = False
            visual_claim["metadata"][
                "composite_member_claim_ids"
            ] = []
            visual_claim["metadata"][
                "composite_member_text_element_ids"
            ] = []
        for member_visual_claim_id in group.get(
            "member_visual_claim_ids", []
        ):
            member_visual_record = visual_record_by_claim_id.get(
                member_visual_claim_id
            )
            if member_visual_record is not None:
                member_visual_record[
                    "embedded_in_parent_visual"
                ] = False
                member_visual_record.pop(
                    "parent_visual_claim_id", None
                )
                member_visual_record[
                    "output_group_active"
                ] = False
            member_visual_markdown = visual_markdown_by_claim_id.get(
                member_visual_claim_id
            )
            if member_visual_markdown is not None:
                member_visual_markdown[
                    "embedded_in_parent_visual"
                ] = False
                member_visual_markdown.pop(
                    "parent_visual_claim_id", None
                )
            member_visual_claim = visual_claim_by_id.get(
                member_visual_claim_id
            )
            if member_visual_claim is not None:
                member_visual_claim["metadata"][
                    "output_group_active"
                ] = False
                member_visual_claim["metadata"].pop(
                    "parent_visual_claim_id", None
                )

    return {
        "active_group_ids": active_group_ids,
        "inactive_group_ids": inactive_group_ids,
        "restored_table_claim_ids": sorted(
            restored_table_claim_ids
        ),
        "active_group_count": len(active_group_ids),
        "inactive_group_count": len(inactive_group_ids),
    }

def restore_inactive_composite_table_outputs(
    restored_table_claim_ids: list[str],
    table_candidates: list[dict],
    page_table_records: list[dict],
    page_table_markdown_elements: list[dict],
    page: fitz.Page,
    page_number: int,
    tables_dir: Path,
    table_images_dir: Path,
    *,
    dependencies: CrossDetectorOwnershipDependencies,
) -> tuple[list[dict], list[dict], float, list[dict]]:
    """恢复失效复合组中的独立表格输出。"""
    if not restored_table_claim_ids:
        return (
            page_table_records,
            page_table_markdown_elements,
            0.0,
            [],
        )

    restored_id_set = set(restored_table_claim_ids)
    candidate_by_claim_id = {
        f"page#{page_number}:table#{candidate['table_number']}": (
            candidate
        )
        for candidate in table_candidates
    }
    records_by_claim_id = {
        record.get("claim_id"): record
        for record in page_table_records
        if record.get("claim_id") is not None
    }
    markdown_by_claim_id = {
        element.get("claim_id"): element
        for element in page_table_markdown_elements
        if element.get("claim_id") is not None
    }
    screenshot_seconds = 0.0
    restore_actions: list[dict] = []

    for claim_id in sorted(restored_id_set):
        candidate = candidate_by_claim_id.get(claim_id)
        if candidate is None:
            continue
        emitted = dependencies.emit_table_candidate(
            candidate=candidate,
            page=page,
            page_number=page_number,
            tables_dir=tables_dir,
            table_images_dir=table_images_dir,
            table_image_dpi=dependencies.table_image_dpi,
            write_table_csv=dependencies.write_table_csv,
            table_to_markdown=dependencies.table_to_markdown,
            padded_table_candidate_clip=dependencies.padded_table_candidate_clip,
            save_rotated_table_image=dependencies.save_rotated_table_image,
        )
        records_by_claim_id[claim_id] = emitted["record"]
        if emitted["markdown_element"] is not None:
            emitted["markdown_element"]["claim_id"] = claim_id
            markdown_by_claim_id[claim_id] = emitted[
                "markdown_element"
            ]
        else:
            markdown_by_claim_id.pop(claim_id, None)
        screenshot_seconds += emitted["screenshot_seconds"]
        restore_actions.append(
            {
                "claim_id": claim_id,
                "reason": "composite_renderer_did_not_survive",
                "restored_output_path": emitted["record"].get(
                    "image_path"
                ),
            }
        )

    ordered_records = [
        records_by_claim_id[record["claim_id"]]
        for record in page_table_records
        if record.get("claim_id") in records_by_claim_id
    ]
    ordered_markdown_elements = [
        markdown_by_claim_id[claim_id]
        for claim_id in sorted(markdown_by_claim_id)
    ]
    return (
        ordered_records,
        ordered_markdown_elements,
        screenshot_seconds,
        restore_actions,
    )
