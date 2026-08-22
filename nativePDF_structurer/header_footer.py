"""检测数字原生PDF中重复出现的页眉与页尾结构。

本模块从原生图片和矢量drawing构建页边候选，聚合碎片化的页尾对象，学习
文档级重复模板，并返回确认排除的drawing索引。模板匹配使用几何位置和
规范化内容摘要，不使用原生文字作为模板键。

处理流程:
    1. 根据对象完整bbox预筛页边候选。
    2. 合并相邻页尾碎片并计算内容摘要。
    3. 从采样页独立学习页眉与页尾模板。
    4. 逐页匹配并返回drawing索引和审计记录。

关键不变量:
    * 页眉与页尾模板独立启用。
    * 不删除原生文字、页码和页尾图片。
    * 不执行OCR或语义文字匹配。

Detect repeated header and footer structures in digital-native PDFs.

This module builds page-edge candidates from native PDF images and vector
drawings, groups fragmented footer drawings, learns repeated document-level
templates, and reports the exact drawing indices confirmed for exclusion.
Matching uses geometry and normalized content signatures; native text is
not used as a template key.

Processing flow:
    1. Pre-filter objects by their complete page-edge bounding boxes.
    2. Merge adjacent footer fragments and compute content signatures.
    3. Learn header and footer templates independently from sampled pages.
    4. Match each page and return drawing indices with audit records.

Key invariants:
    * Header and footer templates are enabled independently.
    * Native text, page numbers, and footer images are not removed.
    * Candidate extraction performs no OCR or semantic text matching.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable

import fitz

from .geometry import rect_area
from .visual_assets import extract_source_image_blocks



def merge_footer_drawing_candidates(
    candidates: list[dict],
    page_rect: fitz.Rect,
    *,
    height_tolerance_points: float,
    x_tolerance_points: float,
) -> list[dict]:
    """把页尾同基线的相邻drawing合并为可审计的复合候选。"""
    drawing_candidates = [
        candidate
        for candidate in candidates
        if candidate["kind"] == "drawing"
    ]
    non_drawing_candidates = [
        candidate
        for candidate in candidates
        if candidate["kind"] != "drawing"
    ]
    ordered_drawings = sorted(
        drawing_candidates,
        key=lambda candidate: (
            candidate["bbox"].x0,
            candidate["bbox"].y0,
            candidate["bbox"].x1,
        ),
    )
    groups: list[list[dict]] = []

    for candidate in ordered_drawings:
        if not groups:
            groups.append([candidate])
            continue

        previous = groups[-1][-1]
        previous_rect = fitz.Rect(previous["bbox"])
        candidate_rect = fitz.Rect(candidate["bbox"])
        horizontal_gap = max(
            0.0,
            candidate_rect.x0 - previous_rect.x1,
        )
        same_baseline = (
            abs(
                candidate_rect.y0
                + candidate_rect.y1
                - previous_rect.y0
                - previous_rect.y1
            )
            / 2.0
            <= height_tolerance_points
        )
        similar_height = (
            abs(candidate_rect.height - previous_rect.height)
            <= height_tolerance_points
        )

        if (
            horizontal_gap <= x_tolerance_points
            and same_baseline
            and similar_height
        ):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    merged_candidates: list[dict] = []

    for group in groups:
        group_rect = fitz.Rect(group[0]["bbox"])

        for member in group[1:]:
            group_rect |= member["bbox"]

        left_edge_offset = group_rect.x0 - page_rect.x0
        right_edge_offset = page_rect.x1 - group_rect.x1
        outer_side = (
            "left"
            if left_edge_offset <= right_edge_offset
            else "right"
        )
        outer_offset = min(left_edge_offset, right_edge_offset)
        member_indices = sorted(
            {
                int(index)
                for member in group
                for index in member.get(
                    "source_drawing_indices",
                    [member.get("source_drawing_index")],
                )
                if index is not None
            }
        )

        if len(group) == 1:
            content_signature = group[0]["content_signature"]
        else:
            width = max(group_rect.width, 1e-6)
            height = max(group_rect.height, 1e-6)
            signature_payload = {
                "member_count": len(group),
                "members": [
                    {
                        "signature": member[
                            "content_signature"
                        ],
                        "relative_bbox": [
                            round(
                                (
                                    member["bbox"].x0
                                    - group_rect.x0
                                )
                                / width,
                                4,
                            ),
                            round(
                                (
                                    member["bbox"].y0
                                    - group_rect.y0
                                )
                                / height,
                                4,
                            ),
                            round(
                                (
                                    member["bbox"].x1
                                    - group_rect.x0
                                )
                                / width,
                                4,
                            ),
                            round(
                                (
                                    member["bbox"].y1
                                    - group_rect.y0
                                )
                                / height,
                                4,
                            ),
                        ],
                    }
                    for member in group
                ],
            }
            content_signature = hashlib.sha256(
                json.dumps(
                    signature_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        merged_candidates.append(
            {
                "kind": "drawing",
                "bbox": group_rect,
                "content_signature": content_signature,
                "height": group_rect.height,
                "width": group_rect.width,
                "x0": group_rect.x0,
                "edge_offset": page_rect.y1 - group_rect.y1,
                "source_drawing_index": (
                    member_indices[0]
                    if len(member_indices) == 1
                    else None
                ),
                "source_drawing_indices": member_indices,
                "composite_member_count": len(group),
                "horizontal_position_mode": "outer_edge",
                "outer_side": outer_side,
                "outer_offset": outer_offset,
            }
        )

    return [*non_drawing_candidates, *merged_candidates]

def extract_header_footer_candidates(
    page: fitz.Page,
    page_dict: dict,
    edge: str,
    drawings: list[dict],
    *,
    edge_page_ratio_by_edge: dict[str, float],
    edge_tolerance_points: float,
    candidate_count: int,
    max_background_area_page_ratio: float,
    image_content_signature: Callable[[bytes | None], str | None],
    drawing_content_signature: Callable[[dict], str],
    height_tolerance_points: float,
    x_tolerance_points: float,
) -> list[dict]:
    """提取页眉或页尾附近的原始图片、矢量候选。"""
    if edge not in {"header", "footer"}:
        raise ValueError(f"unsupported edge: {edge}")

    edge_limit = (
        page.rect.height
        * edge_page_ratio_by_edge[edge]
    )
    page_area = rect_area(page.rect)
    candidates: list[dict] = []

    def candidate_edge_offset(bbox: fitz.Rect) -> float | None:
        edge_offset = (
            bbox.y0 - page.rect.y0
            if edge == "header"
            else page.rect.y1 - bbox.y1
        )
        if edge_offset < -edge_tolerance_points:
            return None
        if edge_offset > edge_limit:
            return None
        if (
            edge == "footer"
            and bbox.y0
            < page.rect.y1
            - edge_limit
            - edge_tolerance_points
        ):
            return None
        return edge_offset

    def add_candidate(
        kind: str,
        bbox: fitz.Rect,
        content_signature: str | None,
        source_drawing_index: int | None = None,
    ) -> None:
        if content_signature is None:
            return

        edge_offset = candidate_edge_offset(bbox)
        if edge_offset is None:
            return

        candidates.append(
            {
                "kind": kind,
                "bbox": bbox,
                "content_signature": content_signature,
                "height": bbox.height,
                "width": bbox.width,
                "x0": bbox.x0,
                "edge_offset": edge_offset,
                "source_drawing_index": source_drawing_index,
                "source_drawing_indices": (
                    [source_drawing_index]
                    if source_drawing_index is not None
                    else []
                ),
                "composite_member_count": 1,
            }
        )

    if edge == "header":
        for image in extract_source_image_blocks(page_dict):
            image_bbox = fitz.Rect(image["bbox"])
            if candidate_edge_offset(image_bbox) is None:
                continue
            add_candidate(
                kind="image",
                bbox=image_bbox,
                content_signature=image_content_signature(
                    image["image_bytes"]
                ),
            )

    page_drawings = drawings
    for drawing_index, drawing in enumerate(page_drawings):
        drawing_rect = fitz.Rect(drawing["rect"])
        drawing_area = rect_area(drawing_rect)

        if drawing_rect.width <= 0 and drawing_rect.height <= 0:
            continue

        if (
            page_area > 0
            and drawing_area / page_area
            >= max_background_area_page_ratio
        ):
            continue

        if candidate_edge_offset(drawing_rect) is None:
            continue

        add_candidate(
            kind="drawing",
            bbox=drawing_rect,
            content_signature=drawing_content_signature(drawing),
            source_drawing_index=drawing_index,
        )

    prepared_candidates = (
        merge_footer_drawing_candidates(
            candidates=candidates,
            page_rect=page.rect,
            height_tolerance_points=height_tolerance_points,
            x_tolerance_points=x_tolerance_points,
        )
        if edge == "footer"
        else candidates
    )
    return sorted(
        prepared_candidates,
        key=lambda candidate: (
            candidate["edge_offset"],
            candidate["x0"],
            candidate["height"],
            candidate["kind"],
        ),
    )[:candidate_count]

def header_footer_candidates_match(
    first: dict,
    second: dict,
    *,
    height_tolerance_points: float,
    x_tolerance_points: float,
    edge_tolerance_points: float,
) -> bool:
    """判断两个图片/矢量候选是否为同一重复页眉或页尾对象。"""
    uses_outer_edge_position = (
        first.get("horizontal_position_mode") == "outer_edge"
        and second.get("horizontal_position_mode")
        == "outer_edge"
    )
    horizontal_position_matches = (
        abs(first["outer_offset"] - second["outer_offset"])
        <= x_tolerance_points
        if uses_outer_edge_position
        else abs(first["x0"] - second["x0"])
        <= x_tolerance_points
    )
    width_matches = (
        abs(first.get("width", 0.0) - second.get("width", 0.0))
        <= x_tolerance_points
        if uses_outer_edge_position
        else True
    )
    return (
        first["kind"] == second["kind"]
        and abs(first["height"] - second["height"])
        <= height_tolerance_points
        and horizontal_position_matches
        and width_matches
        and abs(first["edge_offset"] - second["edge_offset"])
        <= edge_tolerance_points
        and first["content_signature"]
        == second["content_signature"]
    )

def build_header_footer_templates(
    document: fitz.Document,
    drawings_cache: dict[int, list[dict]],
    *,
    sample_start_page: int,
    sample_max_page: int,
    candidate_count: int,
    detect_footer: bool,
    minimum_matched_objects_by_edge: dict[str, int],
    minimum_support_ratio_by_edge: dict[str, float],
    edge_page_ratio_by_edge: dict[str, float],
    edge_tolerance_points: float,
    height_tolerance_points: float,
    x_tolerance_points: float,
    max_background_area_page_ratio: float,
    image_content_signature: Callable[[bytes | None], str | None],
    drawing_content_signature: Callable[[dict], str],
) -> dict[str, dict]:
    """从前若干页构建页眉、页尾图片/矢量模板。"""
    source_page_count = document.page_count
    first_page = (
        sample_start_page
        if source_page_count >= sample_start_page
        else 1
    )
    last_page = min(
        source_page_count,
        sample_max_page,
    )
    sample_page_numbers = list(
        range(first_page, last_page + 1)
    )
    results: dict[str, dict] = {}

    detection_edges = (
        ["header", "footer"]
        if detect_footer
        else ["header"]
    )
    for edge in detection_edges:
        minimum_page_support = max(
            1,
            math.ceil(
                len(sample_page_numbers)
                * minimum_support_ratio_by_edge[edge]
            ),
        )
        minimum_matched_objects = (
            minimum_matched_objects_by_edge[edge]
        )
        clusters: list[dict] = []

        for page_number in sample_page_numbers:
            page = document.load_page(page_number - 1)
            page_dict = page.get_text("dict", sort=False)
            if page.number in drawings_cache:
                page_drawings = drawings_cache[page.number]
            else:
                page_drawings = page.get_drawings()
                drawings_cache[page.number] = page_drawings
            candidates = extract_header_footer_candidates(
                page=page,
                page_dict=page_dict,
                edge=edge,
                drawings=page_drawings,
                edge_page_ratio_by_edge=(
                    edge_page_ratio_by_edge
                ),
                edge_tolerance_points=(
                    edge_tolerance_points
                ),
                candidate_count=candidate_count,
                max_background_area_page_ratio=(
                    max_background_area_page_ratio
                ),
                image_content_signature=image_content_signature,
                drawing_content_signature=drawing_content_signature,
                height_tolerance_points=(
                    height_tolerance_points
                ),
                x_tolerance_points=x_tolerance_points,
            )

            for candidate in candidates:
                matching_cluster = next(
                    (
                        cluster
                        for cluster in clusters
                        if header_footer_candidates_match(
                            candidate,
                            cluster["representative"],
                            height_tolerance_points=(
                                height_tolerance_points
                            ),
                            x_tolerance_points=(
                                x_tolerance_points
                            ),
                            edge_tolerance_points=(
                                edge_tolerance_points
                            ),
                        )
                    ),
                    None,
                )

                if matching_cluster is None:
                    clusters.append(
                        {
                            "representative": candidate,
                            "members": [
                                {
                                    "page_number": page_number,
                                    **candidate,
                                }
                            ],
                        }
                    )
                    continue

                if any(
                    member["page_number"] == page_number
                    for member in matching_cluster["members"]
                ):
                    continue

                matching_cluster["members"].append(
                    {
                        "page_number": page_number,
                        **candidate,
                    }
                )

        stable_clusters = [
            cluster
            for cluster in clusters
            if len(
                {
                    member["page_number"]
                    for member in cluster["members"]
                }
            )
            >= minimum_page_support
        ]
        stable_clusters.sort(
            key=lambda cluster: (
                -len(
                    {
                        member["page_number"]
                        for member in cluster["members"]
                    }
                ),
                cluster["representative"]["edge_offset"],
                cluster["representative"]["x0"],
            )
        )
        selected_clusters = stable_clusters[
            :candidate_count
        ]
        page_match_counts: Counter = Counter()

        for cluster in selected_clusters:
            for member in cluster["members"]:
                page_match_counts[member["page_number"]] += 1

        cooccurring_pages = sorted(
            page_number
            for page_number, matched_count in page_match_counts.items()
            if matched_count >= minimum_matched_objects
        )
        template_enabled = (
            len(selected_clusters)
            >= minimum_matched_objects
            and len(cooccurring_pages)
            >= minimum_page_support
        )
        results[edge] = {
            "enabled": template_enabled,
            "sample_page_numbers": sample_page_numbers,
            "minimum_page_support": minimum_page_support,
            "minimum_matched_objects": minimum_matched_objects,
            "cooccurring_pages": cooccurring_pages,
            "templates": [
                cluster["representative"]
                for cluster in selected_clusters
            ]
            if template_enabled
            else [],
        }

    if not detect_footer:
        results["footer"] = {
            "enabled": False,
            "sample_page_numbers": [],
            "minimum_page_support": 0,
            "minimum_matched_objects": 0,
            "cooccurring_pages": [],
            "templates": [],
        }

    return results

def detect_page_header_footer_bands(
    page: fitz.Page,
    page_dict: dict,
    templates: dict[str, dict],
    drawings: list[dict],
    *,
    detect_footer: bool,
    minimum_matched_objects_by_edge: dict[str, int],
    band_padding_points: float,
    edge_page_ratio_by_edge: dict[str, float],
    edge_tolerance_points: float,
    candidate_count: int,
    max_background_area_page_ratio: float,
    image_content_signature: Callable[[bytes | None], str | None],
    drawing_content_signature: Callable[[dict], str],
    height_tolerance_points: float,
    x_tolerance_points: float,
) -> tuple[list[fitz.Rect], dict]:
    """返回本页需要在后续图像、矢量、公式和文本流中排除的页眉/页尾带。"""
    bands: list[fitz.Rect] = []
    record: dict[str, object] = {
        "header": {
            "matched_object_count": 0,
            "band": None,
            "matched_drawing_indices": [],
            "matched_drawing_bboxes": [],
        },
        "footer": {
            "matched_object_count": 0,
            "band": None,
            "matched_drawing_indices": [],
            "matched_drawing_bboxes": [],
        },
    }

    detection_edges = (
        ["header", "footer"]
        if detect_footer
        else ["header"]
    )
    for edge in detection_edges:
        edge_templates = templates.get(edge, {})

        if not edge_templates.get("enabled"):
            continue

        candidates = extract_header_footer_candidates(
            page=page,
            page_dict=page_dict,
            edge=edge,
            drawings=drawings,
            edge_page_ratio_by_edge=(
                edge_page_ratio_by_edge
            ),
            edge_tolerance_points=(
                edge_tolerance_points
            ),
            candidate_count=candidate_count,
            max_background_area_page_ratio=(
                max_background_area_page_ratio
            ),
            image_content_signature=image_content_signature,
            drawing_content_signature=drawing_content_signature,
            height_tolerance_points=(
                height_tolerance_points
            ),
            x_tolerance_points=x_tolerance_points,
        )
        matched_candidates: list[dict] = []
        used_template_indices: set[int] = set()

        for candidate in candidates:
            for template_index, template in enumerate(
                edge_templates["templates"]
            ):
                if template_index in used_template_indices:
                    continue

                if header_footer_candidates_match(
                    candidate,
                    template,
                    height_tolerance_points=(
                        height_tolerance_points
                    ),
                    x_tolerance_points=(
                        x_tolerance_points
                    ),
                    edge_tolerance_points=(
                        edge_tolerance_points
                    ),
                ):
                    matched_candidates.append(candidate)
                    used_template_indices.add(template_index)
                    break

        record[edge]["matched_object_count"] = len(
            matched_candidates
        )
        minimum_matched_objects = edge_templates.get(
            "minimum_matched_objects",
            minimum_matched_objects_by_edge[edge],
        )
        if len(matched_candidates) < minimum_matched_objects:
            continue

        matched_drawing_candidates = [
            candidate
            for candidate in matched_candidates
            if candidate["kind"] == "drawing"
            and candidate.get("source_drawing_indices")
        ]
        record[edge]["matched_drawing_indices"] = sorted(
            {
                int(index)
                for candidate in matched_drawing_candidates
                for index in candidate.get(
                    "source_drawing_indices", []
                )
            }
        )
        record[edge]["matched_drawing_bboxes"] = [
            [round(value, 3) for value in candidate["bbox"]]
            for candidate in matched_drawing_candidates
        ]
        record[edge]["matched_composite_member_counts"] = [
            int(candidate.get("composite_member_count", 1))
            for candidate in matched_drawing_candidates
        ]
        record[edge]["matched_outer_sides"] = [
            candidate.get("outer_side")
            for candidate in matched_drawing_candidates
            if candidate.get("outer_side") is not None
        ]

        if edge == "header":
            band = fitz.Rect(
                page.rect.x0,
                page.rect.y0,
                page.rect.x1,
                min(
                    page.rect.y1,
                    max(
                        candidate["bbox"].y1
                        for candidate in matched_candidates
                    )
                    + band_padding_points,
                ),
            )
        else:
            band = fitz.Rect(
                page.rect.x0,
                max(
                    page.rect.y0,
                    min(
                        candidate["bbox"].y0
                        for candidate in matched_candidates
                    )
                    - band_padding_points,
                ),
                page.rect.x1,
                page.rect.y1,
            )

        bands.append(band)
        record[edge]["band"] = [
            round(value, 3) for value in band
        ]

    return bands, record

def filter_confirmed_header_footer_drawings(
    drawings: list[dict],
    header_footer_record: dict,
) -> tuple[list[dict], dict]:
    """移除已确认重复页眉/页尾矢量，并返回过滤审计信息。"""
    matched_indices: set[int] = set()
    matched_by_edge: dict[str, list[int]] = {}

    for edge in ("header", "footer"):
        edge_record = header_footer_record.get(edge, {})
        indices = [
            int(index)
            for index in edge_record.get(
                "matched_drawing_indices", []
            )
        ]
        matched_by_edge[edge] = indices
        matched_indices.update(indices)

    filtered_drawings = [
        drawing
        for drawing_index, drawing in enumerate(drawings)
        if drawing_index not in matched_indices
    ]

    header_footer_record["vector_filter"] = {
        "removed_drawing_count": len(matched_indices),
        "remaining_drawing_count": len(filtered_drawings),
        "removed_drawing_indices_by_edge": matched_by_edge,
        "removed_drawing_bboxes_by_edge": {
            edge: [
                [round(value, 3) for value in drawing["rect"]]
                for index, drawing in enumerate(drawings)
                if index in set(indices)
            ]
            for edge, indices in matched_by_edge.items()
        },
    }
    return filtered_drawings, header_footer_record


def extract_header_text_for_markdown(
    text_lines: list[dict],
    header_band: fitz.Rect | None,
    *,
    overlap_threshold: float,
) -> tuple[str, list[dict]]:
    """提取页眉带内原生文字，供可选Markdown字段使用。"""

    if header_band is None:
        return "", []

    selected_lines: list[dict] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    for line in text_lines:
        line_rect = fitz.Rect(line["bbox"])
        line_area = rect_area(line_rect)
        if line_area <= 0:
            continue
        intersection = line_rect & header_band
        if intersection.is_empty:
            continue
        if rect_area(intersection) / line_area < overlap_threshold:
            continue

        normalized_text = re.sub(r"\s+", " ", line["text"]).strip()
        if not normalized_text:
            continue
        key = (
            normalized_text,
            tuple(round(value, 3) for value in line_rect),
        )
        if key in seen:
            continue
        seen.add(key)
        selected_lines.append({**line, "text": normalized_text})

    selected_lines.sort(
        key=lambda line: (
            round(line["bbox"].y0, 3),
            round(line["bbox"].x0, 3),
        )
    )
    return (
        " ".join(line["text"] for line in selected_lines).strip(),
        selected_lines,
    )
