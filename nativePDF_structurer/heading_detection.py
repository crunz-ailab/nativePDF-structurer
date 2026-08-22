"""根据PDF原生文字证据识别Markdown标题。

模块构建文档级字体画像，执行保守
标题判据，匹配PDF目录项，并分配Markdown标题等级；目录同文匹配与字体
推断是相互独立的通道。

关键不变量:
    小于正文基准的字号不会成为标题；页眉页尾、表格、视觉区域、图片和公式
    已拥有的文字不会重复进入标题通道；默认关闭字体推断时仍可使用目录匹配。

Classify Markdown headings from native PDF text evidence.

Builds document typography profiles, evaluates conservative heading
predicates, matches PDF outline entries, and assigns Markdown heading levels.
Outline text matching and font-based inference are independent channels.
"""

from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from collections.abc import Callable

import fitz

from .outline_headings import (
    assign_pdf_outline_entries_to_lines,
    assign_pdf_outline_text_matches_to_lines,
    outline_text_matches_for_audit,
)

HEADING_MIN_PLAUSIBLE_FONT_SIZE_POINTS = 2.0
HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS = 0.35

def weighted_median_font_size(samples: list[tuple[float, int]]) -> float:
    valid = [(float(size), int(weight)) for size, weight in samples if size > 0 and weight > 0]
    if not valid: return 0.0
    valid.sort(key=lambda item: item[0])
    target = sum(weight for _, weight in valid) / 2.0
    accumulated = 0
    for size, weight in valid:
        accumulated += weight
        if accumulated >= target: return round(size, 3)
    return round(valid[-1][0], 3)

def text_line_style_signature(line: dict) -> dict:
    samples = [s for s in line.get("font_stats", []) if float(s.get("size", 0) or 0) >= HEADING_MIN_PLAUSIBLE_FONT_SIZE_POINTS and int(s.get("character_count", 0) or 0) > 0]
    dominant = max(samples, key=lambda s: (int(s.get("character_count", 0) or 0), float(s.get("size", 0) or 0)), default=None)
    font = str(dominant.get("font", "") or "") if dominant else ""
    size = float(line.get("effective_font_size", 0) or 0)
    cluster = round(round(size / HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS) * HEADING_FONT_SIZE_CLUSTER_TOLERANCE_POINTS, 3) if size > 0 else 0.0
    bold = float(line.get("bold_character_ratio", 0) or 0) >= 0.60
    return {"font": font, "size": cluster, "bold": bold, "key": f"{font}\x1f{cluster:.3f}\x1f{int(bold)}"}


def build_page_body_font_context(
    text_lines: list[dict],
    document_body_font_size: float,
    *,
    font_size_cluster_tolerance_points: float,
    minimum_visible_characters: int,
    cluster_weight_ratio: float,
    minimum_cluster_characters: int,
) -> dict:
    """估计当前页面正文可能使用的最高常用字号。"""
    cluster_weights: Counter[float] = Counter()
    visible_character_count = 0
    for line in text_lines:
        for sample in line.get("font_stats", []):
            size = float(sample.get("size", 0.0) or 0.0)
            weight = int(sample.get("character_count", 0) or 0)
            if size <= 0 or weight <= 0:
                continue
            cluster_size = round(round(size / font_size_cluster_tolerance_points) * font_size_cluster_tolerance_points, 3)
            cluster_weights[cluster_size] += weight
            visible_character_count += weight
    if visible_character_count < minimum_visible_characters or not cluster_weights:
        return {"enabled": False, "visible_character_count": visible_character_count, "body_font_size_ceiling": document_body_font_size, "frequent_clusters": []}
    dominant_weight = max(cluster_weights.values())
    minimum_weight = max(minimum_cluster_characters, math.ceil(dominant_weight * cluster_weight_ratio))
    frequent_clusters = sorted(size for size, weight in cluster_weights.items() if weight >= minimum_weight)
    local_body_ceiling = max(frequent_clusters) if frequent_clusters else document_body_font_size
    return {"enabled": True, "visible_character_count": visible_character_count, "dominant_character_weight": dominant_weight, "minimum_cluster_character_weight": minimum_weight, "body_font_size_ceiling": max(document_body_font_size, local_body_ceiling), "frequent_clusters": frequent_clusters}


def looks_like_toc_entry(text: str) -> bool:
    normalized = " ".join(text.split())
    return bool(re.search(r"(?:\.{3,}|…{2,}|·{3,})\s*\d{1,4}\s*$", normalized))


def looks_like_numeric_or_rule_label(text: str) -> bool:
    normalized = "".join(text.split())
    return bool(normalized and re.fullmatch(r"[\d.,%％+\-−–—─━_/()（）:：]+", normalized))


def looks_like_short_colon_label(text: str) -> bool:
    normalized = "".join(str(text).split())
    return bool(normalized and len(normalized) <= 8 and normalized.endswith((":", "：")) and not re.search(r"[。！？!?]", normalized[:-1]))


def looks_like_heading_text(text: str) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if re.match(r"^(?:第\s*[一二三四五六七八九十百千万\d]+\s*[章节篇部]|[一二三四五六七八九十百千万]+[、.)])", normalized):
        return True
    for pattern in (r"^\d+(?:\.\d+){1,3}\s*[.)、]?\s*(.+)$", r"^\d+[.)、]\s*(.+)$", r"^\d+\s+(.+)$"):
        match = re.match(pattern, normalized)
        if match:
            return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", match.group(1)))
    return False

def build_document_typography_profile(
    page_line_sets: list[list[dict]],
    page_rects: list[fitz.Rect] | None = None,
    *,
    enable_heading_detection: bool,
    font_size_cluster_tolerance_points: float,
    body_style_min_characters: int,
    body_style_min_long_line_ratio: float,
    body_style_min_multiline_block_ratio: float,
    body_style_high_character_share: float,
    style_min_short_line_ratio: float,
    running_header_top_page_ratio: float,
    running_header_min_page_support_ratio: float,
    running_header_min_page_support: int,
) -> dict:
    """统计文档级正文字号和可用于标题的字号簇。"""
    tolerance = font_size_cluster_tolerance_points
    cluster_stats: dict[float, dict] = {}
    style_stats: dict[str, dict] = {}
    top_layout_pages: dict[str, set[int]] = defaultdict(set)
    total_visible_character_count = 0

    for page_index, page_lines in enumerate(page_line_sets):
        page_rect = (
            fitz.Rect(page_rects[page_index])
            if page_rects is not None
            and page_index < len(page_rects)
            else None
        )
        page_style_keys: set[str] = set()
        block_style_counts: dict[int, Counter[str]] = defaultdict(
            Counter
        )
        for block_line in page_lines:
            if block_line.get("angle", 0) != 0:
                continue
            block_style_counts[
                int(block_line.get("block_index", -1))
            ][text_line_style_signature(block_line)["key"]] += 1
        for line in page_lines:
            if line.get("angle", 0) != 0:
                continue
            style_signature = text_line_style_signature(line)
            style_key = style_signature["key"]
            character_count = int(
                line.get("visible_character_count", 0) or 0
            )
            line_rect = fitz.Rect(line["bbox"]).normalize()
            line_text = " ".join(
                str(line.get("text", "")).split()
            )
            width_ratio = (
                line_rect.width / page_rect.width
                if page_rect is not None and page_rect.width > 0
                else 0.0
            )
            is_long_line = (
                len(line_text) >= 35 or width_ratio >= 0.55
            )
            is_short_line = (
                len(line_text) <= 30
                and (page_rect is None or width_ratio <= 0.65)
            )
            is_multiline_block = (
                block_style_counts[
                    int(line.get("block_index", -1))
                ][style_key]
                >= 2
            )
            style = style_stats.setdefault(
                style_key,
                {
                    **style_signature,
                    "character_weight": 0,
                    "line_count": 0,
                    "page_count": 0,
                    "long_line_count": 0,
                    "short_line_count": 0,
                    "multiline_block_line_count": 0,
                    "top_line_count": 0,
                },
            )
            style["character_weight"] += character_count
            style["line_count"] += 1
            style["long_line_count"] += int(is_long_line)
            style["short_line_count"] += int(is_short_line)
            style["multiline_block_line_count"] += int(
                is_multiline_block
            )
            total_visible_character_count += character_count
            page_style_keys.add(style_key)

            if (
                page_rect is not None
                and page_rect.height > 0
                and line_rect.y0
                <= page_rect.height
                * running_header_top_page_ratio
            ):
                style["top_line_count"] += 1
                y_ratio_bucket = round(
                    line_rect.y0 / page_rect.height,
                    2,
                )
                top_layout_pages[
                    f"{style_key}\x1f{y_ratio_bucket:.2f}"
                ].add(page_index)

            for sample in line.get("font_stats", []):
                size = float(sample.get("size", 0.0) or 0.0)
                weight = int(sample.get("character_count", 0) or 0)
                if size <= 0 or weight <= 0:
                    continue
                cluster_size = round(
                    round(size / tolerance) * tolerance,
                    3,
                )
                if cluster_size <= 0:
                    continue
                cluster = cluster_stats.setdefault(
                    cluster_size,
                    {
                        "size": cluster_size,
                        "character_weight": 0,
                        "span_count": 0,
                        "bold_character_weight": 0,
                    },
                )
                cluster["character_weight"] += weight
                cluster["span_count"] += 1
                if sample.get("bold"):
                    cluster["bold_character_weight"] += weight

        for style_key in page_style_keys:
            style_stats[style_key]["page_count"] += 1

    clusters = sorted(
        cluster_stats.values(),
        key=lambda item: item["size"],
    )
    if not clusters:
        return {
            "enabled": enable_heading_detection,
            "body_font_size": None,
            "clusters": [],
            "heading_size_clusters": [],
            "body_style_keys": [],
            "heading_style_keys": [],
            "style_roles": [],
            "running_header_layout_templates": [],
            "reason": "no_horizontal_text_font_samples",
        }

    body_cluster = max(
        clusters,
        key=lambda item: (
            item["character_weight"],
            -item["size"],
        ),
    )
    body_font_size = float(body_cluster["size"])
    style_roles: list[dict] = []
    body_style_keys: list[str] = []
    heading_style_keys: list[str] = []
    for style in style_stats.values():
        line_count = max(1, int(style["line_count"]))
        character_share = (
            style["character_weight"]
            / total_visible_character_count
            if total_visible_character_count > 0
            else 0.0
        )
        long_line_ratio = style["long_line_count"] / line_count
        short_line_ratio = style["short_line_count"] / line_count
        multiline_block_ratio = (
            style["multiline_block_line_count"] / line_count
        )
        is_body_style = (
            style["character_weight"]
            >= body_style_min_characters
            and (
                multiline_block_ratio
                >= body_style_min_multiline_block_ratio
                or long_line_ratio
                >= body_style_min_long_line_ratio
                or character_share
                >= body_style_high_character_share
            )
        )
        is_heading_style = (
            not is_body_style
            and style["size"] > 0
            and short_line_ratio
            >= style_min_short_line_ratio
        )
        role = (
            "body"
            if is_body_style
            else "heading_candidate"
            if is_heading_style
            else "other"
        )
        role_record = {
            **style,
            "character_share": round(character_share, 6),
            "long_line_ratio": round(long_line_ratio, 6),
            "short_line_ratio": round(short_line_ratio, 6),
            "multiline_block_ratio": round(
                multiline_block_ratio,
                6,
            ),
            "role": role,
        }
        style_roles.append(role_record)
        if is_body_style:
            body_style_keys.append(style["key"])
        if is_heading_style:
            heading_style_keys.append(style["key"])

    heading_size_clusters = sorted(
        {
            float(style["size"])
            for style in style_roles
            if style["role"] == "heading_candidate"
            and style["character_weight"] >= 2
        },
        reverse=True,
    )

    minimum_running_header_support = max(
        running_header_min_page_support,
        math.ceil(
            len(page_line_sets)
            * running_header_min_page_support_ratio
        ),
    )
    running_header_layout_templates = []
    for template_key, supported_pages in top_layout_pages.items():
        style_key, y_ratio_text = template_key.rsplit("\x1f", 1)
        if (
            style_key not in heading_style_keys
            or len(supported_pages)
            < minimum_running_header_support
        ):
            continue
        running_header_layout_templates.append(
            {
                "style_key": style_key,
                "y_ratio": float(y_ratio_text),
                "page_support": len(supported_pages),
            }
        )

    return {
        "enabled": enable_heading_detection,
        "body_font_size": body_font_size,
        "body_character_weight": body_cluster[
            "character_weight"
        ],
        "clusters": clusters,
        "heading_size_clusters": heading_size_clusters,
        "body_style_keys": sorted(body_style_keys),
        "heading_style_keys": sorted(heading_style_keys),
        "style_roles": sorted(
            style_roles,
            key=lambda item: (
                -item["character_weight"],
                item["font"],
                item["size"],
            ),
        ),
        "running_header_layout_templates": sorted(
            running_header_layout_templates,
            key=lambda item: (
                item["y_ratio"],
                item["style_key"],
            ),
        ),
        "profile_page_count": len(page_line_sets),
        "profile_visible_character_count": (
            total_visible_character_count
        ),
        "reason": "document_wide_multi_style_role_profile",
    }

def annotate_heading_levels(
    text_lines: list[dict],
    typography_profile: dict,
    excluded_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
    outline_entries: list[dict] | None = None,
    page_title_outline_entry_index: int | None = None,
    *,
    enable_font_heading_inference: bool,
    enable_heading_detection: bool,
    enable_pdf_outline_heading_assignment: bool,
    candidate_max_block_line_count: int,
    font_size_cluster_tolerance_points: float,
    local_body_cluster_weight_ratio: float,
    local_body_min_cluster_characters: int,
    local_body_min_size_delta_points: float,
    local_body_min_visible_characters: int,
    max_markdown_level: int,
    max_text_characters: int,
    outline_shared_target_tolerance_points: float,
    outline_target_max_vertical_distance_points: float,
    outline_target_max_x_distance_page_ratio: float,
    same_body_min_bold_character_ratio: float,
    same_body_requires_bold: bool,
    line_matches_running_header_layout: Callable[..., bool],
    rect_is_mostly_inside_any: Callable[..., bool],
) -> list[dict]:
    """为正文文本行标记Markdown标题等级并返回精简审计记录。"""
    if not enable_heading_detection:
        return []
    heading_style_keys = (
        set(typography_profile.get("heading_style_keys", []))
        if enable_font_heading_inference
        else set()
    )
    heading_size_clusters = (
        list(typography_profile.get("heading_size_clusters", []))
        if enable_font_heading_inference
        else []
    )

    outline_entries = outline_entries or []
    records: list[dict] = []
    candidate_lines: list[dict] = []
    candidate_metadata_by_line_id: dict[int, dict] = {}
    for line in text_lines:
        line.pop("heading_level", None)
        line.pop("heading_reason", None)
        line.pop("suppress_as_duplicate_page_title", None)
        line.pop("heading_text_override", None)
        line.pop(
            "suppress_as_outline_heading_continuation",
            None,
        )

    outline_match_lines: list[dict] = []
    for line in text_lines:
        if (
            line.get("kind", "text") != "text"
            or line.get("angle", 0) != 0
            or line.get("figure_role") == "figure_caption"
        ):
            continue
        line_rect = fitz.Rect(line["bbox"]).normalize()
        if any(
            rect_is_mostly_inside_any(
                rect=line_rect,
                container_rects=[fitz.Rect(rect)],
            )
            for rect in excluded_rects
        ):
            continue
        if not " ".join(str(line.get("text", "")).split()):
            continue
        outline_match_lines.append(line)

    outline_text_assignment_by_line_id = (
        assign_pdf_outline_text_matches_to_lines(
            candidate_lines=outline_match_lines,
            outline_entries=outline_entries,
        )
        if enable_pdf_outline_heading_assignment
        else {}
    )
    outline_text_owned_line_ids = {
        id(line)
        for match_record in (
            outline_text_assignment_by_line_id.values()
        )
        for line in match_record.get("lines", [])
    }
    outline_text_assigned_entry_indices = {
        int(
            match_record.get("entry", {}).get(
                "entry_index", -1
            )
        )
        for match_record in (
            outline_text_assignment_by_line_id.values()
        )
    }
    page_body_lines = []
    for body_line in text_lines:
        if (
            body_line.get("kind", "text") != "text"
            or body_line.get("angle", 0) != 0
        ):
            continue
        body_line_rect = fitz.Rect(body_line["bbox"]).normalize()
        if any(
            rect_is_mostly_inside_any(
                rect=body_line_rect,
                container_rects=[fitz.Rect(rect)],
            )
            for rect in excluded_rects
        ):
            continue
        if id(body_line) not in outline_text_owned_line_ids:
            page_body_lines.append(body_line)
    page_body_context = (
        build_page_body_font_context(
            text_lines=page_body_lines,
            document_body_font_size=float(
                typography_profile.get("body_font_size", 0.0) or 0.0
            ),
            font_size_cluster_tolerance_points=(
                font_size_cluster_tolerance_points
            ),
            minimum_visible_characters=(
                local_body_min_visible_characters
            ),
            cluster_weight_ratio=local_body_cluster_weight_ratio,
            minimum_cluster_characters=(
                local_body_min_cluster_characters
            ),
        )
        if enable_font_heading_inference
        else {
            "enabled": False,
            "body_font_size_ceiling": 0.0,
            "reason": "font_heading_inference_disabled",
        }
    )
    page_body_font_size_ceiling = float(
        page_body_context.get("body_font_size_ceiling", 0.0) or 0.0
    )
    block_lines_by_index: dict[int, list[dict]] = defaultdict(list)
    if enable_font_heading_inference:
        for block_line in text_lines:
            if (
                block_line.get("kind", "text") == "text"
                and block_line.get("angle", 0) == 0
            ):
                block_lines_by_index[
                    int(block_line.get("block_index", -1))
                ].append(block_line)

    for line in text_lines:
        # line.pop("heading_level", None)
        # line.pop("heading_reason", None)
        if line.get("figure_role") == "figure_caption":
            continue
        if (
            line.get("kind", "text") != "text"
            or line.get("angle", 0) != 0
        ):
            continue
        line_rect = fitz.Rect(line["bbox"]).normalize()
        if any(
            rect_is_mostly_inside_any(
                rect=line_rect,
                container_rects=[fitz.Rect(rect)],
            )
            for rect in excluded_rects
        ):
            continue

        text = " ".join(line.get("text", "").split())
        if id(line) in outline_text_owned_line_ids:
            continue
        if (
            not enable_font_heading_inference
            or not heading_style_keys
            or not heading_size_clusters
        ):
            continue
        if (
            len(text) < 2
            or len(text) > max_text_characters
            or looks_like_toc_entry(text)
            or looks_like_numeric_or_rule_label(text)
            or looks_like_short_colon_label(text)
            or re.match(r"^[●○•▪◦◆◇■□]", text)
        ):
            continue

        style_signature = text_line_style_signature(line)
        if style_signature["key"] not in heading_style_keys:
            continue

        effective_size = float(
            line.get("effective_font_size", 0.0) or 0.0
        )
        if effective_size <= 0:
            continue
        bold_character_ratio = float(
            line.get("bold_character_ratio", 0.0) or 0.0
        )
        same_body_numbered_heading = (
            same_body_requires_bold
            and bold_character_ratio
            >= same_body_min_bold_character_ratio
            and looks_like_heading_text(text)
        )
        has_meaningful_size_separation = (
            effective_size
            >= page_body_font_size_ceiling
            + local_body_min_size_delta_points
        )
        if (
            page_body_context.get("enabled")
            and not has_meaningful_size_separation
            and not same_body_numbered_heading
        ):
            continue
        block_line_count = int(
            line.get("block_line_count", 1) or 1
        )
        if block_line_count > candidate_max_block_line_count:
            same_block_lines = block_lines_by_index.get(
                int(line.get("block_index", -1)),
                [],
            )
            following_block_lines = [
                block_line
                for block_line in same_block_lines
                if int(block_line.get("block_line_index", 0))
                > int(line.get("block_line_index", 0))
            ]
            leads_mixed_style_block = (
                int(line.get("block_line_index", 0)) == 0
                and bool(following_block_lines)
                and all(
                    text_line_style_signature(block_line)["key"]
                    != style_signature["key"]
                    for block_line in following_block_lines
                )
                and effective_size
                > max(
                    float(
                        block_line.get(
                            "effective_font_size",
                            0.0,
                        )
                        or 0.0
                    )
                    for block_line in following_block_lines
                )
                + font_size_cluster_tolerance_points
            )
            if not leads_mixed_style_block:
                continue
        size_rank = min(
            range(len(heading_size_clusters)),
            key=lambda index: abs(
                heading_size_clusters[index] - effective_size
            ),
            default=0,
        )
        inferred_level = min(
            max_markdown_level,
            size_rank + 1,
        )
        candidate_lines.append(line)
        candidate_metadata_by_line_id[id(line)] = {
            "text": text,
            "style_signature": style_signature,
            "effective_size": effective_size,
            "inferred_level": inferred_level,
            "line_rect": line_rect,
            "body_reference_font_size": page_body_font_size_ceiling,
        }

    running_header_line_ids = {
        id(line)
        for line in candidate_lines
        if line_matches_running_header_layout(
            line=line,
            typography_profile=typography_profile,
            page_rect=page_rect,
        )
    }
    remaining_outline_entries = [
        entry
        for entry in outline_entries
        if int(entry.get("entry_index", -1))
        not in outline_text_assigned_entry_indices
    ]
    outline_assignment_by_line_id = (
        assign_pdf_outline_entries_to_lines(
            candidate_lines=candidate_lines,
            outline_entries=remaining_outline_entries,
            page_rect=page_rect,
            running_header_line_ids=running_header_line_ids,
            target_max_vertical_distance_points=(
                outline_target_max_vertical_distance_points
            ),
            target_max_x_distance_page_ratio=(
                outline_target_max_x_distance_page_ratio
            ),
            shared_target_tolerance_points=(
                outline_shared_target_tolerance_points
            ),
        )
        if (
            enable_pdf_outline_heading_assignment
            and enable_font_heading_inference
        )
        else {}
    )

    for match_record in outline_text_assignment_by_line_id.values():
        outline_match = match_record["entry"]
        matched_lines = match_record["lines"]
        line = matched_lines[0]
        level = min(
            max_markdown_level,
            int(outline_match.get("level", 1)),
        )
        consumed_by_page_title = (
            page_title_outline_entry_index is not None
            and int(outline_match.get("entry_index", -1))
            == int(page_title_outline_entry_index)
        )
        reason = (
            "outline_text_match_consumed_by_page_title"
            if consumed_by_page_title
            else "outline_text_exact_match"
        )
        line["heading_level"] = level
        line["heading_reason"] = reason
        line["heading_effective_font_size"] = round(
            float(line.get("effective_font_size", 0.0) or 0.0),
            3,
        )
        if consumed_by_page_title:
            line["suppress_as_duplicate_page_title"] = True
        else:
            line["heading_text_override"] = match_record[
                "combined_text"
            ]
        for continuation_line in matched_lines[1:]:
            continuation_line[
                "suppress_as_outline_heading_continuation"
            ] = True
        style_signature = text_line_style_signature(line)
        line_rect = fitz.Rect(match_record["bbox"]).normalize()
        bold_ratio = float(
            line.get("bold_character_ratio", 0.0) or 0.0
        )
        records.append(
            {
                "text": match_record["combined_text"],
                "level": level,
                "reason": reason,
                "effective_font_size": round(
                    float(
                        line.get("effective_font_size", 0.0)
                        or 0.0
                    ),
                    3,
                ),
                "bold_character_ratio": round(bold_ratio, 3),
                "body_reference_font_size": None,
                "style_key": style_signature["key"],
                "style_role": "outline_text_match",
                "running_header_layout": False,
                "outline_title": outline_match["title"],
                "outline_entry_index": outline_match.get(
                    "entry_index"
                ),
                "outline_text_matches": True,
                "markdown_emission": (
                    "page_title"
                    if consumed_by_page_title
                    else "body_heading"
                ),
                "matched_text_line_count": len(matched_lines),
                "bbox": [
                    round(value, 3) for value in line_rect
                ],
            }
        )

    for line in candidate_lines:
        metadata = candidate_metadata_by_line_id[id(line)]
        outline_match = outline_assignment_by_line_id.get(id(line))
        is_running_header = id(line) in running_header_line_ids
        if is_running_header and outline_match is None:
            continue

        if outline_match is not None:
            level = min(
                max_markdown_level,
                int(outline_match["level"]),
            )
            reason = "document_style_candidate_with_outline_level"
        else:
            level = int(metadata["inferred_level"])
            reason = "document_style_heading_candidate"

        line["heading_level"] = level
        line["heading_reason"] = reason
        line["heading_effective_font_size"] = round(
            metadata["effective_size"],
            3,
        )
        bold_ratio = float(
            line.get("bold_character_ratio", 0.0) or 0.0
        )
        records.append(
            {
                "text": metadata["text"],
                "level": level,
                "reason": reason,
                "effective_font_size": round(
                    metadata["effective_size"],
                    3,
                ),
                "bold_character_ratio": round(bold_ratio, 3),
                # "body_reference_font_size": None,
                "body_reference_font_size": round(
                    metadata["body_reference_font_size"],
                    3,
                ),
                "style_key": metadata["style_signature"]["key"],
                "style_role": "heading_candidate",
                "running_header_layout": is_running_header,
                "outline_title": (
                    outline_match["title"]
                    if outline_match is not None
                    else None
                ),
                "outline_text_matches": (
                    outline_text_matches_for_audit(
                        metadata["text"],
                        outline_match,
                    )
                    if outline_match is not None
                    else None
                ),
                "markdown_emission": "body_heading",
                "bbox": [
                    round(value, 3)
                    for value in metadata["line_rect"]
                ],
            }
        )

    return records
