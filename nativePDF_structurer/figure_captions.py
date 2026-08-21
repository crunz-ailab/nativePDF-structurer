"""针对最终存活的视觉输出识别图题。

模块在视觉区域和原始图片归属完成后分析
尚未被认领的原生文字；图题必须具有通用图号结构、满足纵向邻接、两者之间
不存在未归属正文，并且只能分配给一个视觉输出。

Classify figure captions against surviving visual outputs.

Evaluates unclaimed native text after visual and source-image ownership is
finalized. A caption requires generic figure-label structure, vertical
adjacency, no intervening unclaimed body line, and unique assignment to one
visual output.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import fitz


#@GPT 图题必须同时满足图号结构、视觉邻接和无竞争正文；靠近图片本身不是充分条件。

def classify_figure_caption_roles(
    text_lines: list[dict],
    visual_records: list[dict],
    excluded_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
    *,
    label_pattern: object,
    max_text_characters: int,
    max_vertical_gap_multiplier: float,
    max_vertical_gap_points: float,
    rect_is_mostly_inside_any: Callable[..., bool],
) -> list[dict]:
    """为视觉区域外的图题行登记figure_caption角色。"""
    for line in text_lines:
        line.pop("figure_role", None)
        line.pop("figure_target_claim_id", None)
        line.pop("figure_role_reason", None)

    external_lines: list[dict] = []
    line_heights: list[float] = []
    for line in text_lines:
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
        text = " ".join(str(line.get("text", "")).split())
        if (
            not text
            or len(text) > max_text_characters
            or int(line.get("block_line_count", 1) or 1) > 1
        ):
            continue
        external_lines.append(line)
        if line_rect.height > 0:
            line_heights.append(line_rect.height)

    if not external_lines or not visual_records:
        return []

    median_line_height = (
        statistics.median(line_heights)
        if line_heights
        else 9.0
    )
    maximum_gap = min(
        max_vertical_gap_points,
        max(
            median_line_height * max_vertical_gap_multiplier,
            median_line_height * 1.5,
        ),
    )
    claimed_caption_line_ids: set[int] = set()
    caption_records: list[dict] = []

    ordered_visual_records = sorted(
        visual_records,
        key=lambda record: (
            fitz.Rect(
                record.get("render_bbox", record.get("bbox"))
            ).y0,
            fitz.Rect(
                record.get("render_bbox", record.get("bbox"))
            ).x0,
            record.get("region_number", 0),
        ),
    )
    for record in ordered_visual_records:
        visual_rect = fitz.Rect(
            record.get("render_bbox", record.get("bbox"))
        ).normalize()
        if visual_rect.is_empty:
            continue
        candidates: list[dict] = []
        for line in external_lines:
            if id(line) in claimed_caption_line_ids:
                continue
            text = " ".join(str(line.get("text", "")).split())
            if not label_pattern.match(text):
                continue
            line_rect = fitz.Rect(line["bbox"]).normalize()
            if line_rect.y1 <= visual_rect.y0:
                side = "before"
                gap = visual_rect.y0 - line_rect.y1
            elif line_rect.y0 >= visual_rect.y1:
                side = "after"
                gap = line_rect.y0 - visual_rect.y1
            else:
                continue
            if gap < 0 or gap > maximum_gap:
                continue

            competing_lines = []
            for other in external_lines:
                if other is line or id(other) in claimed_caption_line_ids:
                    continue
                other_rect = fitz.Rect(other["bbox"]).normalize()
                if side == "before":
                    lies_between = (
                        other_rect.y0 >= line_rect.y1
                        and other_rect.y1 <= visual_rect.y0
                    )
                else:
                    lies_between = (
                        other_rect.y0 >= visual_rect.y1
                        and other_rect.y1 <= line_rect.y0
                    )
                if lies_between:
                    competing_lines.append(other)
            #@GPT 图题与图片之间存在未归属正文时，不因距离接近而强行绑定。
            if competing_lines:
                continue

            x_overlap = max(
                0.0,
                min(line_rect.x1, visual_rect.x1)
                - max(line_rect.x0, visual_rect.x0),
            )
            candidate_score = (
                0 if x_overlap > 0 else 1,
                round(gap, 6),
                abs(
                    (line_rect.x0 + line_rect.x1) / 2
                    - (visual_rect.x0 + visual_rect.x1) / 2
                ),
            )
            candidates.append(
                {
                    "line": line,
                    "line_rect": line_rect,
                    "side": side,
                    "gap": gap,
                    "x_overlap": x_overlap,
                    "score": candidate_score,
                }
            )

        if not candidates:
            continue
        chosen = min(candidates, key=lambda item: item["score"])
        caption_line = chosen["line"]
        caption_line["figure_role"] = "figure_caption"
        caption_line["figure_target_claim_id"] = record.get(
            "claim_id"
        )
        caption_line["figure_role_reason"] = (
            "caption_marker_and_adjacent_visual_without_competing_text"
        )
        claimed_caption_line_ids.add(id(caption_line))
        caption_records.append(
            {
                "text": " ".join(
                    str(caption_line.get("text", "")).split()
                ),
                "target_claim_id": record.get("claim_id"),
                "target_region_number": record.get(
                    "region_number"
                ),
                "side": chosen["side"],
                "gap_points": round(chosen["gap"], 3),
                "x_overlap_points": round(
                    chosen["x_overlap"],
                    3,
                ),
                "reason": caption_line["figure_role_reason"],
            }
        )

    return caption_records
