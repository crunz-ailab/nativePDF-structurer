"""将PDF目录项严格匹配到原生文字行。

模块规范化目录与页面文字，在允许时移除
结构化编号前缀，使用目标几何消解重复匹配，并保证每个目录项最多分配给一
条文字行；不使用模糊匹配或关键词匹配。

Match PDF outline entries to native text lines strictly.

Normalizes outline and page text, removes structural numbering prefixes when
allowed, resolves duplicate matches with target geometry, and assigns each
outline entry to at most one text line. No fuzzy or keyword matching is used.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import fitz

from .geometry import rect_area, union_rects


def normalize_heading_match_text(text: str) -> str:
    """返回用于书签匹配的紧凑规范化文本。"""
    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", "", normalized).strip()


def strip_heading_number_prefix(text: str) -> str:
    """移除书签或标题开头的章节编号。"""
    stripped = re.sub(
        r"^(?:第[一二三四五六七八九十百千万\d]+[章节篇部]|\d+(?:\.\d+){0,3})[-–—.:：、)]*",
        "",
        text,
    )
    return stripped.strip()


def outline_text_matches(text: str, outline_entry: dict) -> bool:
    """检查文本行是否与书签标题严格结构一致。"""
    normalized_text = normalize_heading_match_text(text)
    structural_text = strip_heading_number_prefix(normalized_text)
    return bool(
        normalized_text == outline_entry["normalized_title"]
        or (
            structural_text
            and structural_text == outline_entry["structural_title"]
        )
    )


def outline_text_matches_for_audit(text: str, outline_entry: dict) -> bool:
    """兼容旧审计名称。"""
    return outline_text_matches(text, outline_entry)


def assign_pdf_outline_text_matches_to_lines(
    candidate_lines: list[dict],
    outline_entries: list[dict],
) -> dict[int, dict]:
    """按严格文本一致性将当前页目录项唯一分配给连续文本行。"""

    assignments: dict[int, dict] = {}
    assigned_line_ids: set[int] = set()
    lines_by_block: dict[int, list[dict]] = defaultdict(list)
    isolated_lines: list[list[dict]] = []
    for line in candidate_lines:
        if rect_area(fitz.Rect(line["bbox"]).normalize()) <= 0:
            continue
        block_index = int(line.get("block_index", -1))
        if block_index < 0:
            isolated_lines.append([line])
        else:
            lines_by_block[block_index].append(line)

    ordered_blocks = [
        sorted(
            block_lines,
            key=lambda line: (
                int(line.get("block_line_index", 0)),
                fitz.Rect(line["bbox"]).normalize().y0,
                fitz.Rect(line["bbox"]).normalize().x0,
            ),
        )
        for _, block_lines in sorted(lines_by_block.items())
    ]
    ordered_blocks.extend(isolated_lines)

    for entry in sorted(
        outline_entries,
        key=lambda item: (
            int(item.get("entry_index", 0)),
            int(item.get("level", 1)),
        ),
    ):
        groups: list[dict] = []
        for block_lines in ordered_blocks:
            for start in range(len(block_lines)):
                for length in range(1, min(3, len(block_lines) - start) + 1):
                    group = block_lines[start:start + length]
                    if any(id(line) in assigned_line_ids for line in group):
                        continue
                    if length > 1 and any(
                        int(group[i + 1].get("block_line_index", 0))
                        != int(group[i].get("block_line_index", 0)) + 1
                        for i in range(length - 1)
                    ):
                        continue
                    text = " ".join(
                        " ".join(str(line.get("text", "")).split())
                        for line in group
                    ).strip()
                    if outline_text_matches(text, entry):
                        groups.append({
                            "lines": group,
                            "combined_text": text,
                            "bbox": union_rects([fitz.Rect(line["bbox"]).normalize() for line in group]),
                        })
        if not groups:
            continue
        target = entry.get("target_point")
        if target is not None:
            chosen = min(groups, key=lambda g: (abs(g["bbox"].y0 - target[1]), abs(g["bbox"].x0 - target[0]), len(g["lines"]), g["bbox"].y0, g["bbox"].x0))
        else:
            chosen = min(groups, key=lambda g: (g["bbox"].y0, g["bbox"].x0, len(g["lines"])))
        assignments[id(chosen["lines"][0])] = {"entry": entry, **chosen}
        assigned_line_ids.update(id(line) for line in chosen["lines"])
    return assignments

def assign_pdf_outline_entries_to_lines(
    candidate_lines: list[dict],
    outline_entries: list[dict],
    page_rect: fitz.Rect,
    running_header_line_ids: set[int],
    *,
    target_max_vertical_distance_points: float,
    target_max_x_distance_page_ratio: float,
    shared_target_tolerance_points: float,
) -> dict[int, dict]:
    """按书签目标几何位置给既有标题候选分配书签层级。"""
    candidate_lines = [
        line
        for line in candidate_lines
        if rect_area(fitz.Rect(line["bbox"]).normalize()) > 0
    ]
    positioned_entries = [
        entry
        for entry in outline_entries
        if entry.get("target_point") is not None
    ]
    positioned_entries.sort(
        key=lambda entry: (
            entry["target_point"][1],
            entry["target_point"][0],
            entry["level"],
            entry["entry_index"],
        )
    )

    entry_groups: list[list[dict]] = []
    for entry in positioned_entries:
        if not entry_groups:
            entry_groups.append([entry])
            continue
        previous = entry_groups[-1][0]
        if (
            abs(
                entry["target_point"][0]
                - previous["target_point"][0]
            )
            <= shared_target_tolerance_points
            and abs(
                entry["target_point"][1]
                - previous["target_point"][1]
            )
            <= shared_target_tolerance_points
        ):
            entry_groups[-1].append(entry)
        else:
            entry_groups.append([entry])

    assignments: dict[int, dict] = {}
    assigned_line_ids: set[int] = set()
    maximum_x_distance = (
        page_rect.width
            * target_max_x_distance_page_ratio
    )

    for entry_group in entry_groups:
        target_x, target_y = entry_group[0]["target_point"]
        group_allows_running_header = any(
            int(entry["level"]) == 1
            for entry in entry_group
        )
        nearby_lines = [
            line
            for line in candidate_lines
            if id(line) not in assigned_line_ids
            and (
                id(line) not in running_header_line_ids
                or group_allows_running_header
            )
            and abs(
                fitz.Rect(line["bbox"]).normalize().y0
                - target_y
            )
            <= target_max_vertical_distance_points
            and (
                abs(
                    fitz.Rect(line["bbox"]).normalize().x0
                    - target_x
                )
                <= maximum_x_distance
                or (
                    id(line) in running_header_line_ids
                    and group_allows_running_header
                    and abs(
                        fitz.Rect(line["bbox"]).normalize().y0
                        - target_y
                    )
                    <= target_max_vertical_distance_points
                )
            )
        ]
        if not nearby_lines:
            continue

        if len(entry_group) == 1:
            chosen_lines = [
                min(
                    nearby_lines,
                    key=lambda line: (
                        -float(
                            line.get(
                                "effective_font_size",
                                0.0,
                            )
                            or 0.0
                        ),
                        abs(
                            fitz.Rect(line["bbox"]).normalize().y0
                            - target_y
                        ),
                        abs(
                            fitz.Rect(line["bbox"]).normalize().x0
                            - target_x
                        ),
                    ),
                )
            ]
        else:
            chosen_lines = sorted(
                nearby_lines,
                key=lambda line: (
                    -float(
                        line.get("effective_font_size", 0.0)
                        or 0.0
                    ),
                    abs(
                        fitz.Rect(line["bbox"]).normalize().y0
                        - target_y
                    ),
                    fitz.Rect(line["bbox"]).normalize().y0,
                ),
            )[: len(entry_group)]

        ordered_entries = sorted(
            entry_group,
            key=lambda entry: (
                entry["level"],
                entry["entry_index"],
            ),
        )
        ordered_lines = sorted(
            chosen_lines,
            key=lambda line: (
                -float(
                    line.get("effective_font_size", 0.0) or 0.0
                ),
                fitz.Rect(line["bbox"]).normalize().y0,
            ),
        )
        for entry, line in zip(ordered_entries, ordered_lines):
            assignments[id(line)] = entry
            assigned_line_ids.add(id(line))

    return assignments
