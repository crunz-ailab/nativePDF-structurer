"""输出表格CSV、图片、Markdown和来源记录。

模块序列化单元格数据，检测主要
文字方向，旋转表格矩阵与图片，计算带padding的截图范围，并在归属仲裁后
创建最终Markdown原子和审计记录。

关键不变量:
    * CSV提取使用检测器原始table对象。
    * ``render_bbox`` 控制截图和文字去重范围。
    * ``sort_bbox`` 保持在表格的阅读位置锚点。
    * 被抑制或被父容器拥有的候选不重复输出Markdown。

Emit table CSV, image, Markdown, and provenance artifacts.

Serializes extracted cell data, detects dominant text rotation, rotates table
matrices and images, computes padded render clips, and creates final Markdown
atoms and audit records after ownership resolution.

Key invariants:
    * CSV extraction uses the detector's original table object.
    * ``render_bbox`` controls screenshot and text-deduplication bounds.
    * ``sort_bbox`` remains anchored to the table's reading position.
    * Suppressed or parent-owned candidates do not emit duplicate Markdown.
"""

from __future__ import annotations

import csv
import io
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import fitz
from PIL import Image

from .geometry import rect_center
from .visual_assets import build_top_sort_bbox


TABLE_PADDING_POINTS = 3.0


def padded_table_candidate_clip(
    candidate: dict,
    bbox: fitz.Rect,
    page_rect: fitz.Rect,
) -> fitz.Rect:
    """Add table screenshot padding except on an attached rule-band edge."""

    attachments = candidate.get("adjacent_rule_band_attachments", [])
    suppress_top_padding = any(
        attachment.get("edge") == "top"
        for attachment in attachments
    )
    suppress_bottom_padding = any(
        attachment.get("edge") == "bottom"
        for attachment in attachments
    )
    clip = fitz.Rect(bbox).normalize()
    clip.x0 = max(page_rect.x0, clip.x0 - TABLE_PADDING_POINTS)
    clip.x1 = min(page_rect.x1, clip.x1 + TABLE_PADDING_POINTS)
    clip.y0 = max(
        page_rect.y0,
        clip.y0
        - (0.0 if suppress_top_padding else TABLE_PADDING_POINTS),
    )
    clip.y1 = min(
        page_rect.y1,
        clip.y1
        + (0.0 if suppress_bottom_padding else TABLE_PADDING_POINTS),
    )
    return clip


def write_table_csv(
    path: Path,
    table: list[list[str | None]],
) -> None:
    """将已转正的二维表格数据写为 UTF-8 BOM CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(
            [[cell or "" for cell in row] for row in table]
        )


def table_to_markdown(
    table: list[list[str | None]],
) -> str:
    """将二维文本表格转成不依赖表头语义的Markdown网格。"""
    if not table:
        return "| |\n|---|"

    column_count = max((len(row) for row in table), default=0)
    if column_count <= 0:
        return "| |\n|---|"

    normalized_rows = []
    for row in table:
        padded_row = [*row, *([None] * (column_count - len(row)))]
        normalized_rows.append(
            [
                (cell or "")
                .replace("\\", "\\\\")
                .replace("|", "\\|")
                .replace("\r\n", "<br>")
                .replace("\n", "<br>")
                for cell in padded_row
            ]
        )

    separator = ["---"] * column_count
    lines = [
        "| " + " | ".join(normalized_rows[0]) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in normalized_rows[1:]
    )
    return "\n".join(lines)


def infer_table_rotation(
    text_lines: list[dict],
    table_bbox: tuple[float, float, float, float],
    *,
    minimum_direction_text_chars: int,
    rotation_dominance_threshold: float,
) -> dict:
    """按表格范围内的有效字符数统计主文字方向。"""
    table_rect = fitz.Rect(table_bbox)
    direction_character_counts: Counter[int] = Counter()
    matched_text_line_count = 0
    matched_character_count = 0

    for line in text_lines:
        if not table_rect.contains(rect_center(line["bbox"])):
            continue

        normalized_text = "".join(line["text"].split())
        if not normalized_text:
            continue

        character_count = len(normalized_text)
        direction_character_counts[line["angle"]] += character_count
        matched_text_line_count += 1
        matched_character_count += character_count

    if matched_character_count == 0:
        return {
            "status": "no_text_in_table",
            "dominant_text_angle": None,
            "dominant_ratio": 0.0,
            "suggested_rotation_degrees": 0,
            "matched_text_line_count": 0,
            "matched_character_count": 0,
            "direction_character_counts": {},
        }

    dominant_angle, dominant_character_count = (
        direction_character_counts.most_common(1)[0]
    )
    dominant_ratio = dominant_character_count / matched_character_count
    should_rotate = (
        matched_character_count >= minimum_direction_text_chars
        and dominant_angle != 0
        and dominant_ratio >= rotation_dominance_threshold
    )

    return {
        "status": "rotation_detected" if should_rotate else "keep_original",
        "dominant_text_angle": dominant_angle,
        "dominant_ratio": round(dominant_ratio, 4),
        "suggested_rotation_degrees": dominant_angle if should_rotate else 0,
        "matched_text_line_count": matched_text_line_count,
        "matched_character_count": matched_character_count,
        "direction_character_counts": dict(
            sorted(direction_character_counts.items())
        ),
    }


def normalize_table_shape(
    table: list[list[str | None]],
) -> list[list[str | None]]:
    """将不等长行使用None补齐为矩形。"""
    max_column_count = max((len(row) for row in table), default=0)
    return [
        row + [None] * (max_column_count - len(row))
        for row in table
    ]


def rotate_table_data(
    table: list[list[str | None]],
    rotation_degrees: int,
) -> list[list[str | None]]:
    """按0、±90或180度旋转二维表格数据。"""
    normalized_table = normalize_table_shape(table)
    if not normalized_table or not normalized_table[0]:
        return normalized_table
    if rotation_degrees == 0:
        return normalized_table

    row_count = len(normalized_table)
    column_count = len(normalized_table[0])
    if rotation_degrees == -90:
        return [
            [
                normalized_table[row_index][column_index]
                for row_index in range(row_count - 1, -1, -1)
            ]
            for column_index in range(column_count)
        ]
    if rotation_degrees == 90:
        return [
            [
                normalized_table[row_index][column_index]
                for row_index in range(row_count)
            ]
            for column_index in range(column_count - 1, -1, -1)
        ]
    if rotation_degrees == 180:
        return [
            list(reversed(row))
            for row in reversed(normalized_table)
        ]
    raise ValueError(f"Unsupported table rotation: {rotation_degrees}")


def save_rotated_table_image(
    pixmap: fitz.Pixmap,
    image_path: Path,
    rotation_degrees: int,
) -> tuple[int, int]:
    """保存表格截图，并按需要旋转为可阅读方向。"""
    if rotation_degrees == 0:
        pixmap.save(image_path)
        return pixmap.width, pixmap.height

    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    if rotation_degrees == -90:
        image = image.transpose(Image.Transpose.ROTATE_270)
    elif rotation_degrees == 90:
        image = image.transpose(Image.Transpose.ROTATE_90)
    elif rotation_degrees == 180:
        image = image.transpose(Image.Transpose.ROTATE_180)
    else:
        image = image.rotate(rotation_degrees, expand=True)

    image.save(image_path)
    return image.width, image.height

def emit_table_candidate(
    candidate: dict,
    page: fitz.Page,
    page_number: int,
    tables_dir: Path,
    table_images_dir: Path,
    *,
    table_image_dpi: int,
    padded_table_candidate_clip: Callable[..., fitz.Rect],
    save_rotated_table_image: Callable[..., tuple[int, int]],
    write_table_csv: Callable[..., None] = write_table_csv,
    table_to_markdown: Callable[[list], str] = table_to_markdown,
) -> dict:
    """Emit one table candidate after table/visual ownership arbitration."""
    table = candidate["table"]
    table_detection_bbox = fitz.Rect(table.bbox).normalize()
    table_render_source_bbox = fitz.Rect(
        candidate.get("render_bbox", table_detection_bbox)
    ).normalize()
    table_number = candidate["table_number"]
    display_table_number = candidate.get(
        "display_table_number",
        table_number,
    )
    extracted_raw = candidate["extracted_raw"]
    extracted_reflowed = candidate["extracted_reflowed"]
    rotation_info = candidate["rotation_info"]
    rotation_degrees = candidate["rotation_degrees"]
    csv_name = candidate["csv_name"]
    csv_path = candidate["csv_path"]
    claimed = candidate.get("claimed_by_visual_region", False)

    record = {
        "page_number": page_number,
        "table_number": table_number,
        "display_table_number": display_table_number,
        "claim_id": f"page#{page_number}:table#{table_number}",
        "detection_source": getattr(
            table,
            "detection_source",
            "pymupdf_native",
        ),
        "geometry_fallback_metadata": getattr(
            table,
            "metadata",
            None,
        ),
        "bbox": [round(value, 3) for value in table_detection_bbox],
        "render_bbox": [
            round(value, 3) for value in table_render_source_bbox
        ],
        "raw_row_count": candidate["raw_row_count"],
        "raw_column_count": candidate["raw_column_count"],
        "reflowed_row_count": candidate["reflowed_row_count"],
        "reflowed_column_count": candidate["reflowed_column_count"],
        "grid_cell_count": candidate["table_cell_count"],
        "exceeds_6_cell_threshold": candidate["exceeds_threshold"],
        "force_pure_text_markdown": candidate["force_markdown_table"],
        "rendered_as_image": candidate["should_render_table_image"] and not claimed,
        "has_complex_visual_content": candidate["table_has_complex_visual_content"],
        "visual_content_metadata": candidate["table_visual_metadata"],
        "structurally_valid_table": candidate.get(
            "structurally_valid_table",
            True,
        ),
        "table_structure_audit": candidate.get(
            "table_structure_audit",
            {},
        ),
        "table_render_mode": (
            "full_table_screenshot"
            if candidate.get("table_owns_full_visual_bbox", False)
            else "legacy_parent_child_visual"
        ),
        "suppressed_before_ownership": candidate.get(
            "suppressed_before_ownership",
            False,
        ),
        "csv_rotation_degrees": rotation_degrees,
        "csv_path": None if claimed else str(csv_path),
        "image_path": None,
        "image_width": None,
        "image_height": None,
        "screenshot_seconds": 0.0,
        "text_orientation": rotation_info,
        "raw_data": extracted_raw,
        "reflowed_data": extracted_reflowed,
        "claimed_by_visual_region": claimed,
        "visual_claim_overlap_ratio": round(
            candidate.get("table_overlap_ratio", 0.0),
            6,
        ),
        "visual_overlap_ratio": round(
            candidate.get("visual_overlap_ratio", 0.0),
            6,
        ),
        "precluster_protected": candidate.get(
            "precluster_protected",
            False,
        ),
        "precluster_reason": candidate.get(
            "precluster_reason"
        ),
        "precluster_source_cluster_bbox": [
            round(value, 3)
            for value in candidate.get(
                "source_cluster_bbox",
                fitz.Rect(table.bbox),
            )
        ],
        "precluster_external_drawing_count": candidate.get(
            "precluster_vector_overlap",
            {},
        ).get("external_drawing_count", 0),
        "precluster_overlaps_table_numbers": candidate.get(
            "precluster_overlaps_table_numbers",
            [],
        ),
        "precluster_overlaps_image_groups": candidate.get(
            "precluster_overlaps_image_groups",
            [],
        ),
        "logical_table_container": candidate.get(
            "logical_table_container",
            False,
        ),
        "table_container_element_id": candidate.get(
            "table_container_element_id"
        ),
        "table_skeleton_drawing_ids": candidate.get(
            "table_skeleton_drawing_ids",
            [],
        ),
        "cell_internal_visual_drawing_ids": candidate.get(
            "cell_internal_visual_drawing_ids",
            [],
        ),
        "child_visual_claim_ids": candidate.get(
            "child_visual_claim_ids",
            [],
        ),
        "adjacent_rule_band_attachments": candidate.get(
            "adjacent_rule_band_attachments",
            [],
        ),
        "output_group_id": candidate.get("output_group_id"),
        "output_group_role": candidate.get("output_group_role"),
        "output_group_parent_claim_id": candidate.get(
            "output_group_parent_claim_id"
        ),
        "output_suppressed_by_composite": bool(
            claimed and candidate.get("output_group_id")
        ),
    }
    if candidate.get("is_anomalous_single_value_table"):
        record["nonempty_cell_count"] = candidate["nonempty_cell_count"]
        record["anomalous_single_value"] = True

    if claimed:
        return {
            "record": record,
            "markdown_element": None,
            "excluded_rect": None,
            "embedded": False,
            "auto_rotated": False,
            "screenshot_seconds": 0.0,
        }

    if not candidate["will_emit_if_unclaimed"]:
        write_table_csv(path=csv_path, table=extracted_reflowed)
        record["emitted"] = False
        return {
            "record": record,
            "markdown_element": None,
            "excluded_rect": None,
            "embedded": False,
            "auto_rotated": False,
            "screenshot_seconds": 0.0,
        }

    write_table_csv(path=csv_path, table=extracted_reflowed)
    excluded_rect: fitz.Rect | None = None
    markdown_element: dict | None = None
    image_path: Path | None = None
    image_width: int | None = None
    image_height: int | None = None
    screenshot_elapsed = 0.0
    embedded = False
    auto_rotated = False

    if candidate["is_anomalous_single_value_table"]:
        excluded_rect = fitz.Rect(table_render_source_bbox)
        table_markdown = "\n\n".join(
            [
                (
                    f"### Anomalous Table {display_table_number} "
                    f"({candidate['reflowed_row_count']} × "
                    f"{candidate['reflowed_column_count']})"
                ),
                table_to_markdown(extracted_reflowed),
                f"[CSV data](tables/{csv_name})",
            ]
        )
        table_render_bbox = fitz.Rect(table_render_source_bbox)
        markdown_element = {
            "bbox": table_render_bbox,
            "render_bbox": table_render_bbox,
            "sort_bbox": build_top_sort_bbox(table_detection_bbox),
            "text": table_markdown,
            "direction": (1.0, 0.0),
            "angle": 0,
            "kind": "table",
            "markdown": table_markdown,
            "mode": "anomalous_unclaimed_table",
            "output_path": None,
        }
    elif candidate["should_render_table_image"]:
        excluded_rect = fitz.Rect(table_render_source_bbox)
        screenshot_started = time.perf_counter()
        clip = padded_table_candidate_clip(
            candidate=candidate,
            bbox=table_render_source_bbox,
            page_rect=page.rect,
        )
        pixmap = page.get_pixmap(
            clip=clip,
            dpi=table_image_dpi,
            alpha=False,
        )
        image_name = (
            f"page_{page_number:03d}_table_{display_table_number:02d}"
            f"_rotated_{rotation_degrees:+d}.png"
        )
        image_path = table_images_dir / image_name
        image_width, image_height = save_rotated_table_image(
            pixmap=pixmap,
            image_path=image_path,
            rotation_degrees=rotation_degrees,
        )
        screenshot_elapsed = time.perf_counter() - screenshot_started
        embedded = True
        auto_rotated = rotation_degrees != 0
        rotation_note = (
            f"，PNG 与 CSV 已旋转 {rotation_degrees}°"
            if rotation_degrees != 0
            else ""
        )
        table_markdown = "\n\n".join(
            [
                (
                    f"### Table {display_table_number} "
                    f"({candidate['reflowed_row_count']} × "
                    f"{candidate['reflowed_column_count']}"
                    f"{rotation_note})"
                ),
                (
                    f"![Page {page_number} Table {display_table_number}]"
                    f"(table_images_auto_rotated/{image_name})"
                ),
                f"[CSV data](tables/{csv_name})",
            ]
        )
        table_render_bbox = padded_table_candidate_clip(
            candidate=candidate,
            bbox=table_render_source_bbox,
            page_rect=page.rect,
        )
        markdown_element = {
            "bbox": table_render_bbox,
            "render_bbox": table_render_bbox,
            "sort_bbox": build_top_sort_bbox(table_detection_bbox),
            "text": table_markdown,
            "direction": (1.0, 0.0),
            "angle": 0,
            "kind": "table",
            "markdown": table_markdown,
            "mode": "rendered_table",
            "output_path": image_path,
        }
    elif candidate["force_markdown_table"]:
        excluded_rect = fitz.Rect(table_render_source_bbox)
        table_markdown = "\n\n".join(
            [
                (
                    f"### Table {display_table_number} "
                    f"({candidate['reflowed_row_count']} × "
                    f"{candidate['reflowed_column_count']})"
                ),
                table_to_markdown(extracted_reflowed),
                f"[CSV data](tables/{csv_name})",
            ]
        )
        table_render_bbox = fitz.Rect(table_render_source_bbox)
        markdown_element = {
            "bbox": table_render_bbox,
            "render_bbox": table_render_bbox,
            "sort_bbox": build_top_sort_bbox(table_detection_bbox),
            "text": table_markdown,
            "direction": (1.0, 0.0),
            "angle": 0,
            "kind": "table",
            "markdown": table_markdown,
            "mode": "pure_text_markdown_table",
            "output_path": None,
        }

    record["image_path"] = str(image_path) if image_path else None
    record["image_width"] = image_width
    record["image_height"] = image_height
    record["screenshot_seconds"] = round(screenshot_elapsed, 6)
    return {
        "record": record,
        "markdown_element": markdown_element,
        "excluded_rect": excluded_rect,
        "embedded": embedded,
        "auto_rotated": auto_rotated,
        "screenshot_seconds": screenshot_elapsed,
    }
