"""将已解析的Claim归属结果应用到生成制品。

候选可能因失去必要特征而被淘汰，
也可能因存活父容器统一渲染而作为子成员被抑制；即使Markdown原子不再活跃，
磁盘文件仍可保留用于审计。本模块不运行检测器，也不改变页面几何。

Apply resolved Claim ownership to generated output atoms.

Filters records and Markdown atoms after structural ownership is resolved.
Candidates may be suppressed because they lost required features or because a
surviving parent container renders them as embedded children. Files may remain
on disk for audit even when their Markdown atoms are inactive.
"""

from __future__ import annotations

def apply_claim_resolution_to_outputs(
    records: list[dict],
    markdown_elements: list[dict],
    resolution: dict,
    output_kind: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """按候选是否赢得特征图元过滤最终输出，返回存活记录、元素和淘汰动作。"""
    candidate_by_id = {
        candidate["claim_id"]: candidate
        for candidate in resolution.get("candidates", [])
    }

    def claim_survives(claim_id: str | None) -> bool:
        if claim_id is None:
            return True
        candidate = candidate_by_id.get(claim_id)
        if candidate is None:
            return True
        return bool(candidate.get("survives_preview", False))

    def embedded_in_surviving_parent(item: dict) -> bool:
        if output_kind != "visual_region":
            return False
        parent_claim_id = item.get("parent_table_claim_id")
        if bool(
            item.get("embedded_in_parent_table")
            and parent_claim_id
            and claim_survives(parent_claim_id)
        ):
            return True
        parent_visual_claim_id = item.get("parent_visual_claim_id")
        return bool(
            item.get("embedded_in_parent_visual")
            and parent_visual_claim_id
            and claim_survives(parent_visual_claim_id)
        )

    active_records: list[dict] = []
    suppressed_actions: list[dict] = []
    for record in records:
        claim_id = record.get("claim_id")
        candidate = candidate_by_id.get(claim_id)
        embedded_in_parent = embedded_in_surviving_parent(record)
        active = claim_survives(claim_id) and not embedded_in_parent
        record["claim_resolution"] = {
            "active": active,
            "won_element_count": (
                candidate.get("won_count", 0)
                if candidate is not None
                else None
            ),
            "won_feature_count": (
                candidate.get("won_feature_count", 0)
                if candidate is not None
                else None
            ),
            "embedded_in_parent_table": embedded_in_parent,
            "parent_table_claim_id": record.get(
                "parent_table_claim_id"
            ),
            "embedded_in_parent_visual": bool(
                embedded_in_parent
                and record.get("embedded_in_parent_visual")
            ),
            "parent_visual_claim_id": record.get(
                "parent_visual_claim_id"
            ),
        }
        if active:
            active_records.append(record)
            continue

        suppressed_actions.append(
            {
                "output_kind": output_kind,
                "claim_id": claim_id,
                "output_path": record.get("output_path")
                or record.get("image_path"),
                "reason": (
                    "embedded_in_parent_table"
                    if embedded_in_parent
                    else (
                        candidate.get("survival_reason")
                        if candidate is not None
                        else "missing_candidate_resolution"
                    )
                ),
            }
        )

    active_markdown_elements = [
        element
        for element in markdown_elements
        if claim_survives(element.get("claim_id"))
        and not embedded_in_surviving_parent(element)
    ]
    return (
        active_records,
        active_markdown_elements,
        suppressed_actions,
    )
