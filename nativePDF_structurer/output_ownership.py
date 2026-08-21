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

#@GPT 输出归属模块只消费已完成的Claim裁决，不运行检测器，也不修改页面几何。
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
            #@GPT 未进入台账的输出默认保持可见，并由invalid/missing审计记录暴露契约缺口。
            return True
        return bool(candidate.get("survives_preview", False))

    #@GPT 子视觉Claim仍拥有单元格内部复杂图元，但父表格存活时不应再生成第二张Markdown图片；这不是候选失败，而是层级容器负责统一渲染。
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
        #@GPT 同一正文视觉块的并列visual成员由renderer统一截图；renderer存活时隐藏成员Markdown，renderer失败时由阶段D收口解除该标记。
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
        #@GPT 父容器存活时优先层级嵌入，防止表格截图和单元格子视觉PNG重复进入Markdown。
        embedded_in_parent = embedded_in_surviving_parent(record)
        active = claim_survives(claim_id) and not embedded_in_parent
        #@GPT 原记录原地补充精简仲裁结果，避免另建一份难以与真实文件名对应的大JSON。
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
                #@GPT 淘汰动作直接记录resolver给出的存活原因，区分零面积残片、结构不完整和父容器嵌入。
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

    #@GPT Markdown原子同时检查Claim存活与父容器嵌入；被嵌入成员的磁盘PNG仍可保留用于审计。
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
