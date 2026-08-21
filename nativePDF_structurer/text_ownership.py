"""在阶段F将原生文字分配给存活的结构化输出。

模块先读取显式文字ID，再根据
最终结构边界的overlap补充候选，并按检测器策略处理竞争，保证每个文字图元
最多归属于一个实际输出的Claim；结构图元owner不会在此阶段被修改。

Assign native text to surviving structural outputs in Stage F.

Combines explicit text IDs with overlap against final owned structural bounds,
ranks competing outputs by detector policy, and assigns each text element to
at most one emitted Claim. The function reports ownership without changing
structural element owners.
"""

from __future__ import annotations

import fitz

from .claims import CLAIM_DETECTOR_PRIORITY
from .geometry import rect_area
from .inventory import PageInventory


#@GPT 阶段F为存活结构输出计算唯一文字归属；函数返回审计结果，不直接修改Markdown或结构图元owner。
def build_stage_f_text_ownership_preview(
    inventory: PageInventory,
    claims: list[dict],
    resolution: dict,
    emitted_claim_ids: set[str],
    overlap_threshold: float,
) -> dict:
    """为真正进入输出的存活候选生成唯一文字归属预览。"""
    claims_by_id = {
        claim["claim_id"]: claim
        for claim in claims
    }
    candidates_by_id = {
        candidate["claim_id"]: candidate
        for candidate in resolution.get("candidates", [])
        if candidate.get("survives_preview")
        and candidate["claim_id"] in emitted_claim_ids
        and candidate.get("owned_feature_bbox") is not None
    }
    #@GPT 复合输出组同时读取renderer元数据中的成员表格文字ID，使最终PNG拥有完整文字集合而不改变drawing的表格归属。
    explicit_text_ids_by_claim: dict[str, set[str]] = {
        claim_id: {
            element_id
            for element_id in [
                *claims_by_id[claim_id].get(
                    "claimed_element_ids", []
                ),
                *claims_by_id[claim_id]
                .get("metadata", {})
                .get("composite_member_text_element_ids", []),
            ]
            if element_id in inventory.entries
            and inventory.entry(element_id)["kind"] == "text"
        }
        for claim_id in candidates_by_id
    }
    assigned_text_ids_by_claim: dict[str, list[str]] = {
        claim_id: []
        for claim_id in candidates_by_id
    }
    owner_by_text_element_id: dict[str, str] = {}
    text_conflicts: list[dict] = []
    priorities = CLAIM_DETECTOR_PRIORITY

    for text_id in inventory.ids_of_kind("text"):
        text_bbox = inventory.bbox_of(text_id)
        text_area = rect_area(text_bbox)
        if text_area <= 0:
            continue

        contenders: list[dict] = []
        for claim_id, candidate in candidates_by_id.items():
            structure_bbox = fitz.Rect(
                candidate["owned_feature_bbox"]
            )
            intersection = text_bbox & structure_bbox
            overlap_ratio = (
                rect_area(intersection) / text_area
                if not intersection.is_empty
                else 0.0
            )
            explicit = (
                text_id
                in explicit_text_ids_by_claim[claim_id]
            )
            if (
                not explicit
                and overlap_ratio
                < overlap_threshold
            ):
                continue
            claim = claims_by_id[claim_id]
            contenders.append(
                {
                    "claim_id": claim_id,
                    "detector": claim["detector"],
                    "priority": priorities.get(
                        claim["detector"], 0
                    ),
                    "confidence": claim.get("confidence", 0.0),
                    "explicit_detector_proposal": explicit,
                    "overlap_ratio": round(overlap_ratio, 6),
                }
            )

        if not contenders:
            continue
        ranked = sorted(
            contenders,
            key=lambda contender: (
                -contender["priority"],
                -int(contender["explicit_detector_proposal"]),
                -contender["overlap_ratio"],
                -float(contender["confidence"]),
                contender["claim_id"],
            ),
        )
        winner = ranked[0]
        owner_by_text_element_id[text_id] = winner["claim_id"]
        assigned_text_ids_by_claim[winner["claim_id"]].append(
            text_id
        )
        if len(ranked) > 1:
            text_conflicts.append(
                {
                    "text_element_id": text_id,
                    "bbox": [
                        round(value, 3)
                        for value in text_bbox
                    ],
                    "winner_claim_id": winner["claim_id"],
                    "contenders": ranked,
                }
            )

    return {
        "policy": {
            "applied_to_markdown": False,
            "qualification": (
                "explicit detector text proposal OR text bbox overlap "
                f">= {overlap_threshold:.2f}"
            ),
            "ranking": [
                "detector_priority_desc",
                "explicit_detector_proposal_desc",
                "text_overlap_ratio_desc",
                "confidence_desc",
                "claim_id_asc",
            ],
        },
        "emitted_candidate_count": len(candidates_by_id),
        "assigned_text_count": len(owner_by_text_element_id),
        "unassigned_text_count": (
            len(inventory.ids_of_kind("text"))
            - len(owner_by_text_element_id)
        ),
        "conflict_text_count": len(text_conflicts),
        "owner_by_text_element_id": owner_by_text_element_id,
        "candidates": [
            {
                "claim_id": claim_id,
                "detector": claims_by_id[claim_id]["detector"],
                "assigned_text_count": len(text_ids),
                "assigned_text_element_ids": sorted(text_ids),
            }
            for claim_id, text_ids in sorted(
                assigned_text_ids_by_claim.items()
            )
        ],
        "conflicts": text_conflicts,
    }
