"""定义稳定的Claim清单结构和检测器优先级策略。

模块将检测器提议规范化为
可审计字典，并计算归属仲裁使用的有效优先级；它不读取PDF几何，也不决定
输出边界。

Define the stable Claim schema and detector-priority policy.

Normalizes detector proposals into auditable Claim dictionaries and computes
the effective priority used by ownership arbitration. This module does not
inspect PDF geometry or decide output boundaries.
"""

from __future__ import annotations

from typing import Any

import fitz


#@GPT 检测器优先级是Claim仲裁的公共策略，不属于某个页面检测器；独立模块让表格、视觉区域和原图策略可以在不修改核心流水线的情况下统一调整和测试。
CLAIM_DETECTOR_PRIORITY = {
    "table_detector": 300,
    "text_separator": 250,
    "visual_region": 200,
    "source_image_processor": 100,
}


#@GPT Claim清单创建函数只负责格式归一化：claimed自动进入input，feature只能来自claimed，置信度限制在0到1之间；不承担候选生死或图元归属决策。
def create_claim_manifest(
    detector_name: str,
    output_element_id: str,
    input_element_ids: list[str],
    claimed_element_ids: list[str],
    feature_element_ids: list[str],
    bbox: fitz.Rect,
    confidence: float,
    reason_codes: list[str],
    output_artifacts: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one normalized candidate Claim for structural arbitration."""

    normalized_claimed_ids = sorted(set(claimed_element_ids))
    normalized_feature_ids = sorted(
        set(feature_element_ids).intersection(normalized_claimed_ids)
    )
    normalized_input_ids = sorted(
        set(input_element_ids).union(normalized_claimed_ids)
    )
    normalized_bbox = fitz.Rect(bbox).normalize()
    return {
        "claim_id": output_element_id,
        "detector": detector_name,
        "output_element_id": output_element_id,
        "confidence": round(
            min(1.0, max(0.0, float(confidence))),
            6,
        ),
        "input_count": len(normalized_input_ids),
        "input_element_ids": normalized_input_ids,
        "claimed_count": len(normalized_claimed_ids),
        "claimed_element_ids": normalized_claimed_ids,
        "feature_count": len(normalized_feature_ids),
        "feature_element_ids": normalized_feature_ids,
        "bbox": [round(value, 3) for value in normalized_bbox],
        "reason_codes": sorted(set(reason_codes)),
        "output_artifacts": output_artifacts or [],
        "metadata": metadata or {},
    }


#@GPT 有效优先级保留为独立函数，即使当前等于检测器基础优先级，未来增加结构型降级规则时也能保证仲裁和审计报告共用同一入口。
def claim_effective_priority(
    claim: dict[str, Any],
    detector_priority: dict[str, int],
) -> int:
    """Return the priority used for one Claim during element arbitration."""

    detector = claim.get("detector", "unknown")
    return detector_priority.get(detector, 0)
