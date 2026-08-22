"""管理页面图元的稳定ID和Claim归属仲裁。

模块将原生drawing、图片、文字行及
逻辑容器登记到单页台账，把检测器payload反查为稳定ID，只在父drawing发生
冲突时惰性展开items，并保证每个结构图元最多属于一个存活Claim。

关键不变量:
    * 稳定ID在单页内可重复生成，且不包含Python对象ID。
    * 检测器只能认领其声明输入集合中的图元。
    * 文字归属延迟到阶段F处理。
    * 惰性item细分必须保留父drawing来源信息。

Manage stable page-element IDs and Claim arbitration.

Registers native drawings, images, text lines, and logical containers in one
page-local inventory. It maps detector payloads back to stable IDs, lazily
expands drawing items only on conflict, and assigns each structural element to
at most one surviving Claim.

Key invariants:
    * Stable IDs are deterministic within a page and never contain object IDs.
    * Detector Claims may own only elements present in their declared input.
    * Text ownership is deferred to Stage F.
    * Lazy item expansion must preserve parent-source provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import fitz

from .claims import CLAIM_DETECTOR_PRIORITY, claim_effective_priority
from .geometry import drawing_item_bbox, rect_area, rect_center, union_rects


class PageInventory:
    """Stable page-element inventory and ownership ledger.

    为每个原始图元分配稳定ID，并保存跨检测器的声明与唯一归属结果。
    """

    def __init__(
        self,
        page_number: int,
        page_rect: fitz.Rect,
        drawings: list[dict],
        image_blocks: list[dict],
        text_lines: list[dict],
    ) -> None:
        self.page_number = page_number
        self.page_rect = fitz.Rect(page_rect)
        self.entries: dict[str, dict] = {}
        self._by_kind: dict[str, list[str]] = defaultdict(list)
        self._payload_identity_to_id: dict[tuple[str, int], str] = {}
        self._drawing_item_ids_by_parent_identity: dict[
            int,
            list[str],
        ] = defaultdict(list)

        for index, drawing in enumerate(drawings):
            self._register(
                element_id=f"d:{index}",
                kind="drawing",
                source_index=index,
                bbox=fitz.Rect(drawing["rect"]).normalize(),
                payload=drawing,
            )

        for index, image in enumerate(image_blocks):
            self._register(
                element_id=f"i:{index}",
                kind="image",
                source_index=index,
                bbox=fitz.Rect(image["bbox"]).normalize(),
                payload=image,
            )

        for index, line in enumerate(text_lines):
            self._register(
                element_id=f"t:{index}",
                kind="text",
                source_index=index,
                bbox=fitz.Rect(line["bbox"]).normalize(),
                payload=line,
            )

    def _register(
        self,
        element_id: str,
        kind: str,
        source_index: int | str,
        bbox: fitz.Rect,
        payload: dict,
    ) -> None:
        self.entries[element_id] = {
            "id": element_id,
            "kind": kind,
            "source_index": source_index,
            "bbox": bbox,
            "payload": payload,
            "labels": {},
            "owner": None,
        }
        self._by_kind[kind].append(element_id)
        self._payload_identity_to_id[(kind, id(payload))] = element_id

    def register_table_container(
        self,
        table_number: int,
        bbox: fitz.Rect,
        payload: dict,
    ) -> str:
        """登记一个页面级逻辑表格容器，并返回稳定ID。"""
        element_id = f"tc:{table_number - 1}"
        if element_id in self.entries:
            return element_id

        self._register(
            element_id=element_id,
            kind="table_container",
            source_index=table_number - 1,
            bbox=fitz.Rect(bbox).normalize(),
            payload=payload,
        )
        self.entries[element_id]["labels"] = {
            "logical_table_container": True,
        }
        return element_id

    def register_visual_container(
        self,
        container_number: int,
        bbox: fitz.Rect,
        payload: dict,
    ) -> str:
        """登记一个由闭合规则线构成的逻辑视觉容器。"""
        element_id = f"vc:{container_number - 1}"
        if element_id in self.entries:
            return element_id

        self._register(
            element_id=element_id,
            kind="visual_container",
            source_index=container_number - 1,
            bbox=fitz.Rect(bbox).normalize(),
            payload=payload,
        )
        self.entries[element_id]["labels"] = {
            "closed_visual_container": True,
        }
        return element_id

    def ids_of_kind(self, kind: str) -> list[str]:
        """返回某类图元的全部ID，顺序与原始提取顺序一致。"""
        return list(self._by_kind.get(kind, []))

    def entry(self, element_id: str) -> dict:
        return self.entries[element_id]

    def bbox_of(self, element_id: str) -> fitz.Rect:
        return self.entries[element_id]["bbox"]

    def payload_of(self, element_id: str) -> dict:
        return self.entries[element_id]["payload"]

    def id_for_payload(
        self,
        payload: dict,
        kind: str,
    ) -> str | None:
        return self._payload_identity_to_id.get((kind, id(payload)))

    def ids_for_payloads(
        self,
        payloads: list[dict],
        kind: str,
    ) -> list[str]:
        matched: list[str] = []
        seen: set[str] = set()
        for payload in payloads:
            element_id = self.id_for_payload(payload, kind)
            if element_id is None or element_id in seen:
                continue
            seen.add(element_id)
            matched.append(element_id)
        return matched

    def _ensure_drawing_item_ids(
        self,
        drawing: dict,
    ) -> list[str]:
        existing = self._drawing_item_ids_by_parent_identity.get(
            id(drawing)
        )
        if existing:
            return list(existing)

        parent_id = self.id_for_payload(drawing, "drawing")
        if parent_id is None:
            return []
        parent_index = self.entry(parent_id)["source_index"]
        item_ids: list[str] = []
        for item_index, item in enumerate(drawing.get("items", [])):
            item_bbox = drawing_item_bbox(item)
            if item_bbox is None:
                continue
            item_id = f"di:{parent_index}:{item_index}"
            if item_id not in self.entries:
                self._register(
                    element_id=item_id,
                    kind="drawing_item",
                    source_index=f"{parent_index}:{item_index}",
                    bbox=item_bbox,
                    payload={
                        "parent_drawing": drawing,
                        "parent_drawing_index": parent_index,
                        "item_index": item_index,
                        "item": item,
                    },
                )
            item_ids.append(item_id)
        self._drawing_item_ids_by_parent_identity[id(drawing)] = (
            item_ids
        )
        return list(item_ids)

    def ids_touching_rect(
        self,
        rect: fitz.Rect,
        kinds: tuple[str, ...] = ("drawing", "image", "text"),
        allowed_element_ids: list[str] | None = None,
    ) -> list[str]:
        rect = fitz.Rect(rect).normalize()
        allowed = (
            set(allowed_element_ids)
            if allowed_element_ids is not None
            else None
        )
        matched: list[str] = []
        for kind in kinds:
            for element_id in self._by_kind.get(kind, []):
                if allowed is not None and element_id not in allowed:
                    continue
                element_bbox = self.entries[element_id]["bbox"]
                if (
                    element_bbox.x1 < rect.x0
                    or element_bbox.x0 > rect.x1
                    or element_bbox.y1 < rect.y0
                    or element_bbox.y0 > rect.y1
                ):
                    continue
                matched.append(element_id)
        return matched


    def ids_centred_in(
        self,
        rect: fitz.Rect,
        kinds: tuple[str, ...] = ("drawing", "image", "text"),
    ) -> list[str]:
        """返回中心点落在rect内的图元ID。"""
        rect = fitz.Rect(rect).normalize()
        return [
            element_id
            for kind in kinds
            for element_id in self._by_kind.get(kind, [])
            if rect.contains(rect_center(self.entries[element_id]["bbox"]))
        ]

    def counts(self) -> dict[str, int]:
        return {
            kind: len(ids)
            for kind, ids in sorted(self._by_kind.items())
        }

    def to_audit_record(self, label_summary_only: bool = True) -> dict:
        label_counts: Counter = Counter()
        for entry in self.entries.values():
            for label_name, label_value in entry["labels"].items():
                if label_value:
                    label_counts[label_name] += 1
        record: dict = {
            "page_number": self.page_number,
            "counts": self.counts(),
            "label_counts": dict(sorted(label_counts.items())),
        }
        if not label_summary_only:
            record["entries"] = [
                {
                    "id": entry["id"],
                    "kind": entry["kind"],
                    "bbox": [round(value, 3) for value in entry["bbox"]],
                    "labels": {
                        name: value
                        for name, value in entry["labels"].items()
                        if value
                    },
                }
                for entry in self.entries.values()
            ]
        return record

    def resolve_conflict(
        self,
        element_id: str,
        contenders: list[dict],
        detector_priority: dict[str, int] | None = None,
    ) -> dict:
        """为一个图元选择唯一候选，并返回可审计的裁决依据。"""
        priorities = {
            **CLAIM_DETECTOR_PRIORITY,
            **(detector_priority or {}),
        }
        if not contenders:
            return {
                "element_id": element_id,
                "winner_claim_id": None,
                "resolution_reason": "no_contender",
                "ranked_contenders": [],
            }

        detector_names = {
            claim.get("detector", "unknown")
            for claim in contenders
        }
        ranked = sorted(
            contenders,
            key=lambda claim: (
                -claim_effective_priority(claim, priorities),
                -float(claim.get("confidence", 0.0)),
                -int(claim.get("feature_count", 0)),
                claim.get("claim_id", ""),
            ),
        )
        winner = ranked[0]
        winner_priority = claim_effective_priority(
            winner,
            priorities,
        )
        top_priority = claim_effective_priority(
            ranked[0],
            priorities,
        )
        top_confidence = float(ranked[0].get("confidence", 0.0))
        same_priority = [
            claim
            for claim in ranked
            if claim_effective_priority(claim, priorities)
            == top_priority
        ]
        same_confidence = [
            claim
            for claim in same_priority
            if float(claim.get("confidence", 0.0))
            == top_confidence
        ]
        same_feature_count = [
            claim
            for claim in same_confidence
            if int(claim.get("feature_count", 0))
            == int(winner.get("feature_count", 0))
        ]

        if len(same_priority) < len(ranked):
            resolution_reason = "detector_priority"
        elif len(same_confidence) > 1 and len(same_feature_count) < len(same_confidence):
            resolution_reason = "confidence"
        elif len(same_feature_count) > 1:
            resolution_reason = "feature_count"
        else:
            resolution_reason = "stable_claim_id"

        return {
            "element_id": element_id,
            "winner_claim_id": winner.get("claim_id"),
            "resolution_reason": resolution_reason,
            "ranked_contenders": [
                {
                    "claim_id": claim.get("claim_id"),
                    "detector": claim.get("detector"),
                    "priority": claim_effective_priority(
                        claim,
                        priorities,
                    ),
                    "confidence": claim.get("confidence", 0.0),
                    "feature_count": claim.get("feature_count", 0),
                }
                for claim in ranked
            ],
            "same_detector_conflict": len(detector_names) == 1,
            "winner_priority": winner_priority,
        }

    def resolve_conflicts(
        self,
        claims: list[dict],
        detector_priority: dict[str, int] | None = None,
    ) -> dict:
        """按检测器优先级、置信度和稳定claim_id生成唯一归属预览。"""
        priorities = {
            **CLAIM_DETECTOR_PRIORITY,
            **(detector_priority or {}),
        }
        claims_by_id = {
            claim["claim_id"]: claim
            for claim in claims
        }
        effective_claimed_ids_by_claim: dict[str, set[str]] = {
            claim_id: {
                element_id
                for element_id in claim.get(
                    "claimed_element_ids", []
                )
                if element_id in self.entries
                and self.entries[element_id]["kind"] != "text"
            }
            for claim_id, claim in claims_by_id.items()
        }
        effective_feature_ids_by_claim: dict[str, set[str]] = {
            claim_id: set(
                claim.get("feature_element_ids", [])
            ).intersection(
                effective_claimed_ids_by_claim[claim_id]
            )
            for claim_id, claim in claims_by_id.items()
        }
        contenders_by_element_id: dict[str, list[dict]] = defaultdict(list)
        invalid_element_references: list[dict] = []

        for claim in claims:
            for element_id in claim.get("claimed_element_ids", []):
                if element_id not in self.entries:
                    invalid_element_references.append(
                        {
                            "claim_id": claim["claim_id"],
                            "element_id": element_id,
                        }
                    )
                    continue
                if self.entries[element_id]["kind"] == "text":
                    continue
                contenders_by_element_id[element_id].append(claim)

        refined_parent_drawing_count = 0
        refined_drawing_item_count = 0
        for element_id, contenders in list(
            contenders_by_element_id.items()
        ):
            entry = self.entries[element_id]
            if entry["kind"] != "drawing" or len(contenders) <= 1:
                continue

            drawing = entry["payload"]
            item_ids = self._ensure_drawing_item_ids(drawing)
            if not item_ids:
                continue

            del contenders_by_element_id[element_id]
            refined_parent_drawing_count += 1
            refined_drawing_item_count += len(item_ids)
            for claim in contenders:
                claim_id = claim["claim_id"]
                parent_was_feature = (
                    element_id
                    in effective_feature_ids_by_claim[claim_id]
                )
                effective_claimed_ids_by_claim[claim_id].discard(
                    element_id
                )
                effective_feature_ids_by_claim[claim_id].discard(
                    element_id
                )
                claim_rect = fitz.Rect(claim["bbox"]).normalize()
                for item_id in item_ids:
                    item_rect = self.bbox_of(item_id)
                    if (
                        item_rect.x1 < claim_rect.x0
                        or item_rect.x0 > claim_rect.x1
                        or item_rect.y1 < claim_rect.y0
                        or item_rect.y0 > claim_rect.y1
                    ):
                        continue
                    contenders_by_element_id[item_id].append(claim)
                    effective_claimed_ids_by_claim[claim_id].add(
                        item_id
                    )
                    if parent_was_feature:
                        effective_feature_ids_by_claim[claim_id].add(
                            item_id
                        )

        owner_by_element_id: dict[str, str] = {}
        conflict_records: list[dict] = []
        for element_id, contenders in sorted(
            contenders_by_element_id.items()
        ):
            decision = self.resolve_conflict(
                element_id=element_id,
                contenders=contenders,
                detector_priority=priorities,
            )
            ranked = contenders
            winner_id = decision["winner_claim_id"]
            owner_by_element_id[element_id] = winner_id
            self.entries[element_id]["owner"] = winner_id

            if len(ranked) <= 1:
                continue

            entry = self.entries[element_id]
            conflict_records.append(
                {
                    "element_id": element_id,
                    "kind": entry["kind"],
                    "bbox": [
                        round(value, 3)
                        for value in entry["bbox"]
                    ],
                    "winner_claim_id": winner_id,
                    "resolution_reason": decision[
                        "resolution_reason"
                    ],
                    "same_detector_conflict": decision[
                        "same_detector_conflict"
                    ],
                    "contenders": [
                        {
                            "claim_id": claim["claim_id"],
                            "detector": claim["detector"],
                            "priority": claim["priority"],
                            "confidence": claim.get(
                                "confidence", 0.0
                            ),
                        }
                        for claim in decision[
                            "ranked_contenders"
                        ]
                    ],
                }
            )

        candidate_results: list[dict] = []
        for claim_id, claim in sorted(claims_by_id.items()):
            claimed_ids = effective_claimed_ids_by_claim[claim_id]
            feature_ids = effective_feature_ids_by_claim[claim_id]
            won_ids = sorted(
                element_id
                for element_id in claimed_ids
                if owner_by_element_id.get(element_id) == claim_id
            )
            won_feature_ids = sorted(
                element_id
                for element_id in feature_ids
                if owner_by_element_id.get(element_id) == claim_id
            )
            lost_feature_ids = sorted(
                element_id
                for element_id in feature_ids
                if owner_by_element_id.get(element_id) not in (
                    None,
                    claim_id,
                )
            )
            raw_owned_feature_bbox = (
                union_rects(
                    [
                        self.bbox_of(element_id)
                        for element_id in won_feature_ids
                    ]
                )
                if won_feature_ids
                else None
            )
            proposed_bbox = fitz.Rect(claim.get("bbox", (0, 0, 0, 0)))
            owned_feature_bbox = None
            if raw_owned_feature_bbox is not None:
                clipped_owned_bbox = (
                    raw_owned_feature_bbox & proposed_bbox
                )
                if not clipped_owned_bbox.is_empty:
                    owned_feature_bbox = clipped_owned_bbox
            proposed_area = rect_area(proposed_bbox)
            owned_area = (
                rect_area(owned_feature_bbox)
                if owned_feature_bbox is not None
                else 0.0
            )
            requires_complete_feature_ownership = bool(
                claim.get("metadata", {}).get(
                    "requires_complete_feature_ownership_on_conflict",
                    False,
                )
            )
            survives_preview = (
                bool(won_feature_ids)
                and owned_feature_bbox is not None
                and not (
                    requires_complete_feature_ownership
                    and bool(lost_feature_ids)
                )
            )
            if not won_feature_ids:
                survival_reason = "lost_all_feature_elements"
            elif owned_feature_bbox is None:
                survival_reason = "no_positive_area_owned_feature_bbox"
            elif (
                requires_complete_feature_ownership
                and lost_feature_ids
            ):
                survival_reason = (
                    "incomplete_low_confidence_structure_after_conflict"
                )
            else:
                survival_reason = "survives"
            candidate_results.append(
                {
                    "claim_id": claim_id,
                    "detector": claim["detector"],
                    "claimed_count": len(claimed_ids),
                    "won_count": len(won_ids),
                    "won_element_ids": won_ids,
                    "feature_count": len(feature_ids),
                    "won_feature_count": len(won_feature_ids),
                    "won_feature_element_ids": won_feature_ids,
                    "lost_feature_count": len(lost_feature_ids),
                    "lost_feature_element_ids": lost_feature_ids,
                    "proposed_bbox": [
                        round(value, 3)
                        for value in proposed_bbox
                    ],
                    "raw_owned_feature_bbox": (
                        [
                            round(value, 3)
                            for value in raw_owned_feature_bbox
                        ]
                        if raw_owned_feature_bbox is not None
                        else None
                    ),
                    "owned_feature_bbox": (
                        [
                            round(value, 3)
                            for value in owned_feature_bbox
                        ]
                        if owned_feature_bbox is not None
                        else None
                    ),
                    "owned_to_proposed_area_ratio": (
                        round(owned_area / proposed_area, 6)
                        if proposed_area > 0
                        else None
                    ),
                    "survives_preview": survives_preview,
                    "survival_reason": survival_reason,
                }
            )

        return {
            "policy": {
                "ranking": [
                    "detector_priority_desc",
                    "confidence_desc",
                    "feature_count_desc",
                    "claim_id_asc",
                ],
                "detector_priority": priorities,
                "applied_to_outputs": False,
                "owned_feature_bbox_preview": True,
                "owned_bbox_invariant": (
                    "union(won_feature_elements) intersect proposed_bbox"
                ),
                "text_assignment_deferred_to_stage_f": True,
                "lazy_drawing_item_refinement": {
                    "enabled": True,
                    "refined_parent_drawing_count": (
                        refined_parent_drawing_count
                    ),
                    "registered_drawing_item_count": (
                        refined_drawing_item_count
                    ),
                },
            },
            "claimed_element_count": len(owner_by_element_id),
            "conflict_element_count": len(conflict_records),
            "invalid_element_reference_count": len(
                invalid_element_references
            ),
            "invalid_element_references": invalid_element_references,
            "owner_by_element_id": owner_by_element_id,
            "conflicts": conflict_records,
            "candidates": candidate_results,
        }


def build_page_inventory(
    page_number: int,
    page_rect: fitz.Rect,
    drawings: list[dict],
    image_blocks: list[dict],
    text_lines: list[dict],
) -> PageInventory:
    """按现有提取结果构建页面图元台账。"""
    return PageInventory(
        page_number=page_number,
        page_rect=page_rect,
        drawings=drawings,
        image_blocks=image_blocks,
        text_lines=text_lines,
    )
