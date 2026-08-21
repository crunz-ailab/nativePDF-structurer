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


#@GPT PageInventory集中管理页面图元的稳定ID、局部标签、逻辑容器和唯一归属结果。
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
        #@GPT ID格式为"类型:序号"，序号即该图元在原始提取列表中的下标；
        #@GPT 这样任何阶段都能从ID反查原始对象，且ID在同一页内稳定可复现。
        self.entries: dict[str, dict] = {}
        self._by_kind: dict[str, list[str]] = defaultdict(list)
        #@GPT 原始dict对象在当前页面处理期间保持同一身份；建立临时反向索引，把检测器手中的对象安全映射回可序列化稳定ID。
        self._payload_identity_to_id: dict[tuple[str, int], str] = {}
        #@GPT 父drawing仍保留d:ID供阶段B局部分类；阶段C/D通过此映射取得di:父序号:item序号的原子路径ID。
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
            #@GPT drawing内部item按需展开；没有跨候选冲突的页面无需支付原子路径登记成本。

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
            #@GPT 阶段B在此写入局部标签；阶段A登记时保持为空字典。
            "labels": {},
            #@GPT 阶段D将在此写入唯一归属候选ID；当前阶段恒为None。
            "owner": None,
        }
        self._by_kind[kind].append(element_id)
        #@GPT Python对象id只作为本次运行内的反查键，绝不写入清单；对外始终输出d:/i:/t:稳定ID。
        self._payload_identity_to_id[(kind, id(payload))] = element_id

    #@GPT 表格父容器不是PDF中的单个drawing，而是检测器根据网格拓扑建立的逻辑对象；为它登记稳定tc:ID，使D阶段不再要求表格与单元格内部复杂矢量争抢同一批feature IDs。
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

    #@GPT 相邻闭合视觉容器登记独立稳定ID，使候选合并可以阻止跨容器连接。
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

    #@GPT 检测器已经持有原始图元对象时直接反查稳定ID，避免再次通过bbox中心点猜测对象身份。
    def id_for_payload(
        self,
        payload: dict,
        kind: str,
    ) -> str | None:
        return self._payload_identity_to_id.get((kind, id(payload)))

    #@GPT 批量反查保持输入顺序并去重，供表格簇、视觉连通域和原图组合候选共同使用。
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

    #@GPT 惰性展开保持ID确定性：无论何时触发，同一父drawing的第N个item始终是di:父序号:N；未冲突页面不支付items遍历成本。
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

    #@GPT 候选边界只能在“该检测器实际输入ID集合”内筛选；包含零面积线段的接触关系使用闭区间判断，不再依赖Rect.intersects()对退化矩形的行为。
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

    #@GPT 阶段C/D需要“落在某个候选bbox内的图元”，这里只提供纯几何查询，不含任何归属语义。
    #@GPT 区域查询统一使用ids_touching_rect和ids_centred_in，分别表达边界接触与中心归属语义。

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

    #@GPT 审计输出只保留ID、类型、bbox和标签；payload含PDF路径指令和图片字节，不写入JSON。
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

    #@GPT 单图元仲裁把候选排序与遍历、报告逻辑分离，保证真实归属和审计使用同一决策入口。
    #@GPT 规则按语义分三层：不同检测器先看显式优先级；同检测器再看置信度；仍相同才用候选稳定ID保证确定性。
    #@GPT 不把bbox面积、页码或文字内容作为隐藏特例，避免将局部经验固化为算法条件。
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

        #@GPT 所有竞争候选在此统一排序，避免归属结果与审计报告采用不同优先级口径。
        detector_names = {
            claim.get("detector", "unknown")
            for claim in contenders
        }
        ranked = sorted(
            contenders,
            key=lambda claim: (
                #@GPT 使用集中计算的有效优先级，使策略降级能同时作用于仲裁和报告。
                -claim_effective_priority(claim, priorities),
                -float(claim.get("confidence", 0.0)),
                -int(claim.get("feature_count", 0)),
                claim.get("claim_id", ""),
            ),
        )
        winner = ranked[0]
        #@GPT 获胜者、同优先级集合和报告字段必须复用实际排序的有效优先级。
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
            #@GPT 同优先级比较复用claim_effective_priority()，与获胜者排序保持一致。
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

        #@GPT 只有所有候选有效优先级一致时，才继续使用置信度、特征数量和稳定ID解释裁决原因。
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
                    #@GPT 报告展示实际参与排序的有效优先级；低置信表格会明确显示为150而不是误导性的300。
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
        #@GPT 阶段D使用“有效声明集合”承接惰性细分；原始manifest保持不变，审计时仍能看到检测器最初声明了哪个父drawing。
        effective_claimed_ids_by_claim: dict[str, set[str]] = {
            claim_id: {
                element_id
                for element_id in claim.get(
                    "claimed_element_ids", []
                )
                if element_id in self.entries
                #@GPT 阶段F统一处理文字；D只仲裁结构图元，避免视觉检测器在边界确定前提前锁定text owner。
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
                #@GPT 文字统一延迟到阶段F处理，本循环只建立结构图元的竞争关系。
                if self.entries[element_id]["kind"] == "text":
                    continue
                contenders_by_element_id[element_id].append(claim)

        #@GPT 只有父drawing被多个候选同时声明时才惰性展开items，避免密集但无冲突的页面登记海量原子路径。
        #@GPT 每个候选只获得与自身提议bbox接触的items；父drawing本身从有效声明中移除，防止一个跨区bbox继续整体参与仲裁。
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

        #@GPT 所有图元通过resolve_conflict()取得唯一所有者和可审计裁决原因。
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
                            #@GPT 直接使用resolve_conflict()返回的有效优先级，保证报告与实际排序一致。
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
            #@GPT 候选统计读取惰性细分后的有效集合，使获胜计数和阶段E边界使用真实原子ID。
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
            #@GPT 对候选完整性只读取ID归属结果：lost_feature_ids表示声明为结构证据、但在冲突中判给其他候选的图元，不使用bbox覆盖率补判。
            lost_feature_ids = sorted(
                element_id
                for element_id in feature_ids
                if owner_by_element_id.get(element_id) not in (
                    None,
                    claim_id,
                )
            )
            #@GPT 阶段E边界预览只使用候选真正赢得的feature IDs；不把尚未进入统一F阶段的文字行用于结构边界。
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
            #@GPT 阶段E只负责从检测器提议中剔除失去的图元，不能因一条穿越候选的长线把边界扩到提议之外；因此统一取“获胜特征并集 ∩ proposed_bbox”。
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
            #@GPT 候选必须拥有正面积特征边界，并满足检测器声明的完整性约束，才能进入输出阶段。
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
                    #@GPT 当前仅输出边界预览，不回写真实PNG；面积比可快速暴露候选是否因失去他人图元而大幅收缩。
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
                    #@GPT survives_preview用于过滤Markdown候选；调试图片可保留在磁盘中供归属审计。
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


#@GPT 台账构建独立成函数，使调用点只需一行；drawings/image_blocks/text_lines仍由现有提取逻辑产生，保证零行为变化。
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
