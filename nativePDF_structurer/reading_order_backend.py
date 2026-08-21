"""通过可替换的空间排序后端确定页面原子阅读顺序。

模块先把旋转原子映射到
虚拟转正坐标，调用兼容GapTree的后端，再执行窄列右挂、跨列首行校正、
水平XY-Cut和分隔符优先关系等后处理。

处理流程:
    1. 将 ``sort_bbox`` 转换到虚拟转正坐标。
    2. 在空间切分前合并具有稳定结构的同行文字碎片。
    3. 构建节点并修复窄列和跨列首行归属。
    4. 执行水平分区和显式局部先后关系。
    5. 展平节点并生成Markdown行号与审计元数据。

关键不变量:
    * 排序使用 ``sort_bbox``，渲染继续使用 ``render_bbox``。
    * 文字碎片合并期间非文本原子保持不可拆分。
    * 校正依据几何和重复版式结构，不读取特定文字标签。
    * 空间关系成环时采用稳定的后端节点顺序。

Order page atoms through a pluggable spatial sorting backend.

Transforms rotated atoms into virtual upright coordinates, invokes the
configured GapTree-compatible backend, and applies corrections for narrow
columns, cross-column leading rows, horizontal XY-Cut partitions, and explicit
separator-before-heading relations.

Processing flow:
    1. Convert ``sort_bbox`` values into virtual upright coordinates.
    2. Coalesce stable same-row text fragments before spatial partitioning.
    3. Build nodes and repair narrow-column and leading-row placement.
    4. Apply horizontal partitions and explicit local precedence relations.
    5. Flatten nodes into Markdown row indices and audit metadata.

Key invariants:
    * Sorting uses ``sort_bbox``; rendering continues to use ``render_bbox``.
    * Non-text atoms remain indivisible during text-fragment coalescing.
    * Corrections use geometry and repeated layout structure, not text labels.
    * Cyclic spatial relations fall back to stable backend node order.
"""

from __future__ import annotations

import fitz
from collections import Counter, defaultdict
from ._vendor.gap_tree import GapTree

#@GPT 分隔符与标题候选竞争时复用项目统一的非负矩形面积定义，避免阅读顺序模块产生另一套面积语义。
from .geometry import rect_area

TEXT_ROW_TOLERANCE_POINTS = 3.0


#@GPT 将旋转文本bbox转换到虚拟转正坐标；排序后端只消费该坐标，不修改原PDF bbox。
def transformed_reading_bbox(
    rect: fitz.Rect,
    source_angle: int,
    page_rect: fitz.Rect,
) -> tuple[float, float, float, float]:
    """把原页面矩形转换到对应文字方向的转正坐标系。"""

    page_width = page_rect.width
    page_height = page_rect.height

    def transform_point(x: float, y: float) -> tuple[float, float]:
        if source_angle == 0:
            return x, y
        if source_angle == -90:
            return page_height - y, x
        if source_angle == 90:
            return y, page_width - x
        if source_angle == 180:
            return page_width - x, page_height - y
        return x, y

    points = [
        transform_point(rect.x0, rect.y0),
        transform_point(rect.x1, rect.y0),
        transform_point(rect.x0, rect.y1),
        transform_point(rect.x1, rect.y1),
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


#@GPT 虚拟转正位置与完整bbox必须使用同一方向变换；该函数只负责中心点排序键，不改变原始页面坐标。
#@GPT 通过显式接收页面矩形和旋转角度，避免阅读顺序模块复制核心全局状态。
def transformed_reading_position(
    rect: fitz.Rect,
    source_angle: int,
    page_rect: fitz.Rect,
) -> tuple[float, float]:
    """
    将文本行中心映射到“虚拟转正页面”的阅读坐标。

    返回值：
    - virtual_y：转正后的从上至下顺序
    - virtual_x：同一行内从左至右顺序
    """
    rect = fitz.Rect(rect)
    #@GPT PyMuPDF Rect不提供统一的center接口，直接取两端点均值且不增加几何模块依赖。
    x = (rect.x0 + rect.x1) / 2.0
    y = (rect.y0 + rect.y1) / 2.0

    page_width = page_rect.width
    page_height = page_rect.height

    if source_angle == 0:
        return y, x

    if source_angle == -90:
        # 原页面顺时针旋转 90°：
        # 新 y 对应原 x；新 x 对应 H - 原 y。
        return x, page_height - y

    if source_angle == 90:
        # 原页面逆时针旋转 90°：
        # 新 y 对应 W - 原 x；新 x 对应原 y。
        return page_width - x, y

    if source_angle == 180:
        return page_height - y, page_width - x

    return y, x

from collections.abc import Callable
from typing import Any


#@GPT 适配器只包住GapTree实际使用的三个接口，保持发布版排序输出稳定。
class GapTreeSorter:
    """Expose the small sorter interface consumed by the core reflow stage."""

    backend_name = "gaptree"

    def __init__(
        self,
        implementation: Callable[..., Any],
        *,
        get_bbox: Callable[[dict], Any],
    ) -> None:
        self._implementation = implementation(get_bbox=get_bbox)

    def sort(self, elements: list[dict]) -> Any:
        """Sort elements through the configured backend implementation."""

        return self._implementation.sort(elements)

    def get_nodes_text_blocks(self) -> list[list[dict]]:
        """Return backend node groups used by project-specific post-processing."""

        return self._implementation.get_nodes_text_blocks()

    @property
    def current_rows(self) -> Any:
        """Expose the backend row metadata for the existing audit report."""

        return self._implementation.current_rows

    #@GPT 核心审计报告还读取GapTree当前竖切线；适配器必须原样透传，不能只覆盖排序和节点接口。
    @property
    def current_cuts(self) -> Any:
        """Expose the backend cut metadata for the existing audit report."""

        return self._implementation.current_cuts


#@GPT 发布版只注册经过完整回归验证的GapTree；实验性排序器保存在本地checkpoint，不进入公共运行路径。
SUPPORTED_READING_ORDER_BACKENDS = ("gaptree",)


def create_reading_order_backend(
    backend_name: str,
    *,
    implementation: Callable[..., Any] = GapTree,
    get_bbox: Callable[[dict], Any],
) -> GapTreeSorter:
    """Create the selected reading-order backend behind one stable seam."""

    normalized_name = backend_name.strip().lower()
    if normalized_name == "gaptree":
        return GapTreeSorter(implementation, get_bbox=get_bbox)
    raise ValueError(
        f"unsupported reading-order backend: {backend_name!r}; "
        f"available={SUPPORTED_READING_ORDER_BACKENDS}"
    )


#@GPT “窄文本域按同行右挂”用于处理目录和步骤列表中的窄编号列归属。
#@GPT 该函数只接收页面几何与显式行容差，不依赖核心文件的全局变量，方便后续替换排序后端并保持算法可测试。
def attach_narrow_text_domains_by_rows(
    node_groups: list[list[dict]],
    page_rect: fitz.Rect,
    *,
    row_tolerance_points: float,
) -> tuple[list[list[dict]], dict]:
    """按同一阅读行把窄列元素挂到右侧最近的文本元素。

    ``row_tolerance_points`` 由调用方显式传入，避免阅读顺序模块复制
    全局策略常量。
    """
    copied_groups = [list(group) for group in node_groups]
    if not copied_groups:
        return [], {
            "enabled": False,
            "reason": "no_gap_tree_nodes",
            "merged_node_count": 0,
            "moved_element_count": 0,
            "merges": [],
        }

    text_elements = [
        element
        for group in copied_groups
        for element in group
        if element.get("kind", "text") == "text"
    ]
    text_heights = [
        element["gaptree_bbox"][3] - element["gaptree_bbox"][1]
        for element in text_elements
        if element["gaptree_bbox"][3] > element["gaptree_bbox"][1]
    ]
    median_text_height = (
        float(np.median(text_heights))
        if text_heights
        else max(page_rect.height * 0.01, 1.0)
    )
    #@GPT 窄列宽度和同行间距都按当前页文字高度自适应；同行间距上限只排除跨大栏的远距离误配。
    narrow_domain_threshold = max(
        median_text_height * 2.5,
        page_rect.width * 0.012,
    )
    row_attachment_gap_threshold = max(
        median_text_height * 4.0,
        page_rect.width * 0.05,
    )
    local_pair_gap_threshold = max(
        median_text_height * 2.0,
        page_rect.width * 0.025,
    )

    #@GPT 在每个GapTree节点内筛选单个窄文本元素，再按X左边缘、右边缘或中心的稳定对齐关系聚簇；宽元素不会污染独立编号列。
    #@GPT 对齐容差随当前页文字高度缩放，不读取编号内容、目录标题、页码或固定X坐标。
    narrow_x_alignment_tolerance = max(
        median_text_height * 0.60,
        page_rect.width * 0.004,
    )
    domain_records: list[dict] = []
    for node_index, group in enumerate(copied_groups):
        narrow_group_text = [
            element
            for element in group
            if element.get("kind", "text") == "text"
            and (
                element["gaptree_bbox"][2]
                - element["gaptree_bbox"][0]
            )
            <= narrow_domain_threshold
        ]
        if not narrow_group_text:
            continue

        #@GPT 每个窄元素只进入一个X簇；簇锚点使用当前成员中位数，避免少量PDF浮点误差
        #@GPT 或编号字符宽度变化造成链式漂移。左/右/中心任一对齐即可覆盖左对齐和右对齐编号列。
        x_clusters: list[list[dict]] = []
        for element in sorted(
            narrow_group_text,
            key=lambda item: (
                item["gaptree_bbox"][0],
                (
                    item["gaptree_bbox"][1]
                    + item["gaptree_bbox"][3]
                )
                / 2.0,
            ),
        ):
            element_rect = fitz.Rect(element["gaptree_bbox"])
            element_center_x = (
                element_rect.x0 + element_rect.x1
            ) / 2.0
            matching_clusters: list[tuple[float, list[dict]]] = []
            for cluster in x_clusters:
                cluster_rects = [
                    fitz.Rect(item["gaptree_bbox"])
                    for item in cluster
                ]
                anchor_x0 = float(
                    np.median([rect.x0 for rect in cluster_rects])
                )
                anchor_x1 = float(
                    np.median([rect.x1 for rect in cluster_rects])
                )
                anchor_center_x = float(
                    np.median(
                        [
                            (rect.x0 + rect.x1) / 2.0
                            for rect in cluster_rects
                        ]
                    )
                )
                alignment_distance = min(
                    abs(element_rect.x0 - anchor_x0),
                    abs(element_rect.x1 - anchor_x1),
                    abs(element_center_x - anchor_center_x),
                )
                if alignment_distance <= narrow_x_alignment_tolerance:
                    matching_clusters.append(
                        (alignment_distance, cluster)
                    )

            if matching_clusters:
                min(
                    matching_clusters,
                    key=lambda item: item[0],
                )[1].append(element)
            else:
                x_clusters.append([element])

        for cluster in x_clusters:
            if not cluster:
                continue

            #@GPT 重复窄列要求跨两个Y行；单元素簇只能通过下方严格的同节点同行近距离规则。
            cluster_row_centers: list[float] = []
            for element in sorted(
                cluster,
                key=lambda item: (
                    item["gaptree_bbox"][1]
                    + item["gaptree_bbox"][3]
                )
                / 2.0,
            ):
                center_y = (
                    element["gaptree_bbox"][1]
                    + element["gaptree_bbox"][3]
                ) / 2.0
                if (
                    not cluster_row_centers
                    or abs(center_y - cluster_row_centers[-1])
                    > row_tolerance_points
                ):
                    cluster_row_centers.append(center_y)
            if len(cluster_row_centers) < 2:
                #@GPT 图片或表格可能把列表末项切成独立节点；单个窄元素仅在同节点存在同行、近距离右侧文字时进入局部配对。
                has_local_same_row_pair = any(
                    (
                        target is not source
                        and target.get("kind", "text") == "text"
                        and target["gaptree_bbox"][0]
                        > source["gaptree_bbox"][2]
                        and target["gaptree_bbox"][0]
                        - source["gaptree_bbox"][2]
                        <= local_pair_gap_threshold
                        and (
                            max(
                                0.0,
                                min(
                                    source["gaptree_bbox"][3],
                                    target["gaptree_bbox"][3],
                                )
                                - max(
                                    source["gaptree_bbox"][1],
                                    target["gaptree_bbox"][1],
                                ),
                            )
                            / max(
                                min(
                                    source["gaptree_bbox"][3]
                                    - source["gaptree_bbox"][1],
                                    target["gaptree_bbox"][3]
                                    - target["gaptree_bbox"][1],
                                ),
                                1e-9,
                            )
                            >= 0.50
                        )
                    )
                    for source in cluster
                    for target in group
                )
                if not has_local_same_row_pair:
                    continue

            cluster_bbox = fitz.Rect(cluster[0]["gaptree_bbox"])
            for element in cluster[1:]:
                cluster_bbox |= fitz.Rect(element["gaptree_bbox"])
            domain_records.append(
                {
                    "node_index": node_index,
                    "elements": cluster,
                    "bbox": cluster_bbox,
                    "x_alignment_tolerance": (
                        narrow_x_alignment_tolerance
                    ),
                }
            )

    if not domain_records:
        return copied_groups, {
            "enabled": False,
            "reason": "no_repeated_narrow_text_domain",
            "merged_node_count": 0,
            "moved_element_count": 0,
            "median_text_height": round(median_text_height, 3),
            "narrow_domain_threshold": round(
                narrow_domain_threshold,
                3,
            ),
            #@GPT 即使本页没有形成重复窄簇，也记录元素级X对齐容差供审计排查。
            "narrow_x_alignment_tolerance": round(
                narrow_x_alignment_tolerance,
                3,
            ),
            "row_attachment_gap_threshold": round(
                row_attachment_gap_threshold,
                3,
            ),
            "local_pair_gap_threshold": round(
                local_pair_gap_threshold,
                3,
            ),
            "merges": [],
        }

    #@GPT 候选目标使用原子文本元素而不是目标节点bbox，避免一个单行标题代表整列的垂直范围。
    target_elements = [
        {
            "node_index": node_index,
            "element": element,
        }
        for node_index, group in enumerate(copied_groups)
        for element in group
        if element.get("kind", "text") == "text"
    ]
    #@GPT 右挂以单个文本元素为单位，不整体搬运源节点。
    moved_element_ids: set[int] = set()
    claimed_target_ids: set[int] = set()
    moved_by_target_node: dict[int, list[dict]] = defaultdict(list)
    merge_metadata: list[dict] = []

    for source_domain in domain_records:
        source_node_index = source_domain["node_index"]
        for source_element in source_domain["elements"]:
            source_rect = fitz.Rect(source_element["gaptree_bbox"])
            candidates: list[dict] = []
            for target in target_elements:
                target_element = target["element"]
                target_node_index = target["node_index"]
                #@GPT 窄列与说明可能已被某些后端放入同一节点；仍需建立同行配对和行标记，此时搬运是同节点内的稳定重排。
                if id(target_element) in claimed_target_ids:
                    continue

                target_rect = fitz.Rect(target_element["gaptree_bbox"])
                horizontal_gap = target_rect.x0 - source_rect.x1
                if (
                    horizontal_gap <= 0
                    or horizontal_gap > row_attachment_gap_threshold
                ):
                    continue

                vertical_overlap = max(
                    0.0,
                    min(source_rect.y1, target_rect.y1)
                    - max(source_rect.y0, target_rect.y0),
                )
                minimum_height = min(
                    source_rect.height,
                    target_rect.height,
                )
                overlap_ratio = (
                    vertical_overlap / minimum_height
                    if minimum_height > 0
                    else 0.0
                )
                if overlap_ratio < 0.50:
                    continue

                #@GPT 同一行若存在更近的中间文本元素，不跨过它连接到更远栏位。
                intermediate = [
                    other
                    for other in target_elements
                    if id(other["element"]) not in {
                        id(source_element),
                        id(target_element),
                    }
                    and other["node_index"] != source_node_index
                    #@GPT 只有元素完整落在源列与目标列之间才算真正的中间栏位；
                    #@GPT 目标栏内上一行文字的x0通常等于target_rect.x0，不应因0.05pt浮点重叠阻断当前行配对。
                    and (
                        other["element"]["gaptree_bbox"][0]
                        >= source_rect.x1
                        and other["element"]["gaptree_bbox"][2]
                        < target_rect.x0
                    )
                    and max(
                        0.0,
                        min(
                            source_rect.y1,
                            other["element"]["gaptree_bbox"][3],
                        )
                        - max(
                            source_rect.y0,
                            other["element"]["gaptree_bbox"][1],
                        ),
                    )
                    > 0
                ]
                if intermediate:
                    continue

                candidates.append(
                    {
                        "source_element": source_element,
                        "source_node_index": source_node_index,
                        "target_element": target_element,
                        "target_node_index": target_node_index,
                        "horizontal_gap": horizontal_gap,
                        "vertical_overlap_ratio": overlap_ratio,
                    }
                )

            if not candidates:
                continue

            accepted = min(
                candidates,
                key=lambda item: (
                    item["horizontal_gap"],
                    -item["vertical_overlap_ratio"],
                    item["target_element"]["gaptree_bbox"][0],
                ),
            )
            source_element = accepted["source_element"]
            target_element = accepted["target_element"]
            moved_element_ids.add(id(source_element))
            claimed_target_ids.add(id(target_element))
            moved_by_target_node[accepted["target_node_index"]].append(
                source_element
            )
            merge_metadata.append(
                {
                    "source_node_index": accepted["source_node_index"] + 1,
                    "target_node_index": accepted["target_node_index"] + 1,
                    "source_text": source_element.get("text", ""),
                    "target_text": target_element.get("text", ""),
                    "source_bbox": [
                        round(value, 3)
                        for value in source_rect
                    ],
                    "target_bbox": [
                        round(value, 3)
                        for value in fitz.Rect(
                            target_element["gaptree_bbox"]
                        )
                    ],
                    "horizontal_gap": round(
                        accepted["horizontal_gap"],
                        3,
                    ),
                    "vertical_overlap_ratio": round(
                        accepted["vertical_overlap_ratio"],
                        3,
                    ),
                }
            )

    if not merge_metadata:
        return copied_groups, {
            "enabled": False,
            "reason": "no_same_row_narrow_domain_pair",
            "merged_node_count": 0,
            "moved_element_count": 0,
            "median_text_height": round(median_text_height, 3),
            "narrow_domain_threshold": round(
                narrow_domain_threshold,
                3,
            ),
            #@GPT 候选存在但没有同行目标时仍输出X聚簇容差，避免调试记录缺字段。
            "narrow_x_alignment_tolerance": round(
                narrow_x_alignment_tolerance,
                3,
            ),
            "row_attachment_gap_threshold": round(
                row_attachment_gap_threshold,
                3,
            ),
            "local_pair_gap_threshold": round(
                local_pair_gap_threshold,
                3,
            ),
            "merges": [],
        }

    corrected_groups: list[list[dict]] = []
    target_node_indices = set(moved_by_target_node)
    for node_index, group in enumerate(copied_groups):
        remaining_group = [
            element
            for element in group
            if id(element) not in moved_element_ids
        ]
        if node_index in moved_by_target_node:
            remaining_group.extend(moved_by_target_node[node_index])
            #@GPT 同行合并后按文字中心Y和行容差分桶，再在桶内按X排序，避免微小基线误差颠倒编号和说明。
            ordered_group = sorted(
                remaining_group,
                key=lambda element: (
                    (
                        element["gaptree_bbox"][1]
                        + element["gaptree_bbox"][3]
                    )
                    / 2.0,
                    element["gaptree_bbox"][0],
                ),
            )
            row_buckets: list[list[dict]] = []
            row_centers: list[float] = []
            for element in ordered_group:
                center_y = (
                    element["gaptree_bbox"][1]
                    + element["gaptree_bbox"][3]
                ) / 2.0
                if (
                    row_buckets
                    and abs(center_y - row_centers[-1])
                    <= row_tolerance_points
                ):
                    row_buckets[-1].append(element)
                    row_centers[-1] = (
                        row_centers[-1] + center_y
                    ) / 2.0
                else:
                    row_buckets.append([element])
                    row_centers.append(center_y)
            remaining_group = [
                element
                for bucket in row_buckets
                for element in sorted(
                    bucket,
                    key=lambda item: item["gaptree_bbox"][0],
                )
            ]
            for element in remaining_group:
                element["narrow_domain_row_group"] = True

        if remaining_group:
            corrected_groups.append(remaining_group)

    return corrected_groups, {
        "enabled": True,
        "reason": "same_row_narrow_domain_attached",
        "merged_node_count": len(target_node_indices),
        "moved_element_count": len(moved_element_ids),
        "median_text_height": round(median_text_height, 3),
        "narrow_domain_threshold": round(
            narrow_domain_threshold,
            3,
        ),
    #@GPT 审计输出元素级X聚簇容差，便于解释窄列是否形成稳定对齐。
        "narrow_x_alignment_tolerance": round(
            narrow_x_alignment_tolerance,
            3,
        ),
        "row_attachment_gap_threshold": round(
            row_attachment_gap_threshold,
            3,
        ),
        "local_pair_gap_threshold": round(
            local_pair_gap_threshold,
            3,
        ),
        "merges": merge_metadata,
    }

import math
import numpy as np

def coalesce_horizontal_text_fragments(
    elements: list[dict],
) -> tuple[list[dict], dict]:
    """仅在同行间距形成稳定双峰时合并短间距文本碎片。"""
    #@GPT 碎片配对使用页面中位文字高度推导的中线带，允许同一基线的小字体差异但不会吞入下一行。
    text_heights = [
        item["gaptree_bbox"][3] - item["gaptree_bbox"][1]
        for item in elements
        if item.get("kind", "text") == "text"
        and item["gaptree_bbox"][3] > item["gaptree_bbox"][1]
    ]
    median_text_height = (
        float(np.median(text_heights))
        if text_heights
        else 1.0
    )
    center_tolerance = median_text_height * 0.45
    ordered_elements = sorted(
        elements,
        key=lambda item: (
            (
                item["gaptree_bbox"][1]
                + item["gaptree_bbox"][3]
            )
            / 2,
            item["gaptree_bbox"][0],
        ),
    )
    rows: list[list[dict]] = []
    row_centers: list[float] = []

    for element in ordered_elements:
        element_center = (
            element["gaptree_bbox"][1]
            + element["gaptree_bbox"][3]
        ) / 2

        if (
            not rows
            or abs(element_center - row_centers[-1])
            > center_tolerance
        ):
            rows.append([element])
            row_centers.append(element_center)
            continue

        rows[-1].append(element)
        row_centers[-1] = sum(
            (
                item["gaptree_bbox"][1]
                + item["gaptree_bbox"][3]
            )
            / 2
            for item in rows[-1]
        ) / len(rows[-1])

    for row in rows:
        row.sort(key=lambda item: item["gaptree_bbox"][0])

    candidate_gaps: list[float] = []

    for row in rows:
        text_items = [
            item
            for item in row
            if item.get("kind", "text") == "text"
        ]

        #@GPT 至少四个同行文本框才可能表达“多个记录各含多个字段”；双栏普通正文不参与阈值估计。
        if len(text_items) < 4:
            continue

        text_items.sort(key=lambda item: item["gaptree_bbox"][0])

        for left_item, right_item in zip(
            text_items,
            text_items[1:],
        ):
            #@GPT 使用相邻左边界间距而非框尾空白，避免字段内容宽度变化扰动列间距统计。
            anchor_delta = (
                right_item["gaptree_bbox"][0]
                - left_item["gaptree_bbox"][0]
            )

            #@GPT 改用相邻左边界间距，利用PDF字段起点的重复结构，不受产品号或说明文字长度影响。
            if anchor_delta > 0:
                candidate_gaps.append(anchor_delta)

    #@GPT 样本过少时不推断双峰，保持原始文本块，避免在普通页面上过度合并。
    if len(candidate_gaps) < 20:
        return elements, {
            "enabled": False,
            "reason": "insufficient_repeated_row_gaps",
            "candidate_gap_count": len(candidate_gaps),
            "merged_fragment_count": 0,
        }

    sorted_gaps = sorted(candidate_gaps)
    trimmed_count = max(2, int(len(sorted_gaps) * 0.95))
    #@GPT 使用截尾后的稳定跃迁检测双峰，避免少量超大空白吸引聚类中心。

    trimmed_gaps = sorted_gaps[:trimmed_count]
    minimum_cluster_size = max(
        5,
        int(len(trimmed_gaps) * 0.15),
    )
    split_candidates: list[tuple[float, int]] = []

    #@GPT 只在两侧都至少占15%的平衡位置寻找最大乘法跃迁，忽略少数极端大空白并保留重复结构。
    for split_index in range(
        minimum_cluster_size,
        len(trimmed_gaps) - minimum_cluster_size + 1,
    ):
        lower_gap = trimmed_gaps[split_index - 1]
        upper_gap = trimmed_gaps[split_index]

        if lower_gap <= 0:
            continue

        split_candidates.append(
            (upper_gap / lower_gap, split_index)
        )

    gap_ratio, split_index = max(
        split_candidates,
        default=(1.0, 0),
    )

    #@GPT 最大稳定跃迁不足1.8倍时视为单峰间距，不做碎片合并。
    if gap_ratio < 1.8 or split_index == 0:
        return elements, {
            "enabled": False,
            "reason": "no_stable_bimodal_gap_distribution",
            "candidate_gap_count": len(candidate_gaps),
            "gap_ratio": round(gap_ratio, 3),
            "merged_fragment_count": 0,
        }

    low_cluster = trimmed_gaps[:split_index]
    high_cluster = trimmed_gaps[split_index:]
    low_gap_center = math.exp(
        sum(math.log(gap) for gap in low_cluster)
        / len(low_cluster)
    )
    high_gap_center = math.exp(
        sum(math.log(gap) for gap in high_cluster)
        / len(high_cluster)
    )
    center_ratio = high_gap_center / low_gap_center
    merge_threshold = math.sqrt(
        low_cluster[-1] * high_cluster[0]
    )
    coalesced_elements: list[dict] = []
    merged_fragment_count = 0

    for row in rows:
        row_output: list[dict] = []

        for element in row:
            if (
                row_output
                and element.get("kind", "text") == "text"
                and row_output[-1].get("kind", "text") == "text"
            ):
                previous = row_output[-1]
                #@GPT 使用与阈值估计一致的左边界间距，并保留最后碎片起点以支持多字段链式合并。
                previous_anchor = previous.get(
                    "last_fragment_virtual_x0",
                    previous["gaptree_bbox"][0],
                )
                anchor_delta = (
                    element["gaptree_bbox"][0]
                    - previous_anchor
                )

                if 0 < anchor_delta <= merge_threshold:
                    original_bbox = fitz.Rect(previous["bbox"])
                    original_bbox |= element["bbox"]
                    virtual_bbox = (
                        min(
                            previous["gaptree_bbox"][0],
                            element["gaptree_bbox"][0],
                        ),
                        min(
                            previous["gaptree_bbox"][1],
                            element["gaptree_bbox"][1],
                        ),
                        max(
                            previous["gaptree_bbox"][2],
                            element["gaptree_bbox"][2],
                        ),
                        max(
                            previous["gaptree_bbox"][3],
                            element["gaptree_bbox"][3],
                        ),
                    )
                    previous["bbox"] = original_bbox
                    previous["gaptree_bbox"] = virtual_bbox
                    previous["text"] = (
                        f"{previous['text']} {element['text']}"
                    ).strip()
                    previous["source_fragment_count"] = (
                        previous.get("source_fragment_count", 1)
                        + element.get("source_fragment_count", 1)
                    )
                    previous["last_fragment_virtual_x0"] = (
                        element["gaptree_bbox"][0]
                    )
                    merged_fragment_count += 1
                    continue

            row_output.append({**element})

        coalesced_elements.extend(row_output)

    return coalesced_elements, {
        "enabled": True,
        "reason": "stable_bimodal_gap_distribution",
        "measurement": "adjacent_left_anchor_delta",
        "candidate_gap_count": len(candidate_gaps),
        "low_gap_center": round(low_gap_center, 3),
        "high_gap_center": round(high_gap_center, 3),
        "gap_ratio": round(gap_ratio, 3),
        "center_ratio": round(center_ratio, 3),
        "merge_threshold": round(merge_threshold, 3),
        "merged_fragment_count": merged_fragment_count,
    }

#@GPT 跨列首行校正消费GapTree节点输出，只依据列锚点、页面几何和节点顺序。
#@GPT 规则保留在排序模块能让后续替换GapTree时复用同一后处理，同时避免复制核心全局策略。
def reparent_cross_column_leading_row(
    node_groups: list[list[dict]],
    page_rect: fitz.Rect,
) -> tuple[list[list[dict]], dict]:
    """把被标题节点提前接走的多列首行归回各自稳定正文列。"""
    copied_groups = [list(group) for group in node_groups]
    text_elements = [
        element
        for group in copied_groups
        for element in group
        if element.get("kind", "text") == "text"
    ]
    if not text_elements:
        return copied_groups, {
            "enabled": False,
            "reason": "no_text_elements",
            "stable_column_count": 0,
            "moved_element_count": 0,
            "moves": [],
        }

    text_heights = [
        element["gaptree_bbox"][3]
        - element["gaptree_bbox"][1]
        for element in text_elements
        if element["gaptree_bbox"][3]
        > element["gaptree_bbox"][1]
    ]
    median_text_height = (
        float(np.median(text_heights))
        if text_heights
        else max(page_rect.height * 0.01, 1.0)
    )
    anchor_tolerance = max(
        page_rect.width * 0.008,
        median_text_height * 0.75,
    )
    minimum_node_support = max(
        4,
        int(math.ceil(len(text_elements) * 0.08)),
    )
    minimum_vertical_span = max(
        median_text_height * 8.0,
        page_rect.height * 0.20,
    )

    stable_columns: list[dict] = []
    for node_index, group in enumerate(copied_groups):
        node_text = [
            element
            for element in group
            if element.get("kind", "text") == "text"
        ]
        if len(node_text) < minimum_node_support:
            continue

        vertical_span = (
            max(element["gaptree_bbox"][3] for element in node_text)
            - min(element["gaptree_bbox"][1] for element in node_text)
        )
        if vertical_span < minimum_vertical_span:
            continue

        anchor_x = float(
            np.median(
                [
                    element["gaptree_bbox"][0]
                    for element in node_text
                ]
            )
        )
        anchor_members = [
            element
            for element in node_text
            if abs(element["gaptree_bbox"][0] - anchor_x)
            <= anchor_tolerance
        ]
        if len(anchor_members) < minimum_node_support:
            continue

        stable_columns.append(
            {
                "node_index": node_index,
                "anchor_x": anchor_x,
                "top": min(
                    element["gaptree_bbox"][1]
                    for element in anchor_members
                ),
                "support": len(anchor_members),
                "vertical_span": vertical_span,
            }
        )

    #@GPT 同一X位置若出现多个节点，只保留重复元素更多者；随后要求列锚点横向明显分离。
    distinct_columns: list[dict] = []
    for column in sorted(
        stable_columns,
        key=lambda item: item["anchor_x"],
    ):
        if (
            distinct_columns
            and abs(
                column["anchor_x"]
                - distinct_columns[-1]["anchor_x"]
            )
            <= anchor_tolerance
        ):
            if column["support"] > distinct_columns[-1]["support"]:
                distinct_columns[-1] = column
            continue
        distinct_columns.append(column)

    separated_columns: list[dict] = []
    for column in distinct_columns:
        if (
            not separated_columns
            or column["anchor_x"]
            - separated_columns[-1]["anchor_x"]
            >= page_rect.width * 0.12
        ):
            separated_columns.append(column)

    if len(separated_columns) < 2:
        return copied_groups, {
            "enabled": False,
            "reason": "fewer_than_two_stable_columns",
            "stable_column_count": len(separated_columns),
            "moved_element_count": 0,
            "moves": [],
        }

    leading_window = max(
        median_text_height * 2.5,
        page_rect.height * 0.04,
    )
    candidates: list[dict] = []
    for source_node_index, group in enumerate(copied_groups):
        for element in group:
            if element.get("kind", "text") != "text":
                continue
            element_x0 = element["gaptree_bbox"][0]
            element_y0 = element["gaptree_bbox"][1]
            targets = [
                column
                for column in separated_columns
                if source_node_index < column["node_index"]
                and abs(element_x0 - column["anchor_x"])
                <= anchor_tolerance
                and 0 < column["top"] - element_y0
                <= leading_window
            ]
            if not targets:
                continue
            target = min(
                targets,
                key=lambda item: (
                    item["top"] - element_y0,
                    abs(item["anchor_x"] - element_x0),
                ),
            )
            candidates.append(
                {
                    "element": element,
                    "source_node_index": source_node_index,
                    "target_node_index": target["node_index"],
                    "center_y": (
                        element["gaptree_bbox"][1]
                        + element["gaptree_bbox"][3]
                    )
                    / 2.0,
                }
            )

    #@GPT 单个标题或缩进不能触发移动；同一视觉行必须同时命中至少两条不同稳定列。
    candidate_rows: list[list[dict]] = []
    row_tolerance = median_text_height * 0.65
    for candidate in sorted(
        candidates,
        key=lambda item: item["center_y"],
    ):
        if (
            not candidate_rows
            or abs(
                candidate["center_y"]
                - float(
                    np.median(
                        [
                            item["center_y"]
                            for item in candidate_rows[-1]
                        ]
                    )
                )
            )
            > row_tolerance
        ):
            candidate_rows.append([candidate])
        else:
            candidate_rows[-1].append(candidate)

    accepted = [
        candidate
        for row in candidate_rows
        if len(
            {
                item["target_node_index"]
                for item in row
            }
        )
        >= 2
        for candidate in row
    ]
    if not accepted:
        return copied_groups, {
            "enabled": False,
            "reason": "no_cross_column_leading_row",
            "stable_column_count": len(separated_columns),
            "stable_column_anchors": [
                round(column["anchor_x"], 3)
                for column in separated_columns
            ],
            "moved_element_count": 0,
            "moves": [],
        }

    moved_ids = {
        id(candidate["element"])
        for candidate in accepted
    }
    moved_by_target: dict[int, list[dict]] = defaultdict(list)
    for candidate in accepted:
        moved_by_target[candidate["target_node_index"]].append(
            candidate["element"]
        )

    corrected_groups: list[list[dict]] = []
    for node_index, group in enumerate(copied_groups):
        corrected_group = [
            element
            for element in group
            if id(element) not in moved_ids
        ]
        corrected_group.extend(moved_by_target.get(node_index, []))
        corrected_group.sort(
            key=lambda element: (
                element["gaptree_bbox"][1],
                element["gaptree_bbox"][0],
            )
        )
        if corrected_group:
            corrected_groups.append(corrected_group)

    return corrected_groups, {
        "enabled": True,
        "reason": "cross_column_leading_row_reparented",
        "stable_column_count": len(separated_columns),
        "stable_column_anchors": [
            round(column["anchor_x"], 3)
            for column in separated_columns
        ],
        "anchor_tolerance": round(anchor_tolerance, 3),
        "leading_window": round(leading_window, 3),
        "moved_element_count": len(accepted),
        "moves": [
            {
                "text": candidate["element"].get("text", "")[:80],
                "source_node_index": (
                    candidate["source_node_index"] + 1
                ),
                "target_node_index": (
                    candidate["target_node_index"] + 1
                ),
                "x0": round(
                    candidate["element"]["gaptree_bbox"][0],
                    3,
                ),
            }
            for candidate in accepted
        ],
    }

#@GPT 二级XY-Cut在GapTree节点内部按水平空白分区，页面角度和行容差由调用方显式传入。
def _horizontal_partition_layout_bbox(
    element: dict,
    source_angle: int,
    page_rect: fitz.Rect,
) -> fitz.Rect:
    """返回用于节点内部水平分区的完整实际区域，而不是图片的sort_bbox。"""
    layout_value = element.get("render_bbox")
    if layout_value is None:
        layout_value = element.get("bbox")
    if layout_value is None:
        layout_value = element.get("gaptree_bbox")
    return fitz.Rect(
        transformed_reading_bbox(
            rect=fitz.Rect(layout_value),
            source_angle=source_angle,
            page_rect=page_rect,
        )
    )


#@GPT 节点内水平断层的最小尺度来自当前页文本高度；不新增固定页码规则，也不把普通行间距误当成分区。
def _horizontal_partition_gap_threshold(
    node_groups: list[list[dict]],
    page_rect: fitz.Rect,
    *,
    row_tolerance_points: float,
) -> tuple[float, float]:
    text_heights = [
        fitz.Rect(element.get("bbox", element.get("gaptree_bbox"))).height
        for group in node_groups
        for element in group
        if element.get("kind", "text") == "text"
        and fitz.Rect(element.get("bbox", element.get("gaptree_bbox"))).height > 0
    ]
    median_text_height = (
        float(np.median(text_heights))
        if text_heights
        else max(page_rect.height * 0.01, 1.0)
    )
    #@GPT 以约半个文字高度作为水平断层下限，并保留现有同行容差的两倍，避免拆开普通段落行距。
    gap_threshold = max(
        row_tolerance_points * 2.0,
        median_text_height * 0.45,
    )
    return median_text_height, gap_threshold


#@GPT 先拆分GapTree节点，再依据区块之间的垂直关系和同层左右关系排序；
#@GPT 这是轻量二级XY-Cut，不重写GapTree，也不引入“产品号”等语义特判。
def split_gap_tree_nodes_by_horizontal_gaps(
    node_groups: list[list[dict]],
    source_angle: int,
    page_rect: fitz.Rect,
    *,
    row_tolerance_points: float,
) -> tuple[list[list[dict]], dict]:
    """在GapTree节点内部寻找显著水平空白带并重排所得区块。"""
    if not node_groups:
        return [], {
            "enabled": False,
            "reason": "no_gap_tree_nodes",
            "original_node_count": 0,
            "final_fragment_count": 0,
            "split_records": [],
            "fragment_order": [],
            "cycle_detected": False,
        }

    median_text_height, gap_threshold = _horizontal_partition_gap_threshold(
        node_groups,
        page_rect,
        row_tolerance_points=row_tolerance_points,
    )
    fragment_records: list[dict] = []
    split_records: list[dict] = []

    for original_node_index, group in enumerate(node_groups, start=1):
        if len(group) < 2:
            fragment_records.append(
                {
                    "original_node_index": original_node_index,
                    "fragment_index": 1,
                    "group": list(group),
                    "bbox": (
                        _horizontal_partition_layout_bbox(
                            group[0], source_angle, page_rect
                        )
                        if group
                        else fitz.Rect()
                    ),
                }
            )
            continue

        layout_items = [
            {
                "element": element,
                "bbox": _horizontal_partition_layout_bbox(
                    element,
                    source_angle,
                    page_rect,
                ),
                "original_position": position,
            }
            for position, element in enumerate(group)
        ]
        ordered_items = sorted(
            layout_items,
            key=lambda item: (
                item["bbox"].y0,
                item["bbox"].y1,
                item["bbox"].x0,
                item["original_position"],
            ),
        )

        components: list[list[dict]] = []
        component_gaps: list[float] = []
        current_component: list[dict] = []
        current_end: float | None = None
        for item in ordered_items:
            item_bbox = item["bbox"]
            if current_component and current_end is not None:
                gap = item_bbox.y0 - current_end
                if gap >= gap_threshold:
                    components.append(current_component)
                    component_gaps.append(gap)
                    current_component = []
            current_component.append(item)
            current_end = (
                item_bbox.y1
                if current_end is None
                else max(current_end, item_bbox.y1)
            )
        if current_component:
            components.append(current_component)

        #@GPT 仅当上方含视觉原子且下方是短纯文本带时接受切分，避免把同一视觉簇内部的多个图片拆散。
        accepted_boundary_indices: list[int] = []
        for boundary_index, gap in enumerate(component_gaps):
            if gap < gap_threshold:
                continue
            upper_component = components[boundary_index]
            lower_component = components[boundary_index + 1]
            upper_has_non_text = any(
                item["element"].get("kind", "text") != "text"
                for item in upper_component
            )
            lower_is_text_only = all(
                item["element"].get("kind", "text") == "text"
                for item in lower_component
            )
            if upper_has_non_text and lower_is_text_only and len(lower_component) <= 3:
                accepted_boundary_indices.append(boundary_index)

        if not accepted_boundary_indices:
            fragment_records.append(
                {
                    "original_node_index": original_node_index,
                    "fragment_index": 1,
                    "group": list(group),
                    "bbox": fitz.Rect(
                        min(item["bbox"].x0 for item in layout_items),
                        min(item["bbox"].y0 for item in layout_items),
                        max(item["bbox"].x1 for item in layout_items),
                        max(item["bbox"].y1 for item in layout_items),
                    ),
                }
            )
            continue

        #@GPT 将未被接受的内部候选间隙重新并回同一片段，避免上方多个图片因各自bbox间距较大而被拆散。
        fragment_component_ranges: list[tuple[int, int]] = []
        range_start = 0
        for boundary_index in accepted_boundary_indices:
            fragment_component_ranges.append((range_start, boundary_index + 1))
            range_start = boundary_index + 1
        fragment_component_ranges.append((range_start, len(components)))

        component_element_ids = [
            {
                id(item["element"])
                for component_index in range(start, end)
                for item in components[component_index]
            }
            for start, end in fragment_component_ranges
        ]
        node_split_fragments: list[dict] = []
        for fragment_index, ids in enumerate(component_element_ids, start=1):
            fragment_group = [
                element
                for element in group
                if id(element) in ids
            ]
            fragment_items = [
                item
                for item in layout_items
                if id(item["element"]) in ids
            ]
            fragment_bbox = fitz.Rect(
                min(item["bbox"].x0 for item in fragment_items),
                min(item["bbox"].y0 for item in fragment_items),
                max(item["bbox"].x1 for item in fragment_items),
                max(item["bbox"].y1 for item in fragment_items),
            )
            for element in fragment_group:
                #@GPT 区块来源只用于审计和循环时的稳定排序，不参与内容语义判断。
                element["horizontal_partition_original_node"] = (
                    original_node_index
                )
                element["horizontal_partition_fragment"] = fragment_index
            node_split_fragments.append(
                {
                    "original_node_index": original_node_index,
                    "fragment_index": fragment_index,
                    "group": fragment_group,
                    "bbox": fragment_bbox,
                }
            )

        split_records.append(
            {
                "original_node_index": original_node_index,
                "fragment_count": len(node_split_fragments),
                "gap_threshold": round(gap_threshold, 3),
                "candidate_gaps": [
                    round(value, 3) for value in component_gaps
                ],
                "accepted_boundary_indices": accepted_boundary_indices,
                "accepted_gaps": [
                    round(component_gaps[index], 3)
                    for index in accepted_boundary_indices
                ],
                "fragments": [
                    {
                        "fragment_index": record["fragment_index"],
                        "bbox": [
                            round(value, 3)
                            for value in record["bbox"]
                        ],
                        "element_count": len(record["group"]),
                        "kinds": dict(
                            Counter(
                                element.get("kind", "text")
                                for element in record["group"]
                            )
                        ),
                    }
                    for record in node_split_fragments
                ],
            }
        )
        fragment_records.extend(node_split_fragments)

    if len(fragment_records) <= 1:
        return [record["group"] for record in fragment_records], {
            "enabled": bool(split_records),
            "reason": "single_fragment_after_partition",
            "original_node_count": len(node_groups),
            "final_fragment_count": len(fragment_records),
            "median_text_height": round(median_text_height, 3),
            "gap_threshold": round(gap_threshold, 3),
            "split_records": split_records,
            "fragment_order": [
                {
                    "original_node_index": record["original_node_index"],
                    "fragment_index": record["fragment_index"],
                }
                for record in fragment_records
            ],
            "cycle_detected": False,
        }

    #@GPT 对区块建立一个小型空间DAG：垂直明显分离时按上下，处于同一阅读带时按左右；
    #@GPT 对“薄的下方区块”优先放到高大区块之后，解决产品号落在第二张表前的问题。
    def is_thin_lower_fragment(
        lower: dict,
        upper: dict,
    ) -> bool:
        lower_bbox = lower["bbox"]
        upper_bbox = upper["bbox"]
        return (
            lower_bbox.height <= max(median_text_height * 2.5, 1.0)
            and lower_bbox.y0 >= upper_bbox.y0
            and lower_bbox.y0
            >= upper_bbox.y0 + median_text_height * 0.75
            and lower_bbox.y0
            >= upper_bbox.y1 - median_text_height * 1.5
            and lower_bbox.y1 > upper_bbox.y0
        )

    def relation_before(left: dict, right: dict) -> bool:
        left_bbox = left["bbox"]
        right_bbox = right["bbox"]
        if is_thin_lower_fragment(left, right):
            return False
        if is_thin_lower_fragment(right, left):
            return True

        vertical_overlap = max(
            0.0,
            min(left_bbox.y1, right_bbox.y1)
            - max(left_bbox.y0, right_bbox.y0),
        )
        minimum_height = min(left_bbox.height, right_bbox.height)
        overlap_ratio = (
            vertical_overlap / minimum_height
            if minimum_height > 0
            else 0.0
        )
        if overlap_ratio < 0.25:
            if left_bbox.y1 <= right_bbox.y0 + row_tolerance_points:
                return True
            if right_bbox.y1 <= left_bbox.y0 + row_tolerance_points:
                return False

        left_center_y = (left_bbox.y0 + left_bbox.y1) / 2.0
        right_center_y = (right_bbox.y0 + right_bbox.y1) / 2.0
        if overlap_ratio < 0.25 and abs(left_center_y - right_center_y) > median_text_height:
            return left_center_y < right_center_y
        if left_bbox.x0 != right_bbox.x0:
            return left_bbox.x0 < right_bbox.x0
        return left["original_node_index"] < right["original_node_index"]

    edge_sets: list[set[int]] = [set() for _ in fragment_records]
    indegrees = [0 for _ in fragment_records]
    for left_index in range(len(fragment_records)):
        for right_index in range(left_index + 1, len(fragment_records)):
            left = fragment_records[left_index]
            right = fragment_records[right_index]
            if relation_before(left, right):
                before_index, after_index = left_index, right_index
            else:
                before_index, after_index = right_index, left_index
            if after_index not in edge_sets[before_index]:
                edge_sets[before_index].add(after_index)
                indegrees[after_index] += 1

    available = [
        index
        for index, indegree in enumerate(indegrees)
        if indegree == 0
    ]
    ordered_fragment_indices: list[int] = []
    while available:
        available.sort(
            key=lambda index: (
                fragment_records[index]["bbox"].y0,
                fragment_records[index]["bbox"].x0,
                fragment_records[index]["original_node_index"],
                fragment_records[index]["fragment_index"],
            )
        )
        current = available.pop(0)
        ordered_fragment_indices.append(current)
        for successor in edge_sets[current]:
            indegrees[successor] -= 1
            if indegrees[successor] == 0:
                available.append(successor)

    cycle_detected = len(ordered_fragment_indices) != len(fragment_records)
    if cycle_detected:
        #@GPT 若复杂交错布局形成环，则采用稳定的原GapTree节点顺序，避免强行破坏已有局部关系。
        ordered_fragment_indices = sorted(
            range(len(fragment_records)),
            key=lambda index: (
                fragment_records[index]["original_node_index"],
                fragment_records[index]["fragment_index"],
            ),
        )

    ordered_records = [
        fragment_records[index]
        for index in ordered_fragment_indices
    ]
    return [record["group"] for record in ordered_records], {
        "enabled": bool(split_records),
        "reason": "horizontal_gaps_partitioned_and_reordered",
        "original_node_count": len(node_groups),
        "final_fragment_count": len(ordered_records),
        "median_text_height": round(median_text_height, 3),
        "gap_threshold": round(gap_threshold, 3),
        "split_records": split_records,
        "fragment_order": [
            {
                "original_node_index": record["original_node_index"],
                "fragment_index": record["fragment_index"],
                "bbox": [round(value, 3) for value in record["bbox"]],
            }
            for record in ordered_records
        ],
        "cycle_detected": cycle_detected,
    }

#@GPT separator→heading关系由检测阶段声明；此函数只重排已关联元素，不重新判断标题语义。
#@GPT 目标选择以几何包含为硬条件，文本匹配仅用于同bbox候选的稳定排序。
def apply_attached_separator_heading_precedence(
    node_groups: list[list[dict]],
) -> tuple[list[list[dict]], dict]:
    """把端点附着的Markdown分隔符移动到其标题文字之前。"""
    copied_groups = [list(group) for group in node_groups]
    text_locations: list[dict] = []
    separator_locations: list[dict] = []

    for node_index, group in enumerate(copied_groups):
        for element_index, element in enumerate(group):
            if element.get("kind", "text") == "text":
                text_locations.append(
                    {
                        "node_index": node_index,
                        "element_index": element_index,
                        "element": element,
                    }
                )
            elif (
                element.get("kind") == "text_separator"
                and element.get("sort_before_heading", False)
                and element.get("attached_heading_bbox")
            ):
                separator_locations.append(
                    {
                        "node_index": node_index,
                        "element_index": element_index,
                        "element": element,
                    }
                )

    if not separator_locations:
        return copied_groups, {
            "enabled": False,
            "reason": "no_attached_text_separator",
            "attachment_count": 0,
            "unresolved_count": 0,
            "attachments": [],
        }

    assignments_by_target: dict[int, list[dict]] = defaultdict(list)
    resolved_separator_ids: set[int] = set()
    attachment_records: list[dict] = []
    unresolved_count = 0
    bbox_tolerance = 1.0

    for separator_location in separator_locations:
        separator = separator_location["element"]
        attached_bbox = fitz.Rect(
            separator["attached_heading_bbox"]
        ).normalize()
        attached_text = separator.get(
            "attached_heading_text",
            "",
        )
        target_candidates: list[dict] = []

        for text_location in text_locations:
            text_element = text_location["element"]
            text_bbox = fitz.Rect(text_element["bbox"]).normalize()
            bbox_contains_attachment = (
                text_bbox.x0 - bbox_tolerance
                <= attached_bbox.x0
                and text_bbox.y0 - bbox_tolerance
                <= attached_bbox.y0
                and text_bbox.x1 + bbox_tolerance
                >= attached_bbox.x1
                and text_bbox.y1 + bbox_tolerance
                >= attached_bbox.y1
            )
            #@GPT 文字碎片合并后文本可能变化，因此文本只作为同bbox候选的排序证据，不作为淘汰条件。
            if not bbox_contains_attachment:
                continue

            text_matches_attachment = (
                not attached_text
                or attached_text == text_element.get("text", "")
                or attached_text
                in text_element.get("text", "")
            )

            target_candidates.append(
                {
                    **text_location,
                    "bbox_area_growth": max(
                        0.0,
                        rect_area(text_bbox)
                        - rect_area(attached_bbox),
                    ),
                    "text_length_growth": max(
                        0,
                        len(text_element.get("text", ""))
                        - len(attached_text),
                    ),
                    "text_match_penalty": (
                        0 if text_matches_attachment else 1
                    ),
                }
            )

        if not target_candidates:
            unresolved_count += 1
            continue

        target = min(
            target_candidates,
            key=lambda item: (
                item["text_match_penalty"],
                item["bbox_area_growth"],
                item["text_length_growth"],
                item["node_index"],
                item["element_index"],
            ),
        )
        assignments_by_target[id(target["element"])].append(
            separator
        )
        resolved_separator_ids.add(id(separator))
        attachment_records.append(
            {
                "separator_bbox": [
                    round(value, 3)
                    for value in fitz.Rect(separator["bbox"])
                ],
                "source_rule_bbox": separator.get(
                    "source_rule_bbox"
                ),
                "heading_bbox": separator[
                    "attached_heading_bbox"
                ],
                "heading_text": attached_text,
                "source_node_index": (
                    separator_location["node_index"] + 1
                ),
                "target_node_index": target["node_index"] + 1,
                "attachment_side": separator.get(
                    "attachment_side"
                ),
                "edge_delta_points": round(
                    separator.get("edge_delta_points", 0.0),
                    6,
                ),
                "vertical_overlap_ratio": round(
                    separator.get(
                        "vertical_overlap_ratio",
                        0.0,
                    ),
                    6,
                ),
            }
        )

    if not resolved_separator_ids:
        return copied_groups, {
            "enabled": False,
            "reason": "attached_heading_not_found_after_text_coalescing",
            "attachment_count": 0,
            "unresolved_count": unresolved_count,
            "attachments": [],
        }

    rebuilt_groups: list[list[dict]] = []
    for group in copied_groups:
        rebuilt_group: list[dict] = []
        for element in group:
            if id(element) in resolved_separator_ids:
                continue

            attached_separators = assignments_by_target.get(
                id(element),
                [],
            )
            #@GPT 同一填充长条可能被解析成上下两条边界；先按原始规则条和分隔线Y排序，格式器仍会把相邻`---`去重成一个Markdown分隔符。
            rebuilt_group.extend(
                sorted(
                    attached_separators,
                    key=lambda item: (
                        item.get(
                            "source_rule_bbox",
                            item["bbox"],
                        )[1],
                        fitz.Rect(item["bbox"]).y0,
                        fitz.Rect(item["bbox"]).x0,
                    ),
                )
            )
            rebuilt_group.append(element)
        rebuilt_groups.append(rebuilt_group)

    return rebuilt_groups, {
        "enabled": True,
        "reason": "explicit_attached_separator_before_heading",
        "attachment_count": len(attachment_records),
        "unresolved_count": unresolved_count,
        "attachments": attachment_records,
    }

#@GPT 视觉区域和原始图片内部的原生文字按虚拟转正坐标分行，再按X排序。
#@GPT 行容差由核心绑定后传入，保持visual_assets既有三参数回调接口不变，同时避免复制核心全局常量。
def sort_lines_in_reading_order(
    lines: list[dict],
    source_angle: int,
    page_rect: fitz.Rect,
    *,
    row_tolerance_points: float,
) -> list[dict]:
    """
    将同方向文本行按虚拟转正后的坐标排序。

    对 y 值在 row_tolerance_points 内的行归为同一阅读行，
    再按 virtual_x 排列。
    """
    positioned_lines: list[dict] = []

    for line in lines:
        virtual_y, virtual_x = transformed_reading_position(
            rect=line["bbox"],
            source_angle=source_angle,
            page_rect=page_rect,
        )

        positioned_lines.append(
            {
                **line,
                "virtual_y": virtual_y,
                "virtual_x": virtual_x,
            }
        )

    positioned_lines.sort(
        key=lambda item: (
            item["virtual_y"],
            item["virtual_x"],
        )
    )

    current_row_y: float | None = None
    row_index = 0

    for line in positioned_lines:
        if (
            current_row_y is None
            or abs(line["virtual_y"] - current_row_y)
            > row_tolerance_points
        ):
            row_index += 1
            current_row_y = line["virtual_y"]

        line["virtual_row_index"] = row_index

    positioned_lines.sort(
        key=lambda item: (
            item["virtual_row_index"],
            item["virtual_x"],
        )
    )

    return positioned_lines


def sort_elements_with_gap_tree(
    elements: list[dict],
    source_angle: int,
    page_rect: fitz.Rect,
) -> tuple[list[dict], dict]:
    """使用选定空间后端对同方向原子元素进行阅读顺序排序。"""
    if not elements:
        return [], {
            "backend": "gaptree",
            "row_count": 0,
            "cut_count": 0,
            "node_count": 0,
            "cuts": [],
            "nodes": [],
        }

    positioned_elements: list[dict] = []

    for element in elements:
        #@GPT 非文本元素优先使用顶部sort_bbox参与切分，避免高图片跨越多行；文字使用原始bbox。
        #@GPT 后端公共入口接受fitz.Rect或等价四元组；在坐标变换前统一规范化，保证两种后端输入契约一致。
        sorting_rect = fitz.Rect(
            element.get(
                "sort_bbox",
                element["bbox"],
            )
        ).normalize()
        virtual_bbox = transformed_reading_bbox(
            rect=sorting_rect,
            source_angle=source_angle,
            page_rect=page_rect,
        )
        virtual_y, virtual_x = transformed_reading_position(
            rect=sorting_rect,
            source_angle=source_angle,
            page_rect=page_rect,
        )
        positioned_elements.append(
            {
                **element,
                "gaptree_bbox": virtual_bbox,
                "virtual_y": virtual_y,
                "virtual_x": virtual_x,
            }
        )

    #@GPT 在调用GapTree前仅合并具有稳定短间距的同行文字碎片；表格、图片和视觉区域仍保持原子块。
    positioned_elements, fragment_merge_metadata = (
        coalesce_horizontal_text_fragments(
            elements=positioned_elements,
        )
    )

    sorter = create_reading_order_backend(
        backend_name="gaptree",
        implementation=GapTree,
        get_bbox=lambda item: item["gaptree_bbox"]
    )
    #@GPT 先保留GapTree节点结构完成列首校正和XY-Cut，再展平为最终原子顺序。
    sorter.sort(positioned_elements)

    raw_node_groups = sorter.get_nodes_text_blocks()
    #@GPT 窄列按同一Y行逐元素右挂，避免跨越多个目录行移动整列；未配对元素留在原GapTree节点。
    node_groups, column_anchor_correction = (
        attach_narrow_text_domains_by_rows(
            node_groups=raw_node_groups,
            page_rect=page_rect,
            row_tolerance_points=TEXT_ROW_TOLERANCE_POINTS,
        )
    )
    #@GPT 窄列逐行右挂之后，再校正被纯文本标题节点提前接走的多列首行；
    #@GPT 该阶段必须早于水平分区，使列首先回到稳定正文列，再由后续XY-Cut处理剩余混合节点。
    node_groups, cross_column_leading_row = (
        reparent_cross_column_leading_row(
            node_groups=node_groups,
            page_rect=page_rect,
        )
    )
    #@GPT GapTree节点内部再做一次水平空白分区，解决一个竖向大节点同时包含上方视觉块和下方表头时的阅读顺序错误。
    node_groups, horizontal_partition = (
        split_gap_tree_nodes_by_horizontal_gaps(
            node_groups=node_groups,
            source_angle=source_angle,
            page_rect=page_rect,
            row_tolerance_points=TEXT_ROW_TOLERANCE_POINTS,
        )
    )
    #@GPT 两级空间切分完成后再执行分隔符与标题的显式局部先后关系；若先执行，移动后的合成顺序可能反过来影响GapTree分栏和XY-Cut边界。
    node_groups, attached_separator_heading_order = (
        apply_attached_separator_heading_precedence(
            node_groups=node_groups,
        )
    )
    sorted_elements = [
        element
        for node_group in node_groups
        for element in node_group
    ]
    element_node_index: dict[int, int] = {}

    for node_index, node_group in enumerate(node_groups, start=1):
        for element in node_group:
            element_node_index[id(element)] = node_index

    #@GPT 仅在窄文本域已归入右侧节点后按Y恢复同行；其他节点保持每个原子独占行，避免跨栏重新拼接。
    next_row_index = 0
    for node_group in node_groups:
        row_anchor_y: float | None = None
        row_virtual_index: int | None = None
        row_group_enabled = any(
            element.get("narrow_domain_row_group", False)
            for element in node_group
        )
        for element in node_group:
            next_row_index += 1
            if row_group_enabled and element.get("kind", "text") == "text":
                virtual_y = element.get(
                    "virtual_y",
                    transformed_reading_position(
                        rect=fitz.Rect(element["bbox"]),
                        source_angle=source_angle,
                        page_rect=page_rect,
                    )[0],
                )
                if (
                    row_anchor_y is None
                    or row_virtual_index is None
                    or abs(virtual_y - row_anchor_y)
                    > TEXT_ROW_TOLERANCE_POINTS
                ):
                    row_anchor_y = virtual_y
                    row_virtual_index = next_row_index
                else:
                    next_row_index -= 1
                element["virtual_row_index"] = row_virtual_index
            else:
                element["virtual_row_index"] = next_row_index
            element["gaptree_order_index"] = next_row_index
            element["gaptree_node_index"] = element_node_index.get(
                id(element),
                0,
            )

    cuts = [
        {
            "x0": round(cut[0], 3),
            "x1": round(cut[1], 3),
            "row_start": cut[2],
            "row_end": cut[3],
            "row_span": cut[3] - cut[2] + 1,
        }
        for cut in sorter.current_cuts
    ]
    nodes: list[dict] = []

    for node_index, node_group in enumerate(node_groups, start=1):
        node_bbox = fitz.Rect(node_group[0]["bbox"])

        for element in node_group[1:]:
            node_bbox |= element["bbox"]

        nodes.append(
            {
                "node_index": node_index,
                "bbox": [round(value, 3) for value in node_bbox],
                "element_count": len(node_group),
                "kinds": dict(
                    Counter(
                        element.get("kind", "text")
                        for element in node_group
                    )
                ),
                #@GPT 非文本原子块的text包含完整Markdown链接；报告样例改用类型标签，避免嵌套链接破坏报告表格。
                "samples": [
                    (
                        element.get("text", "")
                        .replace("\n", " ")[:80]
                        if element.get("kind", "text") == "text"
                        else f"<{element.get('kind', 'element')}>"
                    )
                    for element in node_group[:3]
                ],
            }
        )

    return sorted_elements, {
        "backend": sorter.backend_name,
        "row_count": len(sorter.current_rows),
        "cut_count": len(cuts),
        "node_count": len(nodes),
        "cuts": cuts,
        "nodes": nodes,
        "fragment_merge": fragment_merge_metadata,
        #@GPT 将自动学习到的列锚点及实际移动记录写入调试元数据，便于确认规则没有依赖页码或固定坐标。
        "column_anchor_correction": column_anchor_correction,
        #@GPT 记录跨列首行是否触发、稳定列锚点和具体移动；该审计与窄编号列右挂分开。
        "cross_column_leading_row": cross_column_leading_row,
        #@GPT 将二级XY-Cut的断层、区块来源和最终顺序写入审计，便于确认是否发生跨节点重排。
        "horizontal_partition": horizontal_partition,
        #@GPT 单独记录哪些分隔符通过“同Y且端点相接”移动到标题之前；普通GapTree节点顺序不混入该审计。
        "attached_separator_heading_order": (
            attached_separator_heading_order
        ),
    }
