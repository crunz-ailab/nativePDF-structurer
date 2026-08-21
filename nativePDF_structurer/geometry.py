"""提供页面阶段共享的基础几何函数。

本模块负责矩形规范化、中心点、非负面积、并集和drawing item边界，不包含
检测策略、页面特定阈值或归属决策。

Provide small geometric primitives shared by page stages.

Normalizes rectangle-like values and implements centers, non-negative areas,
unions, and drawing-item bounds. The functions contain no detector policy,
page-specific thresholds, or ownership decisions.
"""

from __future__ import annotations

import fitz
from typing import NamedTuple


#@GPT 轴向索引使用轻量四元组代替大量fitz.Rect对象；集中到几何模块后，表格骨架和核心共享同一类型定义。
class AxisRect(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


#@GPT AxisRect只提供只读坐标；进入PyMuPDF API前统一通过本函数转换，避免模块自行假设normalize/contains等Rect方法。
def as_fitz_rect(
    rect: fitz.Rect | AxisRect | tuple[float, float, float, float],
) -> fitz.Rect:
    """Convert a lightweight axis rectangle to a PyMuPDF rectangle."""

    if isinstance(rect, fitz.Rect):
        return rect
    return fitz.Rect(rect)


#@GPT 中心点、面积和并集是多个检测器共同使用的基础运算；集中后，后续台账模块不需要反向依赖巨型核心文件。
def rect_center(rect: fitz.Rect) -> fitz.Point:
    """Return the center point of a rectangle."""

    return fitz.Point(
        (rect.x0 + rect.x1) / 2,
        (rect.y0 + rect.y1) / 2,
    )


def rect_area(rect: fitz.Rect) -> float:
    """Return the non-negative area of a rectangle."""

    return max(0.0, rect.width) * max(0.0, rect.height)


def union_rects(rects: list[fitz.Rect]) -> fitz.Rect:
    """Return the bounding rectangle of one or more rectangles."""

    if not rects:
        raise ValueError("union_rects() requires at least one rectangle")
    return fitz.Rect(
        min(rect.x0 for rect in rects),
        min(rect.y0 for rect in rects),
        max(rect.x1 for rect in rects),
        max(rect.y1 for rect in rects),
    )


#@GPT drawing item可能包含Point、Rect或Quad；该函数保持原有坐标归一化语义，供惰性drawing-item台账和未来几何检测器复用。
def drawing_item_bbox(item: tuple) -> fitz.Rect | None:
    """Return the minimum rectangle covering a PyMuPDF drawing item."""

    points: list[fitz.Point] = []
    for value in item[1:]:
        if isinstance(value, fitz.Point):
            points.append(fitz.Point(value))
            continue
        if isinstance(value, fitz.Rect):
            rect = fitz.Rect(value).normalize()
            points.extend(
                [
                    fitz.Point(rect.x0, rect.y0),
                    fitz.Point(rect.x1, rect.y1),
                ]
            )
            continue
        if isinstance(value, fitz.Quad):
            rect = fitz.Rect(value.rect).normalize()
            points.extend(
                [
                    fitz.Point(rect.x0, rect.y0),
                    fitz.Point(rect.x1, rect.y1),
                ]
            )

    if not points:
        return None
    return fitz.Rect(
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    ).normalize()
