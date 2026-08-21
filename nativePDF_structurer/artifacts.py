"""定义解析制品的输出目录契约。

模块提供Markdown资源、公式、表格、视觉区域
和调试制品的命名路径，并按运行开关创建所需目录；备份或删除策略由调用方
注入，本模块不自行决定覆盖行为。

Define the output-directory contract for parser artifacts.

Provides named paths for Markdown assets and optional diagnostics, plus the
factory that prepares only the directories enabled for a run. Backup or
deletion policy is injected by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ResetOutputDirectory = Callable[[Path], None]


@dataclass(frozen=True)
class OutputLayout:
    """Named locations for Markdown assets and optional diagnostics.

    描述Markdown引用资源和可选调试制品的标准目录布局。
    """

    root: Path
    tables: Path
    table_images: Path
    source_images: Path
    visual_regions: Path
    formula_images: Path
    reading_order_debug: Path


def prepare_output_layout(
    root: Path,
    *,
    reset: ResetOutputDirectory,
    formula_enabled: bool,
    debug_enabled: bool,
) -> OutputLayout:
    """Reset one run directory and create only the directories it can use.

    重置一次运行的输出根目录，并只创建当前开关可能使用的子目录。

    The caller supplies the reset policy because existing runs are currently
    preserved by renaming them to timestamped backups.  Keeping that policy
    outside this module prevents directory naming from becoming coupled to a
    particular backup strategy.  The returned paths are stable contracts for
    rendering stages and for :class:`ArtifactEmitter`.

    调用方负责提供目录重置策略，因此本模块不依赖删除、覆盖或时间戳备份等
    具体实现。返回的路径是渲染阶段和 ``ArtifactEmitter`` 共用的稳定契约。
    """

    root = Path(root)
    reset(root)
    layout = OutputLayout(
        root=root,
        tables=root / "tables",
        table_images=root / "table_images_auto_rotated",
        source_images=root / "source_images",
        visual_regions=root / "visual_regions",
        formula_images=root / "formula_images",
        reading_order_debug=root / "gaptree_debug",
    )

    for directory in (
        layout.tables,
        layout.table_images,
        layout.source_images,
        layout.visual_regions,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if formula_enabled:
        layout.formula_images.mkdir(parents=True, exist_ok=True)
    if debug_enabled:
        layout.reading_order_debug.mkdir(parents=True, exist_ok=True)
    return layout
