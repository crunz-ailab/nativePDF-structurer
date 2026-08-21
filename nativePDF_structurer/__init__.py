"""提供PDF-to-RAG结构化工具的稳定公共接口。

这里只导出受支持的配置对象、
页面上下文、结果模型和执行入口；检测器实现及中间数据结构属于包内细节。

Expose the stable public API for the PDF-to-RAG structuring package.

Exports only the supported configuration objects, page context, result model,
and execution entry point. Detector implementations and intermediate schemas
remain internal package details.
"""

#@GPT 导出运行配置、文档级上下文、页面级上下文和执行入口，调用方无需依赖内部全局变量。
__all__ = ["PageRunContext", "ParseRunContext", "RunConfig", "run"]
__version__ = "0.1.0"

from .config import ParseRunContext, RunConfig
from .pipeline import run
from .page_context import PageRunContext
