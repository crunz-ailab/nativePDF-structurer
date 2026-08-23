"""定义一次PDF结构化任务的已校验配置。

模块解析从1开始的页码表达式，校验
输入输出路径，并将用户可见选项与流水线使用的不可变执行上下文分离。

Define validated configuration for one PDF structuring run.

Parses one-based page selections, validates input and output paths, and
separates user-visible options from the immutable execution context consumed
by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path



def _expand_page_spec(spec: str) -> tuple[int, ...]:
    """Expand ``1-3,8`` into sorted, unique one-based PDF page numbers."""

    pages: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = (part.strip() for part in token.split("-", 1))
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(f"invalid page range: {token}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(token))
    if any(page <= 0 for page in pages):
        raise ValueError("PDF page numbers are one-based and must be positive")
    return tuple(sorted(pages))


@dataclass(frozen=True)
class RunConfig:
    """User-visible options that can be passed to the parser boundary.

    传递给解析器公共入口的用户配置。

    ``pages`` is one-based.  ``None`` means every page in the source PDF.
    Debug artifacts are deliberately opt-in because they can be much larger
    than the Markdown and assets required for retrieval.

    ``pages`` 使用从1开始的页码；``None`` 表示处理整个PDF。调试制品可能
    远大于检索所需的Markdown和资源，因此默认关闭。
    """

    source_pdf: Path
    output_dir: Path
    pages: tuple[int, ...] | None = None
    debug_output: bool = False
    enable_heading_detection: bool = True
    enable_font_heading_inference: bool = False
    enable_table_header_detection: bool = False
    keep_header_text: bool = False
    enable_formula_detection: bool = True
    enable_pdf_outline_heading_assignment: bool = True
    force_pure_text_table_to_markdown: bool = False
    formula_output_mode: str = "image"

    def __post_init__(self) -> None:
        source = Path(self.source_pdf).expanduser()
        output = Path(self.output_dir).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"source PDF does not exist: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"source file is not a PDF: {source}")
        if self.formula_output_mode not in {"image", "latex", "hybrid"}:
            raise ValueError(
                "formula_output_mode must be one of: image, latex, hybrid"
            )
        object.__setattr__(self, "source_pdf", source)
        object.__setattr__(self, "output_dir", output)
        if self.pages is not None:
            normalized = tuple(sorted(set(self.pages)))
            if any(page <= 0 for page in normalized):
                raise ValueError("page numbers must be positive")
            object.__setattr__(self, "pages", normalized)

    @classmethod
    def from_page_spec(
        cls,
        *,
        source_pdf: str | Path,
        output_dir: str | Path,
        page_spec: str | None = None,
        **kwargs: object,
    ) -> "RunConfig":
        """Build a configuration from compact CLI page notation.

        从紧凑的命令行页码表达式构建配置，例如 ``1-20,24,28``。
        """

        pages = None if not page_spec else _expand_page_spec(page_spec)
        return cls(
            source_pdf=Path(source_pdf),
            output_dir=Path(output_dir),
            pages=pages,
            **kwargs,
        )


@dataclass(frozen=True)
class ParseRunContext:
    """Carry validated options into one parser-core invocation.

    将已校验配置传入一次完整的解析执行。

    ``RunConfig.pages`` may be ``None`` until the source PDF is opened.  The
    caller therefore resolves the final one-based page tuple first and stores
    it here.  Keeping that result beside the original config lets the parser
    consume one immutable object instead of loosely related assignments.

    ``RunConfig.pages`` 在PDF打开前可能为 ``None``；调用方需先根据文档页数
    解析最终的一基页码元组，再与原配置一起保存到本对象中。

    The context is intentionally not a replacement for ``RunConfig``.  It is
    an execution boundary for one already validated run and contains no PDF
    document handle, mutable caches, timing counters, or detector records.
    Those remain owned by the parser invocation and its returned
    :class:`~nativePDF_structurer.result.ParseResult`.

    本对象不持有PDF句柄、可变缓存、计时器或检测记录；这些运行期状态由解析
    流程及其返回的 ``ParseResult`` 管理。
    """

    config: RunConfig
    page_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.page_numbers)))
        if any(page <= 0 for page in normalized):
            raise ValueError("page numbers must be positive")
        object.__setattr__(self, "page_numbers", normalized)
