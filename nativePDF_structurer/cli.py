"""提供PDF结构化工具的公开命令行接口。

模块负责构建参数解析器，通过
``RunConfig`` 校验用户选项，并调用包级 ``run`` 入口；显示 ``-h`` 帮助
时不会启动PDF分析。

Provide the public command-line interface for PDF structuring.

Builds the argument parser, validates user-facing options through
:class:`RunConfig`, and invokes the package-level ``run`` entry point. Merely
displaying help does not start PDF analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    """Create the public command-line parser without running PDF analysis.

    创建公开命令行参数解析器，但不启动PDF分析。调用方可以低成本显示
    ``-h`` 帮助或校验参数，而无需执行任何文档处理。
    """

    parser = argparse.ArgumentParser(
        prog="nativePDF-structurer",
        description=(
            "Reconstruct reading order and visual structure from a "
            "digital-native PDF without OCR or a model."
        ),
    )
    parser.add_argument("pdf", type=Path, help="source PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--pages",
        help="one-based pages, for example 1-20,24,28; default: all pages",
    )
    parser.add_argument(
        "--debug-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write JSON, reports and GapTree debug images",
    )
    parser.add_argument(
        "--heading-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable outline/text heading handling",
    )
    parser.add_argument(
        "--font-heading-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable document-wide font-size heading inference",
    )
    parser.add_argument(
        "--table-header-detection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable table-header formatting for pure text tables",
    )
    parser.add_argument(
        "--keep-header-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit matched page-header text in Markdown",
    )
    parser.add_argument(
        "--formula-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render detected display formulas as images",
    )
    parser.add_argument(
        "--formula-output",
        choices=("image", "latex", "hybrid"),
        default="image",
        help="formula output: image (default), latex when safely recoverable, or hybrid",
    )
    parser.add_argument(
        "--pdf-outline-heading-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use matching PDF outline entries for Markdown headings",
    )
    parser.add_argument(
        "--force-pure-text-table-to-markdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit pure-text tables as Markdown regardless of cell count",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate CLI arguments and execute one configured document run.

    校验命令行参数，构建 ``RunConfig``，执行一次文档结构化任务，并返回
    适合shell使用的退出码。
    """

    args = build_parser().parse_args(argv)
    config = RunConfig.from_page_spec(
        source_pdf=args.pdf,
        output_dir=args.output,
        page_spec=args.pages,
        debug_output=args.debug_output,
        enable_heading_detection=args.heading_detection,
        enable_font_heading_inference=args.font_heading_inference,
        enable_table_header_detection=args.table_header_detection,
        keep_header_text=args.keep_header_text,
        enable_formula_detection=args.formula_detection,
        formula_output_mode=args.formula_output,
        enable_pdf_outline_heading_assignment=(
            args.pdf_outline_heading_assignment
        ),
        force_pure_text_table_to_markdown=(
            args.force_pure_text_table_to_markdown
        ),
    )
    run(config)
    return 0
