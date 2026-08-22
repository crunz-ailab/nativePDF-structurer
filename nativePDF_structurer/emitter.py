"""序列化一次解析任务的稳定顶层制品。

模块负责公开文件名、路径校验以及
Markdown和JSON的字节级格式；PDF检测、版面分析、渲染、Claim仲裁和
schema构建均不属于本模块。

关键不变量:
    * Markdown和JSON输出必须保持确定性。
    * 顶层文件名不得越过已准备的输出根目录。
    * 序列化过程不得修改解析器持有的记录。

Serialize stable top-level artifacts for one parser run.

Owns public artifact filenames, path validation, and byte-level Markdown and
JSON formatting. PDF detection, layout analysis, rendering, Claim resolution,
and schema construction remain outside this module.

Key invariants:
    * Markdown and JSON formatting is deterministic.
    * Top-level filenames cannot escape the prepared output root.
    * Serialization never mutates parser-owned records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import OutputLayout
from .result import ParseResult


@dataclass(frozen=True)
class ArtifactEmitter:
    """Write Markdown and JSON artifacts into a prepared output directory.

    将Markdown和JSON制品写入已准备好的输出目录。

    Parameters
    ----------
    layout:
        Directory layout returned by :func:`prepare_output_layout`.  The
        emitter never resets the directory and never creates image assets;
        those side effects remain the responsibility of the parser stages
        that render tables, images, visual regions, and formulas.

    Serialization guarantees / 序列化保证
    -------------------------------------
    Markdown sections are separated by exactly two newline characters and
    the document ends with one newline.  JSON uses UTF-8, keeps Unicode
    characters unescaped, uses an indentation width of two spaces, and does
    not append a trailing newline.  These byte-level rules are part of the
    public output contract and allow regression tests to compare file hashes.

    Markdown和JSON的分隔符、缩进及尾换行规则属于公开输出契约。
    """

    layout: OutputLayout

    def write_markdown_document(
        self,
        sections: Iterable[str],
        *,
        filename: str = "document_with_reflowed_text_and_tables.md",
    ) -> Path:
        """Write the primary Markdown document and return its path.

        写出主Markdown文档并返回文件路径。

        ``sections`` is consumed once and may therefore be a list or a
        generator.  Filename validation prevents callers from escaping the
        prepared output directory; referenced image and CSV assets are
        created separately by their owning parser stages.

        ``sections`` 只遍历一次，可以是列表或生成器。文件名必须位于输出根
        目录；Markdown引用的图片和CSV由对应解析阶段单独生成。
        """

        path = self._root_file(filename)
        path.write_text(
            "\n\n".join(sections) + "\n",
            encoding="utf-8",
        )
        return path

    def write_parse_result_markdown(
        self,
        result: ParseResult,
        *,
        filename: str = "document_with_reflowed_text_and_tables.md",
    ) -> Path:
        """Write the primary Markdown artifact owned by ``result``.

        写出 ``ParseResult`` 持有的主Markdown制品。

        Parameters
        ----------
        result:
            Completed or incrementally accumulated parser result.  Only
            ``markdown_sections`` is read; detector records, timing data,
            and rendered assets are not inspected or modified.
        filename:
            Simple root-level output name.  The default is the stable public
            filename used by the command-line interface.

        Returns
        -------
        pathlib.Path
            Path of the written Markdown document.

        Side effects and compatibility
        ------------------------------
        This method writes one UTF-8 file inside the prepared output root.
        It delegates byte formatting and path validation to
        :meth:`write_markdown_document`, so section separators, the trailing
        newline, and path-safety behavior remain unchanged.  It performs no
        parsing, ordering, rendering, or schema conversion.

        本方法只读取 ``markdown_sections``，不修改检测记录、计时信息或渲染
        资源；实际字节格式和路径校验由 ``write_markdown_document`` 统一完成。
        """

        return self.write_markdown_document(
            result.markdown_sections,
            filename=filename,
        )

    def write_json(self, filename: str, payload: Any) -> Path:
        """Serialize one JSON-compatible value as a root-level artifact.

        将一个JSON兼容值序列化为输出根目录中的制品。

        The method performs no schema conversion.  Callers must first build
        the appropriate public model, such as ``document_ir`` or a diagnostic
        record list.  This separation prevents serialization code from
        acquiring detector-specific branching over time.

        本方法不执行schema转换；调用方必须先构建IR或调试记录等公开模型，
        从而避免通用序列化层依赖检测器内部逻辑。
        """

        path = self._root_file(filename)
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def write_parse_result_debug_json(
        self,
        result: ParseResult,
        *,
        header_footer_detection: Any,
    ) -> dict[str, Path]:
        """Write the machine-readable debug artifacts for one parser run.

        写出一次解析任务的机器可读调试制品，并返回文件名到路径的映射。

        Parameters
        ----------
        result:
            Parser-owned collections and final document-level records.  The
            method reads the fields directly and never mutates them.
        header_footer_detection:
            Already assembled public payload containing serialized templates
            and per-page matches.  Template serialization remains outside
            the emitter because it belongs to the header/footer subsystem,
            not to generic file emission.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping from each stable artifact filename to the path written.
            Returning the mapping supports tests and future callers without
            requiring them to reconstruct output paths.

        Side effects and compatibility
        ------------------------------
        Thirteen UTF-8 JSON files are written at the output root using the
        exact formatting rules of :meth:`write_json`.  Existing filenames
        and payload shapes are preserved.  Human-readable Markdown reports,
        rendered assets, compact-IR construction, parser decisions, and
        debug-mode policy remain outside this method.

        该方法不执行解析、排序、渲染或IR构建；它只按照稳定文件名和
        ``write_json`` 的字节规则写出 ``ParseResult`` 中已有的数据。
        """

        payloads = (
            ("pages_with_text_reflow.json", result.pages),
            ("tables_with_rotation.json", result.tables),
            ("source_images.json", result.source_images),
            ("visual_regions.json", result.visual_regions),
            ("formula_regions.json", result.formula_regions),
            ("header_footer_detection.json", header_footer_detection),
            (
                "logical_horizontal_rules.json",
                result.logical_horizontal_rules,
            ),
            ("page_inventory_audit.json", result.inventory_audit),
            ("detector_claims.json", result.detector_claims),
            (
                "claim_resolution_preview.json",
                result.claim_resolutions,
            ),
            (
                "claim_resolution_actions.json",
                result.claim_resolution_actions,
            ),
            ("document_ir.json", result.document_ir),
            ("summary_with_text_reflow.json", result.summary),
        )
        return {
            filename: self.write_json(filename, payload)
            for filename, payload in payloads
        }

    def _root_file(self, filename: str) -> Path:
        """Resolve a simple filename inside the output root.

        Top-level artifacts intentionally use fixed simple names.  Rejecting
        absolute paths and path separators makes accidental writes outside
        the run directory impossible at this boundary.
        """

        candidate = Path(filename)
        if candidate.is_absolute() or candidate.name != filename:
            raise ValueError(
                "artifact filename must be a simple root-level name: "
                f"{filename}"
            )
        return self.layout.root / candidate
