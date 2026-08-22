from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _qualified_docstring(path: Path, qualified_name: str) -> str | None:
    """Return a top-level or class-method docstring without importing code.

    通过AST读取顶层符号或类方法的docstring，不执行被检查模块。
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes: list[ast.AST] = list(tree.body)
    for part in qualified_name.split("."):
        matched = next(
            (
                node
                for node in nodes
                if isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and node.name == part
            ),
            None,
        )
        if matched is None:
            return None
        nodes = list(getattr(matched, "body", []))
    return ast.get_docstring(matched)


class DocumentationContractTests(unittest.TestCase):
    def test_architecture_document_describes_current_boundaries(self) -> None:
        text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
        for required in (
            "PageRunContext",
            "PageStage",
            "claims.py",
            "geometry.py",
            "inventory.py",
            "output_ownership.py",
            "text_ownership.py",
            "reading_order_backend.py",
            "render_bbox",
            "sort_bbox",
            "page_contract.py",
            "full-range manifests",
        ):
            self.assertIn(required, text)

    def test_readme_links_architecture_guide(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/architecture.md", text)

    def test_public_api_docstrings_are_bilingual(self) -> None:
        public_symbols = {
            "nativePDF_structurer/config.py": (
                "RunConfig",
                "RunConfig.from_page_spec",
                "ParseRunContext",
            ),
            "nativePDF_structurer/page_context.py": ("PageRunContext",),
            "nativePDF_structurer/result.py": ("ParseResult",),
            "nativePDF_structurer/pipeline.py": ("run",),
            "nativePDF_structurer/cli.py": ("build_parser", "main"),
            "nativePDF_structurer/artifacts.py": (
                "OutputLayout",
                "prepare_output_layout",
            ),
            "nativePDF_structurer/emitter.py": (
                "ArtifactEmitter",
                "ArtifactEmitter.write_markdown_document",
                "ArtifactEmitter.write_parse_result_markdown",
                "ArtifactEmitter.write_json",
                "ArtifactEmitter.write_parse_result_debug_json",
            ),
            "nativePDF_structurer/regression_manifest.py": (
                "sha256_file",
                "build_manifest",
                "write_manifest",
                "verify_manifest",
                "build_parser",
                "main",
            ),
        }

        for relative_path, qualified_names in public_symbols.items():
            path = ROOT / relative_path
            for qualified_name in qualified_names:
                with self.subTest(symbol=qualified_name):
                    docstring = _qualified_docstring(path, qualified_name)
                    self.assertTrue(docstring, qualified_name)
                    self.assertRegex(docstring, r"[A-Za-z]", qualified_name)
                    self.assertRegex(
                        docstring,
                        r"[\u4e00-\u9fff]",
                        qualified_name,
                    )

    def test_all_module_docstrings_follow_bilingual_google_style(self) -> None:
        module_paths = sorted((ROOT / "nativePDF_structurer").glob("*.py"))
        self.assertEqual(len(module_paths), 40)

        for path in module_paths:
            with self.subTest(module=path.name):
                tree = ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                )
                docstring = ast.get_docstring(tree, clean=False)
                self.assertTrue(docstring, path.name)
                lines = docstring.splitlines()
                self.assertTrue(lines[0].strip(), path.name)
                self.assertLessEqual(len(lines[0]), 80, path.name)
                self.assertRegex(lines[0], r"[.!?。！？]$", path.name)
                self.assertGreaterEqual(len(lines), 3, path.name)
                self.assertEqual(lines[1], "", path.name)
                self.assertRegex(lines[0], r"[\u4e00-\u9fff]", path.name)
                self.assertNotIn("中文说明:", docstring, path.name)
                self.assertNotIn("English:", docstring, path.name)
                self.assertRegex(docstring, r"[A-Za-z]", path.name)
                self.assertRegex(docstring, r"[\u4e00-\u9fff]", path.name)
                english_summaries = [
                    index
                    for index, line in enumerate(lines)
                    if re.match(r"^[A-Z][^\n]*[.!?]$", line)
                ]
                self.assertTrue(english_summaries, path.name)
                english_index = english_summaries[0]
                self.assertGreater(english_index, 1, path.name)
                self.assertEqual(lines[english_index - 1], "", path.name)
                self.assertEqual(tree.body[0].lineno, 1, path.name)
                closing_line = path.read_text(encoding="utf-8").splitlines()[
                    tree.body[0].end_lineno - 1
                ]
                self.assertEqual(closing_line.strip(), '"""', path.name)


if __name__ == "__main__":
    unittest.main()
