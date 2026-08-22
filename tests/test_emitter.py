"""Byte-level tests for stable Markdown and JSON serialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nativePDF_structurer.artifacts import OutputLayout
from nativePDF_structurer.emitter import ArtifactEmitter
from nativePDF_structurer.result import ParseResult


def _layout(root: Path) -> OutputLayout:
    """Create an OutputLayout for tests that exercise root-level artifacts.

    The emitter writes only into ``root`` in these tests, so the remaining
    paths describe the normal directory contract without requiring image
    directories to be created.
    """

    return OutputLayout(
        root=root,
        tables=root / "tables",
        table_images=root / "table_images_auto_rotated",
        source_images=root / "source_images",
        visual_regions=root / "visual_regions",
        formula_images=root / "formula_images",
        reading_order_debug=root / "gaptree_debug",
    )


class ArtifactEmitterTests(unittest.TestCase):
    """Protect the exact byte format used by existing regression hashes."""

    def test_markdown_separator_and_trailing_newline_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emitter = ArtifactEmitter(_layout(root))
            path = emitter.write_markdown_document(["第一节", "第二节"])
            self.assertEqual(
                path.read_bytes(),
                "第一节\n\n第二节\n".encode("utf-8"),
            )

    def test_json_keeps_unicode_and_has_no_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emitter = ArtifactEmitter(_layout(root))
            path = emitter.write_json("sample.json", {"标题": "示例"})
            self.assertEqual(
                path.read_bytes(),
                '{\n  "标题": "示例"\n}'.encode("utf-8"),
            )

    def test_nested_or_absolute_artifact_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            emitter = ArtifactEmitter(_layout(Path(directory)))
            with self.assertRaises(ValueError):
                emitter.write_json("nested/sample.json", {})
            with self.assertRaises(ValueError):
                emitter.write_json("/tmp/sample.json", {})

    def test_parse_result_markdown_uses_owned_sections(self) -> None:
        """Ensure the result-driven entry preserves Markdown byte output."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = ParseResult(markdown_sections=["第一页", "第二页"])
            path = ArtifactEmitter(_layout(root)).write_parse_result_markdown(
                result
            )
            self.assertEqual(
                path.read_bytes(),
                "第一页\n\n第二页\n".encode("utf-8"),
            )

    def test_parse_result_debug_json_uses_stable_field_mapping(self) -> None:
        """Write every result-owned debug payload to its public filename."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = ParseResult(
                pages=[{"kind": "pages"}],
                tables=[{"kind": "tables"}],
                source_images=[{"kind": "source_images"}],
                visual_regions=[{"kind": "visual_regions"}],
                formula_regions=[{"kind": "formula_regions"}],
                inventory_audit=[{"kind": "inventory_audit"}],
                detector_claims=[{"kind": "detector_claims"}],
                claim_resolutions=[{"kind": "claim_resolutions"}],
                claim_resolution_actions=[{"kind": "resolution_actions"}],
                logical_horizontal_rules=[{"kind": "logical_rules"}],
                document_ir={"kind": "document_ir"},
                summary={"kind": "summary"},
            )
            header_footer = {"kind": "header_footer"}
            paths = ArtifactEmitter(_layout(root)).write_parse_result_debug_json(
                result,
                header_footer_detection=header_footer,
            )

            expected_payloads = {
                "pages_with_text_reflow.json": result.pages,
                "tables_with_rotation.json": result.tables,
                "source_images.json": result.source_images,
                "visual_regions.json": result.visual_regions,
                "formula_regions.json": result.formula_regions,
                "header_footer_detection.json": header_footer,
                "logical_horizontal_rules.json": (
                    result.logical_horizontal_rules
                ),
                "page_inventory_audit.json": result.inventory_audit,
                "detector_claims.json": result.detector_claims,
                "claim_resolution_preview.json": result.claim_resolutions,
                "claim_resolution_actions.json": (
                    result.claim_resolution_actions
                ),
                "document_ir.json": result.document_ir,
                "summary_with_text_reflow.json": result.summary,
            }
            self.assertEqual(set(paths), set(expected_payloads))
            for filename, payload in expected_payloads.items():
                self.assertEqual(
                    json.loads(paths[filename].read_text(encoding="utf-8")),
                    payload,
                )


if __name__ == "__main__":
    unittest.main()
