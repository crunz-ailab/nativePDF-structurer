"""Tests for the compact IR schema and portable provenance paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nativePDF_structurer.ir import build_document_ir


class DocumentIRTests(unittest.TestCase):
    """Lock the stable fields consumed by retrieval and audit tooling."""

    def test_assets_and_claim_artifacts_use_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            asset_path = output_dir / "source_images" / "image.png"
            ir = build_document_ir(
                source_pdf=Path("input.pdf"),
                output_dir=output_dir,
                source_page_count=1,
                summary={
                    "processed_pages": 1,
                    "processed_page_numbers": [1],
                },
                pages=[
                    {
                        "page_number": 1,
                        "markdown_page_title": "## Example",
                        "text_reflow": {"direction_groups": []},
                    }
                ],
                tables=[],
                source_images=[
                    {
                        "page_number": 1,
                        "claim_id": "page#1:source_image#1",
                        "output_path": str(asset_path),
                        "output_bbox": [1, 2, 3, 4],
                        "source_element_ids": ["i:1"],
                    }
                ],
                visual_regions=[],
                formula_regions=[],
                detector_claims=[
                    {
                        "page_number": 1,
                        "claims": [
                            {
                                "claim_id": "page#1:source_image#1",
                                "detector": "source_image_processor",
                                "input_element_ids": ["i:1"],
                                "feature_element_ids": ["i:1"],
                                "output_artifacts": [
                                    {
                                        "type": "direct_copy",
                                        "path": str(asset_path),
                                        "bbox": [1, 2, 3, 4],
                                    }
                                ],
                            }
                        ],
                        "resolution_preview": {
                            "owner_by_element_id": {
                                "i:1": "page#1:source_image#1"
                            }
                        },
                    }
                ],
            )

        self.assertEqual(ir["schema_version"], "0.1")
        self.assertEqual(
            ir["assets"][0]["path"],
            "source_images/image.png",
        )
        self.assertEqual(
            ir["pages"][0]["provenance"]["claims"][0][
                "output_artifacts"
            ][0]["path"],
            "source_images/image.png",
        )
        self.assertEqual(
            ir["pages"][0]["provenance"]["owner_by_element_id"],
            {"i:1": "page#1:source_image#1"},
        )


if __name__ == "__main__":
    unittest.main()
