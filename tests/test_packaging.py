"""Checks that public dependency files describe one coherent runtime."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _package_name(specification: str) -> str:
    """Return a normalized distribution name from a dependency string."""

    name = re.split(r"[<>=!~\[]", specification, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


class PackagingMetadataTests(unittest.TestCase):
    """Prevent the lock file and project metadata from drifting silently."""

    def test_reference_requirements_match_runtime_dependency_names(self) -> None:
        pyproject = tomllib.loads(
            (PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = {
            _package_name(item)
            for item in pyproject["project"]["dependencies"]
        }
        locked = {
            _package_name(line)
            for line in (PROJECT_DIR / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(declared, locked)

    def test_only_headless_opencv_is_declared(self) -> None:
        requirements = (PROJECT_DIR / "requirements.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("opencv-python-headless", requirements)
        self.assertNotRegex(requirements, r"(?m)^opencv-python==")

    def test_license_metadata_and_notice_files_are_complete(self) -> None:
        pyproject = tomllib.loads(
            (PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(pyproject["project"]["license"], "Apache-2.0")
        self.assertEqual(pyproject["project"]["requires-python"], ">=3.12")
        self.assertEqual(
            set(pyproject["project"]["license-files"]),
            {
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "nativePDF_structurer/_vendor/GapTree_LICENSE.txt",
            },
        )
        self.assertIn(
            "GapTree_LICENSE.txt",
            pyproject["tool"]["setuptools"]["package-data"][
                "nativePDF_structurer._vendor"
            ],
        )

        apache_license = (PROJECT_DIR / "LICENSE").read_text(encoding="utf-8")
        notices = (PROJECT_DIR / "THIRD_PARTY_NOTICES.md").read_text(
            encoding="utf-8"
        )
        gap_tree_license = (
            PROJECT_DIR
            / "nativePDF_structurer"
            / "_vendor"
            / "GapTree_LICENSE.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("Apache License", apache_license)
        self.assertIn("Version 2.0", apache_license)
        self.assertIn("PyMuPDF", notices)
        self.assertIn("AGPL-3.0", notices)
        self.assertIn("docs/assets/examples/", notices)
        self.assertIn("MIT License", gap_tree_license)

    def test_runtime_module_package_and_ci_contract_exist(self) -> None:
        pyproject = tomllib.loads(
            (PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertIn(
            "nativePDF_structurer",
            pyproject["tool"]["setuptools"]["packages"],
        )
        for module_name in (
            "claims.py",
            "geometry.py",
            "inventory.py",
            "output_ownership.py",
            "text_ownership.py",
            "reading_order_backend.py",
            "table_skeleton.py",
            "table_detection.py",
            "table_structure.py",
            "cross_detector_ownership.py",
            "pipeline.py",
            "layout_geometry.py",
            "visual_processing.py",
            "table_output.py",
            "header_footer.py",
            "formula_regions.py",
            "figure_captions.py",
            "outline_headings.py",
            "heading_detection.py",
            "visual_assets.py",
        ):
            self.assertTrue(
                (PROJECT_DIR / "nativePDF_structurer" / module_name).is_file(),
                module_name,
            )
        workflow = (PROJECT_DIR / ".github" / "workflows" / "tests.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("python -m unittest discover -s tests -v", workflow)
        self.assertIn("pip install --no-deps -e .", workflow)

    def test_historical_core_and_bridge_are_absent(self) -> None:
        self.assertFalse((PROJECT_DIR / "stage_c_claims_full.py").exists())
        self.assertFalse(
            (PROJECT_DIR / "nativePDF_structurer" / "legacy_bridge.py").exists()
        )
        pyproject = tomllib.loads(
            (PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertNotIn("py-modules", pyproject["tool"]["setuptools"])


if __name__ == "__main__":
    unittest.main()
