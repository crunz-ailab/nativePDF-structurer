"""Tests for ownership and isolation guarantees of ParseResult."""

from __future__ import annotations

import unittest

from nativePDF_structurer.result import ParseResult


class ParseResultTests(unittest.TestCase):
    """Ensure every parser run receives independent mutable collections."""

    def test_instances_do_not_share_lists_or_dictionaries(self) -> None:
        first = ParseResult()
        second = ParseResult()

        first.pages.append({"page_number": 1})
        first.detector_claims.append({"claim_id": "claim#1"})
        first.summary["processed_pages"] = 1

        self.assertEqual(second.pages, [])
        self.assertEqual(second.detector_claims, [])
        self.assertEqual(second.summary, {})

    def test_compatibility_aliases_mutate_the_owned_collection(self) -> None:
        result = ParseResult()
        legacy_pages_output = result.pages
        legacy_pages_output.append({"page_number": 3})

        self.assertIs(legacy_pages_output, result.pages)
        self.assertEqual(result.pages, [{"page_number": 3}])

    def test_final_summary_and_ir_can_be_assigned_after_page_processing(self) -> None:
        result = ParseResult()
        result.summary = {"processed_pages": 2}
        result.document_ir = {"schema_version": "0.1"}

        self.assertEqual(result.summary["processed_pages"], 2)
        self.assertEqual(result.document_ir["schema_version"], "0.1")


if __name__ == "__main__":
    unittest.main()
