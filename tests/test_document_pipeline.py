from __future__ import annotations

import types
import unittest

from nativePDF_structurer.pipeline import (
    DocumentPipelineDependencies,
    process_document_pages,
)


class DocumentPipelineTests(unittest.TestCase):
    def test_process_document_pages_commits_in_order_and_closes(self) -> None:
        events: list[tuple] = []
        result = object()

        class Document:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True
                events.append(("close",))

        document = Document()
        state = types.SimpleNamespace(
            page_indices=[2, 0, 1],
            document=document,
        )

        def process_page(page_index: int, *, dependencies: object) -> dict:
            events.append(("page", page_index, dependencies))
            return {"page_index": page_index}

        def commit_page_context(
            parse_result: object,
            page_context: dict,
            *,
            validate: bool,
        ) -> None:
            events.append(
                ("commit", parse_result, page_context["page_index"], validate)
            )

        process_document_pages(
            dependencies=DocumentPipelineDependencies(
                document_run_state=state,
                parse_result=result,
                page_pipeline_dependencies="page-scope",
                process_page=process_page,
                commit_page_context=commit_page_context,
                validate_page_contracts=True,
            )
        )

        self.assertEqual(
            [event[0:2] for event in events if event[0] == "page"],
            [("page", 2), ("page", 0), ("page", 1)],
        )
        self.assertEqual(
            [event[0:3] for event in events if event[0] == "commit"],
            [
                ("commit", result, 2),
                ("commit", result, 0),
                ("commit", result, 1),
            ],
        )
        self.assertEqual(events[-1], ("close",))
        self.assertTrue(document.closed)


if __name__ == "__main__":
    unittest.main()
