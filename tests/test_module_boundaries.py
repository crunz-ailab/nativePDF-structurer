from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_SYMBOLS = {
    "nativePDF_structurer.claims": {
        "CLAIM_DETECTOR_PRIORITY",
        "claim_effective_priority",
        "create_claim_manifest",
    },
    "nativePDF_structurer.geometry": {
        "AxisRect",
        "as_fitz_rect",
        "drawing_item_bbox",
        "rect_area",
        "rect_center",
        "union_rects",
    },
    "nativePDF_structurer.inventory": {
        "PageInventory",
        "build_page_inventory",
    },
    "nativePDF_structurer.output_ownership": {
        "apply_claim_resolution_to_outputs",
    },
    "nativePDF_structurer.text_ownership": {
        "build_stage_f_text_ownership_preview",
    },
    "nativePDF_structurer.reading_order_backend": {
        "create_reading_order_backend",
        "transformed_reading_bbox",
        "transformed_reading_position",
        "coalesce_horizontal_text_fragments",
        "attach_narrow_text_domains_by_rows",
        "reparent_cross_column_leading_row",
        "split_gap_tree_nodes_by_horizontal_gaps",
        "apply_attached_separator_heading_precedence",
        "sort_lines_in_reading_order",
    },
    "nativePDF_structurer.markdown_formatter": {
        "format_sorted_lines_as_markdown",
    },
    "nativePDF_structurer.page_reflow": {
        "build_reflowed_page_text",
        "append_reflowed_text_to_markdown",
    },
    "nativePDF_structurer.rule_geometry": {
        "build_axis_geometry_index",
        "build_logical_horizontal_rules",
        "drawing_item_axis_segments",
        "merge_exact_collinear_axis_segments",
    },
    "nativePDF_structurer.rule_bands": {
        "detect_rule_line_framework_bbox",
    },
    "nativePDF_structurer.rule_preclassification": {
        "detect_preclassified_rule_bands",
        "build_preclassified_text_separator_outputs",
    },
    "nativePDF_structurer.debug_reports": {
        "write_logical_horizontal_rule_report",
        "write_visual_region_report",
        "write_claim_conflict_report",
        "write_formula_region_report",
        "write_source_image_report",
        "serialize_header_footer_templates",
        "write_header_footer_report",
        "write_heading_detection_report",
        "write_figure_caption_report",
        "render_gaptree_debug_image",
        "write_gaptree_report",
    },
    "nativePDF_structurer.table_skeleton": {
        "select_cluster_table_budget",
        "select_cluster_table_skeleton_drawings",
    },
    "nativePDF_structurer.table_structure": {
        "GeometryFallbackTable",
        "TableStructureDependencies",
        "build_table_claims",
        "classify_table_candidate_structure",
        "combine_native_and_fallback_tables",
        "count_nonempty_table_cells",
        "deduplicate_overlapping_table_candidates",
        "extract_geometry_table_data",
        "table_claim_confidence",
        "table_contains_complex_visual_content",
        "configure_table_container_claim",
        "drawing_is_table_container_structure",
        "merge_horizontal_rules_for_table_skeleton",
        "refine_table_claim_confidence",
        "table_cell_boundary_positions",
        "unique_rule_positions",
    },
    "nativePDF_structurer.table_detection": {
        "TablePipelineDependencies",
        "TablePreclusterDependencies",
        "build_page_table_candidates",
        "build_pre_table_drawing_clusters",
        "detect_geometry_fallback_tables",
        "detect_page_tables",
        "detect_simple_cluster_frame_tables",
        "detect_tables_by_drawing_clusters",
        "finalize_page_table_candidates",
    },
    "nativePDF_structurer.cross_detector_ownership": {
        "CrossDetectorOwnershipDependencies",
        "build_table_visual_composite_output_groups",
        "build_visual_text_block_keys",
        "evaluate_source_cluster_local_bridge",
        "finalize_table_visual_composite_output_groups",
        "resolve_table_visual_candidate_ownership",
        "restore_inactive_composite_table_outputs",
    },
    "nativePDF_structurer.pipeline": {
        "DocumentFinalizeDependencies",
        "DocumentPipelineDependencies",
        "PagePipelineDependencies",
        "PageStageAdapterDependencies",
        "apply_page_boundary_ownership",
        "detect_page_source_images",
        "detect_page_visual_regions",
        "initialize_page_context",
        "prepare_page_geometry",
        "prepare_page_rule_bands",
        "prepare_page_visual_protection",
        "process_page_header_footer",
        "process_page_reading_order",
        "process_page_semantics",
        "propose_page_output_candidates",
        "resolve_page_claims",
        "finalize_document_run",
        "process_document_pages",
        "process_page",
    },
    "nativePDF_structurer.layout_geometry": {
        "drawing_axis_rule",
        "extract_text_lines",
        "meaningful_overlap",
        "rect_is_mostly_inside_any",
    },
    "nativePDF_structurer.visual_processing": {
        "annotate_inventory_labels",
        "process_page_source_images",
        "process_page_visual_regions",
        "render_preclassified_rule_band_visuals",
    },
    "nativePDF_structurer.table_output": {
        "emit_table_candidate",
        "infer_table_rotation",
        "normalize_table_shape",
        "rotate_table_data",
        "save_rotated_table_image",
        "write_table_csv",
        "table_to_markdown",
    },
    "nativePDF_structurer.header_footer": {
        "build_header_footer_templates",
        "detect_page_header_footer_bands",
        "extract_header_footer_candidates",
        "extract_header_text_for_markdown",
        "filter_confirmed_header_footer_drawings",
        "header_footer_candidates_match",
        "merge_footer_drawing_candidates",
    },
    "nativePDF_structurer.formula_regions": {
        "formula_column_key",
        "extract_formula_line_records",
        "cluster_formula_candidates",
        "attach_equation_numbers_to_clusters",
        "process_page_formula_regions",
    },
    "nativePDF_structurer.figure_captions": {
        "classify_figure_caption_roles",
    },
    "nativePDF_structurer.outline_headings": {
        "assign_pdf_outline_entries_to_lines",
        "assign_pdf_outline_text_matches_to_lines",
        "normalize_heading_match_text",
        "strip_heading_number_prefix",
        "outline_text_matches",
        "outline_text_matches_for_audit",
    },
    "nativePDF_structurer.heading_detection": {
        "build_page_body_font_context",
        "build_document_typography_profile",
        "looks_like_toc_entry",
        "looks_like_numeric_or_rule_label",
        "looks_like_short_colon_label",
        "looks_like_heading_text",
    },
    "nativePDF_structurer.visual_assets": {
        "IMAGE_PADDING_POINTS",
        "MIN_IMAGE_THIN_SIDE_POINTS",
        "MIN_IMAGE_LINE_ASPECT_RATIO",
        "MIN_IMAGE_AREA_POINTS_SQUARED",
        "COMPLEX_TRANSFORM_EPSILON",
        "HEADER_FOOTER_IMAGE_HASH_SIZE",
        "VISUAL_REGION_HORIZONTAL_MERGE_PAGE_RATIO",
        "VISUAL_REGION_SAME_LAYER_HORIZONTAL_GAP_PAGE_RATIO",
        "VISUAL_REGION_SAME_LAYER_VERTICAL_OVERLAP_RATIO",
        "VISUAL_REGION_VERTICAL_MERGE_PAGE_RATIO",
        "MAX_HORIZONTAL_LINE_SLOPE_RATIO",
        "VISUAL_REGION_CROP_TEXT_OVERLAP_RATIO",
        "VISUAL_REGION_TEXT_OWNERSHIP_OVERLAP_RATIO",
        "SORT_BBOX_TOP_HEIGHT_POINTS",
        "build_top_sort_bbox",
        "build_visual_region_claims",
        "build_visual_region_output_record",
        "build_visual_regions_from_candidates",
        "clear_rect_on_visual_mask",
        "collect_visual_drawing_candidates",
        "detect_closed_visual_rule_containers",
        "detect_coarse_visual_regions",
        "drawing_extension_segments",
        "draw_rect_on_visual_mask",
        "extract_source_image_blocks",
        "format_visual_region_native_text",
        "image_content_signature",
        "image_has_complex_transform",
        "image_is_large_enough",
        "image_rects_overlap",
        "merge_pure_vector_same_layer_once",
        "merge_visual_region_records",
        "merge_visual_regions_by_horizontal_relations_once",
        "merge_visual_regions_sharing_images",
        "merged_closed_visual_container_ids",
        "merged_region_element_ids",
        "padded_image_clip",
        "point_to_rect_distance",
        "promote_right_attached_crop_text_lines",
        "promote_text_lines_at_region_external_connectors",
        "promote_text_lines_at_visual_endpoints",
        "render_visual_regions",
        "pure_vector_same_layer_should_merge",
        "segment_bridges_opposite_visual_sides",
        "segment_rect_intersection_interval",
        "text_bbox_overlaps_visual_crop",
        "text_is_owned_by_visual_structure",
        "collect_visual_region_terminal_endpoints",
        "collect_region_external_connector_endpoints",
        "visual_regions_cross_distinct_closed_containers",
        "visual_regions_should_merge",
        "visual_regions_should_merge_shared_image",
    },
}


def parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class ModuleBoundaryTests(unittest.TestCase):
    def test_extracted_modules_do_not_import_legacy_core(self) -> None:
        for module_name in EXTRACTED_SYMBOLS:
            module_path = PROJECT_ROOT / Path(*module_name.split("."))
            module_path = module_path.with_suffix(".py")
            tree = parse_file(module_path)
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            imported_modules.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            self.assertNotIn(
                "stage_c_claims_full",
                imported_modules,
                module_name,
            )



if __name__ == "__main__":
    unittest.main()
