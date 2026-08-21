# Engineering architecture

[简体中文](architecture.md) | **English**

NativePDF Structurer is a zero-OCR, zero-model structurer for digital-native
PDFs. Its output is more than a Markdown conversion:

```text
Markdown + referenced assets + structured IR + provenance
```

The parser uses the PDF's native text, font, image, drawing, and coordinate
layers. It does not require OCR, a GPU, or an external language model.

## Repository structure

The following tree shows the main contents of the published repository. Related
modules are grouped on one line to avoid repeating the detailed module boundary
list below, and individual test files are not expanded:

```text
nativePDF-structurer/
├── README.md / README_EN.md          # Primary Chinese and English guides
├── pyproject.toml                    # Package metadata, dependencies, CLI entry
├── LICENSE / THIRD_PARTY_NOTICES.md  # Project and third-party license boundaries
├── environment.yml / requirements.txt
│                                      # Conda environment and validated versions
├── nativePDF_structurer/             # Main Python package
│   ├── __init__.py / __main__.py / cli.py
│   │                                  # Public API, module and CLI entry points
│   ├── config.py / run_state.py / page_context.py
│   │                                  # Configuration, document and page state
│   ├── pipeline.py                    # Document and page-stage orchestration
│   ├── inventory.py / claims.py
│   ├── output_ownership.py / text_ownership.py
│   ├── cross_detector_ownership.py    # Inventory, Claims and detector ownership
│   ├── table_detection.py / table_structure.py
│   ├── table_skeleton.py / table_output.py
│   │                                  # Table detection, structure and output
│   ├── visual_assets.py / visual_processing.py
│   │                                  # Source images, vectors and attachments
│   ├── layout_geometry.py / geometry.py
│   ├── rule_geometry.py / rule_bands.py / rule_preclassification.py
│   │                                  # Layout geometry and rule preprocessing
│   ├── reading_order_backend.py       # Spatial ordering of page atoms
│   ├── header_footer.py / heading_detection.py
│   ├── outline_headings.py / figure_captions.py / formula_regions.py
│   │                                  # Page edges, headings, captions, formulas
│   ├── page_reflow.py / markdown_formatter.py
│   ├── artifacts.py / emitter.py / ir.py / result.py
│   │                                  # Markdown, assets, IR and final results
│   ├── page_contract.py / debug_reports.py / regression_manifest.py
│   │                                  # Debug contracts, audits and regression
│   └── _vendor/
│       ├── gap_tree.py                # Bundled reading-order foundation
│       └── GapTree_LICENSE.txt        # Corresponding MIT license
├── docs/
│   ├── architecture.md / architecture_EN.md
│   │                                  # Chinese and English architecture guides
│   └── assets/examples/               # Compact README output examples
├── tests/                              # Generic unit and contract tests
└── .github/workflows/tests.yml         # GitHub Actions test workflow
```

This tree answers where major code areas live. The sections below remain the
source of truth for algorithm boundaries, state ownership, and module duties.

## Runtime path

```text
CLI / RunConfig
    -> pipeline
    -> document run state
    -> pipeline / one PageRunContext per page
    -> detector and ownership stage callbacks
    -> page commit
    -> ParseResult / ArtifactEmitter
```

The runtime is package-native. Development checkpoints and local regression
assets are excluded from the published source tree; no runtime path imports a
single-file compatibility parser.

## State ownership

| Object | Owns | Must not own |
|---|---|---|
| `ParseRunContext` | validated document options and selected one-based pages | mutable page objects |
| `DocumentRunState` | open document, output layout, shared caches, metrics, outline and typography state | candidate ownership for another page |
| `PageRunContext` | raw/working page objects and all intermediate page results | document-wide result lists |
| `ParseResult` | completed pages, Markdown sections, IR inputs and provenance collections | detector scratch state |

`PageRunContext.raw_drawings` is the untouched page inventory. `drawings` is
the working view after confirmed header/footer vector filtering. Keeping both
prevents later detectors from confusing filtered objects with the source PDF.

## Page stages

Each page records the following stage history in debug mode:

```text
initialization
  -> header_footer
  -> geometry
  -> rule_bands
  -> table_detection
  -> table_candidates
  -> table_finalization
  -> visual_protection
  -> visual_detection
  -> output_proposal
  -> source_images
  -> claim_resolution
  -> boundary_ownership
  -> semantics
  -> reading_order
  -> finalization
```

The stage names are defined once in `nativePDF_structurer.page_contract.PageStage`.
The debug contract reports a mismatch if a stage is skipped, duplicated, or
reordered.

## Extracted modules

The first stable module boundaries are deliberately small:

- `claims.py`: Claim manifest normalization and detector-priority policy.
- `geometry.py`: shared rectangle, center, union, and drawing-item primitives.
- `inventory.py`: stable page element IDs, logical containers, lazy drawing-item
  expansion, and Claim conflict resolution.
- `output_ownership.py`: post-resolution output survival and parent/child
  visual suppression.
- `text_ownership.py`: Stage-F native text assignment to surviving outputs.
- `reading_order_backend.py`: the replaceable reading-order seam, virtual-upright
  bbox/position transforms, backend-independent same-row narrow-column
  attachment, cross-column leading-row repair, and horizontal text fragment
  coalescing primitives. It also owns the second-stage horizontal XY-Cut that
  separates a visual atom from a short lower text band while preserving
  composite visual groups, plus the final explicit separator-before-heading
  precedence step. The same module sorts native text retained inside visual
  regions, source images, and preclassified rule bands; `pipeline.py` binds the
  current row tolerance before passing that sorter through the existing visual
  callback contract. Detector stages create the attachment metadata; the
  reading-order backend only applies that already-declared local relationship.
- `table_detection.py`: the coarse-grained table detection theme. It owns the
  small-mask precluster, native/geometry/fallback/local-frame detectors,
  candidate deduplication, page-level table orchestration, structural
  candidate construction, and table Claim preparation. Its thresholds and
  parser callbacks arrive through explicit dependency records; it does not
  read parser globals or own document/page state.
- `table_structure.py`: the table structure theme. It owns candidate policy,
  Claim construction, geometry-only cell extraction, native/fallback table
  compatibility objects and merging, rule-position grouping, native-cell
  boundary checks, parent-container Claim setup, and table-local evidence
  auditing. Complex internal drawings are not penalized here; a
  full-screenshot table can claim its bbox content while a parent-container
  table leaves child visual content available to later ownership stages.
- `table_skeleton.py`: the intentionally separate, performance-sensitive
  large-cluster axis-skeleton algorithm. Keeping its 524-line topology and
  budget-selection logic isolated makes profiling and future table-geometry
  changes safer without recreating the previous collection of tiny modules.
- `table_output.py`: CSV, Markdown, rotation, PNG rendering, and table output
  artifact helpers. Output serialization remains separate from detection and
  ownership decisions.
- `cross_detector_ownership.py`: cross-detector overlap decisions, visual
  text-block keys, source-cluster bridge evidence, table/visual composite output
  groups, stage-D composite finalization, and independent table-output restore
  fallback. It coordinates table, visual-region, and table-output records but
  does not transfer raw element ownership or resolve inventory conflicts.
- `pipeline.py`: the single editable orchestration boundary. It contains the
  page-stage sequence, concrete stage adapters, configured-page loop, and final
  summary/IR/ArtifactEmitter handling. Keeping these closely related concerns
  together makes later algorithm changes local and avoids a large callback
  graph spread across several tiny modules.
- `table_output.py`: table CSV serialization, pure-text Markdown grid
  serialization, character-weighted orientation inference, rectangular data
  normalization, CSV matrix rotation, lossless right-angle PNG rotation, and
  PNG/Markdown table atom emission. The direct-Markdown branch remains opt-in
  for pure-text tables; complex visual tables continue through the screenshot
  branch. Rotation thresholds are supplied explicitly by the compatibility
  core rather than duplicated in the output module.
- `visual_assets.py`: shared source-image input primitives, image-block
  normalization, size/transform filtering, crop padding, image-content
  signatures, vector candidate filtering, mask coordinate helpers, closed
  visual-container detection, OpenCV initial-region construction, and the
  vector/source-image candidate merge layer. It also owns the geometric text
  attachment predicates, one-hop connector promotion helpers, and native
  visual-text Markdown formatting. It also owns the visual output atom
  builder for `render_bbox`, `sort_bbox`, Markdown links, and compact records;
  visual-region Claim manifests are built from the same stable IDs. Page
  screenshots and coarse-region orchestration are now executed by the same
  module. Cross-detector ownership remains above this boundary.
- `header_footer.py`: local header/footer candidate extraction, adjacent footer
  drawing composition, deterministic hard matching, and document-level sample
  aggregation into independent header/footer templates. It also performs
  per-page template matching, exact confirmed-drawing filtering, and optional
  native header-text extraction for Markdown retention.
- `formula_regions.py`: formula-line extraction from native font/symbol
  evidence, double-column classification, vertical candidate clustering, and
  equation-number attachment. It also performs candidate acceptance, page
  rendering, Markdown atom construction, and precise formula-text ownership.
- `figure_captions.py`: assigns external figure-number lines to final visual
  outputs using label structure, vertical adjacency, competing-text exclusion,
  and one-caption-per-visual ownership.
- `page_contract.py`: debug-only stage, Claim, bbox, and artifact consistency
  checks.
- `page_commit.py` / `page_finalize.py`: transfer of completed page state to
  document-level results.
- `artifacts.py` / `emitter.py` / `ir.py`: output layout, debug artifacts,
  structured intermediate representation, and provenance.
- `markdown_formatter.py`: converts already ordered page atoms into Markdown;
  it owns heading, figure-caption, non-text-atom, text-buffer, and thematic
  separator rendering without reading PDF geometry or detector state.
- `page_reflow.py`: filters already classified text lines, groups them by
  direction, invokes injected reading-order and geometry predicates, and
  appends stable directional Markdown sections. It does not own PDF parsing or
  detector decisions.
- `rule_geometry.py`: converts raw drawing path items into lightweight axis
  segments, merges exact collinear fragments, builds page-level horizontal and
  vertical logical-rule views, and writes the compatibility audit mapping.
  Thickness, coordinate precision, page-width gates, and the legacy drawing
  classifier are injected by the core; table and visual ownership remains in
  downstream stages.
- `rule_bands.py` / `rule_preclassification.py`: detect visual framework
  bands and pre-route paired horizontal-rule bands before table detection.
  Their geometric evidence and thresholds are injected by the compatibility
  core; the modules do not perform OCR or page reads.
- `debug_reports.py`: writes the human-readable debug reports for visual
  regions, source images, formulas, Claim conflicts, logical rules, headings,
  captions, and header/footer templates. It consumes records only and is
  inactive in minimal output mode.
- `layout_geometry.py`: coarse-grained native text extraction, PDF outline page
  naming, rule-line topology, text separators, rectangle overlap, spatial
  indexes, and precluster object relations.
- `visual_processing.py`: source-image handling, vector endpoint/marker rules,
  visual-region orchestration, preclassified visual-band rendering, and Stage
  E/F visual/source-image boundary rebuilding.

Page and document stage order, stage adapters, and final output now run through
`pipeline.py`. It also constructs document-wide resources and assembles the
explicit dependency objects consumed by detector modules. No historical
single-file compatibility core participates in the runtime path.
The visual asset boundary now includes image input, vector candidate filtering,
mask primitives, closed-container detection, initial connected-component
construction, structural candidate merging, and geometric text attachment.
It also formats already-owned native visual text. Cross-detector output
relations are handled by `cross_detector_ownership.py`; screenshot, visual
output records, visual Claim construction, and coarse visual-region
orchestration are part of the visual-processing boundary.

## Claim and ownership model

Every detector Claim distinguishes three ID sets:

- `input_element_ids`: objects the detector examined;
- `claimed_element_ids`: objects the candidate wants to own;
- `feature_element_ids`: objects whose loss invalidates the candidate.

`inventory.py` resolves structural conflicts by detector priority, confidence,
feature count, and stable Claim ID. Text ownership is delayed until structural
boundaries are final. This prevents a text line from being simultaneously
assigned by table, visual, and image detectors.

Output atoms carry two spatial boxes:

- `render_bbox`: the complete screenshot/crop range;
- `sort_bbox`: the reading-order anchor, normally a narrow top strip.

The two boxes must not be collapsed into one field during future changes.

## Debug contracts

Debug mode checks, without changing normal output:

- exact stage order;
- Claim ID existence, uniqueness, and feature/claimed consistency;
- inventory owner references;
- valid, non-reversed, in-page `bbox`, `render_bbox`, and `sort_bbox`;
- Markdown output files, provenance records, and output paths;
- duplicate active output paths.

Minimal mode writes only the normal Markdown and referenced assets. Debug JSON,
reports, stage history, and contract warnings are opt-in.

## Safe extension procedure

When changing a detector or ownership rule:

1. Keep stable element IDs and Claim fields unchanged unless the schema change
   is intentional.
2. Change the detector/module policy, not the page commit or emitter layer.
3. Preserve `render_bbox` versus `sort_bbox` semantics.
4. Add a focused unit test for the new geometry or ownership case.
5. Run the debug contract on focused pages that exercise the changed geometry.
6. Run focused manifests, then the configured full-range manifests.
7. Save a checkpoint before handing the output to visual inspection.

For a local regression run, use the generic manifest command:

```bash
python -m nativePDF_structurer.regression_manifest verify \
  /path/to/parser-output /path/to/manifest.json
```

Project-specific PDFs, page selections, manifests, and visual issue records
belong in a private local regression directory rather than the published
source tree. Byte-level manifests verify the output contract; visual review
is still required because equal hashes do not prove that a page layout is
semantically preferable.

## Reading-order backend

The package exposes one narrow reading-order backend contract. It receives page
atoms in virtual upright coordinates and returns ordered atoms plus layout
metadata:

```text
page atoms -> ordered atoms + layout metadata
```

Any alternative implementation must preserve Markdown, non-text asset hashes,
element ordering, and page-level provenance before it can enter the published
package.
