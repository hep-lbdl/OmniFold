# Metadata Schema Design

## Motivation

OmniFold outputs are often distributed as event-level observables plus weight columns, but without a standardized description of generator setup, training context, normalization choices, or uncertainty semantics. This creates a reproducibility gap: users can apply weights, but they may not know which files support which uncertainty workflows or how to interpret variations consistently.

## Design Goals

The metadata schema is designed to provide:

- reproducibility: explicit file counts, observable definitions, and weight-family meaning,
- reuse: enough context for analysts who did not run the original OmniFold training,
- machine readability: structured fields that can be parsed by scripts and validation tools.

## Schema Structure

The schema uses the following top-level fields:

- `dataset`: human-readable dataset identity and scope.
- `generation`: nominal and alternative generator/sample context.
- `files`: nominal/systematic file mapping and event counts.
- `observables`: per-observable names, descriptions, and units.
- `weights`: nominal/base weights and replica-family prefixes.
- `systematics`: declared uncertainty families and combination guidance.
- `iterations`: optional step1/step2 iteration-weight semantics.
- `normalization`: luminosity/cross-section status and weight-normalization notes.
- `event_selection`: available selection documentation and missing pieces.
- `training`: OmniFold algorithm context and known/unknown training metadata.
- `usage_notes`: practical analysis guidance for downstream users.

## Design Decisions

YAML was chosen because it is:

- readable for physicists during review,
- easy to parse in Python workflows,
- naturally hierarchical for nested metadata (files, observables, weights).

A hierarchical schema was preferred over a flat table because OmniFold outputs mix conceptual levels: file-level provenance, column-level definitions, and analysis-level usage rules.

## What Was Included vs Excluded

Included metadata:

- observable list and units,
- event counts per file,
- nominal/systematic file mapping,
- weight families needed for nominal and uncertainty computations,
- explicit placeholders where information is currently unknown.

In particular, `normalization` and `event_selection` are intentionally marked as
`unknown` in `spec/metadata.yaml` because those details are not present in the
provided HDF5 files. Recording this absence is still useful: it tells
downstream users exactly which external documentation is required before a
fully reproducible physics result can be claimed.

Excluded metadata:

- full ML training logs (optimizer history, checkpoints, random seeds, per-epoch losses),
- full detector/reconstruction configuration payloads.

These were excluded because they are not present in the distributed files and would require external provenance capture systems.

## Expected User Workflow

A typical user workflow is:

1. Load an HDF5 file (`multifold.h5` or a systematic variation file).
2. Read `spec/metadata.yaml` to identify observables, nominal weights, and available variation families.
3. Build nominal histograms with `weights_nominal` (and base MC weight if required by analysis convention).
4. Build variation histograms using `weights_ensemble_*`, `weights_bootstrap_mc_*`, `weights_bootstrap_data_*`, and detector/theory families when available.
5. Propagate uncertainties only for families actually present in the selected file, documenting missing families as analysis limitations.

For packaged outputs, users should query systematics and iteration weights from
package metadata rather than scanning column names. This keeps the analysis
contract explicit and lets validation catch missing or incompatible columns.

## Possible Future Extensions

Potential extensions include:

- JSON Schema validation for strict automated checks,
- HEPData-compatible export helpers for publication pipelines,
- experiment-specific metadata overlays (ATLAS/CMS naming conventions, detector campaign tags),
- explicit uncertainty taxonomies linking each weight family to covariance-building recipes.
