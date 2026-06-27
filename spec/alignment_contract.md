# Event Alignment Contract

## Scope

This contract defines how event rows, observable columns, and weight columns are
associated within one OmniFold publication sample.

The key words MUST, MUST NOT, SHOULD, and MAY are used in their normative sense.

## Row-Order Alignment

The default alignment method is row-order alignment. For a row-order-aligned
sample, row `i` in the event table corresponds to row `i` in every declared
observable column and every declared weight column.

No explicit `event_id` column is required when row-order alignment is declared.

After a package is written, the row order is immutable. Producers MUST NOT
reorder, filter, truncate, shuffle, or independently transform event or weight
columns unless the same operation is applied to the complete event table and the
metadata is regenerated. Consumers are expected to preserve row correspondence
when applying weights to observables.

Analysis-level selections are allowed after loading, provided the same selection
is applied consistently to the observables and weights being used.

## Event Count Rule

The declared `event_count` in `metadata.yaml` MUST exactly match the number of
rows in the event table. Every declared observable and weight column MUST have
exactly `event_count` entries.

Validation fails if the declared event count differs from the stored row count,
or if any declared observable or weight column is missing.

For published artifacts, metadata SHOULD include a checksum or hash for the
event table. Validators SHOULD verify the checksum when it is present.

## Event IDs

The provided HDF5 files do not contain an explicit `event_id` column, so the
current standard permits row-order alignment without event identifiers.

If an analysis provides an `event_id` column and declares column-based
alignment, the metadata MUST name that column. Validators check that the column
Validators MUST check that the column is present, non-null, and unique within the sample.
Duplicate event IDs cause
validation failure.

If `event_id` is absent and row-order alignment is declared, validators MUST NOT
reject the package solely because no event ID is present.

## Aligned Weight Variations

Event-aligned variations, such as replica, bootstrap, or systematic weight
columns, MAY be stored in the same event table as the nominal weights only when
row `i` refers to the same physical event for all declared variations.

Validators MUST check that each declared aligned variation column is present and
has the same row count as the nominal event table.

Implementation note: for columnar formats such as Parquet or HDF5, validation
can check row alignment by comparing the stored row count with `event_count` and
verifying that each declared observable and weight column exists in the same
table.

## Non-Aligned Alternative Samples

Alternative samples, such as different generators or background samples, are not
assumed to be row-aligned with the nominal sample. Examples include Sherpa and
nonDY samples.

Non-aligned samples MUST NOT be stored as additional columns in the nominal
event table. In the current package layout, they MUST be stored as separate
package directories, each with its own event table, `metadata.yaml`, event
count, and alignment contract.

Example layout:

```text
artifacts/
  zjets_nominal/        ← multifold.h5
    events.parquet
    metadata.yaml
  zjets_sherpa/         ← multifold_sherpa.h5
    events.parquet
    metadata.yaml
  zjets_nonDY/          ← multifold_nonDY.h5
    events.parquet
    metadata.yaml
```

Future multi-sample package formats MAY reference several such sample packages
from a higher-level manifest, but MUST preserve the per-sample alignment
contract defined here.
