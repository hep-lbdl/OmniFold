# Gap Analysis of OmniFold Weight Files

This gap analysis answers three core evaluation questions:

1. What is contained in each file?
2. How are columns structured and how do files differ?
3. What missing metadata blocks reproducible reuse?

## 1. File Overview

The repository contains three OmniFold-related HDF5 outputs:

- `multifold.h5`: nominal OmniFold output with the most complete set of weights.
- `multifold_sherpa.h5`: alternative-generator variation (Sherpa) with reduced weight content.
- `multifold_nonDY.h5`: non-DY alternative sample with minimal weight content.

Together, these represent a nominal result plus systematic/alternative modeling variations. The central practical difference is that weight completeness is highly asymmetric across files.

## 2. Column Categories

Columns can be grouped into three categories.

### Observables

Observable columns encode physics quantities used for distributions and unfolding studies. They are event-level kinematic variables.

Examples include:

- `pT_ll`
- `eta_ll` (common naming style in HEP tables)
- `lepton_pt1`, `lepton_pt2` (equivalent in this dataset to `pT_l1`, `pT_l2`)

In this dataset, observables such as `pT_ll`, `pT_l1`, `pT_l2`, jet-track features, and multiplicities define the phase-space where weighted comparisons are performed.

### Weight Columns

Weight columns contain OmniFold reweighting factors and uncertainty replicas applied to events.

Examples:

- `weights_nominal`
- `weights_bootstrap_mc_*`
- `weights_ensemble_*`

These weights encode corrections that transform simulated event distributions toward the target data distribution and support uncertainty propagation through replica ensembles.

### Metadata Columns

Metadata columns carry event identifiers, labels, or auxiliary attributes rather than primary observables.

Examples:

- `event_id` (common identifier field in HEP ntuples; not confirmed in provided files, listed as a common example)
- `target_dd`
- sample labels

In the current files, `target_dd` is the clearest metadata-like field. It appears to be a data-driven target/label used during OmniFold-related training, but its semantic definition is not provided in-file.

## 3. Comparison Across Files

The three files differ substantially in weight information:

- **Concrete column summary**

| File                  |  Events | Observable Columns | Weight Columns | Metadata Columns |
| --------------------- | ------: | -----------------: | -------------: | ---------------- |
| `multifold.h5`        | 418,014 |                 24 |           ~175 | `target_dd`      |
| `multifold_sherpa.h5` | 326,430 |                 24 |            ~27 | none             |
| `multifold_nonDY.h5`  | 433,397 |                 24 |              2 | none             |

Observable columns are identical across all files. The weight column counts
differ significantly, with only `multifold.h5` providing full uncertainty
replica families.

- `multifold.h5` has the richest set (nominal, ensemble replicas, MC/data bootstraps, and many detector/theory weights).
- `multifold_sherpa.h5` keeps a smaller subset (nominal plus MC bootstrap replicas).
- `multifold_nonDY.h5` contains only `weight_mc` and `weights_nominal`.

These differences are consistent with different generators/alternative samples and different systematic packaging choices. They also imply that some files cannot support full uncertainty workflows on their own because required weight replicas are missing.

## 4. Missing Information Needed for Reuse

For publication-grade reproducibility, the following metadata are still required:

- generator configuration: needed to reproduce nominal and variation samples.
- OmniFold training parameters: needed to understand optimization behavior.
- number of training iterations: affects convergence and final weights.
- neural network architecture: controls model capacity and potential bias.
- event selection criteria: defines the fiducial phase-space being unfolded.
- luminosity normalization: required for absolute-yield interpretation.
- observable units: required for correct plotting and interpretation.
- recommended binning: needed for consistent cross-analysis comparisons.
- uncertainty interpretation: clarifies how each weight family should be propagated.

Without these, downstream users can compute weighted plots but cannot reliably reproduce full analysis-level results.

## 5. Challenges for Standardization

Standardizing OmniFold outputs across experiments is difficult because of:

- inconsistent column naming conventions,
- very large numbers of weight replicas,
- large file sizes and I/O costs,
- experiment-specific detector simulations,
- differing event selections and fiducial definitions,
- different machine-learning training procedures and hyperparameters.

These factors make a common metadata layer essential for interoperability.

## 6. Summary

The files are useful but not self-describing enough for robust reuse across analyses. A structured metadata schema is needed to define observables, weight families, normalization, training context, and uncertainty semantics so OmniFold outputs remain reproducible and comparable.

## Summary Table

| File                | Events  | Weight Columns | Key Finding                       |
| ------------------- | ------- | -------------- | --------------------------------- |
| multifold.h5        | 418,014 | 175            | Full replica families present     |
| multifold_sherpa.h5 | 326,430 | 27             | Reduced subset only               |
| multifold_nonDY.h5  | 433,397 | 2              | Nominal only, no uncertainty info |

**Key conclusion:** Weight structure is asymmetric across files. Fixed column
prefix assumptions cannot work. A metadata-driven taxonomy is required.

## 7. Open Questions

1. Should iteration count be a fixed default (e.g. 4) or always
   analysis-specific metadata?

2. Should replica/bootstrap weights be supported inside systematic
   samples, or only in the nominal file?

3. What is `target_dd` — training target, classifier score, or
   analysis-specific auxiliary field?

4. Should the schema require explicit `event_id`, or is row-order
   alignment sufficient?

5. Should the schema anticipate additional files beyond the three
   provided (e.g. per-iteration weights, model checkpoints)?
