# Canonical Weight Formula

## Final Weight

```text
w_final = w_mc * w_omnifold_final
```

## What Each Term Means

| Term | Column | Description |
|---|---|---|
| `w_mc` | `weight_mc` | MC generator/base event weight, present in all provided files |
| `w_omnifold_final` | `weights_nominal` | Learned final OmniFold reweighting factor |
| `w_final` | derived | Canonical downstream event weight for central-value observables |

## Nominal vs Iteration Weights

- **Nominal**: the final truth-level OmniFold weight after training is complete.
  This is the required publication object for central values.
- **Iteration weights**: optional intermediate weights, such as
  `weights_iter{N}_step{1,2}` if an analysis stores them. If present, they must
  declare both the iteration number and the step.

The provided HDF5 files expose `weight_mc` and `weights_nominal`, but do not
store explicit iteration or step columns.

## Downstream Usage

Until the reader API explicitly exposes a derived `w_final`, downstream users
should apply the formula directly:

```python
import numpy as np
from omnifold_publication import load_package

pkg = load_package("artifacts/zjets/")
df = pkg.load_events(columns=["pT_ll", "weight_mc", "weights_nominal"])

w_final = df["weight_mc"].to_numpy() * df["weights_nominal"].to_numpy()
hist, edges = np.histogram(df["pT_ll"], bins=30, weights=w_final)
```

Future API versions may expose this as:

```python
w_final = pkg.get_weights("final")
```
