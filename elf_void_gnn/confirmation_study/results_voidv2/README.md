# Confirmation study: all-descriptor GNNs on v2 void features

Same trainer, split and seeds (20261028–30) as `../results/`, on the dataset
with corrected void descriptors (`elfnet-gnn-2026-11-10-voidv2`). The
atom-only runs were not repeated: their void descriptors are zeroed, so
their inputs are identical on v1 and v2. `w64_d2_baseline` is new (the
original study only had the all-descriptor model at width 128). Test was
scored with the original frozen selection (w64 atom-only) carried over.

Mean over 3 seeds (pooled RMSE; macro = mean over held-out families):

| Model | Val RMSE | Val macro | Test RMSE | Test macro | Test R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| w64 atom-only (unchanged) | 0.0600 | 0.0615 | **0.0593** | **0.0638** | **0.822** |
| w64 all descriptors, v2 | **0.0580** | **0.0611** | 0.0636 | 0.0672 | 0.795 |
| w128 atom-only (unchanged) | 0.0636 | 0.0636 | 0.0603 | 0.0651 | 0.816 |
| w128 all descriptors, v2 | 0.0600 | 0.0581 | 0.0675 | 0.0683 | 0.769 |
| w128 all descriptors, v1 (old) | 0.0629 | 0.0947 | 0.0701 | 0.1004 | 0.751 |

- Fixing the void descriptors removed the per-family failures of the
  all-descriptor model (macro RMSE 0.095–0.100 → 0.058–0.068).
- With correct descriptors, all-descriptor models are now better on
  validation, but atom-only is still better on test, on every seed. Void
  descriptors help on the validation families and not on the test
  families; there is no consistent gain.

Runs are on Negishi under
`/home/USER/elfnet-runs/voidv2-2026-11-10/confirmation-study/`.
