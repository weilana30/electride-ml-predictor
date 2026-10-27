# ELF-center feature study plan

## Question and rationale

Measure which input information helps predict ELF at a detected void center: atom chemistry, void-local geometry, and geometric edge distances. This is a controlled feature study of the registered `hetero_mpnn`, not a new architecture benchmark. Crystal GNNs commonly represent atoms and bonds together (CGCNN); distance is a standard edge signal, while ALIGNN demonstrates that angle/local geometry can add predictive information. Here the void node already carries local geometric descriptors, so the study asks whether those void descriptors add value alongside atom chemistry and distance messages. The feature study should report predictive utility, not imply causal or physically unique importance.

Primary references:

- Xie & Grossman, *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*, PRL (2018): [paper](https://doi.org/10.1103/PhysRevLett.120.145301), [arXiv](https://arxiv.org/abs/1710.10324).
- Choudhary & DeCost, *Atomistic Line Graph Neural Network for Improved Materials Property Predictions*, npj Computational Materials (2021): [paper](https://doi.org/10.1038/s41524-021-00650-1), [arXiv](https://arxiv.org/abs/2106.01829). It explicitly models bond-angle information through line-graph message passing; this motivates testing the existing local-geometry features, without adding a line graph here.
- Chen et al., *A universal graph deep learning interatomic potential for the periodic table* (M3GNet), Nature Computational Science (2022): [paper](https://doi.org/10.1038/s43588-022-00349-3). Atomic-number embeddings and distance-expanded edges are established minimal structural inputs.
- Xie & Grossman also demonstrate atom-level chemical interpretation in CGCNN; this supports treating feature attribution as interpretation rather than a substitute for controlled retraining. The primary study here is the prespecified held-out ablation comparison.

## Fixed data and protocol

Use the already validated dataset and its existing `splits.parquet` assignment, loaded through `read_split_table`; do not regenerate, rebalance, or filter the split per variant. Preserve the family-safe assignment and its stated limitation: exact named phase folders are held out, while related composition/pressure variants may cross splits. All variants see identical graph topology, void targets, sample order, target weighting, optimizer, seed, batch size, early stopping, and maximum epoch budget from `elf-center-v1.yaml`. Keep the target `elf_at_center` out of every input and statistic.

Fit input/target scaling and feature correlations on training rows only. Never use validation or test distributions to define scaling, feature selection, thresholds, or correlations. Select each run's checkpoint by validation pooled RMSE, with the existing early-stopping rule. Validation metrics may rank feature variants; test is sealed until the study design and variant selection are frozen. Then evaluate each member of the frozen shortlist (full, minimal, and up to one distinct candidate) once on test, and report these plus the baseline. No repeated test inspection or feature redesign after seeing test results.

## Preserve the exact architecture and parameter count

Keep the current two message layers, hidden width 64, 8-channel 0–6 Å radial expansion, first-layer four relation-specific message MLPs, and second-layer atom→void and void→void message MLPs, atom/void self updates, LayerNorm, dropout 0.1, and the same prediction head. Use the study adapter's fixed 14 atom channels, 9 void channels, and 8 edge channels; richer groups replace fixed slot values rather than widening any encoder or message MLP. Keep every relation, edge index, and edge-distance tensor present. A masked node feature is zero in normalized coordinates (the training mean in raw coordinates); validity slots are masked with the associated descriptor. The baseline edge context is the current RBF representation; rich edge context replaces those 8 channels with an 8-channel fixed-width encoding of the selected geometric edge fields. Thus all variants instantiate identical parameterized modules, tensor shapes, and exactly 80,001 trainable parameters. Record/assert the parameter count and module names for every variant. Do not add a branch, embedding table, or parameterized feature-specific projection.

## Feature groups and variants

The validated Negishi schema maps to fixed encoder slots as follows. For atom inputs the five validity flags are appended after the nine values, giving the existing 14-channel encoder input. Void inputs already have nine columns. Edge channels in the baseline are the existing eight Gaussian RBF values of each relation's stored distance (fixed centers spanning 0–6 Å). This is the exact slot map; variants mask/reuse these slots and never change input widths.

| Type | Slot | Schema column / meaning | Minimal variant |
|---|---:|---|---|
| atom | 0 | `atomic_number` (Z) | retain |
| atom | 1 | `electronegativity` | mask |
| atom | 2 | `atomic_radius_angstrom` | mask |
| atom | 3 | `period` | mask |
| atom | 4 | `group` | mask |
| atom | 5 | `is_s_block` | mask |
| atom | 6 | `is_p_block` | mask |
| atom | 7 | `is_d_block` | mask |
| atom | 8 | `is_f_block` | mask |
| atom | 9 | `electronegativity_valid` | mask with slot 1 |
| atom | 10 | `atomic_radius_valid` | mask with slot 2 |
| atom | 11 | `period_valid` | mask with slot 3 |
| atom | 12 | `group_valid` | mask with slot 4 |
| atom | 13 | `block_valid` | mask with slots 5–8 |
| void | 0 | `clearance_angstrom` | mask |
| void | 1 | `nearest_atom_distance_angstrom` | mask (duplicate of slot 5) |
| void | 2 | `atom_count_within_3A` | mask |
| void | 3 | `atom_count_within_4A` | mask |
| void | 4 | `atom_count_within_5A` | mask |
| void | 5 | `nearest_1_distance_angstrom` | retain |
| void | 6 | `nearest_2_distance_angstrom` | retain |
| void | 7 | `nearest_3_distance_angstrom` | retain |
| void | 8 | `nearest_4_distance_angstrom` | retain |
| edge | 0–7 | Gaussian RBF of stored relation distance, eight fixed centers from 0 to 6 Å | retain |

Minimal retains Z only in the atom input, void slots 5–8 only, and the eight RBF channels for every relation. Note that void slot 1 repeats the nearest-atom distance represented by slot 5; it is excluded from the minimal variant to avoid duplicate encoding. Atom block validity is shared by the four block-indicator slots. Define groups from this mapping and retain/mask validity flags with their associated descriptor:

1. **Full reference:** all atom descriptors/validity flags, all nine void descriptors, and all distance signals.
2. **Required minimal:** atom atomic number only; void nearest-atom distances only; edge distances only. All other slots are masked. Do not replace the underlying graph or recompute distances. This is the minimal-information anchor.
3. **Z-only atoms, full void:** remove all atom descriptors except atomic number; retain full void geometry and edges.
4. **No atom descriptors:** mask all 14 atom channels; retain full void geometry and edges.
5. **Void geometry ablation:** remove all void descriptors; retain full atom features and edges.
6. **Edge-distance ablation:** remove distance signal for every relation (atom–atom, void–atom/atom–void, void–void); retain nodes and topology.
7. **Relation-specific edge ablations:** independently mask atom–atom distance, void–atom/atom–void distance (as one paired relation), and void–void distance; use full node features.
8. **Atom descriptor group ablations:** one at a time, mask radius; mask electronegativity; mask period/group; mask block indicators, retaining Z and other groups. State mappings before running.
9. **Void descriptor group ablations:** one at a time mask nearest-neighbor distances (slots 5–8); coordination/count descriptors (slots 2–4); or clearance/nearest-distance descriptors (slots 0–1). If a group is redundant or a derived feature rather than schema input, call that out explicitly.

Run grouped ablations first, then only the interactions needed to interpret them: (a) atom group × void group; (b) atom group × all edge distances; (c) void-distance group × void–atom edges. For these, use the 2×2 cells (both present, each absent separately, both absent) to distinguish additive loss from interaction. Define interaction as the difference-in-differences in validation RMSE relative to the full reference; use paired bootstrap over family IDs for uncertainty. No exhaustive search across individual columns. The confirmatory set is limited to schema-defined existing inputs and these prespecified groups/interactions. Any newly derived void features (for example, shape ratios or angle summaries not already stored) are exploratory follow-up only: specify formulas in advance, fit no data-derived thresholds outside train, retain the same fixed channels/parameter count, and do not mix those results into confirmatory conclusions.

## Train-only correlation and complementary interpretation

On training rows only, compute Pearson and Spearman correlation matrices for numeric input columns and target, both pooled and summarized by sample first (avoid letting samples with many voids dominate); report missingness/validity rates alongside correlations. For each correlated group, note redundancy (for example, high absolute pairwise correlation) and interpret ablation jointly rather than claiming a low single-feature score means irrelevance. Correlation is descriptive and does not remove/merge features in this confirmatory run. After model selection, optionally calculate grouped input permutation importance on validation predictions, permuting within sample/family where feasible; report it as a stability check, not as a model-selection loop. Do not use test for attribution.

## Compute budget and run order

Use the baseline seed/config and existing 30-epoch cap, batch size 8, patience 5. A previous full fit took about 18m41s wall time; use approximately 3.1 GPU-hours per 10 fits and 8.1 GPU-hours for the 26-fit ceiling, before setup/analysis overhead. Stage A: one seed for full reference, required minimal, Z-only atoms/full void, no atom descriptors, no void descriptors, no-edge-distance, and three relation-specific edge ablations: 9 fits (~2.8 GPU-hours). Go to Stage B only if full reference reproduces its expected validation behavior and all runs pass shape/parameter-count checks; otherwise diagnose adapter/data errors and stop for review. Stage B: schema-supported atom and void subgroups plus the three prespecified 2×2 interaction grids; skip cells already run in Stage A: maximum 11 additional fits (~3.4 GPU-hours). Freeze a validation shortlist of at most three candidates (full, required minimal, and the best distinct ablation/interaction candidate). Stage C: rerun those candidates with two additional fixed seeds (6 fits; ~1.9 GPU-hours). Total ceiling 26 fits × 30 epochs = 780 epochs; early stopping reduces expected work. Log elapsed GPU time after each stage and stop at the 8.1 GPU-hour fit budget, reporting unrun candidates. After the shortlist and all choices are frozen, evaluate each shortlisted candidate once on test, without further tuning. The study is authorized to proceed through the staged runs; budget limits are stop/go criteria, not a request for additional approval.

## Compact deliverables

- `run_manifest.csv`: variant, feature mask, seed, parameter count, best epoch, validation metrics, wall time, and run directory.
- `feature_correlations.csv`: training-only correlation and missingness summary.
- `ablation_summary.csv`: validation pooled/per-family MAE, RMSE, R², family-bootstrap intervals, and delta from full reference; after freeze, the same test columns only for selected candidates.
- One compact horizontal plot of validation RMSE deltas from full, ordered by feature group, with family-bootstrap intervals; distinguish primary ablations and interactions by marker/color.
- One small heatmap for the three interaction grids and one training-only correlation heatmap (group-aggregated or top correlated pairs to stay legible).
- A short report with schema-to-group mapping, identical-parameter assertion, protocol/split caveat, selected model, test-once results, and limitations. Include a table comparing full, minimal, and selected model against the training-mean baseline.
