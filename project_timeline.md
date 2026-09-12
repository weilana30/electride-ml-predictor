# Electride ML Predictor: Project Timeline

This project should proceed as a dependency-based sequence rather than a calendar schedule. Each stage should establish the information needed for the next stage.

## 1. Define the prediction problem

Decide what counts as an electride, whether the fundamental example is a crystal, a void, or a crystal–void pair, which electronic descriptors will be used, and how confirmed, predicted, ambiguous, and negative materials will be labeled.

The recommended basic unit is a **crystal–candidate-void pair**.

## 2. Design the dataset schema

Create a consistent format for storing:

- Structure files.
- Composition and pressure.
- Structure source.
- Polymorph and structural-prototype groups.
- Candidate-void coordinates.
- Geometric descriptors.
- Electronic descriptors.
- Electride confidence labels.
- Calculation-quality flags.

This schema should be fixed before large-scale data collection begins.

## 3. Collect known electride structures

Gather experimentally confirmed electrides, computationally predicted electrides, and examples spanning 0D, 1D, 2D, and 3D electride behavior. Include metallic, semiconducting, magnetic, nonmagnetic, polymorphic, and pressure-induced examples where available.

The objective is chemical and structural diversity rather than simply maximizing the number of structures.

## 4. Collect non-electride and difficult negative structures

Add chemically similar non-electrides, structures with large cavities but no interstitial electron, structures with high ELF caused by ordinary bonding, and structures with lone-pair density near—but not within—voids.

Unreported structures should not automatically be considered true negatives. Maintain ambiguous or unlabeled categories where appropriate.

## 5. Standardize and deduplicate structures

For every structure:

- Standardize the cell representation.
- Remove exact duplicates.
- Identify near-duplicates.
- Record composition and structural prototype.
- Group related structures for later train/test splitting.
- Check for missing or unreasonable structural data.

This stage reduces inconsistent inputs and prevents information leakage.

## 6. Develop candidate-void generation

Build a periodic geometric method, such as a Voronoi or Delaunay approach, to identify possible void sites.

The method should generate candidate positions, remove redundant sites, group symmetry-equivalent sites, calculate void size and shape, measure local coordination and bottlenecks, and determine void-network connectivity and dimensionality.

Before proceeding, verify that the method identifies sensible sites in known electrides.

## 7. Generate or collect electronic reference data

For selected structures, calculate or collect ELF, electron density, spin density where relevant, density of states, band gap or metallicity, non-nuclear attractors, interstitial basin charge, and electron-density connectivity.

Associate these electronic features with the candidate voids generated in the previous stage.

## 8. Establish the electride labeling protocol

Define void-level continuous targets and crystal-level electride confidence labels. Also define separate labels for localization dimensionality, magnetism, metallicity, and ambiguity.

The labeling rules should distinguish genuine interstitial density from ordinary bonding or lone-pair density. Avoid reducing the problem to an arbitrary binary label before understanding the underlying electronic descriptors.

## 9. Train simple baseline models

Train and compare:

- Composition-only models.
- Global structure-only models.
- Void geometry-only models.
- Combined geometry and composition models.

Start with interpretable methods such as gradient-boosted trees or small neural networks. These baselines establish whether the targets are learnable and how much information geometry and chemistry contribute independently.

## 10. Build the structure-only model

Use atomic positions, lattice vectors, candidate-void locations, and local geometry, while excluding element identities.

The model should predict whether a void is spatially reasonable for hosting an electron. Begin with a void-centered descriptor model, then consider an atom–void graph or equivariant model if the data support the additional complexity.

## 11. Build the chemical/electronic model

Add element identities and chemical information, including valence, electronegativity, ionization energy, electron affinity, atomic and ionic radii, electron-count features, local bonding, coordination, and global composition.

This model should predict whether the surrounding chemistry can donate, stabilize, and delocalize an electron in the candidate void.

## 12. Compare the two models

Analyze structures where:

- Both models predict high likelihood.
- Geometry is favorable but chemistry is unfavorable.
- Chemistry is favorable but geometry is unfavorable.
- Both models fail.

These comparisons will show whether the two-model decomposition captures a meaningful physical distinction.

## 13. Combine the models

Initially use a cascade:

```text
Structure-only model
        ↓
Geometrically plausible voids
        ↓
Chemical/electronic model
        ↓
Final electride ranking
```

After establishing the cascade, test a joint multitask model in which the chemical branch receives the geometric representation or score from the structure branch.

Keep the geometry and chemistry outputs separate so that predictions remain interpretable.

## 14. Validate on unseen chemistry

Use test sets separated by composition, chemical family, structural prototype, polymorph group, and pressure regime.

Evaluate recall of known electrides, precision among top-ranked candidates, false positives from ordinary bonding, calibration, uncertainty, and performance on ambiguous examples.

## 15. Expand through active learning

Use the initial models to select structures for additional electronic-structure calculations. Prioritize candidates with high uncertainty, disagreement between the two models, unusual chemistry, or a mismatch between large geometric voids and weak electronic evidence.

Add the new calculations to the dataset, retrain the models, and repeat the cycle.

## Overall sequence

```text
Define labels
   ↓
Design dataset
   ↓
Collect structures
   ↓
Generate voids
   ↓
Generate electronic targets
   ↓
Train baselines
   ↓
Build structure-only model
   ↓
Build chemical/electronic model
   ↓
Combine models
   ↓
Validate on unseen chemistry
   ↓
Active learning
```

The first major milestone should be a validated crystal–void dataset with reliable labels and baseline performance. Complex neural architectures should follow only after that foundation is established.
